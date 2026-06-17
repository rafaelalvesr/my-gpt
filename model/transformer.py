import numpy as np
import json
from .embedding import PositionalEncoding, RoPe
from train import TensorValue, no_grad

#https://github.com/stanford-cs336/assignment1-basics/blob/main/tests/adapters.py
#https://github.com/liaoyanqing666/Decoder-only-transformer_Time_Series_Prediction/blob/main/model.py

ATTENTION_MASK_VALUE = -1e9
LAYER_NORM_EPS = 1e-5


def causal_mask(seq_len: int) -> np.ndarray:
    """Create a lower-triangular decoder mask."""
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))

def mask_to_bias(mask: np.ndarray) -> TensorValue:
    """Convert a boolean mask (True = attend) into an additive bias (0.0 or -1e9)."""
    mask_array = np.asarray(mask, dtype=bool)
    return np.where(mask_array, 0.0, ATTENTION_MASK_VALUE)

def scale_dot_product(q: TensorValue, k: TensorValue, v: TensorValue, mask_bias: TensorValue = None):
    """
    Scaled dot-product attention (supports 2D and batched 3D tensors).
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    Parameters:
        Q: [seq_len x d_k] or [num_heads x seq_len x d_k]
        K: [seq_len x d_k] or [num_heads x seq_len x d_k]
        V: [seq_len x d_v] or [num_heads x seq_len x d_v]
        mask_bias: [seq_len x seq_len] additive bias — 0.0 = include, -1e9 = ignore
    Returns:
        output:  [seq_len x d_v] or [num_heads x seq_len x d_v]
        weights: [seq_len x seq_len] or [num_heads x seq_len x seq_len]
    """
    d_k = q.shape[-1]
    scale = np.sqrt(d_k)
    scores = (q @ k.T()) / scale  # [seq_len x seq_len] or [num_heads x seq_len x seq_len]

    if mask_bias is not None:
        scores = scores.add_const(mask_bias) # Masked positions get large negative scores

    scores = softmax(scores)  # [seq_len x seq_len] or [num_heads x seq_len x seq_len]
    sdpa = scores @ v  # [seq_len x d_v] or [num_heads x seq_len x d_v]
    return sdpa, scores


def softmax(x: TensorValue, axis=-1):
    """
    Numerically stable softmax.
    Parameters:
        x: [seq_len x d_k] or [num_heads x seq_len x d_k]
        axis: int, the axis to apply softmax on
    Returns:
        output: [seq_len x d_k] or [num_heads x seq_len x d_k]
    """
    x_max = x.data.max(axis=axis, keepdims=True)
    shifted_x = x - TensorValue(x_max, require_grads=False)  # Shift for numerical stability
    exp = shifted_x.exp()
    return exp/exp.sum(axis=axis, keepdims=True)


def layer_norm(x: TensorValue, gamma: TensorValue = None, beta: TensorValue = None, eps=LAYER_NORM_EPS):
    """Layer normalization across the last dimension.
    Compute:
    normalized = (x - mean) / sqrt(var + eps)
    LayerNorm(x) = gamma * normalized + beta
    where gamma and beta are learnable parameters.
    Parameters:
        x: [seq_len x d_model] or [num_heads x seq_len x d_model]
        gamma: [d_model] or [num_heads x d_model], scale parameter
        beta: [d_model] or [num_heads x d_model], shift parameter
        eps: float, small constant to prevent division by zero
    Returns:        
    normalized: same shape as x, with mean 0 and variance 1 across the last dimension
    """
    feature_dim = x.shape[-1]
    mean = x.sum(axis=-1, keepdims=True) / feature_dim
    centered = x - mean
    var = (centered * centered).sum(axis=-1, keepdims=True) / feature_dim
    normalized = centered * (var + eps).pow(-0.5)
    if gamma is not None:
        if gamma.shape[-1] != feature_dim:
            raise ValueError(f"gamma last dimension must be {feature_dim}, got {gamma.shape}")
        normalized = normalized * gamma
    if beta is not None:
        if beta.shape[-1] != feature_dim:
            raise ValueError(f"beta last dimension must be {feature_dim}, got {beta.shape}")
        normalized = normalized + beta
    return normalized


class MultiHeadAttention:
    """
    Multi-head attention. Why Multiple Heads?
    A single attention head performs a single projection of the representation space. 
    This limits its ability to capture different types of relationships simultaneously.
    Multi-Head Attention runs `h` attention operations in parallel, each in a subspace:
    MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O 
    where 
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    Each head uses different weights, allowing it to learn to focus on different aspects 
    (e.g., syntax, semantics, long-range dependencies) of the input.
    Parameters:
        d_model: int, the dimensionality of the model (e.g., 512)
        num_heads: int, the number of attention heads (e.g., 8)
    Create:
        weight matrices for Q, K, V and output projection with [d_model x d_model] shape.
        weight of output projection is used to combine the outputs of all heads back to d_model dimension.
    """
    def __init__(self,d_model: int, num_heads: int, ):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # Initialize weight matrices for Q, K, V and output projection
        self.w_qkv = TensorValue(np.random.randn(d_model, 3*d_model) * (d_model ** -0.5)) #[d_model,3*d_model]
        self.w_o = TensorValue(np.random.randn(d_model, d_model) * (d_model ** -0.5))

        #parameters for optimization, grouped in a list for easy access wit flat format
        self.parameters = [self.w_qkv, self.w_o]
        self.total_params = sum(p.data.size for p in self.parameters)

    def forward(self, x: TensorValue, mask_bias: TensorValue, rope=None)-> TensorValue:
        """
        Forward pass for multi-head attention.
        Parameters:
            x: [batch_size, seq_len x d_model]  — input sequence
            mask_bias: [batchx seq_len x seq_len] additive attention bias (0.0 = include, -1e9 = ignore)
            rope: optional RoPe instance applied to q and k before attention
        Returns:
            output: [batch_size x seq_len x d_model]

        batch_size = B
        seq_len = S
        vocab_size = V
        d_model = D
        num_heads = H
        d_head = D_H
        """
        B, S, D = x.shape #[B,S,D]
        H = self.num_heads
        D_H = self.d_head
        assert D == self.d_model, "Input feature dimension must match d_model"
        # Linear projections for Q, K, V
        qkv = x @ self.w_qkv  # [B,S,D] x [D,3D] -> [B,S,3*D]

        # [B,S,3*D] -> [B,S,3, H,D_H]-> [3,B,H,S,D_H]
        qkv = qkv.reshape(B,S,3,H,D_H).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2] # [B,H,S,D_H]

        if rope is not None: #apply RoPE to q and k before attention
            if hasattr(rope, "d_model") and rope.d_model != self.d_head:
                raise ValueError(f"RoPE dimension {rope.d_model} must match attention head size {self.d_head}")
            q, k = rope.apply(q, k)
        
        # Single batched attention call
        heads_output, _ = scale_dot_product(q, k, v, mask_bias) # [B,H,S,D_H]

        # [B,H,S,D_H] -> [B,S,H,D_H] -> [B,S,D] ( D = D_H*D)
        concat = heads_output.permute(0, 2, 1, 3).reshape(B,S,D)

        # Output projection
        return concat @ self.w_o  # [B,S,D] x[D,D] -> [B,S,D]
    
class MLP:
    """Simple 2-layer MLP with selectable activation: ReLU or SwiGLU."""
    def __init__(self, d_model: int, d_ff: int, activation: str = "relu"):
        if activation not in ("relu", "swiglu"):
            raise ValueError("Unsupported activation. Choose 'relu' or 'swiglu'.")
        self.w1 = TensorValue(np.random.randn(d_model, d_ff) * (d_model ** -0.5)) #[d_model x d_ff]
        self.w2 = TensorValue(np.random.randn(d_ff, d_model) * (d_ff ** -0.5)) #[d_ff x d_model]
        self.parameters = [self.w1, self.w2]
        if activation == "swiglu":
            self.w3 = TensorValue(np.random.randn(d_model, d_ff) * (d_model ** -0.5))  # Extra weights for gating
            self.parameters.append(self.w3)

        self.total_params = sum(p.data.size for p in self.parameters)
        self.activation = activation

    def forward(self, x: TensorValue):
        """
        Forward pass through the MLP. Supports ReLU and SwiGLU activations.
        Input: [seq_len x d_model]
        Return: [seq_len x d_model]
        """
        if self.activation == "relu":
            return self.forward_relu(x)
        elif self.activation == "swiglu":
            return self.forward_swiglu(x)
      
    def forward_relu(self, x: TensorValue):
        """forward pass with ReLU activation: output = ReLU(x @ w1) @ w2"""
        return (x @ self.w1).relu() @ self.w2
    
    def forward_swiglu(self, x: TensorValue):
        """SwiGLU activation: output = (silu(x @ w1) * (x @ w3)) @ w2
        where silu(z) = z * sigmoid(z). The linear branch (x @ w3) gates the
        activated branch element-wise, letting the model learn dynamic gating
        of the feedforward activations.
        """
        activated = x @ self.w1  # [seq_len x d_ff]
        activated = activated * activated.sigmoid()  # silu(x @ w1)
        gate = x @ self.w3  # [seq_len x d_ff], linear gate
        return (activated * gate) @ self.w2  # [seq_len x d_model]
    

class TransformerBlock:
    """
    The Transformer block combines all the components we have seen with residual connections.
    Residual Connections (skip connections): output = x + sublayer(x)
    Pre-LN (more stable): x → Norm → sublayer → Add  <- we use this here

        x ──┬──── LayerNorm ──► MultiHeadAttn ──► + ─── y
            └────────────────────────────────────►┘

        y ─┬──── LayerNorm ──► FeedForward ───► + ─── z
            └────────────────────────────────────►┘
        y = x + MultiHeadAttn(LayerNorm(x))
        z = y + FFN(LayerNorm(y))
    Parameters:
        d_model: int, the dimensionality of the model (e.g., 512)
        num_heads: int, the number of attention heads (e.g., 8)
        d_ff: int, the dimensionality of the feedforward layer (e.g., 2048)
        activation: str, "relu" or "swiglu" for the MLP activation function
        use_rope: bool, whether to apply RoPE positional encoding in attention (default True)
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int , activation: str = "relu"):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.mlp = MLP(d_model, d_ff, activation)
        #LayerNorm parameters (gamma and beta) for both sublayers
        self.ln1_gamma = TensorValue(np.ones(d_model))
        self.ln1_beta = TensorValue(np.zeros(d_model))
        self.ln2_gamma = TensorValue(np.ones(d_model))
        self.ln2_beta = TensorValue(np.zeros(d_model))
        self.parameters = [self.ln1_gamma, self.ln1_beta, self.ln2_gamma, self.ln2_beta] + self.mha.parameters + self.mlp.parameters
        self.total_params = sum(p.data.size for p in self.parameters)

    def forward(self, x: TensorValue, mask_bias: TensorValue, rope=None) -> TensorValue:
        """Forward pass through the Transformer block.

        Args:
            x (TensorValue): Input tensor of shape [batch_size x seq_len x d_model]
            mask_bias (TensorValue): Additive attention bias of shape [batch_size x seq_len x seq_len]
                (0.0 = include, -1e9 = ignore).
            rope: optional RoPe instance applied inside attention.

        Returns:
            TensorValue: Output tensor of shape [seq_len x d_model]

        batch_size = B
        seq_len = S
        d_model = D
        num_heads = H
        """
        # Multi-head attention sublayer with pre-LN and residual connection

        ln1_out = layer_norm(x, self.ln1_gamma, self.ln1_beta)  #  [B,S,D]
        mha_out = self.mha.forward(ln1_out, mask_bias, rope)  # [B,S,D]
        x = x + mha_out  # Residual connection

        # Feedforward sublayer with pre-LN and residual connection
        ln2_out = layer_norm(x, self.ln2_gamma, self.ln2_beta)  # [seq_len x d_model]
        mlp_out = self.mlp.forward(ln2_out)  # [seq_len x d_model]
        return x + mlp_out  # Residual connection

class TransformerLM:
    """
    Transformer Language Model (decoder-only, GPT-style).
    Predict next token given previous tokens.
    Predicts P(token_t | token_0, ..., token_{t-1}) for each position.
    Parameters:
        vocab_size: int, size of the vocabulary
        d_model: int, dimensionality of the model (e.g., 512)
        num_heads: int, number of attention heads (e.g., 8)
        d_ff: int, dimensionality of the feedforward layer (e.g., 2048)
        num_layers: int, number of Transformer blocks (e.g., 6)
        max_seq_len: int, maximum sequence length for positional encoding (if using absolute positional encoding)
        uuse_rope: bool, whether to apply RoPE positional encoding in attention (default True)
    
      Weight tying (Press & Wolf, 2017): the output projection reuses the token
      embedding matrix — logits = x @ E.T — so there is NO separate output_projection
      parameter. This cuts ~V*D parameters and ties the input/output vocabulary
      representations. E receives gradient from BOTH paths (input lookup E[ids] and
      output projection x @ E.T) and the autograd accumulates them into E.grad.
      Caveat: changes the checkpoint layout (output_projection is gone) — checkpoints
      saved before tying are not loadable under the current param ordering.


                    ┌─────────────────────────────────────-┐
   Input            │          TRANSFORMER LM              │
   tokens ─────────►  Embedding + Pos. Encoding            │
                    │         │                            │
                    │  ┌──────▼──────────────────────┐     │
                    │  │      Transformer Block      │     │
                    │  │  ┌───────────────────────┐  │     │
                    │  │  │ Masked Self-Attention │  │ xN  │
                    │  │  │   + Add & Layer Norm  │  │     │
                    │  │  └───────────────────────┘  │     │
                    │  │  ┌───────────────────────┐  │     │
                    │  │  │  Feed-Forward Network │  │     │
                    │  │  │   + Add & Layer Norm  │  │     │
                    │  │  └───────────────────────┘  │     │
                    │  └──────┬──────────────────────┘     │
                    │         │                            │
                    │  Tied Projection (E^T) → Logits       │
                    │         │                            │
                    └─────────┼────────────────────────────┘
                              │
                          Softmax
                              │
                     Next-token probabilities
    """
    def __init__(self,vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_len: int, use_rope: bool = True, mask: np.ndarray = None ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        #
        if use_rope:
            self.rope = RoPe(max_seq_len, d_model // num_heads)  # RoPE is applied per attention head
        else:
            self.rope = None
            self.pos_encoding = PositionalEncoding(max_seq_len, d_model)  # Absolute positional encoding if not using RoPE
        
        # causal mask
        if mask is None:
            causal_mask_array = causal_mask(max_seq_len)
        else:
            causal_mask_array = np.asarray(mask, dtype=bool)
            expected_mask_shape = (max_seq_len, max_seq_len)
            if causal_mask_array.shape != expected_mask_shape:
                raise ValueError(f"mask must have shape {expected_mask_shape}, got {causal_mask_array.shape}")

        # causal mask bias (additive)
        self._mask_bias = mask_to_bias(causal_mask_array)
   

        # Token embedding and positional encoding (using RoPE, so no separate positional embedding)
        self.token_embedding = TensorValue(np.random.randn(vocab_size, d_model) * (d_model ** -0.5))  # [vocab_size x d_model]

        # Stack of Transformer blocks
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

        # Collect all parameters for optimization
        self.parameters = [self.token_embedding]
        for block in self.blocks:
            self.parameters.extend(block.parameters)

        self.total_params = sum(p.data.size for p in self.parameters)

    def forward(self, ids: np.ndarray):
        """
        Forward pass through the Transformer LM.
        Parameters:
            ids: [batch_size x seq_len] array of token indices per batch
        Returns:
            logits: [batch_size xseq_len x vocab_size] unnormalized log probabilities for next token prediction
            
        Notations: batch_size = B, seq_len = S, vocab_size = V, d_model = D
        """
        seq_len = ids.shape[-1]
        assert seq_len <= self.max_seq_len, "Input sequence length exceeds model's maximum sequence length"
        
        
        #Tokens → embeddings
        x = self.token_embedding[ids]  #[B,S] indexando [V,D] -> [B,S,D]

        # Use absolute positional encoding if not using RoPE, otherwise RoPE is applied inside attention
        if not self.use_rope:
            x = x + self.pos_encoding.apply(np.arange(seq_len, dtype=int))

        #mask bias applied only during the trainin
        mask_bias = self._mask_bias if seq_len == self.max_seq_len else self._mask_bias[:seq_len,:seq_len]

        # Transformer blocks (attention + FFN, N times)
        for block in self.blocks:
            x = block.forward(x, mask_bias, self.rope)  # [B,S,D]

        # Final linear projection to vocabulary size
        #Press & Wolf (2017) show that tying the token embedding and output projection weights can improve performance 
        # and reduce parameters, so we reuse the token embedding matrix for the output projection.
        # W_o = W_token_embedding^T, so we can compute logits with a single matrix multiplication:
        logits = x @ self.token_embedding.T()  # [B,S,D] @ [D,V] -> [B,S,V]  (E^T, weight-tied)
        return logits
    
    def loss(self,ids_input:list[int], ids_target: np.ndarray):
        """
        Mean cross-entropy loss for a given (input, target) pair:

            loss = -1/T * Σ_t log P(target_t | token_0..token_t)

        where P comes from softmax(logits). The softmax + log + mean chain is
        fused into a single graph node (TensorValue.cross_entropy), which uses
        the log-sum-exp trick for numerical stability and the closed-form
        gradient ∂loss/∂logits = (softmax - one_hot) / T.
        Input:
            - ids_input:  [batch_size x seq_len] array of token indices per batch
            - ids_target:  [batch_size x seq_len]

        Notations: batch_size = B, seq_len = S, vocab_size = V, , T = B*S
        """
        #For each batch, we got the log-probabilities matrix of the sequence input
        # for the vocab size
        logits = self.forward(ids_input)  # [B,S] -> [B,S,V] 
        B,S,V = logits.shape
        flat = logits.reshape(B*S,V) # [T,V]
        targets = np.asarray(ids_target).reshape(-1) # [B,S] -> [B*S]
        return flat.cross_entropy(targets) #
    
    def predict_next_token(self, input_tokens: np.ndarray, temperature: float = 0.0) -> int:
        """Predict the next token given input tokens.
        Parameters:
            input_tokens: [seq_len] array of token indices
            temperature: Sampling temperature. 0 for greedy decoding.
        Returns:
            next_token_index: Index of the predicted next token

        notation: seq_len = S, vocab_size = V
        """
        if temperature < 0:
            raise ValueError("temperature must be non-negative")

        with no_grad():  # Disable gradient tracking for inference
            #change format of input_token from [S] -> [B,S] 
            logits = self.forward(input_tokens[None,:])  # [None, S] -> [None, S,V]
            last_token_logits = logits[0,-1]  # Get logits for the last position
        if temperature == 0:
            # Greedy decoding: return the index of the highest logit.
            next_token_index = np.argmax(last_token_logits.data)
        else:
            # Sample from the distribution with temperature scaling
            scaled_logits = last_token_logits / temperature
            probs = softmax(scaled_logits).data # numpy data
            probs /= probs.sum()
            next_token_index = np.random.choice(self.vocab_size, p=probs)  # Sample next token index
        return next_token_index
    
    def save(self, path:str):
        config = {
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "num_heads": self.num_heads, "d_ff": self.d_ff,
            "num_layers": self.num_layers, "max_seq_len": self.max_seq_len,
            "use_rope": self.use_rope
        }
        arrays = {f"param_{i}": p.data for i,p in enumerate(self.parameters)}
        np.savez(path,config=json.dumps(config), **arrays)

    @classmethod
    def load(cls,path:str) -> "TransformerLM":
        ckpt = np.load(path)
        config = json.loads(str(ckpt["config"]))
        model = cls(**config) #build the model TransformerLM(**config)
        for i,p in enumerate(model.parameters):
            p.data[...] = ckpt[f"param_{i}"] #copia in-place
        return model
