import numpy as np
from train import TensorValue


# Positional Encoding (PE)
class PositionalEncoding:
    """Sinusoidal positional encoding with correct even/odd frequency pairing."""

    def __init__(self, max_len: int, d_model: int):
        self.max_len = max_len
        self.d_model = d_model

        # Standard Transformer PE: even (2k) and odd (2k+1) share the same frequency for each pair k.
        position = np.arange(max_len, dtype=np.float32).reshape(max_len, 1)
        even_dims = np.arange(0, d_model, 2, dtype=np.float32)
        div_term = np.exp(-np.log(10000.0) * (even_dims / d_model))  # equal  to 1 / (10000 ** (2k/d_model))

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = np.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.positional_encoding = TensorValue(pe, require_grads=False)

    def apply(self, positions: list[int] | np.ndarray) -> TensorValue:
        """Returns positional-encoding rows for absolute positions."""
        return self.positional_encoding[positions]
    

class RoPe:
    """Rotary Positional Encoding (ROPE) for 2D or batched 3D TensorValue matrices.

    Supported shapes:
    - [seq_len, d_model] and [batch_size, seq_len, d_model]
    - [num_heads, seq_len, d_model] and [batch_size, num_heads, seq_len, d_model]
    Simple notations:
    S = seq_len
    D = d_model
    B = batch_size
    H = num_heads
    """

    def __init__(self, max_len: int, d_model: int):
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for RoPe")

        self.d_model = d_model
        self.max_len = max_len

        pair_dims = np.arange(0, d_model, 2, dtype=np.float32)
        inv_freq = np.exp(-np.log(10000.0) * (pair_dims / d_model))  # [d_model/2]
        positions = np.arange(max_len, dtype=np.float32).reshape(max_len, 1)
        angles = positions * inv_freq.reshape(1, -1)  # [max_len, d_model/2]

        self.cos_pos = np.cos(angles).astype(np.float32)
        self.sin_pos = np.sin(angles).astype(np.float32)

    def _interleave(self, even: TensorValue, odd: TensorValue) -> TensorValue:
        # Rebuild [..., d_model] by interleaving last-dimension even/odd channels.
        stacked = TensorValue.stack([even, odd], axis=-1)
        return stacked.reshape(*even.shape[:-1], even.shape[-1] * 2)

    def _apply_rope(self, x: TensorValue, cos: np.array, sin: np.array) -> TensorValue:
        x_even = x[..., 0::2] #even columns
        x_odd = x[..., 1::2]  #odd columns

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_odd * cos + x_even * sin

        return self._interleave(x_even_rot, x_odd_rot)

    def apply(self, q_matrix: TensorValue, k_matrix: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Apply ROPE rotation to query and key tensors."""
        if q_matrix.shape != k_matrix.shape:
            raise ValueError("q_matrix and k_matrix must have the same shape")

        S,D  = q_matrix.shape[-2], q_matrix.shape[-1]

        if D != self.d_model:
            raise ValueError("Last dimension must equal d_model")
        if S > self.max_len:
            raise ValueError("Sequence length exceeds RoPe max_len")

        cos = self.cos_pos[:S]  # [S, D/2]
        sin = self.sin_pos[:S]  # [S, D/2]

        q_rot = self._apply_rope(q_matrix, cos, sin)
        k_rot = self._apply_rope(k_matrix, cos, sin)
        return q_rot, k_rot