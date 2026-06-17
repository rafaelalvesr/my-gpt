import numpy as np
import pytest

from model.transformer import (
    MLP,
    MultiHeadAttention,
    TransformerBlock,
    TransformerLM,
    causal_mask,
    layer_norm,
    mask_to_bias,
    scale_dot_product,
    softmax,
)
from train import TensorValue


def test_softmax_rows_sum_to_one_and_preserve_ordering():
    x = TensorValue([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    y = softmax(x, axis=1)

    assert y.shape == (2, 3)
    assert np.allclose(y.data.sum(axis=1), np.ones(2), atol=1e-8)
    assert y.data[0, 2] > y.data[0, 1] > y.data[0, 0]
    assert np.allclose(y.data[1], np.array([1 / 3, 1 / 3, 1 / 3]), atol=1e-8)


def test_scale_dot_product_matches_numpy_without_mask():
    q_np = np.array([[1.0, 0.0], [0.0, 1.0]])
    k_np = np.array([[1.0, 0.0], [0.0, 1.0]])
    v_np = np.array([[2.0, 1.0], [0.0, 3.0]])

    q = TensorValue(q_np)
    k = TensorValue(k_np)
    v = TensorValue(v_np)

    out, weights = scale_dot_product(q, k, v, mask_bias=None)

    scores_np = q_np @ k_np.T / np.sqrt(q_np.shape[-1])
    scores_np = scores_np - np.max(scores_np, axis=-1, keepdims=True)
    weights_np = np.exp(scores_np) / np.sum(np.exp(scores_np), axis=-1, keepdims=True)
    out_np = weights_np @ v_np

    assert np.allclose(weights.data, weights_np, atol=1e-8)
    assert np.allclose(out.data, out_np, atol=1e-8)


def test_scale_dot_product_respects_causal_mask():
    q = TensorValue([[1.0, 0.0], [0.0, 1.0]])
    k = TensorValue([[1.0, 0.0], [0.0, 1.0]])
    v = TensorValue([[5.0, 1.0], [2.0, 3.0]])
    mask = np.tril(np.ones((2, 2)))
    mask_bias = mask_to_bias(mask)

    out, weights = scale_dot_product(q, k, v, mask_bias)

    assert out.shape == (2, 2)
    assert weights.shape == (2, 2)
    assert weights.data[0, 1] < 1e-8
    assert np.allclose(weights.data.sum(axis=1), np.ones(2), atol=1e-8)


def test_scale_dot_product_supports_batched_3d_inputs_and_masking():
    np.random.seed(42)
    q = TensorValue(np.random.randn(2, 3, 4))
    k = TensorValue(np.random.randn(2, 3, 4))
    v = TensorValue(np.random.randn(2, 3, 5))
    mask = np.tril(np.ones((3, 3)))
    mask_bias = mask_to_bias(mask)

    out, weights = scale_dot_product(q, k, v, mask_bias)

    assert out.shape == (2, 3, 5)
    assert weights.shape == (2, 3, 3)
    assert np.allclose(weights.data[:, 0, 1:], 0.0, atol=1e-8)
    assert np.allclose(weights.data.sum(axis=-1), np.ones((2, 3)), atol=1e-8)


def test_layer_norm_normalizes_last_dimension_without_affine_params():
    x = TensorValue(np.array([[1.0, 2.0, 3.0], [3.0, 3.0, 6.0]]))

    y = layer_norm(x)

    assert y.shape == x.shape
    assert np.allclose(y.data.mean(axis=-1), np.zeros(2), atol=1e-7)
    assert np.allclose(y.data.var(axis=-1), np.ones(2), atol=1e-4)


def test_layer_norm_applies_gamma_and_beta():
    x_np = np.array([[1.0, 2.0, 4.0]], dtype=float)
    x = TensorValue(x_np)
    gamma = TensorValue(np.array([1.0, 2.0, 0.5]))
    beta = TensorValue(np.array([-1.0, 0.0, 2.0]))

    y = layer_norm(x, gamma=gamma, beta=beta, eps=1e-5)

    mean = x_np.mean(axis=-1, keepdims=True)
    var = x_np.var(axis=-1, keepdims=True)
    expected = ((x_np - mean) / np.sqrt(var + 1e-5)) * gamma.data + beta.data

    assert np.allclose(y.data, expected, atol=1e-8)


def test_layer_norm_backward_matches_finite_difference():
    x_data = np.array([[1.5, -0.5, 2.0]], dtype=float)
    weight = np.array([[0.2, -0.3, 0.5]], dtype=float)
    x = TensorValue(x_data.copy())

    loss = (layer_norm(x) * TensorValue(weight)).sum()
    loss.backward()

    numerical_grad = np.zeros_like(x_data)
    epsilon = 1e-3  # TensorValue is float32: smaller steps drown in rounding noise
    for index in range(x_data.size):
        plus = x_data.copy()
        minus = x_data.copy()
        plus.flat[index] += epsilon
        minus.flat[index] -= epsilon

        loss_plus = (layer_norm(TensorValue(plus)).data * weight).sum()
        loss_minus = (layer_norm(TensorValue(minus)).data * weight).sum()
        numerical_grad.flat[index] = (loss_plus - loss_minus) / (2 * epsilon)

    assert np.allclose(x.grad, numerical_grad, atol=1e-4, rtol=1e-3)


def test_multihead_attention_init_raises_when_heads_do_not_divide_model():
    with pytest.raises(AssertionError):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_multihead_attention_forward_and_backward_shapes_and_grads():
    np.random.seed(0)
    seq_len = 5
    d_model = 8
    num_heads = 2

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = TensorValue(np.random.randn(1, seq_len, d_model))
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask_bias = mask_to_bias(mask)

    out = mha.forward(x, mask_bias)
    loss = out.sum()
    loss.backward()

    assert out.shape == (1, seq_len, d_model)
    assert x.grad.shape == x.shape
    assert np.all(np.isfinite(x.grad))

    for p in mha.parameters:
        assert p.grad.shape == p.shape
        assert np.all(np.isfinite(p.grad))


def test_multihead_attention_forward_raises_on_wrong_input_dimension():
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = TensorValue(np.random.randn(1, 4, 6))

    with pytest.raises(AssertionError):
        mha.forward(x, None)


def test_multihead_attention_forward_calls_rope_apply_when_provided():
    class DummyRoPE:
        def __init__(self):
            self.called = False

        def apply(self, q, k):
            self.called = True
            return q, k

    np.random.seed(10)
    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = TensorValue(np.random.randn(1, 4, 8))
    mask = np.tril(np.ones((4, 4)))
    mask_bias = mask_to_bias(mask)
    rope = DummyRoPE()

    out = mha.forward(x, mask_bias, rope=rope)

    assert rope.called is True
    assert out.shape == (1, 4, 8)


def test_multihead_attention_forward_raises_on_rope_dimension_mismatch():
    class BadRoPE:
        d_model = 8

        def apply(self, q, k):
            return q, k

    mha = MultiHeadAttention(d_model=8, num_heads=2)
    x = TensorValue(np.random.randn(1, 4, 8))

    with pytest.raises(ValueError, match="must match attention head size"):
        mha.forward(x, None, rope=BadRoPE())


def test_mlp_relu_forward_and_backward():
    np.random.seed(1)
    seq_len = 4
    d_model = 8
    d_ff = 16

    mlp = MLP(d_model=d_model, d_ff=d_ff, activation="relu")
    x = TensorValue(np.random.randn(seq_len, d_model))

    out = mlp.forward(x)
    loss = out.sum()
    loss.backward()

    assert out.shape == (seq_len, d_model)
    assert x.grad.shape == x.shape
    assert np.all(np.isfinite(x.grad))

    for p in mlp.parameters:
        assert p.grad.shape == p.shape
        assert np.all(np.isfinite(p.grad))


def test_mlp_forward_dispatch_matches_forward_relu():
    np.random.seed(7)
    mlp = MLP(d_model=8, d_ff=16, activation="relu")
    x = TensorValue(np.random.randn(3, 8))

    y_dispatch = mlp.forward(x)
    y_direct = mlp.forward_relu(x)

    assert np.allclose(y_dispatch.data, y_direct.data, atol=1e-8)


def test_mlp_swiglu_forward_and_backward():
    np.random.seed(2)
    seq_len = 3
    d_model = 6
    d_ff = 12

    mlp = MLP(d_model=d_model, d_ff=d_ff, activation="swiglu")
    x = TensorValue(np.random.randn(seq_len, d_model))

    out = mlp.forward(x)
    loss = out.sum()
    loss.backward()

    assert out.shape == (seq_len, d_model)
    assert x.grad.shape == x.shape
    assert np.all(np.isfinite(x.grad))

    for p in mlp.parameters:
        assert p.grad.shape == p.shape
        assert np.all(np.isfinite(p.grad))


def test_mlp_forward_dispatch_matches_forward_swiglu():
    np.random.seed(11)
    mlp = MLP(d_model=6, d_ff=12, activation="swiglu")
    x = TensorValue(np.random.randn(2, 6))

    y_dispatch = mlp.forward(x)
    y_direct = mlp.forward_swiglu(x)

    assert np.allclose(y_dispatch.data, y_direct.data, atol=1e-8)

def test_mlp_invalid_activation_raises_error():
    with pytest.raises(ValueError):
        MLP(d_model=8, d_ff=16, activation="invalid")


def test_transformer_block_forward_and_backward():
    np.random.seed(21)
    seq_len = 5
    d_model = 8
    block = TransformerBlock(d_model=d_model, num_heads=2, d_ff=16, activation="relu")
    x = TensorValue(np.random.randn(1, seq_len, d_model))
    mask = np.tril(np.ones((seq_len, seq_len)))
    mask_bias = mask_to_bias(mask)

    out = block.forward(x, mask_bias)
    loss = out.sum()
    loss.backward()

    assert out.shape == (1, seq_len, d_model)
    assert x.grad.shape == x.shape
    assert np.all(np.isfinite(x.grad))

    for p in block.parameters:
        assert p.grad.shape == p.shape
        assert np.all(np.isfinite(p.grad))


def test_transformer_lm_forward_without_rope_returns_logits_and_grads():
    np.random.seed(123)
    lm = TransformerLM(
        vocab_size=15,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_seq_len=8,
        use_rope=False,
    )
    input_tokens = np.array([[1, 2, 4, 3]], dtype=int)

    logits = lm.forward(input_tokens)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (1, 4, 15)
    # Weight tying (Press & Wolf, 2017): the output projection reuses the token
    # embedding (logits = x @ E.T), so there is no separate output_projection param.
    assert not hasattr(lm, "output_projection")
    assert lm.token_embedding.grad.shape == lm.token_embedding.shape
    assert np.all(np.isfinite(lm.token_embedding.grad))


def test_transformer_lm_mask_bias_matches_causal_mask_for_shorter_sequences():
    lm = TransformerLM(
        vocab_size=12,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=6,
        use_rope=False,
    )
    seq_len = 4
    sliced_bias = lm._mask_bias[:seq_len, :seq_len]
    expected_bias = mask_to_bias(causal_mask(seq_len))

    assert np.allclose(sliced_bias.data, expected_bias.data, atol=1e-8)


def test_transformer_lm_init_raises_on_invalid_mask_shape():
    with pytest.raises(ValueError, match="mask must have shape"):
        TransformerLM(
            vocab_size=10,
            d_model=8,
            num_heads=2,
            d_ff=16,
            num_layers=1,
            max_seq_len=5,
            use_rope=False,
            mask=np.ones((3, 3), dtype=bool),
        )


def test_transformer_lm_forward_with_rope_returns_logits_and_grads():
    np.random.seed(321)
    lm = TransformerLM(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=6,
        use_rope=True,
    )
    input_tokens = np.array([[1, 2, 3]], dtype=int)

    logits = lm.forward(input_tokens)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (1, 3, 10)
    # Weight tying: no separate output_projection; E receives gradient via both
    # the input lookup E[ids] and the tied output projection x @ E.T.
    assert not hasattr(lm, "output_projection")
    assert lm.token_embedding.grad.shape == lm.token_embedding.shape
    assert np.all(np.isfinite(lm.token_embedding.grad))


def test_transformer_lm_forward_raises_when_sequence_exceeds_max_length():
    lm = TransformerLM(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=2,
        use_rope=False,
    )

    with pytest.raises(AssertionError, match="maximum sequence length"):
        lm.forward(np.array([0, 1, 2], dtype=int))


def test_transformer_lm_predict_next_token_greedy_and_sampling(monkeypatch):
    lm = TransformerLM(
        vocab_size=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=5,
        use_rope=False,
    )

    def fake_forward(_input_tokens):
        return TensorValue(
            np.array(
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [1.0, 3.0, 0.5, -2.0],
                    ]
                ]
            )
        )

    lm.forward = fake_forward
    input_tokens = np.array([0, 1], dtype=int)

    greedy_idx = lm.predict_next_token(input_tokens, temperature=0.0)
    assert greedy_idx == 1

    captured = {}

    def fake_choice(vocab_size, p):
        captured["vocab_size"] = vocab_size
        captured["probabilities"] = p
        return 2

    monkeypatch.setattr(np.random, "choice", fake_choice)
    sampled_idx = lm.predict_next_token(input_tokens, temperature=0.7)

    assert sampled_idx == 2
    assert captured["vocab_size"] == lm.vocab_size
    assert np.allclose(captured["probabilities"].sum(), 1.0, atol=1e-8)
    assert np.all(captured["probabilities"] >= 0.0)


def test_transformer_lm_predict_next_token_rejects_negative_temperature():
    lm = TransformerLM(
        vocab_size=4,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
        max_seq_len=5,
        use_rope=False,
    )

    with pytest.raises(ValueError, match="temperature must be non-negative"):
        lm.predict_next_token(np.array([0, 1], dtype=int), temperature=-0.1)


def test_transformer_lm_uses_weight_tying():
    lm = TransformerLM(
        vocab_size=15,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_seq_len=8,
        use_rope=False,
    )
    # No separate output projection: logits = x @ token_embedding.T (Press & Wolf, 2017).
    assert not hasattr(lm, "output_projection")
    # The token embedding is a trainable parameter and appears exactly once.
    assert sum(p is lm.token_embedding for p in lm.parameters) == 1


def test_transformer_lm_loss_backward_propagates_to_embedding():
    np.random.seed(7)
    lm = TransformerLM(
        vocab_size=12,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_seq_len=8,
        use_rope=False,
    )
    inputs = np.array([[1, 2, 3, 4]], dtype=int)
    targets = np.array([[2, 3, 4, 5]], dtype=int)

    loss = lm.loss(inputs, targets)
    loss.backward()

    # cross_entropy returns a scalar mean negative log-likelihood
    assert loss.shape == ()
    assert np.isfinite(loss.data)
    assert loss.data > 0.0
    # E receives gradient from both the input lookup and the tied output projection
    assert lm.token_embedding.grad.shape == lm.token_embedding.shape
    assert np.all(np.isfinite(lm.token_embedding.grad))
    assert np.any(lm.token_embedding.grad != 0.0)


def test_transformer_lm_save_load_roundtrip(tmp_path):
    np.random.seed(0)
    lm = TransformerLM(
        vocab_size=11,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=2,
        max_seq_len=6,
        use_rope=False,
    )
    tokens = np.array([[1, 2, 3, 4]], dtype=int)
    logits_before = lm.forward(tokens).data

    path = str(tmp_path / "model.npz")
    lm.save(path)
    restored = TransformerLM.load(path)

    # config and (tied) parameters are restored exactly
    assert restored.vocab_size == lm.vocab_size
    assert restored.use_rope == lm.use_rope
    assert len(restored.parameters) == len(lm.parameters)
    for p_old, p_new in zip(lm.parameters, restored.parameters):
        assert np.array_equal(p_old.data, p_new.data)
    # and the forward pass is identical
    assert np.allclose(restored.forward(tokens).data, logits_before, atol=1e-8)
