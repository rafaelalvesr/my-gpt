# Study Notebooks — Building a GPT from Scratch

A didactic, self-contained track that builds the theory behind a decoder-only
Transformer (GPT) **from first principles** using only NumPy / pure Python.
Read the notebooks **in order** — each one assumes the previous ones.

All notebooks are in **English**. Code is intentionally small and readable; the
goal is to understand *why* each piece exists, not to be fast (production uses
PyTorch/JAX, which are ~1000× faster on GPU).

---

## Reading order

### Part I — Machine-learning foundations

| # | Notebook | What you learn |
|---|----------|----------------|
| 01 | [`01_learning_theory.ipynb`](01_learning_theory.ipynb) | Probability, supervised/unsupervised learning, capacity, over/under-fitting, bias–variance, MLE, Bayes. The conceptual ground for everything else. |
| 02 | [`02_loss_functions.ipynb`](02_loss_functions.ipynb) | What we minimize: MSE/MAE/Huber for regression, **cross-entropy** for classification (the loss the LM uses). |
| 03 | [`03_autograd.ipynb`](03_autograd.ipynb) | Automatic differentiation and **backpropagation** from scratch (`Value` → `TensorValue`). The engine that computes gradients. |
| 04 | [`04_optimizers.ipynb`](04_optimizers.ipynb) | How gradients update parameters: SGD → Momentum → Adagrad → RMSProp → **Adam → AdamW**, plus learning-rate scheduling. |

### Part II — Language-model building blocks

| # | Notebook | What you learn |
|---|----------|----------------|
| 05 | [`05_tokenization.ipynb`](05_tokenization.ipynb) | Text → integers: Unicode/UTF-8, **Byte-Pair Encoding (BPE)**, and the GPT-2 regex pre-tokenizer. |
| 06 | [`06_embeddings.ipynb`](06_embeddings.ipynb) | Token IDs → dense vectors (**embeddings**), plus **positional encoding** (sinusoidal) and **RoPE**. |

### Part III — Sequence architectures

| # | Notebook | What you learn |
|---|----------|----------------|
| 07 | [`07_rnn_language_models.ipynb`](07_rnn_language_models.ipynb) | The pre-Transformer baseline: **RNN/LSTM** language models (PyTorch), and why recurrence struggles with long context. |
| 08 | [`08_attention.ipynb`](08_attention.ipynb) | **Attention** as it first appeared (Seq2Seq, Bahdanau, Luong) — the bottleneck it solved and scaled dot-product. |
| 09 | [`09_transformer.ipynb`](09_transformer.ipynb) | The full **decoder-only Transformer**: self-attention, multi-head, FFN, LayerNorm, residuals, training, generation. |

---

## How the notebooks connect

```
01 theory ─► 02 loss ─► 03 autograd ─► 04 optimizers
                                  │
                                  ▼   (the training machinery)
05 tokenization ─► 06 embeddings ─► 07 RNN ─► 08 attention ─► 09 transformer
                                  │                                 ▲
                                  └──────── building blocks ────────┘
```

The final model (`09_transformer.ipynb`) reuses `TensorValue` (autograd, NB 03),
cross-entropy (NB 02), AdamW (NB 04), embeddings + positional encoding (NB 06),
and the scaled-dot-product idea (NB 08).

## Environment notes

- Most notebooks need only `numpy` and `matplotlib`.
- `01_learning_theory` also uses `scikit-learn`; `07_rnn_language_models` uses `torch`.
- `05_tokenization` uses the `regex` and `requests` packages.
- Notebooks 03/06/09 import `TensorValue` from the repository's `train` package
  (run them from the `notebooks/` folder so `sys.path.append('..')` resolves).
