# LLM didática — A teaching-oriented GPT

> 🇧🇷 **Introdução.** Apresenta-se um modelo de linguagem GPT (apenas decodificador)
> implementado do zero em **NumPy puro**, com motor de **autograd próprio**. O objetivo é
> didático: o código foi separado em componentes e comentado para facilitar o entendimento.
> A única biblioteca relevante é a **NumPy**, usada apenas para não nos preocuparmos com a
> implementação das operações matemáticas sobre vetores e matrizes. Este código pode ser
> entendido como uma versão intermediária entre uma implementação em **PyTorch** e a
> implementação simplificada e didática do
> [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) de Andrej
> Karpathy. É possível treinar, salvar e retomar checkpoints e gerar textos com o modelo
> treinado. Os notebooks com a teoria estão em [`notebooks/`](notebooks/README.md).
> *(O restante deste README está em inglês.)*

> 🇬🇧 **Introduction.** A decoder-only GPT language model built from scratch in **pure NumPy**,
> with its **own autograd engine**. The goal is didactic: the code is split into commented
> components for clarity. **NumPy is the only relevant dependency**, used solely so we don't
> have to reimplement the vector/matrix math — everything else is written by hand. Think of it
> as a middle ground between a full **PyTorch** implementation and Andrej Karpathy's simplified,
> didactic [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): you
> can train, save/resume checkpoints and generate text with the trained model.

> 📚 The theory behind each piece lives in the study notebooks under
> [`notebooks/`](notebooks/README.md) (a numbered 01→09 track: ML theory, autograd,
> optimizers, tokenization, embeddings, attention and the Transformer).

---

## How the program works

The project is organized in **layers**, from the most fundamental (automatic differentiation)
to the most composed (the training loop). Each layer depends only on the previous ones:

```
                    ┌─────────────────────────────────────────────┐
   text   ──►  tokenizer/  ──►  ids (np.uint16)                    │
                    │                                              │
                    ▼                                              │
        ┌───────────────────────────────────────────────┐        │
        │  train/tensorvalue.py — TensorValue (autograd)  │  ◄─── train/autograd.py
        │  NumPy tensors that build the computation       │       (Value: the same
        │  graph and know how to run .backward()          │        idea, scalar)
        └───────────────────────────────────────────────┘        │
                    │                                              │
                    ▼                                              │
        ┌───────────────────────────────────────────────┐        │
        │  model/  — Transformer built on TensorValue:    │        │
        │  embeddings + PE/RoPE, multi-head attention,    │        │
        │  MLP, LayerNorm, blocks, the LM                 │        │
        └───────────────────────────────────────────────┘        │
                    │                                              │
                    ▼                                              │
        ┌───────────────────────────────────────────────┐        │
        │  train/  — AdamW, LR schedule, clipping and     │        │
        │  the training loop (batches, eval, checkpoint)  │        │
        └───────────────────────────────────────────────┘        │
                    │                                              │
                    ▼                                              │
            checkpoints/  (model.npz + trainer.npz)  ──►  generate.py
```

**Core idea — the autograd.** `TensorValue` wraps an `ndarray` and, for every operation
(`+`, `@`, `relu`, `softmax`, indexing…), records how to propagate the gradient to its
operands. Calling `loss.backward()` walks the graph in reverse topological order and fills
the `.grad` of every parameter. The optimizer then uses those gradients to update the
weights — the same idea PyTorch implements, but in ~450 readable lines of NumPy (`tensorvalue.py`).

**One training step:** `ids → forward (embedding → blocks → logits) → cross-entropy →
backward → AdamW.step() → zero_grad`.

## Code reading track

Recommended order to study the implementation, from the most basic concept to the most
composed:

1. **`train/autograd.py`** — **scalar** autograd ([micrograd](https://github.com/karpathy/micrograd)
   style): the `Value` class shows the computation-graph idea, per-operation `_backward`, and
   backpropagation via topological ordering, one number at a time.
2. **`train/tensorvalue.py`** — the same idea generalized to NumPy **tensors**: `TensorValue`
   with broadcasting (`_unbroadcast`), `matmul`, softmax-friendly ops, gradient-aware indexing
   (`__getitem__`), a fused `cross_entropy`, and `no_grad()`. This is the engine the model uses.
3. **`tokenizer/`** — BPE tokenization: `base.py` (base class), `basic.py` (plain BPE),
   `regex.py` (BPE with a GPT-2-style regex pre-split), `gpt2.py`/`gpt4.py` (load pretrained
   vocabularies) and `char.py` (character level).
4. **`model/embedding.py`** — positional encoding: absolute sinusoidal (`PositionalEncoding`)
   and rotary (`RoPe`).
5. **`model/transformer.py`** — the Transformer pieces: `scale_dot_product`, `softmax`,
   `layer_norm`, `MultiHeadAttention`, `MLP` (ReLU/SwiGLU), `TransformerBlock`
   (pre-LN + residual) and `TransformerLM` (the full model, with `loss`,
   `predict_next_token`, `save`/`load` and **weight tying**).
6. **`train/optimizer.py`** — `SGD` and `AdamW`, the learning-rate scheduler
   (warmup + cosine) and gradient clipping.
7. **`train/train.py`** — the training loop: token preparation, batches, backward, clipping,
   schedule, periodic validation and checkpoint/resume.

Reference material: `microgpt.py` (a minimal pure-Python GPT, by A. Karpathy).

## How to use

### 1. Install dependencies

```bash
pip install -r requirements.txt    # numpy, pandas, pyarrow, regex, tiktoken
```

### 2. Train

Hyperparameters live in [`train/config.py`](train/config.py) (`GPTConfig`: number of layers,
`model_dim`, `batch_size`, `learning_rate`, `num_steps`, data/checkpoint paths, etc.). To train
with the default config:

```bash
python train/train.py
```

The loop:
- tokenizes the corpus (`train_data_path`) and **caches the tokens** to `token_path` (`.npy`);
- trains for `num_steps`, with warmup + cosine schedule and gradient clipping;
- every `eval_interval` steps it validates and logs to `train.log`;
- saves `checkpoints/model.npz` (architecture + weights) and `checkpoints/trainer.npz`
  (current step + AdamW state).

#### Experiment pipeline (`main.py`)

To compare hyperparameters, **`main.py`** drives the same `train()` over an explicit list of
experiments declared in `experiments.json`. Each experiment runs in its own folder, with the
**same seed** (identical init + batches, so the comparison is fair), and leaves a checkpoint,
`history.csv`, the resolved `config.json` and a `train.log`. A top-level manifest
(`results.csv` / `results.json`) plus an overlaid `curves.png` summarize the runs.

```bash
python main.py                                            # reads experiments.json
python main.py --file my_exps.json --out checkpoints/experiments --seed 0
python main.py --resume                                   # continue without wiping run folders
```

`experiments.json` is a list of `{"name", "overrides"}`, where `overrides` may set **any**
`GPTConfig` field (`learning_rate`, `num_layers`, `model_dim`, `num_steps`, `warmup_steps`, …):

```json
[
  {"name": "baseline", "overrides": {"num_steps": 500}},
  {"name": "lr_3e-3",  "overrides": {"num_steps": 500, "learning_rate": 3e-3}}
]
```

### 3. Resume training

Because the optimizer state is stored in `trainer.npz`, just run training again: it resumes
from the saved `step`, restoring the AdamW moments.

### 4. Generate text

```bash
python generate.py        # loads checkpoints/model.npz and generates from a prompt
```

Edit the `prompt` and `temperature` in [`generate.py`](generate.py). Generation is
autoregressive: at each step the model predicts the next token and appends it to the context
(clipped to `max_seq_len`).

### 5. Tests

```bash
pytest test/              # tests the Transformer and the tokenizers
```

### 6. Dataset
For training, we use the corpus `wikipedia_pt_1M.txt`, available at
[`TucanoBR/wikipedia-PT`](https://huggingface.co/datasets/TucanoBR/wikipedia-PT/blob/main/wikipedia_pt_1M.txt)
on Hugging Face. Download it and place it in `database/`.

> ⚠️ This file is ~163 MB and exceeds GitHub's 100 MB limit, so it is excluded from the
> repository via `.gitignore`. You must download it manually before training.



## Structure

| Path           | Contents                                                             |
|----------------|---------------------------------------------------------------------|
| `train/`       | scalar (`autograd.py`) and tensor (`tensorvalue.py`) autograd, optimizers, schedule/clipping, the training loop and `config.py` |
| `model/`       | Transformer (attention, MLP, blocks, `TransformerLM`) and positional embeddings (sinusoidal + RoPE) |
| `tokenizer/`   | BPE tokenizers (basic, regex/GPT-2, GPT-4, char) and vocabulary loaders |
| `data/`        | corpus reading/preparation (`prepare.py`: text and parquet, in chunks) |
| `test/`        | tests (pytest) for the Transformer and the tokenizers               |
| `notebooks/`   | **theory study track** (01→09) — see [`notebooks/README.md`](notebooks/README.md) |
| `checkpoints/` | saved weights (`model.npz`) and training state (`trainer.npz`)      |
| `database/`    | training corpora (text/parquet)                                     |
| `main.py`      | experiment pipeline: runs `experiments.json` over `train()`, with manifest + curves |
| `generate.py`  | text generation from a checkpoint                                   |
| `microgpt.py`  | minimal reference GPT (Karpathy)                                    |
