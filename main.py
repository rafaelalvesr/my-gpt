"""
Pipeline de experimentos de treino (AVALIACAO.md, receita §3/§6/§7).

Orquestra a função `train()` de train/train.py para rodar uma LISTA EXPLÍCITA de
experimentos definida num JSON externo, cada um com seu próprio conjunto de
overrides do GPTConfig. Varia qualquer hiperparâmetro (não só o learning rate).

Cada experimento roda numa pasta própria e deixa tudo que é preciso para avaliar:
    <out>/<name>/model.npz       pesos (checkpoint)
    <out>/<name>/trainer.npz     estado do AdamW + step (permite --resume)
    <out>/<name>/history.csv     train_loss/val_loss por step
    <out>/<name>/config.json     config resolvida (proveniência)
    <out>/<name>/train.log       log do experimento
    <out>/results.csv            manifesto: 1 linha por experimento (tracker §7)
    <out>/results.json           idem, detalhado
    <out>/curves.png             curvas train/val sobrepostas (se houver matplotlib)

Garantias de comparação justa:
    - pasta limpa por experimento (sem resume acidental nem history.csv misturado);
    - mesma seed antes de cada train() → init e batches idênticos entre runs.

Formato do JSON (lista explícita):
    [
      {"name": "baseline", "overrides": {"num_steps": 500}},
      {"name": "lr_3e-3",  "overrides": {"num_steps": 500, "learning_rate": 3e-3}}
    ]
Os `overrides` aceitam qualquer campo do GPTConfig (learning_rate, num_layers,
model_dim, weight_decay, num_steps, warmup_steps, eval_interval, ...).

Uso:
    python main.py                          # lê experiments.json
    python main.py --file meus_exps.json --seed 0 --out checkpoints/experiments
    python main.py --resume                 # continua de onde parou (não limpa as pastas)
"""
import argparse
import csv
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train.config import GPTConfig
from train.train import train, prepare_dataset
from tokenizer import load_gpt2_tokenizer
from data.prepare import ReadTextFile

VALID_FIELDS = {f.name for f in fields(GPTConfig)}


def parse_args():
    p = argparse.ArgumentParser(description="Pipeline de experimentos de treino")
    p.add_argument("--file", type=str, default="experiments.json",
                   help="JSON com a lista de experimentos (default: experiments.json)")
    p.add_argument("--seed", type=int, default=0, help="seed (init + batches)")
    p.add_argument("--out", type=str, default="checkpoints/experiments",
                   help="pasta-raiz das saídas")
    p.add_argument("--resume", action="store_true",
                   help="continua de checkpoints existentes (não limpa as pastas)")
    return p.parse_args()


def load_experiments(path: Path):
    """Lê e valida a lista de experimentos do JSON."""
    exps = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(exps, list) or not exps:
        raise ValueError(f"{path}: esperava uma lista não-vazia de experimentos")
    names = set()
    for e in exps:
        if "name" not in e or "overrides" not in e:
            raise ValueError(f"experimento sem 'name'/'overrides': {e}")
        if e["name"] in names:
            raise ValueError(f"nome de experimento duplicado: {e['name']!r}")
        names.add(e["name"])
    return exps


def resolve_config(base: GPTConfig, vocab: int, overrides: dict) -> GPTConfig:
    """Aplica os overrides ao GPTConfig, validando os nomes dos campos."""
    bad = set(overrides) - VALID_FIELDS
    if bad:
        raise ValueError(f"overrides desconhecidos {sorted(bad)}; "
                         f"campos válidos: {sorted(VALID_FIELDS)}")
    return replace(base, **{"vocab_size": vocab, **overrides})


def setup_tokens_and_vocab(base: GPTConfig) -> int:
    """Garante o cache de tokens e devolve o vocab_size resolvido pelo tokenizer."""
    token_path = Path(base.token_path).with_suffix(".npy")
    if not token_path.exists():
        tok = load_gpt2_tokenizer()
        prepare_dataset(tok, ReadTextFile(base.train_data_path), base.token_path)
    return len(load_gpt2_tokenizer().vocab)


def compute_metrics(train_curve, val_curve) -> dict:
    """Métricas-resumo de uma curva + status (NaN / DIVERGE / ok)."""
    tail = float(np.mean(train_curve[-max(1, len(train_curve) // 5):]))  # média ~20% finais
    vmin = float(np.nanmin(val_curve)) if np.isfinite(val_curve).any() else float("nan")
    if np.isnan(train_curve).any():
        status = "NaN"
    elif train_curve[-1] > train_curve[0]:
        status = "DIVERGE"
    else:
        status = "ok"
    return {"final_train": float(train_curve[-1]), "val_min": vmin,
            "train_tail": tail, "status": status}


def run_experiment(exp: dict, base: GPTConfig, vocab: int, args) -> dict:
    """Roda um experimento; devolve seu registro (config, métricas, curvas)."""
    name = exp["name"]
    run_dir = Path(args.out) / name
    if run_dir.exists() and not args.resume:
        shutil.rmtree(run_dir)               # baseline limpo (a menos de --resume)
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config(base, vocab, exp["overrides"])
    cfg = replace(cfg, checkpoint_path=str(run_dir / "model.npz"))
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # log dedicado deste experimento (anexado ao logger do train, removido ao fim)
    train_logger = logging.getLogger("train.train")
    handler = logging.FileHandler(run_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    train_logger.addHandler(handler)

    print(f"\n{'='*60}\n  EXPERIMENTO: {name}\n  overrides: {exp['overrides']}\n"
          f"  dir: {run_dir}\n{'='*60}")
    t0 = time.time()
    np.random.seed(args.seed)                # init + batches idênticos entre experimentos
    try:
        train(cfg)
    finally:
        train_logger.removeHandler(handler)
        handler.close()
    seconds = time.time() - t0

    hist = np.atleast_2d(np.loadtxt(run_dir / "history.csv"))
    train_curve, val_curve = hist[:, 0], hist[:, 1]
    rec = {"name": name, "overrides": exp["overrides"], "num_steps": cfg.num_steps,
           "seconds": round(seconds, 1), **compute_metrics(train_curve, val_curve),
           "_train": train_curve, "_val": val_curve}
    return rec


def write_manifest(results, out_root: Path):
    """Grava o tracker de experimentos (CSV legível + JSON detalhado)."""
    cols = ["name", "status", "final_train", "val_min", "train_tail",
            "num_steps", "seconds", "overrides"]
    with open(out_root / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in results:
            w.writerow([r["name"], r["status"], f"{r['final_train']:.4f}",
                        f"{r['val_min']:.4f}", f"{r['train_tail']:.4f}",
                        r["num_steps"], r["seconds"], json.dumps(r["overrides"])])
    detailed = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (out_root / "results.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    print(f"\n[manifesto] {out_root/'results.csv'}  +  results.json")


def plot_curves(results, out_root: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib indisponível ({e}); usando só CSV/JSON.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for r in results:
        x = np.arange(1, len(r["_train"]) + 1)
        ax[0].plot(x, r["_train"], label=r["name"])
        ax[1].plot(x, r["_val"], label=r["name"])
    ax[0].set_title("train_loss"); ax[1].set_title("val_loss")
    for a in ax:
        a.set_xlabel("step"); a.set_ylabel("loss"); a.grid(True, alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(out_root / "curves.png", dpi=120)
    print(f"[plot] {out_root/'curves.png'}")


def print_summary(results):
    print(f"\n{'='*60}\n  RESUMO DOS EXPERIMENTOS\n{'='*60}")
    print(f"{'name':>16} {'status':>8} {'train@fim':>10} {'val_min':>9} {'train_cauda':>12}")
    ok = []
    for r in results:
        print(f"{r['name']:>16} {r['status']:>8} {r['final_train']:>10.3f} "
              f"{r['val_min']:>9.3f} {r['train_tail']:>12.3f}")
        if r["status"] == "ok":
            ok.append((r["train_tail"], r["name"]))
    if ok:
        ok.sort()
        print(f"\n  → melhor train_loss (cauda) sem divergir: {ok[0][1]!r}")
        print("    (confirme também o val_min: menor train com val acompanhando)")


def main():
    args = parse_args()
    exps = load_experiments(Path(args.file))
    base = GPTConfig()
    vocab = setup_tokens_and_vocab(base)
    print(f"{len(exps)} experimento(s) | vocab={vocab} | ln(V)={np.log(vocab):.3f} "
          f"(loss esperado no step 1)")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    results = [run_experiment(e, base, vocab, args) for e in exps]

    write_manifest(results, out_root)
    plot_curves(results, out_root)
    print_summary(results)


if __name__ == "__main__":
    main()
