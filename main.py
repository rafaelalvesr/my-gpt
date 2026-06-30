"""
    Experiments pipeline for training GPT models.
- Applied the train() functio from train/train.py to run an EXPLICIT LIST of experiments 
defined in an external JSON, each with its own set of GPTConfig overrides. Varies any hyperparameter 
(not just learning rate). Each experiment runs in its own folder and leaves everything needed for evaluation:
    <out>/<name>/model.npz       weights (checkpoint)
    <out>/<name>/trainer.npz     AdamW state + step (allows --resume)
    <out>/<name>/history.csv     train_loss/val_loss per step
    <out>/<name>/config.json     resolved config (provenance)
    <out>/<name>/train.log       experiment log
    <out>/results.csv            manifest: 1 line per experiment (tracker §7)
    <out>/results.json           same, detailed
    <out>/curves.png             overlaid train/val curves (if matplotlib available)

- All experiments are run with the same seed (init + batches) to ensure fair comparison.

Json format:
    [{"name": "baseline", "overrides": {"num_steps": 500}},
      {"name": "lr_3e-3",  "overrides": {"num_steps": 500, "learning_rate": 3e-3}}]

The 'overrides' can contain any field of GPTConfig (learning_rate, num_layers, model_dim, ...).

Usage:
    python main.py --help
    python main.py --file experiments.json --seed 0 --out runs
    python main.py --resume    continues from where it left off (does not clean folders)
    python main.py --file meus_exps.json --seed 0 --out runs
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


def get_train_logger() -> logging.Logger:
    """Returns the logger used by train/train.py."""
    return logging.getLogger("train.train")


def emit(message: str, logger: logging.Logger = None, log_paths=None):
    """Prints to stdout and mirrors the message to one or more train.log files."""
    print(message)
    target_logger = logger or get_train_logger()
    if logger is not None:
        target_logger.info(message)
        return
    if not log_paths:
        return

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for log_path in log_paths:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        target_logger.addHandler(handler)
        try:
            target_logger.info(message)
        finally:
            target_logger.removeHandler(handler)
            handler.close()


def prepare_run_dir(run_dir: Path, resume: bool):
    """Creates a clean output directory unless resuming."""
    if run_dir.exists() and not resume:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def parse_args():
    """Parses the command line arguments."""
    p = argparse.ArgumentParser(description="Pipeline of training experiments")
    p.add_argument("--file", type=str, default="experiments.json",
                   help="JSON with the list of experiments (default: experiments.json)")
    p.add_argument("--seed", type=int, default=0, help="seed (init + batches)")
    p.add_argument("--out", type=str, default="runs",
                   help="root folder for outputs")
    p.add_argument("--resume", action="store_true",
                   help="resume from existing checkpoints (does not clean folders)")
    return p.parse_args()


def load_experiments(path: Path):
    """Reads and validates the list of experiments from the JSON."""
    exps = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(exps, list) or not exps:
        raise ValueError(f"{path}: expected a non-empty list of experiments")
    names = set()
    for e in exps:
        if "name" not in e or "overrides" not in e:
            raise ValueError(f"experiment missing 'name'/'overrides': {e}")
        if e["name"] in names:
            raise ValueError(f"duplicate experiment name: {e['name']!r}")
        names.add(e["name"])
    return exps


def resolve_config(base: GPTConfig, vocab: int, overrides: dict) -> GPTConfig:
    """Applies the overrides to the GPTConfig, validating the field names."""
    bad = set(overrides) - VALID_FIELDS
    if bad:
        raise ValueError(f"unknown overrides {sorted(bad)}; "
                         f"valid fields: {sorted(VALID_FIELDS)}")
    return replace(base, **{"vocab_size": vocab, **overrides})


def setup_tokens_and_vocab(base: GPTConfig) -> int:
    """Ensures the token cache and returns the vocab_size resolved by the tokenizer."""
    token_path = Path(base.token_path).with_suffix(".npy")
    if not token_path.exists():
        tok = load_gpt2_tokenizer()
        prepare_dataset(tok, ReadTextFile(base.train_data_path), base.token_path)
    return len(load_gpt2_tokenizer().vocab)


def compute_metrics(train_curve, val_curve) -> dict:
    """Summary metrics of a curve + status (NaN / DIVERGE / ok)."""
    tail = float(np.mean(train_curve[-max(1, len(train_curve) // 5):]))  # mean of ~20% final steps
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
    """Runs an experiment; returns its record (config, metrics, curves)."""
    name = exp["name"]
    run_dir = Path(args.out) / name

    cfg = resolve_config(base, vocab, exp["overrides"])
    cfg = replace(cfg, checkpoint_path=str(run_dir / "model.npz"))
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    # dedicated log for this experiment (attached to the train logger, removed at the end)
    train_logger = logging.getLogger("train.train")
    handler = logging.FileHandler(run_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    train_logger.addHandler(handler)

    emit(f"\n{'='*60}\n  EXPERIMENT: {name}\n  overrides: {exp['overrides']}\n"
         f"  dir: {run_dir}\n{'='*60}", logger=train_logger)
    t0 = time.time()
    np.random.seed(args.seed)                # init + identical batches across experiments
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


def write_manifest(results, out_root: Path, log_paths=None):
    """Writes the experiment tracker (human-readable CSV + detailed JSON)."""
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
    emit(f"\n[manifest] {out_root/'results.csv'}  +  results.json", log_paths=log_paths)


def plot_curves(results, out_root: Path, log_paths=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        emit(f"[plot] matplotlib unavailable ({e}); using only CSV/JSON.", log_paths=log_paths)
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
    emit(f"[plot] {out_root/'curves.png'}", log_paths=log_paths)


def print_summary(results, log_paths=None):
    emit(f"\n{'='*60}\n  EXPERIMENT SUMMARY\n{'='*60}", log_paths=log_paths)
    emit(f"{'name':>16} {'status':>8} {'train@end':>10} {'val_min':>9} {'train_tail':>12}", log_paths=log_paths)
    ok = []
    for r in results:
        emit(f"{r['name']:>16} {r['status']:>8} {r['final_train']:>10.3f} "
             f"{r['val_min']:>9.3f} {r['train_tail']:>12.3f}", log_paths=log_paths)
        if r["status"] == "ok":
            ok.append((r["train_tail"], r["name"]))
    if ok:
        ok.sort()
        emit(f"\n  → best train_loss (tail) without diverging: {ok[0][1]!r}", log_paths=log_paths)
        emit("    (also check val_min: lowest train with val following)", log_paths=log_paths)


def main():
    args = parse_args()
    exps = load_experiments(Path(args.file))
    base = GPTConfig()
    vocab = setup_tokens_and_vocab(base)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    run_dirs = [out_root / exp["name"] for exp in exps]
    for run_dir in run_dirs:
        prepare_run_dir(run_dir, args.resume)
    run_log_paths = [run_dir / "train.log" for run_dir in run_dirs]

    emit(f"{len(exps)} experiment(s) | vocab={vocab} | ln(V)={np.log(vocab):.3f} "
         f"(expected loss at step 1)", log_paths=run_log_paths)

    results = [run_experiment(e, base, vocab, args) for e in exps]

    write_manifest(results, out_root, log_paths=run_log_paths)
    plot_curves(results, out_root, log_paths=run_log_paths)
    print_summary(results, log_paths=run_log_paths)


if __name__ == "__main__":
    main()
