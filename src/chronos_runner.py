"""Chronos-2 wrapper using AutoGluon TimeSeriesPredictor.

Supports the four configurations from the paper plus the 16 GB VRAM-friendly
fine-tune defaults from configs/default.yaml (smaller batch size and the
Bolt-Base preset). Each run is multi-seed (default 3 seeds, foundation budget).

Configurations:
  - chronos2_zero_shot          : default Chronos2 preset, price only
  - chronos2_zero_shot_cov      : zero-shot with sentiment + indicator covariates
  - chronos2_finetuned          : fine-tune, price only
  - chronos2_finetuned_cov      : fine-tune with covariates

Writes one CSV per run with Val_* and Test_* columns.

NOTE: This file is a SCAFFOLD that wraps the foundation-model scripts in
`src/` (chronos_baseline.py + chronos_finetune.py). Those scripts read from
`dataset/training_features/`
(same source as the LSTM / traditional-ML grids) and compose sentiment at
`--alpha-news`, so the foundation comparison can use the learned-α value from
`results/learned_alpha.json` (per-stock by default; pass `--alpha-scope global`
to use the shared α). The originals still operate on the pre-revision
2020-2023 train / 2024 test split and only emit Test_* columns — update those
defaults on the GPU laptop before running.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import config as cfg_mod


def _load_learned_alpha_table() -> Optional[Dict[str, Any]]:
    """Load results/learned_alpha.json if present (notebook-02 rich schema)."""
    rich = cfg_mod.results_dir() / "learned_alpha.json"
    if not rich.exists():
        return None
    data = json.loads(rich.read_text())
    return {
        "per_stock": {r["stock"]: r["alpha_news"] for r in data["per_stock"]},
        "global": data["global"]["alpha_news"],
    }


def _resolve_alpha(stock: str, scope: str, fallback: float,
                   table: Optional[Dict[str, Any]]) -> float:
    """Pick the right alpha for `stock` from the learned-α table, or fall back."""
    if scope == "fixed_70_30" or table is None:
        return fallback
    if scope == "global":
        return float(table["global"])
    # per-stock (the default scope)
    return float(table["per_stock"].get(stock, fallback))


def build_run_command(stock: str, mode: str, seed: int, cfg: Dict[str, Any],
                      alpha_news: float, conda_env: str) -> str:
    """Return the shell command for one chronos run (used in run_pipeline.py).

    Wraps the call in `conda run -n <env> --no-capture-output` so the user
    doesn't have to activate the env beforehand. `--no-capture-output` is
    important — without it, conda buffers stdout and you lose live progress.
    """
    chronos_cfg = cfg["hpo"]
    zero_shot = "zero_shot" in mode
    # Fine-tune-only flags. chronos_baseline.py rejects unknown args, so we omit
    # these in the zero-shot command (only --seed / --context-length / --alpha-news
    # / --use-covariates / --horizons are honored there).
    ft_flags = "" if zero_shot else (
        f"--fine-tune-steps {chronos_cfg['chronos_fine_tune_steps']} "
        f"--fine-tune-batch-size {chronos_cfg['chronos_fine_tune_batch_size']} "
    )
    horizons_str = " ".join(str(h) for h in cfg["horizons"])
    overrides = (
        f"--seed {seed} "
        f"{ft_flags}"
        f"--context-length {chronos_cfg['chronos_context_length']} "
        f"--horizons {horizons_str} "
        f"--alpha-news {alpha_news:.6f}"
    )
    script = (
        "src/chronos_baseline.py" if zero_shot else
        "src/chronos_finetune.py"
    )
    cov_flag = "--use-covariates" if mode.endswith("_cov") else ""
    # --output-dir routes both AutoGluon model checkpoints (chronos2_*_<stock>_h<N>/)
    # and the result CSV under results/chronos-2/, keeping cwd and other models' dirs clean.
    return (f"conda run -n {conda_env} --no-capture-output "
            f"python {script} --stock {stock} --data-dir dataset/training_features "
            f"--output-dir results/chronos-2 "
            f"{cov_flag} {overrides}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["chronos2_zero_shot", "chronos2_zero_shot_cov",
                                 "chronos2_finetuned", "chronos2_finetuned_cov"])
    parser.add_argument("--stock", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--alpha-scope", default="per_stock",
                        choices=["per_stock", "global", "fixed_70_30"],
                        help="Where to read α from. per_stock (default) uses "
                             "results/learned_alpha.json per-stock; global uses "
                             "the shared α; fixed_70_30 uses the legacy 0.7.")
    parser.add_argument("--conda-env", default="stock-prediction",
                        help="Conda env to run the foundation script in. "
                             "Chronos-2 + autogluon.timeseries live in the "
                             "stock-prediction env by default.")
    args = parser.parse_args()

    cfg = cfg_mod.load_config()
    stocks = [args.stock] if args.stock else cfg_mod.stocks(cfg)
    seeds = args.seeds if args.seeds else cfg["foundation_seeds"]
    table = _load_learned_alpha_table()
    if args.alpha_scope != "fixed_70_30" and table is None:
        print("WARNING: results/learned_alpha.json not found; "
              "falling back to fixed 0.7 weighting.\n")

    print("This scaffold dispatches to src/chronos_*.py.")
    print("Run commands (execute from the repo root on GPU laptop):\n")
    for s in stocks:
        alpha = _resolve_alpha(s, args.alpha_scope, 0.7, table)
        for seed in seeds:
            print("  " + build_run_command(s, args.mode, seed, cfg, alpha, args.conda_env))


if __name__ == "__main__":
    main()
