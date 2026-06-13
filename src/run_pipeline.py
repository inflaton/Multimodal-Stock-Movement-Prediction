"""End-to-end pipeline driver. Use this on the GPU laptop.

Orchestrates the 5 steps of the revision plan in order:

  1. Fit learned-alpha per stock + global on the 2023 validation slice.
  2. Run traditional ML HPO (all models × all ablations × multi-seed).
  3. Run LSTM HPO (representative horizon, then broadcast).
  4. Run Chronos-2 + FinCast (foundation models).
  5. Lock model/horizon selection by val scores → selected_config.json.
  6. Aggregate + write summary table.

Use --steps to run a subset (e.g., --steps 1 to only fit learned alphas).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd

from . import config as cfg_mod
from . import learned_alpha as la


def step1_learned_alpha(cfg) -> None:
    print("\n=== Step 1: learned-alpha fitting ===")
    out = []
    for stock in cfg_mod.stocks(cfg):
        r = la.fit_alpha_per_stock(stock, cfg)
        print(f"  {stock}: alpha_news = {r.alpha_news:.3f}  (val AUC {r.val_auc:.3f})")
        out.append({"stock": r.stock, "alpha_news": r.alpha_news,
                    "val_auc": r.val_auc, "scope": "per_stock"})
    g = la.fit_alpha_global(cfg)
    print(f"  global: alpha_news = {g.alpha_news:.3f}  (val AUC {g.val_auc:.3f})")
    out.append({"stock": None, "alpha_news": g.alpha_news,
                "val_auc": g.val_auc, "scope": "global"})
    # Notebook 02 is the canonical writer of `learned_alpha.json` (rich schema with
    # baselines + L2 sensitivity). The pipeline writes a separate slim file so the
    # two never clobber each other.
    out_path = cfg_mod.results_dir() / "learned_alpha_pipeline.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


def step2_traditional(cfg) -> None:
    print("\n=== Step 2: traditional ML HPO ===")
    for model in ["logistic_regression", "svm", "random_forest",
                  "gradient_boosting", "xgboost", "lightgbm"]:
        cmd = [sys.executable, "-m", "src.hpo_traditional", "--model", model]
        print("RUN:", " ".join(cmd))
        subprocess.check_call(cmd)


def step3_lstm(cfg) -> None:
    print("\n=== Step 3: LSTM HPO (reduced grid) ===")
    cmd = [sys.executable, "-m", "src.hpo_lstm"]
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)


def step4_foundation(cfg) -> None:
    print("\n=== Step 4: foundation models ===")
    for mode in ["chronos2_zero_shot", "chronos2_zero_shot_cov",
                 "chronos2_finetuned", "chronos2_finetuned_cov"]:
        print(f"  -> {mode} (see chronos_runner.py for shell commands)")
    for mode in ["fincast_zero_shot", "fincast_finetuned"]:
        print(f"  -> {mode} (see fincast_runner.py for shell commands)")


def step5_select(cfg) -> None:
    print("\n=== Step 5: lock val-based selection ===")
    cmd = [sys.executable, "-m", "src.select_best_config"]
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)


def step6_aggregate(cfg) -> None:
    print("\n=== Step 6: aggregate results ===")
    raw_dir = cfg_mod.results_dir() / "raw"
    if not raw_dir.exists():
        print("  no raw results yet — skipping")
        return
    frames = []
    for f in sorted(raw_dir.glob("*.csv")):
        frames.append(pd.read_csv(f))
    if not frames:
        print("  no CSVs found")
        return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(cfg_mod.results_dir() / "all_results.csv", index=False)
    print(f"  combined {len(all_df):,} rows -> results/all_results.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5, 6],
                        help="Subset of steps to run (1..6)")
    args = parser.parse_args()
    cfg = cfg_mod.load_config()
    if 1 in args.steps:
        step1_learned_alpha(cfg)
    if 2 in args.steps:
        step2_traditional(cfg)
    if 3 in args.steps:
        step3_lstm(cfg)
    if 4 in args.steps:
        step4_foundation(cfg)
    if 5 in args.steps:
        step5_select(cfg)
    if 6 in args.steps:
        step6_aggregate(cfg)


if __name__ == "__main__":
    main()
