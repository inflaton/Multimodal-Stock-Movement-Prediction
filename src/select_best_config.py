"""Lock the per-(stock, ablation, criterion) best (model, horizon) by VAL scores.

This is the substantive fix to Reviewer 5's selection-bias critique
(Section 2.1 of the revision plan). It reads the new Val_* columns
written by every HPO/scoring script and writes a JSON lock file that
every downstream backtest/inference/table-generation script must read
INSTEAD OF scanning test CSVs for the maximum.

Expected input: one CSV per (model, ablation) under results/, with at
least these columns:
    Stock, Ablation, Model, Horizon, Seed,
    Val_ROC_AUC, Val_Sharpe, Val_Trades, Val_WinRate,
    Test_ROC_AUC, Test_Sharpe, ...

Output: results/selected_config.json keyed by (Stock, Ablation, Criterion)
where Criterion is one of {"val_auc", "val_sharpe"}.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config as cfg_mod


REQUIRED_VAL_COLS = ["Val_ROC_AUC", "Val_Sharpe"]


def load_all_results(results_glob: str) -> pd.DataFrame:
    files = sorted(glob.glob(results_glob))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"WARN: could not read {f}: {e}")
            continue
        if not all(c in df.columns for c in REQUIRED_VAL_COLS):
            print(f"WARN: skipping {f} (missing one of {REQUIRED_VAL_COLS})")
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError(f"No results files matched glob {results_glob}")
    return pd.concat(frames, ignore_index=True)


def best_config_per_group(df: pd.DataFrame, criterion: str) -> pd.DataFrame:
    """For each (Stock, Ablation), pick the (Model, Horizon) with best mean Val_<criterion>.

    Mean is taken across seeds so we don't pick the best seed (that would just
    move the selection bias from horizon to seed). The chosen (Model, Horizon)
    must therefore be one whose AVERAGE across seeds beats every alternative
    average.
    """
    val_col = {"val_auc": "Val_ROC_AUC", "val_sharpe": "Val_Sharpe"}[criterion]
    grouped = (
        df.groupby(["Stock", "Ablation", "Model", "Horizon"], dropna=False)
          .agg(mean_val=(val_col, "mean"),
               std_val=(val_col, "std"),
               n_seeds=(val_col, "count"))
          .reset_index()
    )
    picks = (
        grouped.sort_values(["Stock", "Ablation", "mean_val"], ascending=[True, True, False])
               .groupby(["Stock", "Ablation"], as_index=False)
               .head(1)
    )
    picks["Criterion"] = criterion
    return picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-glob",
        default=None,
        help="Glob for the HPO result CSVs (default: results/raw/*.csv)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: results/selected_config.json)",
    )
    args = parser.parse_args()

    cfg = cfg_mod.load_config()
    results_glob = args.results_glob or str(cfg_mod.results_dir() / "raw" / "*.csv")
    out_path = Path(args.out) if args.out else cfg_mod.selected_config_path()

    print(f"Reading results from {results_glob}")
    df = load_all_results(results_glob)
    print(f"  loaded {len(df):,} rows across "
          f"{df['Stock'].nunique()} stocks, "
          f"{df['Ablation'].nunique()} ablations, "
          f"{df['Model'].nunique()} models, "
          f"{df['Horizon'].nunique()} horizons")

    picks_auc = best_config_per_group(df, "val_auc")
    picks_sharpe = best_config_per_group(df, "val_sharpe")
    picks = pd.concat([picks_auc, picks_sharpe], ignore_index=True)

    # Build nested JSON: selected[stock][ablation][criterion] = {Model, Horizon, mean_val, std_val, n_seeds}
    selected: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for _, row in picks.iterrows():
        s = str(row["Stock"])
        a = str(row["Ablation"])
        c = str(row["Criterion"])
        selected.setdefault(s, {}).setdefault(a, {})[c] = {
            "Model": str(row["Model"]),
            "Horizon": int(row["Horizon"]),
            "mean_val": float(row["mean_val"]),
            "std_val": float(row["std_val"]) if not np.isnan(row["std_val"]) else None,
            "n_seeds": int(row["n_seeds"]),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"  {len(picks)} (stock, ablation, criterion) selections")


if __name__ == "__main__":
    main()
