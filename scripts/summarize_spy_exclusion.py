# [CA-SPY] Added to substantiate the limitations-section robustness claim.
"""Reproduce the manuscript's SPY-excluded summary from saved experiment results.

No training or selection is rerun. Global fusion weights retain their original
five-stock validation fit. The best-per-stock view uses the existing lock file.
Run from any directory with: python scripts/summarize_spy_exclusion.py
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean


# [CA-SPY] Fix the paper's original grid so missing or extra cells cannot alter the comparison.
ROOT = Path(__file__).resolve().parents[1]
STOCKS = ("AAPL", "META", "NVDA", "SPY", "TSLA")
MODELS = (
    "logistic_regression", "svm", "random_forest", "gradient_boosting",
    "xgboost", "lightgbm", "lstm",
)
ABLATIONS = (
    "technical_only", "full_70_30", "equal_50_50", "news_only", "social_only",
    "sentiment_only", "learned_alpha_per_stock", "learned_alpha_global",
)
HORIZONS = (2, 5, 10)
SEEDS = (42, 7, 1337, 2024, 31415)


def main() -> None:
    # [CA-SPY] Read existing results and selections; do not train, refit, or reselect.
    results_path = ROOT / "results/all_results.csv"
    selection_path = ROOT / "results/selected_config.json"
    with results_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = json.loads(selection_path.read_text())
    # [CA-SPY] Require unique, finite scores and complete coverage of the 4,200-run grid.
    keyed = {}
    for row in rows:
        key = (row["Stock"], row["Ablation"], row["Model"],
               int(row["Horizon"]), int(row["Seed"]))
        if key in keyed:
            raise ValueError(f"Duplicate experiment: {key}")
        for metric in ("Test_Sharpe", "Test_ROC_AUC"):
            row[metric] = float(row[metric])
            if not math.isfinite(row[metric]):
                raise ValueError(f"Non-finite {metric}: {key}")
        keyed[key] = row
    expected = set(itertools.product(STOCKS, ABLATIONS, MODELS, HORIZONS, SEEDS))
    if set(keyed) != expected:
        raise ValueError("Results do not match the complete 4,200-run paper grid")

    # [CA-SPY] Change only the evaluation universe; preserve both paper aggregation schemes.
    summaries = {}
    for label, stocks in (("all_stocks", STOCKS),
                          ("excluding_spy", tuple(s for s in STOCKS if s != "SPY"))):
        summary = {}
        for ablation in ABLATIONS:
            cells = [keyed[(s, ablation, m, h, seed)]
                     for s, m, h, seed in itertools.product(stocks, MODELS, HORIZONS, SEEDS)]
            # [CA-SPY] Reuse each saved validation choice and average all seeds, then stocks.
            locked = []
            for stock in stocks:
                pick = selected[stock][ablation]["val_sharpe"]
                locked.append(mean(
                    keyed[(stock, ablation, pick["Model"], pick["Horizon"], seed)]["Test_Sharpe"]
                    for seed in SEEDS
                ))
            summary[ablation] = {
                "runs": len(cells),
                "mean_test_sharpe": mean(c["Test_Sharpe"] for c in cells),
                "mean_test_auc": mean(c["Test_ROC_AUC"] for c in cells),
                "validation_locked_mean_test_sharpe": mean(locked),
            }
        # [CA-SPY] Compute Sharpe gaps before rounding the values for the manuscript.
        baseline = summary["technical_only"]["mean_test_sharpe"]
        for metrics in summary.values():
            metrics["technical_only_minus_configuration_sharpe"] = baseline - metrics["mean_test_sharpe"]
        summaries[label] = summary

    # [CA-SPY] JSON has no comment syntax: retain scope, aggregation, and hashes as metadata.
    output = {
        "scope": "Exclude SPY from evaluation only; no refitting or reselection. "
                 "Global fusion weights retain their original five-stock validation fit.",
        "aggregation": "Grid means weight every model/horizon/stock/seed cell equally. "
                       "Validation-locked means average seeds for each saved per-stock choice, then stocks.",
        "inputs_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (results_path, selection_path)
        },
        "summaries": summaries,
    }
    output_path = ROOT / "results/spy_excluded_robustness.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(rows):,} unique experiment cells; wrote {output_path}")


if __name__ == "__main__":
    main()
