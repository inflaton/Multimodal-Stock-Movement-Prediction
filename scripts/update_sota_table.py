#!/usr/bin/env python3
"""
Update SOTA Comparison Table in main.tex

This script updates the baseline values in the SOTA comparison table (\\label{tab:sota})
with the latest results from Chronos-2 experiments.

Usage:
    python update_sota_table.py
    python update_sota_table.py --tex-file main.tex
    python update_sota_table.py --results-file chronos_all_results_combined.csv
"""

import pandas as pd
import argparse
import re
import os


def load_chronos_results(results_file: str) -> pd.DataFrame:
    """Load and prepare Chronos results."""
    df = pd.read_csv(results_file)

    # Filter for long_short strategy only
    df = df[df["Strategy"] == "long_short"].copy()

    return df


def calculate_baseline_metrics(df: pd.DataFrame, model_types: list) -> dict:
    """
    Calculate average metrics for a specific baseline configuration.

    Returns metrics for both AUC-selected and Sharpe-selected models.

    Args:
        df: DataFrame with results
        model_types: List of model type strings to include (e.g., ['zero-shot'] or ['multivariate'])
    """
    # Filter by model types
    subset = df[df["ModelType"].isin(model_types)].copy()

    if len(subset) == 0:
        return None

    # Group by stock to find best per stock
    best_by_auc = []
    best_by_sharpe = []

    for stock in subset["Stock"].unique():
        stock_data = subset[subset["Stock"] == stock]

        if len(stock_data) > 0:
            # Best by AUC
            best_auc_idx = stock_data["Test_ROC_AUC"].idxmax()
            best_by_auc.append(stock_data.loc[best_auc_idx])

            # Best by Sharpe
            best_sharpe_idx = stock_data["Sharpe"].idxmax()
            best_by_sharpe.append(stock_data.loc[best_sharpe_idx])

    if not best_by_auc or not best_by_sharpe:
        return None

    best_auc_df = pd.DataFrame(best_by_auc)
    best_sharpe_df = pd.DataFrame(best_by_sharpe)

    # Calculate average metrics
    metrics = {
        "auc": {
            "acc": best_auc_df["Test_Accuracy"].mean(),
            "auc": best_auc_df["Test_ROC_AUC"].mean(),
            "trades": best_auc_df["Trades"].mean(),
            "winrate": best_auc_df["WinRate"].mean() * 100,
            "sharpe": best_auc_df["Sharpe"].mean(),
        },
        "sharpe": {
            "acc": best_sharpe_df["Test_Accuracy"].mean(),
            "auc": best_sharpe_df["Test_ROC_AUC"].mean(),
            "trades": best_sharpe_df["Trades"].mean(),
            "winrate": best_sharpe_df["WinRate"].mean() * 100,
            "sharpe": best_sharpe_df["Sharpe"].mean(),
        }
    }

    return metrics


def generate_baseline_rows(df: pd.DataFrame) -> dict:
    """Generate all baseline rows for the SOTA table."""

    baselines = {}

    # Chronos-2 Zero-shot
    # Price only: model_type = 'zero-shot'
    # With sentiment: model_type = 'multivariate' (zero-shot with covariates)
    chronos_zeroshot_price = calculate_baseline_metrics(df, ["zero-shot"])
    chronos_zeroshot_cov = calculate_baseline_metrics(df, ["multivariate"])

    # Chronos-2 Fine-tuned
    # Price only: model_type = 'finetuned'
    # With sentiment: model_type = 'finetuned_cov'
    chronos_finetuned_price = calculate_baseline_metrics(df, ["finetuned"])
    chronos_finetuned_cov = calculate_baseline_metrics(df, ["finetuned_cov"])

    baselines["chronos_zeroshot_price"] = chronos_zeroshot_price
    baselines["chronos_zeroshot_cov"] = chronos_zeroshot_cov
    baselines["chronos_finetuned_price"] = chronos_finetuned_price
    baselines["chronos_finetuned_cov"] = chronos_finetuned_cov

    return baselines


def format_baseline_rows(baselines: dict) -> str:
    """Format baseline rows as LaTeX table content."""

    lines = []

    # Chronos-2 Zero-shot section
    lines.append("\\multicolumn{7}{l}{\\textit{Chronos-2~\\cite{ansari2025chronos} Zero-shot}} \\\\")

    if baselines["chronos_zeroshot_price"]:
        m = baselines["chronos_zeroshot_price"]["auc"]
        lines.append(f"\\quad Price only & AUC & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    if baselines["chronos_zeroshot_cov"]:
        m = baselines["chronos_zeroshot_cov"]["auc"]
        lines.append(f"\\quad + Sentiment & AUC & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    if baselines["chronos_zeroshot_price"]:
        m = baselines["chronos_zeroshot_price"]["sharpe"]
        lines.append(f"\\quad Price only & Sharpe & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    if baselines["chronos_zeroshot_cov"]:
        m = baselines["chronos_zeroshot_cov"]["sharpe"]
        lines.append(f"\\quad + Sentiment & Sharpe & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    lines.append("\\midrule")

    # Chronos-2 Fine-tuned section
    lines.append("\\multicolumn{7}{l}{\\textit{Chronos-2~\\cite{ansari2025chronos} Fine-tuned}} \\\\")

    if baselines["chronos_finetuned_price"]:
        m = baselines["chronos_finetuned_price"]["auc"]
        lines.append(f"\\quad Price only & AUC & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    if baselines["chronos_finetuned_cov"]:
        m = baselines["chronos_finetuned_cov"]["auc"]
        lines.append(f"\\quad + Sentiment & AUC & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    if baselines["chronos_finetuned_price"]:
        m = baselines["chronos_finetuned_price"]["sharpe"]
        lines.append(f"\\quad Price only & Sharpe & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    if baselines["chronos_finetuned_cov"]:
        m = baselines["chronos_finetuned_cov"]["sharpe"]
        lines.append(f"\\quad + Sentiment & Sharpe & {m['acc']:.3f} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    return "\n".join(lines)


def update_sota_table(tex_file: str, new_content: str) -> bool:
    """Update the SOTA table in the LaTeX file."""

    if not os.path.exists(tex_file):
        print(f"Error: LaTeX file not found: {tex_file}")
        return False

    with open(tex_file, 'r') as f:
        content = f.read()

    # Pattern to match Chronos-2 sections in the SOTA table
    # We'll replace everything from the first Chronos-2 Zero-shot line to the midrule before "Ours"
    pattern = (
        r"(\\multicolumn\{7\}\{l\}\{\\textit\{Chronos-2.*?Zero-shot\}\} \\\\)"  # First occurrence of Chronos-2 Zero-shot
        r"(.*?)"  # Old Chronos-2 content (capture all between first and last)
        r"(\\midrule\s*\n\\multicolumn\{7\}\{l\}\{\\textit\{Ours)"  # Start of "Ours" section
    )

    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Error: Could not find Chronos-2 sections in SOTA table")
        print("Looking for pattern starting with: \\multicolumn{7}{l}{\\textit{Chronos-2")
        return False

    # Replace the Chronos-2 sections, including the first header line
    new_content_full = content[:match.start(1)] + new_content + "\n" + content[match.start(3):]

    with open(tex_file, 'w') as f:
        f.write(new_content_full)

    print(f"Successfully updated SOTA table in {tex_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update SOTA Comparison Table in main.tex")
    parser.add_argument(
        "--results-file",
        type=str,
        default="chronos_all_results_combined.csv",
        help="Path to combined Chronos results CSV",
    )
    parser.add_argument(
        "--tex-file",
        type=str,
        default="main.tex",
        help="Path to main.tex file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("UPDATING SOTA TABLE")
    print("=" * 80)
    print(f"Results file: {args.results_file}")
    print(f"LaTeX file: {args.tex_file}")
    print()

    # Load results
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return

    print("Loading Chronos results...")
    df = load_chronos_results(args.results_file)
    print(f"Loaded {len(df)} result rows")
    print(f"Model types: {df['ModelType'].unique()}")
    print(f"Stocks: {df['Stock'].unique()}")
    print()

    # Generate baseline rows
    print("Calculating baseline metrics...")
    baselines = generate_baseline_rows(df)

    # Print summary
    print("\nBaseline Metrics Summary:")
    print("-" * 80)
    for name, metrics in baselines.items():
        if metrics:
            print(f"\n{name}:")
            print(f"  AUC-selected: Acc={metrics['auc']['acc']:.3f}, AUC={metrics['auc']['auc']:.3f}, Sharpe={metrics['auc']['sharpe']:.2f}")
            print(f"  Sharpe-selected: Acc={metrics['sharpe']['acc']:.3f}, AUC={metrics['sharpe']['auc']:.3f}, Sharpe={metrics['sharpe']['sharpe']:.2f}")
    print()

    # Format as LaTeX
    print("Generating LaTeX table content...")
    latex_content = format_baseline_rows(baselines)
    print("\nGenerated LaTeX content:")
    print("-" * 80)
    print(latex_content)
    print("-" * 80)
    print()

    # Update tex file
    print("Updating main.tex...")
    success = update_sota_table(args.tex_file, latex_content)

    if success:
        print("\n" + "=" * 80)
        print("SOTA TABLE UPDATED SUCCESSFULLY")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED TO UPDATE SOTA TABLE")
        print("=" * 80)


if __name__ == "__main__":
    main()
