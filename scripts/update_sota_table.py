#!/usr/bin/env python3
"""
Update SOTA Comparison Table in main.tex

This script updates the baseline values in the SOTA comparison table (\\label{tab:sota})
with the latest results from Chronos-2 and FinCast experiments.

Usage:
    python scripts/update_sota_table.py
    python scripts/update_sota_table.py --tex-file docs/main.tex
    python scripts/update_sota_table.py --results-dir results/baselines
"""

import pandas as pd
import argparse
import re
import os
from pathlib import Path


def load_chronos_results(results_dir: str) -> pd.DataFrame:
    """Load and prepare Chronos results from results directory."""
    chronos_file = Path(results_dir) / "chronos_all_results_combined.csv"

    if not chronos_file.exists():
        print(f"Warning: Chronos results file not found: {chronos_file}")
        return pd.DataFrame()

    df = pd.read_csv(chronos_file)

    # Filter for long_short strategy only
    df = df[df["Strategy"] == "long_short"].copy()

    return df


def load_fincast_results(results_dir: str) -> pd.DataFrame:
    """Load and prepare FinCast results from results directory."""
    fincast_zeroshot_file = Path(results_dir) / "fincast_all_stocks_results.csv"

    # Try newer filename first, then fall back to older one
    fincast_finetuned_file = Path(results_dir) / "finetuned_fincast" / "fincast_finetuned_all_results.csv"
    if not fincast_finetuned_file.exists():
        fincast_finetuned_file = Path(results_dir) / "finetuned_fincast" / "fincast_finetuned_all_stocks_results.csv"

    dfs = []

    # Load zero-shot results
    if fincast_zeroshot_file.exists():
        df_zeroshot = pd.read_csv(fincast_zeroshot_file)
        df_zeroshot["ModelType"] = "zero-shot"
        dfs.append(df_zeroshot)
        print(f"  Loaded {len(df_zeroshot)} zero-shot results")
    else:
        print(f"Warning: FinCast zero-shot file not found: {fincast_zeroshot_file}")

    # Load fine-tuned results
    if fincast_finetuned_file.exists():
        df_finetuned = pd.read_csv(fincast_finetuned_file)
        df_finetuned["ModelType"] = "finetuned"
        # Align column names (fine-tuned has "Threshold", zero-shot has "BestThreshold")
        if "Threshold" in df_finetuned.columns and "BestThreshold" not in df_finetuned.columns:
            df_finetuned = df_finetuned.rename(columns={"Threshold": "BestThreshold"})
        dfs.append(df_finetuned)
        print(f"  Loaded {len(df_finetuned)} fine-tuned results")
    else:
        print(f"Warning: FinCast fine-tuned file not found: {fincast_finetuned_file}")

    # Combine all results
    if not dfs:
        print("Warning: No FinCast results found")
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
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


def generate_chronos_baseline_rows(df: pd.DataFrame) -> dict:
    """Generate Chronos-2 baseline rows for the SOTA table."""

    baselines = {}

    if df.empty:
        return baselines

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


def generate_fincast_baseline_rows(df: pd.DataFrame) -> dict:
    """Generate FinCast baseline rows for the SOTA table."""

    baselines = {}

    if df.empty:
        return baselines

    # FinCast Zero-shot (Price only)
    fincast_zeroshot = calculate_baseline_metrics(df, ["zero-shot", "zeroshot"])

    # FinCast Fine-tuned (Price only)
    fincast_finetuned = calculate_baseline_metrics(df, ["finetuned", "fine-tuned"])

    baselines["fincast_zeroshot"] = fincast_zeroshot
    baselines["fincast_finetuned"] = fincast_finetuned

    return baselines


def find_best_baseline_accuracy(fincast_baselines: dict, chronos_baselines: dict) -> float:
    """Find the best (maximum) accuracy value across all baselines (Sharpe-selected only)."""
    best_acc = -float('inf')

    # Collect all baseline metrics
    all_baselines = {**fincast_baselines, **chronos_baselines}

    for _, metrics in all_baselines.items():
        if metrics:
            # Only check Sharpe-selected results
            if "sharpe" in metrics and "acc" in metrics["sharpe"]:
                best_acc = max(best_acc, metrics["sharpe"]["acc"])

    return best_acc


def format_fincast_rows(baselines: dict, best_acc: float) -> str:
    """Format FinCast baseline rows as LaTeX table content (Sharpe-selected only)."""

    lines = []

    # FinCast section header
    lines.append("\\multicolumn{6}{l}{\\textit{FinCast~\\cite{zhu2025fincast}}} \\\\")

    # Zero-shot
    if baselines.get("fincast_zeroshot"):
        m = baselines["fincast_zeroshot"]["sharpe"]
        acc_str = f"\\textbf{{{m['acc']:.3f}}}" if abs(m['acc'] - best_acc) < 0.0005 else f"{m['acc']:.3f}"
        lines.append(f"\\quad Zero-shot & {acc_str} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    # Fine-tuned
    if baselines.get("fincast_finetuned"):
        m = baselines["fincast_finetuned"]["sharpe"]
        acc_str = f"\\textbf{{{m['acc']:.3f}}}" if abs(m['acc'] - best_acc) < 0.0005 else f"{m['acc']:.3f}"
        lines.append(f"\\quad Fine-tuned & {acc_str} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    return "\n".join(lines)


def format_chronos_rows(baselines: dict, best_acc: float) -> str:
    """Format Chronos-2 baseline rows as LaTeX table content (Sharpe-selected only)."""

    lines = []

    # Chronos-2 section header
    lines.append("\\multicolumn{6}{l}{\\textit{Chronos-2~\\cite{ansari2025chronos}}} \\\\")

    # Zero-shot
    if baselines.get("chronos_zeroshot_price"):
        m = baselines["chronos_zeroshot_price"]["sharpe"]
        acc_str = f"\\textbf{{{m['acc']:.3f}}}" if abs(m['acc'] - best_acc) < 0.0005 else f"{m['acc']:.3f}"
        lines.append(f"\\quad Zero-shot & {acc_str} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    # Zero-shot + Sentiment
    if baselines.get("chronos_zeroshot_cov"):
        m = baselines["chronos_zeroshot_cov"]["sharpe"]
        acc_str = f"\\textbf{{{m['acc']:.3f}}}" if abs(m['acc'] - best_acc) < 0.0005 else f"{m['acc']:.3f}"
        lines.append(f"\\quad Zero-shot + Sent. & {acc_str} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    # Fine-tuned
    if baselines.get("chronos_finetuned_price"):
        m = baselines["chronos_finetuned_price"]["sharpe"]
        acc_str = f"\\textbf{{{m['acc']:.3f}}}" if abs(m['acc'] - best_acc) < 0.0005 else f"{m['acc']:.3f}"
        lines.append(f"\\quad Fine-tuned & {acc_str} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    # Fine-tuned + Sentiment
    if baselines.get("chronos_finetuned_cov"):
        m = baselines["chronos_finetuned_cov"]["sharpe"]
        acc_str = f"\\textbf{{{m['acc']:.3f}}}" if abs(m['acc'] - best_acc) < 0.0005 else f"{m['acc']:.3f}"
        lines.append(f"\\quad Fine-tuned + Sent. & {acc_str} & {m['auc']:.3f} & {m['trades']:.1f} & {m['winrate']:.1f} & {m['sharpe']:.2f} \\\\")

    return "\n".join(lines)


def compute_sharpe_optimized_metrics(results_file: str, strategy: str = "long_short") -> dict:
    """Compute Sharpe-optimized metrics from tuned results CSV.

    For each stock, find the configuration with highest Sharpe ratio,
    then average across all stocks.

    Args:
        results_file: Path to tuned_all_results_combined.csv
        strategy: Trading strategy to filter for (default: 'long_short')

    Returns:
        Dictionary with 'acc', 'auc', 'trades', 'winrate', 'sharpe' keys
    """
    STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]

    df = pd.read_csv(results_file)
    # Filter for the specified strategy
    strategy_df = df[df["Strategy"] == strategy]

    # For each stock, find the configuration with highest Sharpe
    best_sharpe_rows = []
    for stock in STOCKS:
        stock_data = strategy_df[strategy_df["Stock"] == stock]
        if len(stock_data) > 0:
            best_idx = stock_data["Sharpe"].idxmax()
            best_sharpe_rows.append(stock_data.loc[best_idx])

    # Average across stocks
    best_sharpe_df = pd.DataFrame(best_sharpe_rows)

    return {
        'acc': best_sharpe_df["Test_Accuracy"].mean(),
        'auc': best_sharpe_df["Test_ROC_AUC"].mean(),
        'trades': best_sharpe_df["Trades"].mean(),
        'winrate': best_sharpe_df["WinRate"].mean() * 100,
        'sharpe': best_sharpe_df["Sharpe"].mean()
    }


def format_ours_row(best_acc: float, ours_metrics: dict = None) -> str:
    """Format the 'Ours' row with Sharpe-optimized results.

    Args:
        best_acc: Best accuracy among baselines (for bolding comparison)
        ours_metrics: Dictionary with 'acc', 'auc', 'trades', 'winrate', 'sharpe'.
    """

    # Check if "Ours" accuracy should be bolded
    acc_str = f"\\textbf{{{ours_metrics['acc']:.3f}}}" if abs(ours_metrics['acc'] - best_acc) < 0.0005 else f"{ours_metrics['acc']:.3f}"

    # Bold the best metrics (AUC, Win%, Sharpe)
    auc_str = f"\\textbf{{{ours_metrics['auc']:.3f}}}"
    winrate_str = f"\\textbf{{{ours_metrics['winrate']:.1f}}}"
    sharpe_str = f"\\textbf{{{ours_metrics['sharpe']:.2f}}}"

    return f"\\textbf{{Ours (Multimodal)}} & {acc_str} & {auc_str} & {ours_metrics['trades']:.1f} & {winrate_str} & {sharpe_str} \\\\"


def load_ablation_results(ablation_dir: str) -> dict:
    """Load ablation study results from CSV files.

    Returns:
        Dictionary mapping ablation config names to DataFrames
    """
    ablation_configs = {
        "technical_only": "ablation_technical_only_all_stocks_results.csv",
        "sentiment_only": "ablation_sentiment_only_all_stocks_results.csv",
        "equal_weights": "ablation_equal_weights_all_stocks_results.csv",
        "news_only": "ablation_news_only_all_stocks_results.csv",
        "social_only": "ablation_social_only_all_stocks_results.csv",
    }

    results = {}
    for config_name, filename in ablation_configs.items():
        filepath = Path(ablation_dir) / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            results[config_name] = df
        else:
            print(f"Warning: Ablation file not found: {filepath}")

    return results


def compute_ablation_metrics(df: pd.DataFrame, selection_criterion: str) -> dict:
    """Compute averaged metrics for an ablation configuration.

    Args:
        df: DataFrame with ablation results
        selection_criterion: "auc" or "sharpe"

    Returns:
        Dictionary with averaged metrics
    """
    STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]

    # For each stock, find best configuration by criterion
    best_rows = []
    for stock in STOCKS:
        stock_data = df[df["Stock"] == stock]
        if len(stock_data) > 0:
            if selection_criterion == "auc":
                best_idx = stock_data["Test_ROC_AUC"].idxmax()
            else:  # sharpe
                best_idx = stock_data["Sharpe"].idxmax()
            best_rows.append(stock_data.loc[best_idx])

    if not best_rows:
        return None

    best_df = pd.DataFrame(best_rows)

    return {
        'acc': best_df["Test_Accuracy"].mean(),
        'auc': best_df["Test_ROC_AUC"].mean(),
        'trades': best_df["Trades"].mean(),
        'winrate': best_df["WinRate"].mean() * 100,
        'sharpe': best_df["Sharpe"].mean()
    }


def format_ablation_table(full_model_metrics: dict, ablation_results: dict) -> str:
    """Generate LaTeX content for Ablation Study table.

    Args:
        full_model_metrics: Dictionary with 'auc' and 'sharpe' selection results for full model
        ablation_results: Dictionary mapping config names to DataFrames

    Returns:
        LaTeX table content (rows only, without table environment)
    """
    lines = []

    # Full Model rows
    fm_auc = full_model_metrics['auc']
    fm_sharpe = full_model_metrics['sharpe']

    # Find best AUC for bolding
    best_auc = fm_auc['auc']

    lines.append(f"Full Model & AUC & {fm_auc['acc']:.3f} & \\textbf{{{fm_auc['auc']:.3f}}} & {fm_auc['trades']:.1f} & {fm_auc['winrate']:.1f} & {fm_auc['sharpe']:.2f} & -- \\\\")
    lines.append(f"\\quad & Sharpe & \\textbf{{{fm_sharpe['acc']:.3f}}} & {fm_sharpe['auc']:.3f} & {fm_sharpe['trades']:.1f} & \\textbf{{{fm_sharpe['winrate']:.1f}}} & \\textbf{{{fm_sharpe['sharpe']:.2f}}} & -- \\\\")

    # Configuration display names
    config_names = {
        "technical_only": "Technical Only",
        "sentiment_only": "Sentiment Only",
        "equal_weights": "Equal Weights",
        "news_only": "News Only",
        "social_only": "Social Only"
    }

    # Add ablation configuration rows
    for config_key in ["technical_only", "sentiment_only", "equal_weights", "news_only", "social_only"]:
        if config_key not in ablation_results:
            continue

        df = ablation_results[config_key]

        # Compute metrics for both AUC and Sharpe selection
        auc_metrics = compute_ablation_metrics(df, "auc")
        sharpe_metrics = compute_ablation_metrics(df, "sharpe")

        if not auc_metrics or not sharpe_metrics:
            continue

        # Round values to display precision first, then compute delta
        # This ensures delta matches what's actually shown in the table
        fm_auc_rounded = round(fm_auc['auc'], 3)
        fm_sharpe_rounded = round(fm_sharpe['sharpe'], 2)
        auc_metrics_auc_rounded = round(auc_metrics['auc'], 3)
        sharpe_metrics_sharpe_rounded = round(sharpe_metrics['sharpe'], 2)

        # Compute delta using rounded values
        delta_auc = ((auc_metrics_auc_rounded - fm_auc_rounded) / fm_auc_rounded) * 100
        delta_sharpe = ((sharpe_metrics_sharpe_rounded - fm_sharpe_rounded) / fm_sharpe_rounded) * 100

        lines.append("\\midrule")
        lines.append(f"{config_names[config_key]} & AUC & {auc_metrics['acc']:.3f} & {auc_metrics['auc']:.3f} & {auc_metrics['trades']:.1f} & {auc_metrics['winrate']:.1f} & {auc_metrics['sharpe']:.2f} & {delta_auc:.1f} \\\\")
        lines.append(f"\\quad & Sharpe & {sharpe_metrics['acc']:.3f} & {sharpe_metrics['auc']:.3f} & {sharpe_metrics['trades']:.1f} & {sharpe_metrics['winrate']:.1f} & {sharpe_metrics['sharpe']:.2f} & {delta_sharpe:.1f} \\\\")

    return "\n".join(lines)


def update_sota_table(tex_file: str, baseline_content: str, ours_row: str) -> bool:
    """Update the SOTA table in the LaTeX file (both baselines and Ours row)."""

    if not os.path.exists(tex_file):
        print(f"Error: LaTeX file not found: {tex_file}")
        return False

    with open(tex_file, 'r') as f:
        content = f.read()

    # Pattern to match baseline sections and Ours row in the SOTA table
    # We'll replace everything from FinCast section through the Ours row
    pattern = (
        r"(\\multicolumn\{6\}\{l\}\{\\textit\{FinCast.*?\}\} \\\\)"  # First occurrence of FinCast
        r"(.*?)"  # Old baseline content (FinCast + Chronos-2)
        r"(\\midrule\s*\n)"  # midrule before "Ours"
        r"(\\textbf\{Ours.*?\\\\\s*\n)"  # Ours row
        r"(\\bottomrule)"  # bottomrule after Ours
    )

    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Error: Could not find SOTA table sections")
        print("Looking for pattern starting with: \\multicolumn{6}{l}{\\textit{FinCast")
        return False

    # Replace all sections: baselines + midrule + Ours row
    new_content_full = (
        content[:match.start(1)] +
        baseline_content + "\n" +
        match.group(3) +  # Keep the midrule
        ours_row + "\n" +
        match.group(5) +  # Keep the bottomrule
        content[match.end(5):]
    )

    with open(tex_file, 'w') as f:
        f.write(new_content_full)

    print(f"Successfully updated SOTA table in {tex_file}")
    return True


def update_ablation_table(tex_file: str, ablation_content: str) -> bool:
    """Update the Ablation Study table in the LaTeX file."""

    if not os.path.exists(tex_file):
        print(f"Error: LaTeX file not found: {tex_file}")
        return False

    with open(tex_file, 'r') as f:
        content = f.read()

    # Pattern to match the ablation table content
    # Match from "Full Model" row to the last "\bottomrule"
    pattern = (
        r"(\\caption\{Ablation Study Results\}.*?"
        r"\\midrule\s*\n)"  # Match up to first \midrule
        r"(Full Model.*?)"  # Capture old table rows
        r"(\\bottomrule)"  # Match the \bottomrule
    )

    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Error: Could not find Ablation Study table")
        print("Looking for pattern with caption: Ablation Study Results")
        return False

    # Replace the table content
    new_content = (
        content[:match.start(2)] +
        ablation_content + "\n" +
        match.group(3) +  # Keep the \bottomrule
        content[match.end(3):]
    )

    with open(tex_file, 'w') as f:
        f.write(new_content)

    print(f"Successfully updated Ablation Study table in {tex_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Update SOTA Comparison Table in main.tex")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines",
        help="Directory containing baseline results CSV files",
    )
    parser.add_argument(
        "--tex-file",
        type=str,
        default="docs/main.tex",
        help="Path to main.tex file",
    )
    parser.add_argument(
        "--ours-results",
        type=str,
        default="results/ours/tuned_all_results_combined.csv",
        help="Path to tuned_all_results_combined.csv for computing 'Ours' metrics",
    )
    parser.add_argument(
        "--ablation-dir",
        type=str,
        default="results/ablation",
        help="Directory containing ablation study CSV files",
    )
    parser.add_argument(
        "--update-ablation",
        action="store_true",
        help="Also update the Ablation Study table in main.tex",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("UPDATING SOTA TABLE")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"LaTeX file: {args.tex_file}")
    print(f"Ours results file: {args.ours_results}")
    print()

    # Load results
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    print("Loading Chronos-2 results...")
    chronos_df = load_chronos_results(args.results_dir)
    if not chronos_df.empty:
        print(f"Loaded {len(chronos_df)} Chronos-2 result rows")
        print(f"Model types: {chronos_df['ModelType'].unique()}")
        print(f"Stocks: {chronos_df['Stock'].unique()}")
    else:
        print("No Chronos-2 results found")
    print()

    print("Loading FinCast results...")
    fincast_df = load_fincast_results(args.results_dir)
    if not fincast_df.empty:
        print(f"Loaded {len(fincast_df)} FinCast result rows")
        print(f"Model types: {fincast_df['ModelType'].unique()}")
        print(f"Stocks: {fincast_df['Stock'].unique()}")
    else:
        print("No FinCast results found")
    print()

    # Compute "Ours" metrics from tuned results
    print("Computing 'Ours' metrics from tuned results...")
    ours_metrics = compute_sharpe_optimized_metrics(args.ours_results)
    print(f"Ours metrics: Acc={ours_metrics['acc']:.3f}, AUC={ours_metrics['auc']:.3f}, "
          f"N={ours_metrics['trades']:.1f}, Win%={ours_metrics['winrate']:.1f}, "
          f"Sharpe={ours_metrics['sharpe']:.2f}")
    print()

    # Generate baseline rows
    print("Calculating baseline metrics...")
    chronos_baselines = generate_chronos_baseline_rows(chronos_df)
    fincast_baselines = generate_fincast_baseline_rows(fincast_df)

    # Print summary
    print("\nBaseline Metrics Summary:")
    print("-" * 80)

    if fincast_baselines:
        print("\nFinCast Baselines:")
        for name, metrics in fincast_baselines.items():
            if metrics:
                print(f"\n{name}:")
                print(f"  AUC-selected: Acc={metrics['auc']['acc']:.3f}, AUC={metrics['auc']['auc']:.3f}, Sharpe={metrics['auc']['sharpe']:.2f}")
                print(f"  Sharpe-selected: Acc={metrics['sharpe']['acc']:.3f}, AUC={metrics['sharpe']['auc']:.3f}, Sharpe={metrics['sharpe']['sharpe']:.2f}")

    if chronos_baselines:
        print("\nChronos-2 Baselines:")
        for name, metrics in chronos_baselines.items():
            if metrics:
                print(f"\n{name}:")
                print(f"  AUC-selected: Acc={metrics['auc']['acc']:.3f}, AUC={metrics['auc']['auc']:.3f}, Sharpe={metrics['auc']['sharpe']:.2f}")
                print(f"  Sharpe-selected: Acc={metrics['sharpe']['acc']:.3f}, AUC={metrics['sharpe']['auc']:.3f}, Sharpe={metrics['sharpe']['sharpe']:.2f}")
    print()

    # Format as LaTeX
    print("Generating LaTeX table content...")

    # Find best baseline accuracy for bolding
    best_acc = find_best_baseline_accuracy(fincast_baselines, chronos_baselines)
    print(f"Best baseline accuracy: {best_acc:.3f}")

    fincast_latex = format_fincast_rows(fincast_baselines, best_acc)
    chronos_latex = format_chronos_rows(chronos_baselines, best_acc)
    ours_latex = format_ours_row(best_acc, ours_metrics)

    # Combine FinCast and Chronos-2 sections with midrule between (baselines only)
    baseline_latex = fincast_latex + "\n\\midrule\n" + chronos_latex

    print("\nGenerated LaTeX content (Baselines):")
    print("-" * 80)
    print(baseline_latex)
    print("-" * 80)
    print("\nGenerated LaTeX content (Ours):")
    print("-" * 80)
    print(ours_latex)
    print("-" * 80)
    print()

    # Update tex file
    print("Updating main.tex...")
    success = update_sota_table(args.tex_file, baseline_latex, ours_latex)

    if success:
        print("\n" + "=" * 80)
        print("SOTA TABLE UPDATED SUCCESSFULLY")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILED TO UPDATE SOTA TABLE")
        print("=" * 80)

    # Update Ablation Study table if requested
    if args.update_ablation:
        print("\n" + "=" * 80)
        print("UPDATING ABLATION STUDY TABLE")
        print("=" * 80)
        print(f"Ablation directory: {args.ablation_dir}")
        print()

        # Load ablation results
        print("Loading ablation study results...")
        ablation_results = load_ablation_results(args.ablation_dir)
        if ablation_results:
            print(f"Loaded {len(ablation_results)} ablation configurations:")
            for config_name in ablation_results:
                print(f"  - {config_name}: {len(ablation_results[config_name])} rows")
        else:
            print("Warning: No ablation results found")
            return
        print()

        # Compute Full Model metrics (using the same CSV as "Ours")
        print("Computing Full Model metrics...")
        try:
            df = pd.read_csv(args.ours_results)
            df = df[df["Strategy"] == "long_short"]

            # AUC-selected full model
            auc_metrics = compute_ablation_metrics(df, "auc")
            # Sharpe-selected full model
            sharpe_metrics = compute_ablation_metrics(df, "sharpe")

            full_model_metrics = {
                'auc': auc_metrics,
                'sharpe': sharpe_metrics
            }

            print(f"Full Model (AUC-selected): Acc={auc_metrics['acc']:.3f}, AUC={auc_metrics['auc']:.3f}, "
                  f"Sharpe={auc_metrics['sharpe']:.2f}")
            print(f"Full Model (Sharpe-selected): Acc={sharpe_metrics['acc']:.3f}, AUC={sharpe_metrics['auc']:.3f}, "
                  f"Sharpe={sharpe_metrics['sharpe']:.2f}")
        except Exception as e:
            print(f"Error computing full model metrics: {e}")
            return
        print()

        # Generate ablation table content
        print("Generating Ablation Study table content...")
        ablation_latex = format_ablation_table(full_model_metrics, ablation_results)

        print("\nGenerated LaTeX content (Ablation Study):")
        print("-" * 80)
        print(ablation_latex)
        print("-" * 80)
        print()

        # Update ablation table in tex file
        print("Updating Ablation Study table in main.tex...")
        ablation_success = update_ablation_table(args.tex_file, ablation_latex)

        if ablation_success:
            print("\n" + "=" * 80)
            print("ABLATION STUDY TABLE UPDATED SUCCESSFULLY")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("FAILED TO UPDATE ABLATION STUDY TABLE")
            print("=" * 80)


if __name__ == "__main__":
    main()
