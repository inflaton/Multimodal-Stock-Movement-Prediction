"""
Analyze Chronos Baseline Results

This script analyzes all Chronos baseline results across:
- 4 model types: zero-shot, multivariate, finetuned, finetuned_cov
- 4 trading strategies: long_short, long_only, long_short_confidence, long_only_confidence
- 5 stocks: AAPL, META, NVDA, SPY, TSLA

Outputs:
- Summary statistics by strategy
- Summary statistics by stock
- Summary statistics by model type
- Overall averages
- Best performing configurations

Usage:
    python analyze_chronos_results.py
    python analyze_chronos_results.py --results-dir ./paper
"""

import pandas as pd
import numpy as np
import os
import argparse
from glob import glob

# Configuration
STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
MODEL_TYPES = ["zero-shot", "multivariate", "finetuned", "finetuned_cov"]
STRATEGIES = [
    "long_short",
    "long_only",
    "long_short_confidence",
    "long_only_confidence",
]

# Metrics to analyze
METRICS = [
    "Test_Accuracy",
    "Test_ROC_AUC",
    "Trades",
    "WinRate",
    "Sharpe",
    "TotalReturn",
]


def get_strategy_suffix(strategy):
    """Get file suffix for a strategy."""
    if strategy == "long_short":
        return ""  # Default strategy has no suffix
    return f"_{strategy}"


def load_all_results(results_dir):
    """Load all Chronos results into a single DataFrame.

    File naming convention from chronos_baseline.py:
    - long_short (default): {model_type}_chronos-2_{stock}_results.csv
    - other strategies: {model_type}_chronos-2_{stock}_{strategy}_results.csv

    Finetuned models are in subfolders:
    - finetuned_single/Chronos2_finetuned_all_stocks_results.csv
    - finetuned_covariates/Chronos2_finetuned_cov_all_stocks_results.csv

    We need to be careful to match files correctly:
    - For long_short: match files WITHOUT any strategy suffix
    - For other strategies: match files WITH that specific strategy suffix
    """
    all_data = []

    for model_type in MODEL_TYPES:
        # Handle finetuned models from subfolders
        if model_type == "finetuned":
            # Load from finetuned_single folder
            filename = f"{results_dir}/finetuned_single/Chronos2_finetuned_all_stocks_results.csv"
            try:
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    df["ModelType"] = model_type
                    if "Strategy" not in df.columns:
                        df["Strategy"] = "long_short"
                    all_data.append(df)
                    print(f"Loaded {len(df)} rows from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
            continue

        elif model_type == "finetuned_cov":
            # Load from finetuned_covariates folder
            # filename = f"{results_dir}/finetuned_covariates/Chronos2_finetuned_cov_all_stocks_inference_results.csv"
            filename = f"{results_dir}/finetuned_covariates/Chronos2_finetuned_cov_all_stocks_results.csv"
            try:
                if os.path.exists(filename):
                    df = pd.read_csv(filename)
                    df["ModelType"] = model_type
                    if "Strategy" not in df.columns:
                        df["Strategy"] = "long_short"
                    all_data.append(df)
                    print(f"Loaded {len(df)} rows from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
            continue

        # Handle zero-shot and multivariate models
        # First try loading combined all_stocks file (for long_short strategy)
        all_stocks_file = f"{results_dir}/{model_type}_chronos-2_all_stocks_results.csv"
        if os.path.exists(all_stocks_file):
            try:
                df = pd.read_csv(all_stocks_file)
                df["ModelType"] = model_type
                if "Strategy" not in df.columns:
                    df["Strategy"] = "long_short"
                all_data.append(df)
                print(f"Loaded {len(df)} rows from {all_stocks_file}")
            except Exception as e:
                print(f"Error loading {all_stocks_file}: {e}")
        else:
            # Fall back to individual stock files
            for stock in STOCKS:
                for strategy in STRATEGIES:
                    # Build the exact filename based on strategy
                    if strategy == "long_short":
                        # Default strategy has no suffix
                        filename = (
                            f"{results_dir}/{model_type}_chronos-2_{stock}_results.csv"
                        )
                    else:
                        # Other strategies have explicit suffix
                        filename = f"{results_dir}/{model_type}_chronos-2_{stock}_{strategy}_results.csv"

                    try:
                        if os.path.exists(filename):
                            df = pd.read_csv(filename)
                            df["ModelType"] = model_type
                            # Ensure Strategy column is set correctly
                            # (some files may not have it or have different values)
                            if "Strategy" not in df.columns:
                                df["Strategy"] = strategy
                            all_data.append(df)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")

    if not all_data:
        print("No data files found!")
        return None

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def analyze_by_strategy(df):
    """Analyze results grouped by strategy."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY STRATEGY")
    print("=" * 80)

    strategy_stats = df.groupby("Strategy")[METRICS].agg(["mean", "std", "min", "max"])

    for strategy in STRATEGIES:
        if strategy in df["Strategy"].values:
            print(f"\n--- {strategy.upper()} ---")
            subset = df[df["Strategy"] == strategy]
            print(f"  Samples: {len(subset)}")
            print(
                f"  Avg Accuracy:    {subset['Test_Accuracy'].mean():.4f} (+/- {subset['Test_Accuracy'].std():.4f})"
            )
            print(
                f"  Avg ROC-AUC:     {subset['Test_ROC_AUC'].mean():.4f} (+/- {subset['Test_ROC_AUC'].std():.4f})"
            )
            print(f"  Avg Trades:      {subset['Trades'].mean():.1f}")
            print(
                f"  Avg Win Rate:    {subset['WinRate'].mean():.4f} ({subset['WinRate'].mean()*100:.1f}%)"
            )
            print(f"  Avg Sharpe:      {subset['Sharpe'].mean():.4f}")
            print(
                f"  Avg Total Return: {subset['TotalReturn'].mean():.4f} ({subset['TotalReturn'].mean()*100:.2f}%)"
            )

    return strategy_stats


def analyze_by_stock(df):
    """Analyze results grouped by stock."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY STOCK")
    print("=" * 80)

    for stock in STOCKS:
        if stock in df["Stock"].values:
            print(f"\n--- {stock} ---")
            subset = df[df["Stock"] == stock]
            print(f"  Samples: {len(subset)}")
            print(f"  Avg Accuracy:    {subset['Test_Accuracy'].mean():.4f}")
            print(f"  Avg ROC-AUC:     {subset['Test_ROC_AUC'].mean():.4f}")
            print(f"  Avg Win Rate:    {subset['WinRate'].mean()*100:.1f}%")
            print(f"  Avg Sharpe:      {subset['Sharpe'].mean():.4f}")
            print(f"  Avg Total Return: {subset['TotalReturn'].mean()*100:.2f}%")


def analyze_by_model_type(df):
    """Analyze results grouped by model type (zero-shot vs multivariate)."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL TYPE")
    print("=" * 80)

    for model_type in MODEL_TYPES:
        if model_type in df["ModelType"].values:
            print(f"\n--- {model_type.upper()} ---")
            subset = df[df["ModelType"] == model_type]
            print(f"  Samples: {len(subset)}")
            print(f"  Avg Accuracy:    {subset['Test_Accuracy'].mean():.4f}")
            print(f"  Avg ROC-AUC:     {subset['Test_ROC_AUC'].mean():.4f}")
            print(f"  Avg Win Rate:    {subset['WinRate'].mean()*100:.1f}%")
            print(f"  Avg Sharpe:      {subset['Sharpe'].mean():.4f}")
            print(f"  Avg Total Return: {subset['TotalReturn'].mean()*100:.2f}%")


def analyze_by_model_and_strategy(df):
    """Analyze results grouped by model type AND strategy."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL TYPE & STRATEGY")
    print("=" * 80)

    summary_data = []

    for model_type in MODEL_TYPES:
        for strategy in STRATEGIES:
            subset = df[(df["ModelType"] == model_type) & (df["Strategy"] == strategy)]
            if len(subset) > 0:
                row = {
                    "ModelType": model_type,
                    "Strategy": strategy,
                    "Samples": len(subset),
                    "Avg_Accuracy": subset["Test_Accuracy"].mean(),
                    "Avg_ROC_AUC": subset["Test_ROC_AUC"].mean(),
                    "Avg_Trades": subset["Trades"].mean(),
                    "Avg_WinRate": subset["WinRate"].mean(),
                    "Avg_Sharpe": subset["Sharpe"].mean(),
                    "Avg_TotalReturn": subset["TotalReturn"].mean(),
                }
                summary_data.append(row)

                print(f"\n--- {model_type.upper()} + {strategy.upper()} ---")
                print(f"  Samples: {len(subset)}")
                print(f"  Avg Accuracy:    {row['Avg_Accuracy']:.4f}")
                print(f"  Avg ROC-AUC:     {row['Avg_ROC_AUC']:.4f}")
                print(f"  Avg Trades:      {row['Avg_Trades']:.1f}")
                print(f"  Avg Win Rate:    {row['Avg_WinRate']*100:.1f}%")
                print(f"  Avg Sharpe:      {row['Avg_Sharpe']:.4f}")
                print(f"  Avg Total Return: {row['Avg_TotalReturn']*100:.2f}%")

    return pd.DataFrame(summary_data)


def find_best_configurations(df):
    """Find best performing configurations."""
    print("\n" + "=" * 80)
    print("BEST PERFORMING CONFIGURATIONS")
    print("=" * 80)

    # Best by Sharpe Ratio
    print("\n--- Top 5 by Sharpe Ratio ---")
    top_sharpe = df.nlargest(5, "Sharpe")[
        [
            "Stock",
            "Horizon",
            "ModelType",
            "Strategy",
            "Sharpe",
            "TotalReturn",
            "WinRate",
        ]
    ]
    print(top_sharpe.to_string(index=False))

    # Best by Total Return
    print("\n--- Top 5 by Total Return ---")
    top_return = df.nlargest(5, "TotalReturn")[
        [
            "Stock",
            "Horizon",
            "ModelType",
            "Strategy",
            "TotalReturn",
            "Sharpe",
            "WinRate",
        ]
    ]
    print(top_return.to_string(index=False))

    # Best by Win Rate (with min trades filter)
    print("\n--- Top 5 by Win Rate (min 10 trades) ---")
    filtered = df[df["Trades"] >= 10]
    if len(filtered) > 0:
        top_winrate = filtered.nlargest(5, "WinRate")[
            ["Stock", "Horizon", "ModelType", "Strategy", "WinRate", "Trades", "Sharpe"]
        ]
        print(top_winrate.to_string(index=False))

    # Best by Accuracy
    print("\n--- Top 5 by Test Accuracy ---")
    top_acc = df.nlargest(5, "Test_Accuracy")[
        ["Stock", "Horizon", "ModelType", "Strategy", "Test_Accuracy", "Test_ROC_AUC"]
    ]
    print(top_acc.to_string(index=False))


def create_comparison_table(df):
    """Create a comparison table grouped by strategy, similar to the paper format."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Grouped by Strategy)")
    print("=" * 80)
    print()
    print(
        f"{'Method':<45} {'Acc.':<7} {'AUC':<7} {'N':<6} {'Win%':<7} {'Sharpe':<8} {'Return%':<8}"
    )
    print("-" * 95)

    # For each strategy
    for strategy in STRATEGIES:
        strategy_df = df[df["Strategy"] == strategy]
        if len(strategy_df) == 0:
            continue

        strategy_label = strategy.replace("_", " ").title()
        print(f"\n{strategy_label}:")

        # For each model type within strategy
        for model_type in MODEL_TYPES:
            subset = strategy_df[strategy_df["ModelType"] == model_type]
            if len(subset) == 0:
                continue

            model_labels = {
                "zero-shot": "Zero-shot",
                "multivariate": "Multivariate",
                "finetuned": "Fine-tuned",
                "finetuned_cov": "Fine-tuned + Cov",
            }
            model_label = model_labels.get(model_type, model_type)

            # Average across all stocks
            avg_acc = subset["Test_Accuracy"].mean()
            avg_auc = subset["Test_ROC_AUC"].mean()
            avg_trades = subset["Trades"].mean()
            avg_winrate = subset["WinRate"].mean() * 100
            avg_sharpe = subset["Sharpe"].mean()
            avg_return = subset["TotalReturn"].mean() * 100

            method_name = f"  {model_label} (avg all stocks)"
            print(
                f"{method_name:<45} {avg_acc:<7.3f} {avg_auc:<7.3f} {avg_trades:<6.0f} {avg_winrate:<7.1f} {avg_sharpe:<8.2f} {avg_return:<8.2f}"
            )

            # Best by AUC: find best per stock, then average
            best_auc_rows = []
            for stock in STOCKS:
                stock_data = subset[subset["Stock"] == stock]
                if len(stock_data) > 0:
                    best_idx = stock_data["Test_ROC_AUC"].idxmax()
                    best_auc_rows.append(stock_data.loc[best_idx])
            if best_auc_rows:
                best_auc_df = pd.DataFrame(best_auc_rows)
                method_name = f"  {model_label} Best by AUC"
                print(
                    f"{method_name:<45} {best_auc_df['Test_Accuracy'].mean():<7.3f} {best_auc_df['Test_ROC_AUC'].mean():<7.3f} {best_auc_df['Trades'].mean():<6.0f} {best_auc_df['WinRate'].mean()*100:<7.1f} {best_auc_df['Sharpe'].mean():<8.2f} {best_auc_df['TotalReturn'].mean()*100:<8.2f}"
                )

            # Best by Sharpe: find best per stock, then average
            best_sharpe_rows = []
            for stock in STOCKS:
                stock_data = subset[subset["Stock"] == stock]
                if len(stock_data) > 0:
                    best_idx = stock_data["Sharpe"].idxmax()
                    best_sharpe_rows.append(stock_data.loc[best_idx])
            if best_sharpe_rows:
                best_sharpe_df = pd.DataFrame(best_sharpe_rows)
                method_name = f"  {model_label} Best by Sharpe"
                print(
                    f"{method_name:<45} {best_sharpe_df['Test_Accuracy'].mean():<7.3f} {best_sharpe_df['Test_ROC_AUC'].mean():<7.3f} {best_sharpe_df['Trades'].mean():<6.0f} {best_sharpe_df['WinRate'].mean()*100:<7.1f} {best_sharpe_df['Sharpe'].mean():<8.2f} {best_sharpe_df['TotalReturn'].mean()*100:<8.2f}"
                )

            # Best by Return: find best per stock, then average
            best_return_rows = []
            for stock in STOCKS:
                stock_data = subset[subset["Stock"] == stock]
                if len(stock_data) > 0:
                    best_idx = stock_data["TotalReturn"].idxmax()
                    best_return_rows.append(stock_data.loc[best_idx])
            if best_return_rows:
                best_return_df = pd.DataFrame(best_return_rows)
                method_name = f"  {model_label} Best by Return"
                print(
                    f"{method_name:<45} {best_return_df['Test_Accuracy'].mean():<7.3f} {best_return_df['Test_ROC_AUC'].mean():<7.3f} {best_return_df['Trades'].mean():<6.0f} {best_return_df['WinRate'].mean()*100:<7.1f} {best_return_df['Sharpe'].mean():<8.2f} {best_return_df['TotalReturn'].mean()*100:<8.2f}"
                )

    print()
    print("-" * 95)
    print(
        "Note: Results averaged across 5 stocks (AAPL, META, NVDA, SPY, TSLA). N = avg trades."
    )


def create_summary_table(df, output_dir):
    """Create and save summary tables."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    # Summary by Strategy
    strategy_summary = (
        df.groupby("Strategy")
        .agg(
            {
                "Test_Accuracy": "mean",
                "Test_ROC_AUC": "mean",
                "Trades": "mean",
                "WinRate": "mean",
                "Sharpe": "mean",
                "TotalReturn": "mean",
            }
        )
        .round(4)
    )
    strategy_summary.columns = [
        "Avg_Accuracy",
        "Avg_ROC_AUC",
        "Avg_Trades",
        "Avg_WinRate",
        "Avg_Sharpe",
        "Avg_TotalReturn",
    ]

    print("\n--- Summary by Strategy ---")
    print(strategy_summary.to_string())

    # Summary by Stock
    stock_summary = (
        df.groupby("Stock")
        .agg(
            {
                "Test_Accuracy": "mean",
                "Test_ROC_AUC": "mean",
                "Trades": "mean",
                "WinRate": "mean",
                "Sharpe": "mean",
                "TotalReturn": "mean",
            }
        )
        .round(4)
    )
    stock_summary.columns = [
        "Avg_Accuracy",
        "Avg_ROC_AUC",
        "Avg_Trades",
        "Avg_WinRate",
        "Avg_Sharpe",
        "Avg_TotalReturn",
    ]

    print("\n--- Summary by Stock ---")
    print(stock_summary.to_string())

    # Summary by Model Type
    model_summary = (
        df.groupby("ModelType")
        .agg(
            {
                "Test_Accuracy": "mean",
                "Test_ROC_AUC": "mean",
                "Trades": "mean",
                "WinRate": "mean",
                "Sharpe": "mean",
                "TotalReturn": "mean",
            }
        )
        .round(4)
    )
    model_summary.columns = [
        "Avg_Accuracy",
        "Avg_ROC_AUC",
        "Avg_Trades",
        "Avg_WinRate",
        "Avg_Sharpe",
        "Avg_TotalReturn",
    ]

    print("\n--- Summary by Model Type ---")
    print(model_summary.to_string())

    # Overall averages
    print("\n--- Overall Averages ---")
    print(f"  Total Samples:    {len(df)}")
    print(f"  Avg Accuracy:     {df['Test_Accuracy'].mean():.4f}")
    print(f"  Avg ROC-AUC:      {df['Test_ROC_AUC'].mean():.4f}")
    print(f"  Avg Trades:       {df['Trades'].mean():.1f}")
    print(f"  Avg Win Rate:     {df['WinRate'].mean()*100:.1f}%")
    print(f"  Avg Sharpe:       {df['Sharpe'].mean():.4f}")
    print(f"  Avg Total Return: {df['TotalReturn'].mean()*100:.2f}%")

    # Save summaries to CSV
    os.makedirs(output_dir, exist_ok=True)
    strategy_summary.to_csv(f"{output_dir}/chronos_summary_by_strategy.csv")
    stock_summary.to_csv(f"{output_dir}/chronos_summary_by_stock.csv")
    model_summary.to_csv(f"{output_dir}/chronos_summary_by_model.csv")

    print(f"\nSummary tables saved to {output_dir}/")

    return strategy_summary, stock_summary, model_summary


def main():
    parser = argparse.ArgumentParser(description="Analyze Chronos Baseline Results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Directory containing Chronos result CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Directory to save summary CSV files",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CHRONOS BASELINE RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load all results
    df = load_all_results(args.results_dir)

    if df is None or len(df) == 0:
        print("No results found!")
        return

    print(f"\nLoaded {len(df)} result rows")
    print(f"Stocks: {df['Stock'].unique().tolist()}")
    print(f"Strategies: {df['Strategy'].unique().tolist()}")
    print(f"Model Types: {df['ModelType'].unique().tolist()}")
    print(f"Horizons: {sorted(df['Horizon'].unique().tolist())}")

    # Run analyses
    analyze_by_strategy(df)
    analyze_by_stock(df)
    analyze_by_model_type(df)
    model_strategy_summary = analyze_by_model_and_strategy(df)
    find_best_configurations(df)
    create_comparison_table(df)
    create_summary_table(df, args.output_dir)

    # Save model+strategy summary
    model_strategy_summary.to_csv(
        f"{args.output_dir}/chronos_summary_by_model_strategy.csv", index=False
    )

    # Save full combined results
    df.to_csv(f"{args.output_dir}/chronos_all_results_combined.csv", index=False)
    print(
        f"\nFull combined results saved to: {args.output_dir}/chronos_all_results_combined.csv"
    )

    # Save separate CSV for each model type (long_short strategy only for comparison)
    for model_type in df["ModelType"].unique():
        model_df = df[
            (df["ModelType"] == model_type) & (df["Strategy"] == "long_short")
        ]
        if len(model_df) > 0:
            output_file = (
                f"{args.output_dir}/{model_type}_chronos-2_all_stocks_results.csv"
            )
            model_df.to_csv(output_file, index=False)
            print(f"Saved {len(model_df)} rows to: {output_file}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
