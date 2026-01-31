"""
Analyze Hyperparameter Tuned ML Model Results

This script analyzes all hyperparameter-tuned ML model results across:
- 7 models: LSTM, XGBoost, LightGBM, RandomForest, GradientBoosting, LogisticRegression, SVM
- 4 trading strategies: long_short, long_only, long_short_confidence, long_only_confidence
- 5 stocks: AAPL, META, NVDA, SPY, TSLA

Outputs:
- Summary statistics by model
- Summary statistics by strategy
- Summary statistics by stock
- Comparison table similar to paper format
- Best performing configurations
- Optional: Update LaTeX tables in main.tex

Usage:
    python analyze_tuned_results.py
    python analyze_tuned_results.py --results-dir results/ours
    python analyze_tuned_results.py --update-tex --tex-file docs/main.tex
    python analyze_tuned_results.py --update-tex --update-sota --tex-file docs/main.tex
"""

import pandas as pd
import numpy as np
import os
import argparse
import re
from glob import glob

# Configuration
STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
MODELS = ["LSTM", "XGBoost", "LightGBM", "RandomForest", "GradientBoosting", "LogisticRegression", "SVM"]
STRATEGIES = ["long_short", "long_only", "long_short_confidence", "long_only_confidence"]

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
    """Load all tuned ML model results into a single DataFrame.

    File naming convention:
    - long_short (default): {Model}_tuned_{Stock}_results.csv
    - other strategies: {Model}_tuned_{Stock}_results_{strategy}.csv
    """
    all_data = []

    for model in MODELS:
        for stock in STOCKS:
            for strategy in STRATEGIES:
                # Build the exact filename based on strategy
                suffix = get_strategy_suffix(strategy)
                filename = f"{results_dir}/{model}_tuned_{stock}_results{suffix}.csv"

                try:
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        df["Model"] = model
                        # Ensure Stock column is set correctly
                        if "Stock" not in df.columns:
                            df["Stock"] = stock
                        # Ensure Strategy column is set correctly
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


def analyze_by_model(df):
    """Analyze results grouped by model."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL")
    print("=" * 80)

    for model in MODELS:
        if model in df["Model"].values:
            print(f"\n--- {model} ---")
            subset = df[df["Model"] == model]
            print(f"  Samples: {len(subset)}")
            print(f"  Avg Accuracy:    {subset['Test_Accuracy'].mean():.4f} (+/- {subset['Test_Accuracy'].std():.4f})")
            print(f"  Avg ROC-AUC:     {subset['Test_ROC_AUC'].mean():.4f} (+/- {subset['Test_ROC_AUC'].std():.4f})")
            print(f"  Avg Trades:      {subset['Trades'].mean():.1f}")
            print(f"  Avg Win Rate:    {subset['WinRate'].mean():.4f} ({subset['WinRate'].mean()*100:.1f}%)")
            print(f"  Avg Sharpe:      {subset['Sharpe'].mean():.4f}")
            print(f"  Avg Total Return: {subset['TotalReturn'].mean():.4f} ({subset['TotalReturn'].mean()*100:.2f}%)")


def analyze_by_strategy(df):
    """Analyze results grouped by strategy."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY STRATEGY")
    print("=" * 80)

    for strategy in STRATEGIES:
        if strategy in df["Strategy"].values:
            print(f"\n--- {strategy.upper()} ---")
            subset = df[df["Strategy"] == strategy]
            print(f"  Samples: {len(subset)}")
            print(f"  Avg Accuracy:    {subset['Test_Accuracy'].mean():.4f} (+/- {subset['Test_Accuracy'].std():.4f})")
            print(f"  Avg ROC-AUC:     {subset['Test_ROC_AUC'].mean():.4f} (+/- {subset['Test_ROC_AUC'].std():.4f})")
            print(f"  Avg Trades:      {subset['Trades'].mean():.1f}")
            print(f"  Avg Win Rate:    {subset['WinRate'].mean():.4f} ({subset['WinRate'].mean()*100:.1f}%)")
            print(f"  Avg Sharpe:      {subset['Sharpe'].mean():.4f}")
            print(f"  Avg Total Return: {subset['TotalReturn'].mean():.4f} ({subset['TotalReturn'].mean()*100:.2f}%)")


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


def analyze_by_model_and_strategy(df):
    """Analyze results grouped by model AND strategy."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL & STRATEGY")
    print("=" * 80)

    summary_data = []

    for model in MODELS:
        for strategy in STRATEGIES:
            subset = df[(df["Model"] == model) & (df["Strategy"] == strategy)]
            if len(subset) > 0:
                row = {
                    "Model": model,
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

                print(f"\n--- {model} + {strategy.upper()} ---")
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
    print("\n--- Top 10 by Sharpe Ratio ---")
    top_sharpe = df.nlargest(10, "Sharpe")[["Stock", "Horizon", "Model", "Strategy", "Sharpe", "TotalReturn", "WinRate", "Test_ROC_AUC"]]
    print(top_sharpe.to_string(index=False))

    # Best by Total Return
    print("\n--- Top 10 by Total Return ---")
    top_return = df.nlargest(10, "TotalReturn")[["Stock", "Horizon", "Model", "Strategy", "TotalReturn", "Sharpe", "WinRate", "Test_ROC_AUC"]]
    print(top_return.to_string(index=False))

    # Best by Win Rate (with min trades filter)
    print("\n--- Top 10 by Win Rate (min 10 trades) ---")
    filtered = df[df["Trades"] >= 10]
    if len(filtered) > 0:
        top_winrate = filtered.nlargest(10, "WinRate")[["Stock", "Horizon", "Model", "Strategy", "WinRate", "Trades", "Sharpe", "Test_ROC_AUC"]]
        print(top_winrate.to_string(index=False))

    # Best by AUC
    print("\n--- Top 10 by Test AUC ---")
    top_auc = df.nlargest(10, "Test_ROC_AUC")[["Stock", "Horizon", "Model", "Strategy", "Test_ROC_AUC", "Test_Accuracy", "Sharpe", "TotalReturn"]]
    print(top_auc.to_string(index=False))


def create_comparison_table(df):
    """Create comparison table using direct per-stock best model selection.

    For each strategy:
    1. For each stock, find the single best model configuration (across all models and horizons)
    2. Average results across all 5 stocks

    This matches main.tex Tables 1 & 3 methodology.
    """
    print("\n" + "=" * 80)
    print("COMPARISON TABLE (Direct Per-Stock Best Model Selection)")
    print("=" * 80)
    print()
    print(f"{'Method':<45} {'Acc.':<7} {'AUC':<7} {'N':<6} {'Win%':<7} {'Sharpe':<8} {'Return%':<8}")
    print("-" * 95)

    # For each strategy - use direct per-stock selection
    for strategy in STRATEGIES:
        strategy_df = df[df["Strategy"] == strategy]
        if len(strategy_df) == 0:
            continue

        strategy_label = strategy.replace("_", " ").title()
        print(f"\n{strategy_label} (Direct Per-Stock Selection):")

        # Best by AUC: For each stock, find the single best model by AUC (across all models/horizons)
        best_auc_rows = []
        for stock in STOCKS:
            stock_data = strategy_df[strategy_df["Stock"] == stock]
            if len(stock_data) > 0:
                best_idx = stock_data["Test_ROC_AUC"].idxmax()
                best_auc_rows.append(stock_data.loc[best_idx])

        if best_auc_rows:
            best_auc_df = pd.DataFrame(best_auc_rows)
            method_name = f"  Best by AUC"
            print(f"{method_name:<45} {best_auc_df['Test_Accuracy'].mean():<7.3f} {best_auc_df['Test_ROC_AUC'].mean():<7.3f} {best_auc_df['Trades'].mean():<6.0f} {best_auc_df['WinRate'].mean()*100:<7.1f} {best_auc_df['Sharpe'].mean():<8.2f} {best_auc_df['TotalReturn'].mean()*100:<8.2f}")

        # Best by Sharpe: For each stock, find the single best model by Sharpe (across all models/horizons)
        best_sharpe_rows = []
        for stock in STOCKS:
            stock_data = strategy_df[strategy_df["Stock"] == stock]
            if len(stock_data) > 0:
                best_idx = stock_data["Sharpe"].idxmax()
                best_sharpe_rows.append(stock_data.loc[best_idx])

        if best_sharpe_rows:
            best_sharpe_df = pd.DataFrame(best_sharpe_rows)
            method_name = f"  Best by Sharpe"
            print(f"{method_name:<45} {best_sharpe_df['Test_Accuracy'].mean():<7.3f} {best_sharpe_df['Test_ROC_AUC'].mean():<7.3f} {best_sharpe_df['Trades'].mean():<6.0f} {best_sharpe_df['WinRate'].mean()*100:<7.1f} {best_sharpe_df['Sharpe'].mean():<8.2f} {best_sharpe_df['TotalReturn'].mean()*100:<8.2f}")

    print()
    print("-" * 95)
    print("Note: For each stock, single best model selected across all ML types and horizons, then averaged.")


def create_model_comparison_table(df):
    """Create a comparison table across all models (best per stock, then averaged)."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE (Best by metric per stock, then averaged)")
    print("=" * 80)
    print()
    print(f"{'Model':<20} {'Strategy':<25} {'Acc.':<7} {'AUC':<7} {'N':<6} {'Win%':<7} {'Sharpe':<8} {'Return%':<8}")
    print("-" * 100)

    results = []

    for model in MODELS:
        model_df = df[df["Model"] == model]
        if len(model_df) == 0:
            continue

        for strategy in STRATEGIES:
            subset = model_df[model_df["Strategy"] == strategy]
            if len(subset) == 0:
                continue

            # Best by Sharpe: find best per stock, then average
            best_rows = []
            for stock in STOCKS:
                stock_data = subset[subset["Stock"] == stock]
                if len(stock_data) > 0:
                    best_idx = stock_data["Sharpe"].idxmax()
                    best_rows.append(stock_data.loc[best_idx])

            if best_rows:
                best_df = pd.DataFrame(best_rows)
                row = {
                    "Model": model,
                    "Strategy": strategy,
                    "Acc": best_df["Test_Accuracy"].mean(),
                    "AUC": best_df["Test_ROC_AUC"].mean(),
                    "N": best_df["Trades"].mean(),
                    "Win%": best_df["WinRate"].mean() * 100,
                    "Sharpe": best_df["Sharpe"].mean(),
                    "Return%": best_df["TotalReturn"].mean() * 100,
                }
                results.append(row)
                print(f"{model:<20} {strategy:<25} {row['Acc']:<7.3f} {row['AUC']:<7.3f} {row['N']:<6.0f} {row['Win%']:<7.1f} {row['Sharpe']:<8.2f} {row['Return%']:<8.2f}")

    print("-" * 100)

    return pd.DataFrame(results)


def create_summary_table(df, output_dir):
    """Create and save summary tables."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLES")
    print("=" * 80)

    # Summary by Model
    model_summary = df.groupby("Model").agg({
        "Test_Accuracy": "mean",
        "Test_ROC_AUC": "mean",
        "Trades": "mean",
        "WinRate": "mean",
        "Sharpe": "mean",
        "TotalReturn": "mean",
    }).round(4)
    model_summary.columns = ["Avg_Accuracy", "Avg_ROC_AUC", "Avg_Trades", "Avg_WinRate", "Avg_Sharpe", "Avg_TotalReturn"]

    print("\n--- Summary by Model ---")
    print(model_summary.to_string())

    # Summary by Strategy
    strategy_summary = df.groupby("Strategy").agg({
        "Test_Accuracy": "mean",
        "Test_ROC_AUC": "mean",
        "Trades": "mean",
        "WinRate": "mean",
        "Sharpe": "mean",
        "TotalReturn": "mean",
    }).round(4)
    strategy_summary.columns = ["Avg_Accuracy", "Avg_ROC_AUC", "Avg_Trades", "Avg_WinRate", "Avg_Sharpe", "Avg_TotalReturn"]

    print("\n--- Summary by Strategy ---")
    print(strategy_summary.to_string())

    # Summary by Stock
    stock_summary = df.groupby("Stock").agg({
        "Test_Accuracy": "mean",
        "Test_ROC_AUC": "mean",
        "Trades": "mean",
        "WinRate": "mean",
        "Sharpe": "mean",
        "TotalReturn": "mean",
    }).round(4)
    stock_summary.columns = ["Avg_Accuracy", "Avg_ROC_AUC", "Avg_Trades", "Avg_WinRate", "Avg_Sharpe", "Avg_TotalReturn"]

    print("\n--- Summary by Stock ---")
    print(stock_summary.to_string())

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
    model_summary.to_csv(f"{output_dir}/tuned_summary_by_model.csv")
    strategy_summary.to_csv(f"{output_dir}/tuned_summary_by_strategy.csv")
    stock_summary.to_csv(f"{output_dir}/tuned_summary_by_stock.csv")

    print(f"\nSummary tables saved to {output_dir}/")

    return model_summary, strategy_summary, stock_summary


def generate_latex_table_by_auc(df, strategy="long_short"):
    """Generate LaTeX table for best model by AUC per stock."""
    strategy_df = df[df["Strategy"] == strategy]

    # For each stock, find best model by AUC
    best_rows = []
    for stock in STOCKS:
        stock_data = strategy_df[strategy_df["Stock"] == stock]
        if len(stock_data) > 0:
            best_idx = stock_data["Test_ROC_AUC"].idxmax()
            best_rows.append(stock_data.loc[best_idx])

    if not best_rows:
        return None

    best_df = pd.DataFrame(best_rows)

    # Model name mapping
    model_map = {
        "LSTM": "LSTM",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "RandomForest": "RF",
        "GradientBoosting": "GB",
        "LogisticRegression": "LR",
        "SVM": "SVM"
    }

    # Find max values across all stocks for bolding
    max_acc = best_df["Test_Accuracy"].max()
    max_auc = best_df["Test_ROC_AUC"].max()
    max_winrate = (best_df["WinRate"] * 100).max()
    max_sharpe = best_df["Sharpe"].max()

    # Build LaTeX table
    latex_lines = []

    for _, row in best_df.iterrows():
        stock = row["Stock"]
        model = model_map.get(row["Model"], row["Model"])
        horizon = int(row["Horizon"])
        acc = row["Test_Accuracy"]
        auc = row["Test_ROC_AUC"]
        trades = int(row["Trades"])
        winrate = row["WinRate"] * 100
        sharpe = row["Sharpe"]

        # Format with bold for best values (across all stocks)
        acc_str = f"\\textbf{{{acc:.3f}}}" if acc == max_acc else f"{acc:.3f}"
        auc_str = f"\\textbf{{{auc:.3f}}}" if auc == max_auc else f"{auc:.3f}"
        winrate_str = f"\\textbf{{{winrate:.1f}}}" if winrate == max_winrate else f"{winrate:.1f}"
        sharpe_str = f"\\textbf{{{sharpe:.2f}}}" if sharpe == max_sharpe else f"{sharpe:.2f}"

        line = f"{stock} & {model} & {horizon} & {acc_str} & {auc_str} & {trades} & {winrate_str} & {sharpe_str} \\\\"
        latex_lines.append(line)

    # Add average row
    avg_horizon = best_df["Horizon"].mean()
    avg_acc = best_df["Test_Accuracy"].mean()
    avg_auc = best_df["Test_ROC_AUC"].mean()
    avg_trades = best_df["Trades"].mean()
    avg_winrate = best_df["WinRate"].mean() * 100
    avg_sharpe = best_df["Sharpe"].mean()

    latex_lines.append("\\midrule")
    latex_lines.append(f"\\multicolumn{{2}}{{l}}{{Average}} & {avg_horizon:.1f} & {avg_acc:.3f} & {avg_auc:.3f} & {avg_trades:.1f} & {avg_winrate:.1f} & {avg_sharpe:.2f} \\\\")

    return "\n".join(latex_lines)


def generate_latex_table_by_sharpe(df, strategy="long_short"):
    """Generate LaTeX table for best model by Sharpe per stock."""
    strategy_df = df[df["Strategy"] == strategy]

    # For each stock, find best model by Sharpe
    best_rows = []
    for stock in STOCKS:
        stock_data = strategy_df[strategy_df["Stock"] == stock]
        if len(stock_data) > 0:
            best_idx = stock_data["Sharpe"].idxmax()
            best_rows.append(stock_data.loc[best_idx])

    if not best_rows:
        return None

    best_df = pd.DataFrame(best_rows)

    # Model name mapping
    model_map = {
        "LSTM": "LSTM",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "RandomForest": "RF",
        "GradientBoosting": "GB",
        "LogisticRegression": "LR",
        "SVM": "SVM"
    }

    # Find max values across all stocks for bolding
    max_acc = best_df["Test_Accuracy"].max()
    max_auc = best_df["Test_ROC_AUC"].max()
    max_winrate = (best_df["WinRate"] * 100).max()
    max_sharpe = best_df["Sharpe"].max()

    # Build LaTeX table
    latex_lines = []

    for _, row in best_df.iterrows():
        stock = row["Stock"]
        model = model_map.get(row["Model"], row["Model"])
        horizon = int(row["Horizon"])
        acc = row["Test_Accuracy"]
        auc = row["Test_ROC_AUC"]
        trades = int(row["Trades"])
        winrate = row["WinRate"] * 100
        sharpe = row["Sharpe"]

        # Format with bold for best values (across all stocks)
        acc_str = f"\\textbf{{{acc:.3f}}}" if acc == max_acc else f"{acc:.3f}"
        auc_str = f"\\textbf{{{auc:.3f}}}" if auc == max_auc else f"{auc:.3f}"
        winrate_str = f"\\textbf{{{winrate:.1f}}}" if winrate == max_winrate else f"{winrate:.1f}"
        sharpe_str = f"\\textbf{{{sharpe:.2f}}}" if sharpe == max_sharpe else f"{sharpe:.2f}"

        line = f"{stock} & {model} & {horizon} & {acc_str} & {auc_str} & {trades} & {winrate_str} & {sharpe_str} \\\\"
        latex_lines.append(line)

    # Add average row
    avg_horizon = best_df["Horizon"].mean()
    avg_acc = best_df["Test_Accuracy"].mean()
    avg_auc = best_df["Test_ROC_AUC"].mean()
    avg_trades = best_df["Trades"].mean()
    avg_winrate = best_df["WinRate"].mean() * 100
    avg_sharpe = best_df["Sharpe"].mean()

    latex_lines.append("\\midrule")
    latex_lines.append(f"\\multicolumn{{2}}{{l}}{{Average}} & {avg_horizon:.1f} & {avg_acc:.3f} & {avg_auc:.3f} & {avg_trades:.1f} & {avg_winrate:.1f} & {avg_sharpe:.2f} \\\\")

    return "\n".join(latex_lines)


def update_latex_table_in_file(tex_file, caption_text, new_table_content):
    """Update a specific table in the LaTeX file."""
    if not os.path.exists(tex_file):
        print(f"LaTeX file not found: {tex_file}")
        return False

    with open(tex_file, 'r') as f:
        content = f.read()

    # Find the table by caption
    caption_pattern = re.escape(caption_text)

    # Pattern to match the entire table content including the old average row
    # Match from first \midrule to \bottomrule (which comes after the average row)
    pattern = (
        f"(\\\\caption\\{{{caption_pattern}\\}}.*?"
        f"\\\\midrule\\s*\\n)"  # Match up to first \midrule
        f"(.*?)"  # Capture old table rows AND old average section
        f"(\\\\bottomrule)"  # Match the \bottomrule after average
    )

    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print(f"Could not find table with caption: {caption_text}")
        return False

    # Replace the table content (new content already includes \midrule and average row)
    new_content = content[:match.start(2)] + new_table_content + "\n" + content[match.start(3):]

    with open(tex_file, 'w') as f:
        f.write(new_content)

    print(f"Updated table: {caption_text}")
    return True


def generate_ablation_full_model_rows(df, strategy="long_short"):
    """Generate Full Model rows for the Ablation Study table."""
    strategy_df = df[df["Strategy"] == strategy]

    # Best by AUC
    best_auc_rows = []
    for stock in STOCKS:
        stock_data = strategy_df[strategy_df["Stock"] == stock]
        if len(stock_data) > 0:
            best_idx = stock_data["Test_ROC_AUC"].idxmax()
            best_auc_rows.append(stock_data.loc[best_idx])

    # Best by Sharpe
    best_sharpe_rows = []
    for stock in STOCKS:
        stock_data = strategy_df[strategy_df["Stock"] == stock]
        if len(stock_data) > 0:
            best_idx = stock_data["Sharpe"].idxmax()
            best_sharpe_rows.append(stock_data.loc[best_idx])

    if not best_auc_rows or not best_sharpe_rows:
        return None

    best_auc_df = pd.DataFrame(best_auc_rows)
    best_sharpe_df = pd.DataFrame(best_sharpe_rows)

    # Calculate metrics for AUC row
    auc_acc = best_auc_df["Test_Accuracy"].mean()
    auc_auc = best_auc_df["Test_ROC_AUC"].mean()
    auc_trades = best_auc_df["Trades"].mean()
    auc_winrate = best_auc_df["WinRate"].mean() * 100
    auc_sharpe = best_auc_df["Sharpe"].mean()
    auc_ret = best_auc_df["TotalReturn"].mean() * 100

    # Calculate metrics for Sharpe row
    sharpe_acc = best_sharpe_df["Test_Accuracy"].mean()
    sharpe_auc = best_sharpe_df["Test_ROC_AUC"].mean()
    sharpe_trades = best_sharpe_df["Trades"].mean()
    sharpe_winrate = best_sharpe_df["WinRate"].mean() * 100
    sharpe_sharpe = best_sharpe_df["Sharpe"].mean()
    sharpe_ret = best_sharpe_df["TotalReturn"].mean() * 100

    # Format rows with proper LaTeX
    auc_row = f"Full Model & AUC & {auc_acc:.3f} & \\textbf{{{auc_auc:.3f}}} & {auc_trades:.1f} & {auc_winrate:.1f} & {auc_sharpe:.2f} & -- \\\\"

    # Add \textbf for best values in Sharpe row
    sharpe_acc_str = f"\\textbf{{{sharpe_acc:.3f}}}" if sharpe_acc > auc_acc else f"{sharpe_acc:.3f}"
    sharpe_auc_str = f"\\textbf{{{sharpe_auc:.3f}}}" if sharpe_auc > auc_auc else f"{sharpe_auc:.3f}"
    sharpe_winrate_str = f"\\textbf{{{sharpe_winrate:.1f}}}" if sharpe_winrate > auc_winrate else f"{sharpe_winrate:.1f}"
    sharpe_sharpe_str = f"\\textbf{{{sharpe_sharpe:.2f}}}" if sharpe_sharpe > auc_sharpe else f"{sharpe_sharpe:.2f}"
    sharpe_ret_str = f"\\textbf{{{sharpe_ret:.1f}}}" if sharpe_ret > auc_ret else f"{sharpe_ret:.1f}"

    sharpe_row = f"\\quad & Sharpe & {sharpe_acc_str} & {sharpe_auc_str} & {sharpe_trades:.1f} & {sharpe_winrate_str} & {sharpe_sharpe_str} & -- \\\\"

    return f"{auc_row}\n{sharpe_row}"


def update_ablation_full_model(tex_file, df, strategy="long_short"):
    """Update the Full Model rows in the Ablation Study table."""
    full_model_rows = generate_ablation_full_model_rows(df, strategy)

    if full_model_rows is None:
        print("Could not generate Full Model rows - missing data")
        return False

    print("\nGenerated Full Model rows:")
    print(full_model_rows)

    if not os.path.exists(tex_file):
        print(f"LaTeX file not found: {tex_file}")
        return False

    with open(tex_file, 'r') as f:
        content = f.read()

    # Pattern to match the Full Model rows in Ablation Study table
    pattern = (
        r"(\\caption\{Ablation Study Results\}.*?"
        r"\\midrule\s*\n)"  # Match up to first \midrule
        r"(Full Model & AUC.*?\n.*?\\quad & Sharpe.*?\n)"  # Capture old Full Model rows
        r"(\\midrule)"  # Next \midrule after Full Model
    )

    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Could not find Full Model rows in Ablation Study table")
        return False

    # Replace the Full Model rows
    new_content = content[:match.start(2)] + full_model_rows + "\n" + content[match.start(3):]

    with open(tex_file, 'w') as f:
        f.write(new_content)

    print("Updated Full Model rows in Ablation Study table")
    return True


def compute_sota_ours_metrics(df, strategy="long_short"):
    """Compute Sharpe-optimized metrics for SOTA table 'Ours' row.

    For each stock, find the configuration with highest Sharpe ratio,
    then average across all stocks.

    Args:
        df: DataFrame with tuned results
        strategy: Trading strategy to filter for (default: 'long_short')

    Returns:
        Dictionary with 'acc', 'auc', 'trades', 'winrate', 'sharpe' keys
    """
    # Filter for the specified strategy
    strategy_df = df[df["Strategy"] == strategy]

    if len(strategy_df) == 0:
        print(f"Warning: No results found for strategy '{strategy}'")
        return None

    # For each stock, find the configuration with highest Sharpe
    best_sharpe_rows = []
    for stock in STOCKS:
        stock_data = strategy_df[strategy_df["Stock"] == stock]
        if len(stock_data) > 0:
            best_idx = stock_data["Sharpe"].idxmax()
            best_sharpe_rows.append(stock_data.loc[best_idx])

    if not best_sharpe_rows:
        print("Warning: Could not find best Sharpe configurations")
        return None

    # Average across stocks
    best_sharpe_df = pd.DataFrame(best_sharpe_rows)

    return {
        'acc': best_sharpe_df["Test_Accuracy"].mean(),
        'auc': best_sharpe_df["Test_ROC_AUC"].mean(),
        'trades': best_sharpe_df["Trades"].mean(),
        'winrate': best_sharpe_df["WinRate"].mean() * 100,
        'sharpe': best_sharpe_df["Sharpe"].mean()
    }


def format_sota_ours_row(ours_metrics):
    """Format the 'Ours' row for the SOTA comparison table.

    Args:
        ours_metrics: Dictionary with 'acc', 'auc', 'trades', 'winrate', 'sharpe'

    Returns:
        LaTeX table row for 'Ours (Multimodal)'
    """
    if ours_metrics is None:
        return None

    # Bold the best metrics (we assume Ours has best AUC, Win%, Sharpe)
    # Accuracy bolding depends on baseline comparison
    acc_str = f"{ours_metrics['acc']:.3f}"  # Will be bolded manually if needed
    auc_str = f"\\textbf{{{ours_metrics['auc']:.3f}}}"
    winrate_str = f"\\textbf{{{ours_metrics['winrate']:.1f}}}"
    sharpe_str = f"\\textbf{{{ours_metrics['sharpe']:.2f}}}"

    return f"\\textbf{{Ours (Multimodal)}} & {acc_str} & {auc_str} & {ours_metrics['trades']:.1f} & {winrate_str} & {sharpe_str} \\\\"


def update_sota_ours_row(tex_file, df, strategy="long_short"):
    """Update the 'Ours' row in the SOTA comparison table.

    Args:
        tex_file: Path to main.tex
        df: DataFrame with tuned results
        strategy: Trading strategy to use

    Returns:
        True if successful, False otherwise
    """
    # Compute Ours metrics
    ours_metrics = compute_sota_ours_metrics(df, strategy)

    if ours_metrics is None:
        print("Could not compute Ours metrics for SOTA table")
        return False

    # Format the Ours row
    ours_row = format_sota_ours_row(ours_metrics)

    if ours_row is None:
        print("Could not format Ours row for SOTA table")
        return False

    print("\nGenerated Ours row for SOTA table:")
    print(ours_row)
    print(f"Metrics: Acc={ours_metrics['acc']:.3f}, AUC={ours_metrics['auc']:.3f}, "
          f"N={ours_metrics['trades']:.1f}, Win%={ours_metrics['winrate']:.1f}, "
          f"Sharpe={ours_metrics['sharpe']:.2f}")

    if not os.path.exists(tex_file):
        print(f"LaTeX file not found: {tex_file}")
        return False

    with open(tex_file, 'r') as f:
        content = f.read()

    # Pattern to match the Ours row in SOTA table
    # Look for the row starting with \textbf{Ours
    pattern = (
        r"(\\textbf\{Ours \(Multimodal\)\}.*?\\\\)"
    )

    match = re.search(pattern, content, re.DOTALL)

    if not match:
        print("Could not find Ours row in SOTA table")
        print("Looking for pattern: \\textbf{Ours (Multimodal)}")
        return False

    # Replace the Ours row
    new_content = content[:match.start(1)] + ours_row + content[match.end(1):]

    with open(tex_file, 'w') as f:
        f.write(new_content)

    print("Updated Ours row in SOTA comparison table")
    return True


def update_main_tex_tables(df, tex_file="main.tex", strategy="long_short", update_sota=False):
    """Generate and update all tables in main.tex.

    Args:
        df: DataFrame with tuned results
        tex_file: Path to main.tex file
        strategy: Trading strategy to use (default: 'long_short')
        update_sota: If True, also update the SOTA comparison table 'Ours' row

    Returns:
        True if all updates successful, False otherwise
    """
    print("\n" + "=" * 80)
    print("UPDATING MAIN.TEX TABLES")
    print("=" * 80)

    # Generate tables
    auc_table = generate_latex_table_by_auc(df, strategy)
    sharpe_table = generate_latex_table_by_sharpe(df, strategy)

    if auc_table is None or sharpe_table is None:
        print("Could not generate tables - missing data")
        return False

    print("\nGenerated LaTeX table (Best by AUC):")
    print(auc_table)
    print("\nGenerated LaTeX table (Best by Sharpe):")
    print(sharpe_table)

    # Update main.tex tables
    success1 = update_latex_table_in_file(
        tex_file,
        "Best Model Performance by Stock (Selected by AUC)",
        auc_table
    )

    success2 = update_latex_table_in_file(
        tex_file,
        "Best Model Performance by Stock (Selected by Sharpe Ratio)",
        sharpe_table
    )

    # Update Ablation Study Full Model rows
    success3 = update_ablation_full_model(tex_file, df, strategy)

    # Optionally update SOTA comparison table
    success4 = True
    if update_sota:
        print("\n" + "=" * 80)
        print("UPDATING SOTA COMPARISON TABLE")
        print("=" * 80)
        success4 = update_sota_ours_row(tex_file, df, strategy)

    if success1 and success2 and success3 and success4:
        print(f"\nSuccessfully updated all tables in {tex_file}")
        return True
    else:
        print(f"\nFailed to update some tables in {tex_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Analyze Hyperparameter Tuned ML Model Results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results/ours",
        help="Directory containing tuned result CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results/ours",
        help="Directory to save summary CSV files",
    )
    parser.add_argument(
        "--update-tex",
        action="store_true",
        help="Update LaTeX tables in main.tex (AUC, Sharpe, Ablation Full Model)",
    )
    parser.add_argument(
        "--update-sota",
        action="store_true",
        help="Also update the SOTA comparison table 'Ours' row (requires --update-tex)",
    )
    parser.add_argument(
        "--tex-file",
        type=str,
        default="docs/main.tex",
        help="Path to main.tex file to update",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("HYPERPARAMETER TUNED ML MODEL RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load all tuning results
    df = load_all_results(args.results_dir)

    if df is None or len(df) == 0:
        print("No new tuning results found!")
        return

    print(f"\nLoaded {len(df)} result rows from new tuning scripts")
    print(f"Stocks: {df['Stock'].unique().tolist()}")
    print(f"Models: {df['Model'].unique().tolist()}")
    print(f"Strategies: {df['Strategy'].unique().tolist()}")
    if 'Horizon' in df.columns:
        print(f"Horizons: {sorted(df['Horizon'].unique().tolist())}")

    # Run analyses
    analyze_by_model(df)
    analyze_by_strategy(df)
    analyze_by_stock(df)
    model_strategy_summary = analyze_by_model_and_strategy(df)
    find_best_configurations(df)
    create_comparison_table(df)
    model_comparison = create_model_comparison_table(df)
    create_summary_table(df, args.output_dir)

    # Save model+strategy summary
    model_strategy_summary.to_csv(f"{args.output_dir}/tuned_summary_by_model_strategy.csv", index=False)
    model_comparison.to_csv(f"{args.output_dir}/tuned_model_comparison.csv", index=False)

    # Save full combined results
    df.to_csv(f"{args.output_dir}/tuned_all_results_combined.csv", index=False)
    print(f"\nFull combined results saved to: {args.output_dir}/tuned_all_results_combined.csv")

    # Update LaTeX tables if requested
    if args.update_tex:
        update_main_tex_tables(df, tex_file=args.tex_file, strategy="long_short", update_sota=args.update_sota)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
