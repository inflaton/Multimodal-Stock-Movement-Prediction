"""
FinCast Baseline for Stock Prediction

This script creates a baseline using FinCast foundation model
for time series forecasting. It uses the same evaluation methodology as
the LSTM and other models in the paper.

Methodology:
1. Use FinCast to predict the Close price H days ahead
2. Convert the price prediction to a binary classification:
   - Predict 1 (up) if predicted return > threshold
   - Predict 0 (down) otherwise
3. Trading simulation with long/short strategy
4. Evaluate using the same metrics: Accuracy, AUC, Sharpe, Win Rate

Installation:
    # Clone FinCast-fts repository and install dependencies
    cd FinCast-fts
    ./env_setup.sh
    ./dep_install.sh

    # Download model weights from HuggingFace
    # Model is available at: https://huggingface.co/Vincent05R/FinCast

Usage:
    python fincast_baseline.py --all-stocks
    python fincast_baseline.py --stock AAPL
    python fincast_baseline.py --stock AAPL --model-path /path/to/fincast_v1.pth
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from types import SimpleNamespace

# Add FinCast-fts to path
FINCAST_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "FinCast-fts", "src"
)
if FINCAST_PATH not in sys.path:
    sys.path.insert(0, FINCAST_PATH)

import torch
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================================
# Configuration
# ============================================================================

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR = 2024
FEE_BPS_ROUND_TRIP = 10  # 10 basis points = 0.10%
RANDOM_SEED = 42


# Default FinCast model path - check multiple locations
def get_default_model_path():
    """Find FinCast model weights in standard locations."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    possible_paths = [
        os.path.join(base_dir, "FinCast-fts", "model_weights", "v1.pth"),
        os.path.join(base_dir, "FinCast-fts", "checkpoints", "v1.pth"),
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--Vincent05R--FinCast/snapshots/*/v1.pth"
        ),
    ]
    for path in possible_paths:
        if "*" in path:
            import glob

            matches = glob.glob(path)
            if matches:
                return matches[0]
        elif os.path.exists(path):
            return path
    # Return first path as default (will need to be downloaded)
    return possible_paths[0]


DEFAULT_MODEL_PATH = get_default_model_path()

# Default thresholds per horizon (averaged across all stocks from hyperparameter tuning)
DEFAULT_THRESHOLDS = {
    2: 0.0025,
    3: 0.0025,
    4: 0.0050,
    5: 0.0050,
    6: 0.0075,
    7: 0.0100,
    8: 0.0150,
    9: 0.0175,
    10: 0.0170,
}
ALL_HORIZONS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# ============================================================================
# Utility Functions
# ============================================================================


def load_data(stock: str, data_dir: str) -> pd.DataFrame:
    """Load and preprocess stock data."""
    df = pd.read_csv(f"{data_dir}/{stock}_data_model_training.csv")
    df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Year"] = df["__DateDT__"].dt.year
    return df


def calculate_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """
    Calculate binary target variable.

    Target = 1 if price increases by more than threshold over horizon days.
    """
    future_return = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    return (future_return > threshold).astype(int)


# Trading strategy constants
STRATEGY_LONG_SHORT = "long_short"


def non_overlap_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    horizon: int,
    fee_bps: int = FEE_BPS_ROUND_TRIP,
    predicted_probs: np.ndarray = None,
) -> dict:
    """
    Simulate trading with non-overlapping positions using long/short strategy.
    """
    prices = test_df.sort_values("__DateDT__")[["__DateDT__", "Close"]].reset_index(
        drop=True
    )
    fee_mult = 1 - fee_bps / 10000

    long_trades = []
    short_trades = []
    equity = 1.0
    pred_idx = 0
    i = 0

    while i + horizon < len(prices) and pred_idx < len(predictions):
        p_in = prices.iloc[i]["Close"]
        p_out = prices.iloc[i + horizon]["Close"]
        pred = predictions[pred_idx]

        # Long/short strategy: long if pred=1, short if pred=0
        if pred == 1:
            gross_return = p_out / p_in - 1
            net_return = (1 + gross_return) * fee_mult - 1
            long_trades.append(net_return)
            equity *= 1 + net_return
        else:
            gross_return = p_in / p_out - 1
            net_return = (1 + gross_return) * fee_mult - 1
            short_trades.append(net_return)
            equity *= 1 + net_return

        i += horizon
        pred_idx += 1

    all_trades = np.array(long_trades + short_trades)

    if len(all_trades) == 0:
        return {
            "Trades": 0,
            "LongTrades": 0,
            "ShortTrades": 0,
            "WinRate": 0.0,
            "Sharpe": 0.0,
            "TotalReturn": 0.0,
        }

    win_rate = (all_trades > 0).mean()
    sharpe = (
        all_trades.mean() / all_trades.std() * np.sqrt(len(all_trades))
        if all_trades.std() > 0
        else 0
    )
    total_return = equity - 1

    return {
        "Trades": len(all_trades),
        "LongTrades": len(long_trades),
        "ShortTrades": len(short_trades),
        "WinRate": win_rate,
        "Sharpe": sharpe,
        "TotalReturn": total_return,
    }


# ============================================================================
# FinCast Prediction Functions
# ============================================================================


def download_fincast_model(target_dir: str) -> str:
    """
    Download FinCast model from HuggingFace.

    Returns the path to the downloaded model file.
    """
    try:
        from huggingface_hub import hf_hub_download

        print("Downloading FinCast model from HuggingFace (~4GB)...")
        print("This may take several minutes...")
        model_path = hf_hub_download(
            repo_id="Vincent05R/FinCast",
            filename="v1.pth",
            local_dir=target_dir,
        )
        print(f"Model downloaded to: {model_path}")
        return model_path
    except ImportError:
        print(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        raise
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise


def load_fincast_model(model_path: str, config: SimpleNamespace):
    """
    Load FinCast model.
    """
    from tools.model_utils import get_model_FFM
    from ffm import FFmHparams

    ffm_hparams = FFmHparams(
        backend=config.backend,
        per_core_batch_size=32,
        horizon_len=config.horizon_len,
        context_len=config.context_len,
        use_positional_embedding=False,
        num_experts=config.num_experts,
        gating_top_n=config.gating_top_n,
        load_from_compile=config.load_from_compile,
        point_forecast_mode=config.forecast_mode,
    )

    model_actual, ffm_config, ffm_api = get_model_FFM(model_path, ffm_hparams)
    ffm_api.model_eval_mode()

    return ffm_api


def predict_with_fincast(
    model,
    context: np.ndarray,
    prediction_length: int,
    freq: int = 0,  # 0 = daily (high frequency)
) -> tuple:
    """
    Generate predictions using FinCast.

    Parameters:
    -----------
    model : FFM API instance
    context : array of historical prices
    prediction_length : number of steps to predict
    freq : frequency indicator (0=high/daily, 1=medium, 2=low)

    Returns:
    --------
    tuple of (mean_prediction, low_quantile, high_quantile)
    """
    # FinCast expects list of time series
    inputs = [context.astype(np.float32)]
    freqs = [freq]

    # Get forecasts
    mean_outputs, full_outputs = model.forecast(inputs, freqs)

    # mean_outputs: [1, H]
    # full_outputs: [1, H, 1+Q] where Q=9 quantiles
    mean = mean_outputs[0, :prediction_length]

    # Extract quantiles (index 0=mean, 1-9=quantiles q1-q9)
    # q1 ~ 0.1 quantile, q5 ~ 0.5 (median), q9 ~ 0.9 quantile
    if full_outputs is not None and full_outputs.shape[2] > 1:
        low = full_outputs[0, :prediction_length, 1]  # q1 ~ 0.1
        high = full_outputs[0, :prediction_length, 9]  # q9 ~ 0.9
    else:
        low = mean
        high = mean

    return mean, low, high


def generate_predictions_for_test(
    model,
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    horizon: int,
    context_length: int = 128,
) -> tuple:
    """
    Generate predictions for all test data points.

    Uses a rolling window approach: for each test point,
    use the last `context_length` prices as context.
    """
    # Combine train and test for rolling context
    all_prices = np.concatenate([train_prices, test_prices])
    train_len = len(train_prices)

    predictions = []
    predicted_probs = []

    for i in range(len(test_prices) - horizon):
        # Current position in the combined array
        current_idx = train_len + i

        # Get context: last context_length prices before current point
        start_idx = max(0, current_idx - context_length)
        context = all_prices[start_idx:current_idx]

        # Predict
        mean, low, high = predict_with_fincast(
            model,
            context,
            prediction_length=horizon,
        )

        # Current price
        current_price = all_prices[current_idx]

        # Predicted price at horizon
        predicted_price = mean[-1]  # Last prediction in the horizon

        # Predicted return
        predicted_return = (predicted_price - current_price) / current_price

        # Store results
        predictions.append(predicted_return)

        # For probability, use the proportion of samples predicting positive return
        if low[-1] > current_price:
            prob = 0.9
        elif high[-1] < current_price:
            prob = 0.1
        else:
            prob = 0.5 + 0.4 * (predicted_return / (0.05 + abs(predicted_return)))
            prob = np.clip(prob, 0.1, 0.9)

        predicted_probs.append(prob)

    return np.array(predictions), np.array(predicted_probs)


# ============================================================================
# Main Evaluation Function
# ============================================================================


def run_fincast_baseline_for_stock(
    stock: str,
    data_dir: str,
    model_path: str,
    context_length: int = 128,
    output_dir: str = ".",
    model=None,
) -> pd.DataFrame:
    """
    Run FinCast baseline for all horizons of a given stock.
    """
    print(f"\n{'='*70}")
    print(f"FINCAST BASELINE: {stock}")
    print(f"Model: {model_path}")
    print(f"Context length: {context_length}")
    print(f"{'='*70}")

    # Load FinCast model if not provided
    if model is None:
        config = SimpleNamespace(
            backend="gpu" if torch.cuda.is_available() else "cpu",
            context_len=context_length,
            horizon_len=max(ALL_HORIZONS),  # Max horizon we'll need
            num_experts=4,
            gating_top_n=2,
            load_from_compile=True,
            forecast_mode="mean",
        )
        model = load_fincast_model(model_path, config)

    # Load data
    df = load_data(stock, data_dir)

    # Split train/test
    train_df = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Year"] == TEST_YEAR].copy()

    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Get prices
    train_prices = train_df.sort_values("__DateDT__")["Close"].values
    test_prices = test_df.sort_values("__DateDT__")["Close"].values

    # Use all horizons with default thresholds
    print(f"Using all horizons with default thresholds")
    horizons_thresholds = [(h, DEFAULT_THRESHOLDS[h]) for h in ALL_HORIZONS]

    results = []

    for horizon, threshold in horizons_thresholds:
        horizon = int(horizon)
        print(f"\n{'-'*50}")
        print(f"Horizon: {horizon} days, Threshold: {threshold:.4f}")
        print(f"{'-'*50}")

        # Generate FinCast predictions
        print("Generating FinCast predictions...")
        predicted_returns, predicted_probs = generate_predictions_for_test(
            model,
            train_prices,
            test_prices,
            horizon=horizon,
            context_length=context_length,
        )

        # Convert predicted returns to binary predictions
        y_pred = (predicted_returns > threshold).astype(int)

        # Calculate actual target
        test_copy = test_df.sort_values("__DateDT__").copy()
        test_copy["target"] = calculate_target(test_copy, horizon, threshold)
        test_copy = test_copy.dropna(subset=["target"])

        # Align predictions with targets
        n_predictions = len(y_pred)
        n_targets = len(test_copy) - horizon

        # Use the minimum length
        n_valid = min(n_predictions, n_targets)
        y_pred = y_pred[:n_valid]
        predicted_probs = predicted_probs[:n_valid]
        y_true = test_copy["target"].values[:n_valid]

        if len(y_pred) == 0:
            print(f"Skipping horizon {horizon}: no valid predictions")
            continue

        # Calculate metrics
        test_acc = accuracy_score(y_true, y_pred)
        try:
            test_auc = roc_auc_score(y_true, predicted_probs)
        except ValueError:
            test_auc = 0.5

        # Backtest
        backtest = non_overlap_backtest(
            test_copy,
            y_pred,
            horizon,
        )

        result = {
            "Stock": stock,
            "Horizon": horizon,
            "BestThreshold": threshold,
            "Model": "FinCast",
            "ContextLength": context_length,
            "Test_Accuracy": test_acc,
            "Test_ROC_AUC": test_auc,
            "Trades": backtest["Trades"],
            "LongTrades": backtest["LongTrades"],
            "ShortTrades": backtest["ShortTrades"],
            "WinRate": backtest["WinRate"],
            "Sharpe": backtest["Sharpe"],
            "TotalReturn": backtest["TotalReturn"],
        }
        results.append(result)

        print(f"\nTest Results:")
        print(f"  Accuracy: {test_acc:.3f}")
        print(f"  AUC: {test_auc:.3f}")
        print(
            f"  Trades: {backtest['Trades']} (Long: {backtest['LongTrades']}, Short: {backtest['ShortTrades']})"
        )
        print(f"  Sharpe: {backtest['Sharpe']:.2f}")
        print(f"  Win Rate: {backtest['WinRate']*100:.1f}%")
        print(f"  Total Return: {backtest['TotalReturn']*100:.1f}%")

    # Save results
    results_df = pd.DataFrame(results)
    output_file = f"{output_dir}/fincast_{stock}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="FinCast Baseline for Stock Prediction"
    )
    parser.add_argument(
        "--stock",
        type=str,
        default="AAPL",
        choices=STOCKS,
        help="Stock symbol to evaluate",
    )
    parser.add_argument(
        "--all-stocks", action="store_true", help="Run evaluation for all stocks"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to FinCast model weights (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./dataset/training_data",
        help="Directory containing *_data_model_training.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/baselines",
        help="Directory to save output results",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        help="Number of historical days to use as context (default: 128)",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Check if model exists, download if needed
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        # Try to download from HuggingFace
        target_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "FinCast-fts", "model_weights"
        )
        try:
            model_path = download_fincast_model(target_dir)
        except Exception as e:
            print(f"Failed to download model: {e}")
            print(
                "\nPlease manually download from: https://huggingface.co/Vincent05R/FinCast"
            )
            print(f"And place v1.pth in: {target_dir}")
            return

    # Load model once (shared across all stocks)
    print(f"Loading FinCast model from: {model_path}")
    config = SimpleNamespace(
        backend="gpu" if torch.cuda.is_available() else "cpu",
        context_len=args.context_length,
        horizon_len=max(ALL_HORIZONS),
        num_experts=4,
        gating_top_n=2,
        load_from_compile=True,
        forecast_mode="mean",
    )

    try:
        model = load_fincast_model(model_path, config)
    except Exception as e:
        print(f"Error loading FinCast model: {e}")
        print("Make sure FinCast-fts dependencies are installed.")
        print("Run: cd FinCast-fts && ./env_setup.sh && ./dep_install.sh")
        return

    if args.all_stocks:
        all_results = []
        for stock in STOCKS:
            results = run_fincast_baseline_for_stock(
                stock=stock,
                data_dir=args.data_dir,
                model_path=args.model_path,
                context_length=args.context_length,
                output_dir=args.output_dir,
                model=model,
            )
            all_results.append(results)

        # Combine and save all results
        combined = pd.concat(all_results, ignore_index=True)
        combined_file = f"{args.output_dir}/fincast_all_stocks_results.csv"
        combined.to_csv(combined_file, index=False)

        # Print summary
        print(f"\n\n{'='*70}")
        print("SUMMARY (Best by Criterion per Stock, then Averaged)")
        print(f"{'='*70}")
        print(f"\nModel: FinCast")

        # Compute "Best by AUC" summary
        auc_selected = []
        for stock in STOCKS:
            stock_data = combined[combined["Stock"] == stock]
            if len(stock_data) > 0:
                best_idx = stock_data["Test_ROC_AUC"].idxmax()
                auc_selected.append(stock_data.loc[best_idx])
        auc_df = pd.DataFrame(auc_selected)

        print(f"\nBest by AUC (per stock, then averaged):")
        print(f"  Accuracy:     {auc_df['Test_Accuracy'].mean():.3f}")
        print(f"  AUC:          {auc_df['Test_ROC_AUC'].mean():.3f}")
        print(f"  Sharpe:       {auc_df['Sharpe'].mean():.2f}")
        print(f"  Win%:         {auc_df['WinRate'].mean()*100:.1f}%")

        # Compute "Best by Sharpe" summary
        sharpe_selected = []
        for stock in STOCKS:
            stock_data = combined[combined["Stock"] == stock]
            if len(stock_data) > 0:
                best_idx = stock_data["Sharpe"].idxmax()
                sharpe_selected.append(stock_data.loc[best_idx])
        sharpe_df = pd.DataFrame(sharpe_selected)

        print(f"\nBest by Sharpe (per stock, then averaged):")
        print(f"  Accuracy:     {sharpe_df['Test_Accuracy'].mean():.3f}")
        print(f"  AUC:          {sharpe_df['Test_ROC_AUC'].mean():.3f}")
        print(f"  Sharpe:       {sharpe_df['Sharpe'].mean():.2f}")
        print(f"  Win%:         {sharpe_df['WinRate'].mean()*100:.1f}%")

        # Print LaTeX table rows
        print(f"\n{'='*70}")
        print("LaTeX Table Rows:")
        print(f"{'='*70}")
        print(
            f"FinCast (Best by AUC) & {auc_df['Test_Accuracy'].mean():.3f} & {auc_df['Test_ROC_AUC'].mean():.3f} & {auc_df['Sharpe'].mean():.2f} & {auc_df['WinRate'].mean()*100:.1f} \\\\"
        )
        print(
            f"FinCast (Best by Sharpe) & {sharpe_df['Test_Accuracy'].mean():.3f} & {sharpe_df['Test_ROC_AUC'].mean():.3f} & {sharpe_df['Sharpe'].mean():.2f} & {sharpe_df['WinRate'].mean()*100:.1f} \\\\"
        )

        print(f"\nCombined results saved to: {combined_file}")
    else:
        run_fincast_baseline_for_stock(
            stock=args.stock,
            data_dir=args.data_dir,
            model_path=args.model_path,
            context_length=args.context_length,
            output_dir=args.output_dir,
            model=model,
        )


if __name__ == "__main__":
    main()
