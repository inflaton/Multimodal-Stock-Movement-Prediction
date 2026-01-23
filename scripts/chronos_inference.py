"""
Chronos-2 Inference Script for Stock Prediction

This script runs inference using pre-trained Chronos-2 models with covariates.
It loads models from the finetuned_covariates directory and generates predictions.

Usage:
    python chronos_inference.py --all-stocks
    python chronos_inference.py --stock AAPL
    python chronos_inference.py --all-stocks --model-dir finetuned_covariates
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
from sklearn.metrics import accuracy_score, roc_auc_score

# ============================================================================
# Configuration
# ============================================================================

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR = 2024
FEE_BPS_ROUND_TRIP = 10  # 10 basis points = 0.10%

# Trading strategy constants
STRATEGY_LONG_SHORT = "long_short"

# Default thresholds per horizon
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

# Covariate columns
COVARIATE_COLS = ["Filtered Sentiment Score"]


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
    """Calculate binary target variable."""
    future_return = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    return (future_return > threshold).astype(int)


def non_overlap_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    horizon: int,
    fee_bps: int = FEE_BPS_ROUND_TRIP,
    strategy: str = STRATEGY_LONG_SHORT,
    predicted_probs: np.ndarray = None,
) -> dict:
    """Simulate trading with non-overlapping positions."""
    prices = test_df.sort_values("__DateDT__")[["__DateDT__", "Close"]].reset_index(
        drop=True
    )
    fee_mult = 1 - fee_bps / 10000

    long_trades = []
    short_trades = []
    skipped = 0
    equity = 1.0
    pred_idx = 0
    i = 0

    while i + horizon < len(prices) and pred_idx < len(predictions):
        p_in = prices.iloc[i]["Close"]
        p_out = prices.iloc[i + horizon]["Close"]
        pred = predictions[pred_idx]

        if strategy == STRATEGY_LONG_SHORT:
            action = "long" if pred == 1 else "short"
        else:
            action = "long" if pred == 1 else None

        if action == "long":
            gross_return = p_out / p_in - 1
            net_return = (1 + gross_return) * fee_mult - 1
            long_trades.append(net_return)
            equity *= 1 + net_return
        elif action == "short":
            gross_return = p_in / p_out - 1
            net_return = (1 + gross_return) * fee_mult - 1
            short_trades.append(net_return)
            equity *= 1 + net_return
        else:
            skipped += 1

        i += horizon
        pred_idx += 1

    all_trades = np.array(long_trades + short_trades)

    if len(all_trades) == 0:
        return {
            "Trades": 0,
            "LongTrades": 0,
            "ShortTrades": 0,
            "Skipped": skipped,
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
        "Skipped": skipped,
        "WinRate": win_rate,
        "Sharpe": sharpe,
        "TotalReturn": total_return,
    }


# ============================================================================
# Inference Function
# ============================================================================


def generate_rolling_predictions(
    predictor: TimeSeriesPredictor,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    context_length: int = 60,
    covariate_cols: list = None,
) -> tuple:
    """
    Generate rolling predictions using a fine-tuned AutoGluon predictor.

    Uses a rolling window approach: for each test point,
    use the last `context_length` prices as context.

    Parameters:
    -----------
    predictor : TimeSeriesPredictor instance (fine-tuned)
    train_df : DataFrame with training data (sorted by date, with Close and optional covariates)
    test_df : DataFrame with test data (sorted by date, with Close and optional covariates)
    horizon : prediction horizon
    context_length : number of historical days to use as context
    covariate_cols : list of covariate column names (optional)

    Returns:
    --------
    tuple of (predicted_returns, predicted_probs)
    """
    # Combine train and test for rolling context
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    train_len = len(train_df)

    all_prices = all_df["Close"].values

    predictions = []
    predicted_probs = []

    for i in range(len(test_df) - horizon):
        # Current position in the combined array
        current_idx = train_len + i

        # Get context: last context_length rows before current point
        start_idx = max(0, current_idx - context_length)
        context_data = all_df.iloc[start_idx:current_idx].copy()

        # Create TimeSeriesDataFrame for this context
        data_dict = {
            "item_id": "STOCK",
            "timestamp": pd.RangeIndex(len(context_data)),
            "target": context_data["Close"].values,
        }

        # Add covariates if specified
        if covariate_cols:
            for col in covariate_cols:
                if col in context_data.columns:
                    # Handle NaN values
                    values = context_data[col].ffill().bfill().fillna(0).values
                    data_dict[col] = values

        context_df = pd.DataFrame(data_dict)
        context_ts = TimeSeriesDataFrame.from_data_frame(
            context_df,
            id_column="item_id",
            timestamp_column="timestamp",
        )

        # Prepare known_covariates for the forecast horizon if covariates are used
        known_covariates_ts = None
        if covariate_cols:
            # Get future covariate values for the horizon period
            future_start = current_idx
            future_end = min(current_idx + horizon, len(all_df))
            future_data = all_df.iloc[future_start:future_end].copy()

            if len(future_data) > 0:
                # Build future covariates DataFrame
                future_cov_dict = {
                    "item_id": ["STOCK"] * len(future_data),
                    "timestamp": pd.RangeIndex(
                        len(context_data), len(context_data) + len(future_data)
                    ),
                }
                for col in covariate_cols:
                    if col in future_data.columns:
                        values = future_data[col].ffill().bfill().fillna(0).values
                        future_cov_dict[col] = values

                future_cov_df = pd.DataFrame(future_cov_dict)
                known_covariates_ts = TimeSeriesDataFrame.from_data_frame(
                    future_cov_df,
                    id_column="item_id",
                    timestamp_column="timestamp",
                )

        # Predict using AutoGluon predictor
        try:
            if known_covariates_ts is not None:
                pred_df = predictor.predict(
                    context_ts, known_covariates=known_covariates_ts
                )
            else:
                pred_df = predictor.predict(context_ts)
            pred_df = pred_df.reset_index()

            # Extract mean prediction
            if "mean" in pred_df.columns:
                forecast = pred_df["mean"].values
            else:
                numeric_cols = pred_df.select_dtypes(include=[np.number]).columns
                forecast = pred_df[numeric_cols[0]].values

            # Get predicted price at horizon (last value in forecast)
            predicted_price = forecast[-1]

            # Current price
            current_price = all_prices[current_idx]

            # Predicted return
            predicted_return = (predicted_price - current_price) / current_price
            predictions.append(predicted_return)

            # Estimate probability based on return magnitude
            prob = 0.5 + 0.4 * (predicted_return / (0.05 + abs(predicted_return)))
            prob = np.clip(prob, 0.1, 0.9)
            predicted_probs.append(prob)

        except Exception as e:
            # If prediction fails, use neutral prediction
            print(f"  Warning: Prediction failed at index {i}: {e}")
            predictions.append(0.0)
            predicted_probs.append(0.5)

    return np.array(predictions), np.array(predicted_probs)


# ============================================================================
# Main Inference Function
# ============================================================================


def run_inference_for_stock(
    stock: str,
    data_dir: str,
    model_dir: str,
    output_dir: str = ".",
    covariate_cols: list = None,
    context_length: int = 60,
) -> pd.DataFrame:
    """
    Run inference using pre-trained Chronos-2 models for all horizons of a given stock.
    """
    print(f"\n{'='*70}", flush=True)
    print(f"CHRONOS-2 INFERENCE: {stock}", flush=True)
    if covariate_cols:
        print(f"Covariates: {covariate_cols}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load data
    print(f"Loading data from {data_dir}/{stock}_data_model_training.csv...", flush=True)
    df = load_data(stock, data_dir)

    # Split train/test
    train_df = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Year"] == TEST_YEAR].copy()

    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}", flush=True)

    # Sort DataFrames by date for rolling predictions
    train_df_sorted = train_df.sort_values("__DateDT__").reset_index(drop=True)
    test_df_sorted = test_df.sort_values("__DateDT__").reset_index(drop=True)

    results = []

    for horizon in ALL_HORIZONS:
        threshold = DEFAULT_THRESHOLDS[horizon]
        print(f"\n{'-'*50}", flush=True)
        print(f"Horizon: {horizon} days, Threshold: {threshold:.4f}", flush=True)
        print(f"{'-'*50}", flush=True)

        # Load pre-trained model
        model_path = Path(model_dir) / f"chronos2_finetuned_{stock}_h{horizon}"
        if not model_path.exists():
            print(f"  Model not found at {model_path}, skipping...", flush=True)
            continue

        print(f"Loading model from {model_path}...", flush=True)
        predictor = TimeSeriesPredictor.load(str(model_path))
        print(f"Model loaded successfully.", flush=True)

        # Generate rolling predictions
        print("Generating rolling predictions...", flush=True)
        predicted_returns, predicted_probs = generate_rolling_predictions(
            predictor=predictor,
            train_df=train_df_sorted,
            test_df=test_df_sorted,
            horizon=horizon,
            context_length=context_length,
            covariate_cols=covariate_cols,
        )

        # Convert to binary predictions
        binary_predictions = (predicted_returns > threshold).astype(int)

        # Calculate targets
        test_df_sorted["target"] = calculate_target(test_df_sorted, horizon, threshold)

        # Get actual targets for evaluation
        y_true = test_df_sorted["target"].values[: len(binary_predictions)]

        # Calculate metrics
        accuracy = accuracy_score(y_true, binary_predictions)
        try:
            auc = roc_auc_score(y_true, predicted_probs)
        except ValueError:
            auc = 0.5

        # Run backtest
        backtest = non_overlap_backtest(
            test_df=test_df_sorted,
            predictions=binary_predictions,
            horizon=horizon,
            strategy=STRATEGY_LONG_SHORT,
            predicted_probs=predicted_probs,
        )

        print(f"\nTest Results:", flush=True)
        print(f"  Predictions: {len(binary_predictions)}", flush=True)
        print(f"  Accuracy: {accuracy:.3f}", flush=True)
        print(f"  AUC: {auc:.3f}", flush=True)
        print(f"  Trades: {backtest['Trades']}", flush=True)
        print(f"  Long/Short: {backtest['LongTrades']}/{backtest['ShortTrades']}", flush=True)
        print(f"  Sharpe: {backtest['Sharpe']:.2f}", flush=True)

        results.append(
            {
                "Stock": stock,
                "Horizon": horizon,
                "BestThreshold": threshold,
                "Mode": "Fine-tuned",
                "Strategy": STRATEGY_LONG_SHORT,
                "ConfidenceThreshold": "",
                "FineTuneLR": 0.0001,
                "FineTuneSteps": 1000,
                "UseCovariates": True,
                "Test_Accuracy": accuracy,
                "Test_ROC_AUC": auc,
                **backtest,
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    cov_suffix = "_cov" if covariate_cols else ""
    output_path = Path(output_dir) / f"Chronos2_finetuned{cov_suffix}_{stock}_inference_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}", flush=True)

    return results_df


def main():
    import sys
    print("Starting inference script...", flush=True)
    sys.stdout.flush()

    parser = argparse.ArgumentParser(
        description="Run Chronos-2 inference with pre-trained models"
    )
    parser.add_argument("--stock", type=str, help="Stock symbol (e.g., AAPL)")
    parser.add_argument(
        "--all-stocks", action="store_true", help="Run for all stocks"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing stock data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="finetuned_covariates",
        help="Directory containing pre-trained models",
    )
    parser.add_argument(
        "--use-covariates",
        action="store_true",
        help="Use covariates (sentiment) for prediction",
    )
    parser.add_argument(
        "--no-covariates",
        action="store_true",
        help="Run without covariates (price only)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="finetuned_covariates",
        help="Directory for output results",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=60,
        help="Context length for predictions (default: 60)",
    )

    args = parser.parse_args()

    # Determine which stocks to process
    if args.all_stocks:
        stocks = STOCKS
    elif args.stock:
        stocks = [args.stock]
    else:
        parser.error("Must specify --stock or --all-stocks")

    # Determine covariate usage
    # Default: use covariates if model_dir contains "covariate"
    if args.no_covariates:
        covariate_cols = None
        use_cov_suffix = ""
    elif args.use_covariates or "covariate" in args.model_dir.lower():
        covariate_cols = COVARIATE_COLS
        use_cov_suffix = "_cov"
    else:
        covariate_cols = None
        use_cov_suffix = ""

    print(f"Covariate mode: {'WITH covariates' if covariate_cols else 'NO covariates (price only)'}", flush=True)

    # Run inference for each stock
    all_results = []
    for stock in stocks:
        result_df = run_inference_for_stock(
            stock=stock,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            covariate_cols=covariate_cols,
            context_length=args.context_length,
        )
        all_results.append(result_df)

    # Combine all results
    if args.all_stocks and all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = (
            Path(args.output_dir) / f"Chronos2_finetuned{use_cov_suffix}_all_stocks_inference_results.csv"
        )
        combined_df.to_csv(combined_path, index=False)
        print(f"\n{'='*70}", flush=True)
        print(f"All results saved to: {combined_path}", flush=True)
        print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
