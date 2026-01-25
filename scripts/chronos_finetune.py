"""
Chronos-2 Fine-Tuning for Stock Prediction

This script fine-tunes Chronos-2 on your stock data using AutoGluon's
TimeSeriesPredictor for improved forecasting performance.

Requirements:
    pip install autogluon.timeseries

Usage:
    # Fine-tune models for all stocks (loads existing models by default)
    python chronos_finetune.py --all-stocks

    # Fine-tune for a single stock
    python chronos_finetune.py --stock AAPL

    # Fine-tune with covariates
    python chronos_finetune.py --all-stocks --use-covariates

    # Force re-training even if models exist
    python chronos_finetune.py --all-stocks --overwrite-if-exists
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import argparse
import shutil
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
RANDOM_SEED = 42

# Trading strategy constants
STRATEGY_LONG_SHORT = "long_short"
STRATEGY_LONG_ONLY = "long_only"
STRATEGY_LONG_SHORT_CONFIDENCE = "long_short_confidence"
STRATEGY_LONG_ONLY_CONFIDENCE = "long_only_confidence"

VALID_STRATEGIES = [
    STRATEGY_LONG_SHORT,
    STRATEGY_LONG_ONLY,
    STRATEGY_LONG_SHORT_CONFIDENCE,
    STRATEGY_LONG_ONLY_CONFIDENCE,
]

DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # For confidence strategies: long when p(up) > 0.6, short when p(up) < 0.4

# Fine-tuning hyperparameters
DEFAULT_FINE_TUNE_LR = 1e-4
DEFAULT_FINE_TUNE_STEPS = 1000
DEFAULT_FINE_TUNE_BATCH_SIZE = 32

# Default thresholds per horizon (averaged across all stocks from hyperparameter tuning)
# These are used when running with --use-all-horizons flag
DEFAULT_THRESHOLDS = {
    2: 0.0025,
    3: 0.0025,
    4: 0.0050,
    5: 0.0050,
    6: 0.0075,
    7: 0.0100,
    8: 0.0150,
    9: 0.0175,
    10: 0.0170
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


def non_overlap_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    horizon: int,
    fee_bps: int = FEE_BPS_ROUND_TRIP,
    strategy: str = STRATEGY_LONG_SHORT,
    predicted_probs: np.ndarray = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """
    Simulate trading with non-overlapping positions using various strategies.

    Strategies:
    - long_short: Go long when pred=1, go short when pred=0 (always in market)
    - long_only: Go long when pred=1, stay in cash when pred=0
    - long_short_confidence: Long/short only when confidence > threshold
    - long_only_confidence: Long only when p(up) > threshold

    Parameters:
    -----------
    test_df : DataFrame with '__DateDT__' and 'Close' columns
    predictions : array of binary predictions (1 for up, 0 for down)
    horizon : int, holding period in days
    fee_bps : int, round-trip transaction fee in basis points
    strategy : str, one of VALID_STRATEGIES
    predicted_probs : array of predicted probabilities for p(up), required for confidence strategies
    confidence_threshold : float, threshold for confidence-based strategies (default: 0.6)

    Returns:
    --------
    dict with Trades, LongTrades, ShortTrades, Skipped, WinRate, Sharpe, TotalReturn
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {VALID_STRATEGIES}")

    if strategy in [STRATEGY_LONG_SHORT_CONFIDENCE, STRATEGY_LONG_ONLY_CONFIDENCE]:
        if predicted_probs is None:
            raise ValueError(f"predicted_probs required for strategy: {strategy}")

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
        prob = predicted_probs[pred_idx] if predicted_probs is not None else None

        # Determine action based on strategy
        action = None  # None = skip, "long" = go long, "short" = go short

        if strategy == STRATEGY_LONG_SHORT:
            # Always trade: long if pred=1, short if pred=0
            action = "long" if pred == 1 else "short"

        elif strategy == STRATEGY_LONG_ONLY:
            # Long if pred=1, skip if pred=0
            action = "long" if pred == 1 else None

        elif strategy == STRATEGY_LONG_SHORT_CONFIDENCE:
            # Long if p(up) > threshold, short if p(up) < (1 - threshold), else skip
            if prob is not None:
                if prob > confidence_threshold:
                    action = "long"
                elif prob < (1 - confidence_threshold):
                    action = "short"
                # else: skip (no trade)

        elif strategy == STRATEGY_LONG_ONLY_CONFIDENCE:
            # Long only if p(up) > threshold, else skip
            if prob is not None and prob > confidence_threshold:
                action = "long"

        # Execute the action
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
# Data Preparation for AutoGluon
# ============================================================================


def prepare_autogluon_data(
    df: pd.DataFrame,
    stock: str,
    covariate_cols: list = None,
) -> TimeSeriesDataFrame:
    """
    Prepare data in AutoGluon TimeSeriesDataFrame format.

    Parameters:
    -----------
    df : DataFrame with stock data
    stock : stock symbol (used as item_id)
    covariate_cols : list of covariate column names

    Returns:
    --------
    TimeSeriesDataFrame
    """
    # Sort by date
    df = df.sort_values("__DateDT__").copy().reset_index(drop=True)

    # Create base DataFrame with required columns
    # Resample to business day frequency to handle gaps
    data = {
        "item_id": stock,
        "timestamp": df["__DateDT__"].values,
        "target": df["Close"].values,
    }

    # Add covariates if specified
    if covariate_cols:
        for col in covariate_cols:
            if col in df.columns:
                # Handle NaN values
                values = df[col].ffill().bfill().fillna(0).values
                data[col] = values

    result_df = pd.DataFrame(data)

    # Use integer index instead of datetime to avoid frequency resampling issues
    # This preserves the exact number of trading days (1006 for training)
    result_df["timestamp"] = pd.RangeIndex(len(result_df))

    # Convert to TimeSeriesDataFrame
    ts_df = TimeSeriesDataFrame.from_data_frame(
        result_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    return ts_df


def prepare_multi_stock_data(
    stocks: list,
    data_dir: str,
    train_years: list,
    covariate_cols: list = None,
) -> TimeSeriesDataFrame:
    """
    Prepare training data from multiple stocks.

    Using multiple stocks provides more training data for fine-tuning.
    """
    all_data = []

    for stock in stocks:
        df = load_data(stock, data_dir)
        train_df = df[df["Year"].isin(train_years)].copy()

        # Prepare data for this stock
        ts_df = prepare_autogluon_data(train_df, stock, covariate_cols)
        all_data.append(ts_df)

    # Concatenate all stocks
    combined = pd.concat(all_data, ignore_index=False)
    return TimeSeriesDataFrame(combined)


# ============================================================================
# Fine-Tuning and Prediction
# ============================================================================


def finetune_model(
    train_data: TimeSeriesDataFrame,
    horizon: int,
    stock: str = "STOCK",
    covariate_cols: list = None,
    fine_tune: bool = True,
    fine_tune_lr: float = DEFAULT_FINE_TUNE_LR,
    fine_tune_steps: int = DEFAULT_FINE_TUNE_STEPS,
    time_limit: int = 600,
    output_dir: str = "./chronos_models",
    skip_if_exists: bool = False,
) -> TimeSeriesPredictor:
    """
    Fine-tune Chronos-2 model.

    Parameters:
    -----------
    train_data : TimeSeriesDataFrame for training
    horizon : prediction horizon
    stock : stock symbol
    covariate_cols : list of covariate column names
    fine_tune : whether to fine-tune (True) or use zero-shot (False)
    fine_tune_lr : learning rate for fine-tuning
    fine_tune_steps : number of fine-tuning steps
    time_limit : time limit in seconds for fitting
    output_dir : directory to save the model
    skip_if_exists : if True, load existing model instead of re-training

    Returns:
    --------
    TimeSeriesPredictor instance
    """
    # Set up known covariates if provided
    known_covariates = covariate_cols if covariate_cols else []

    # Configure hyperparameters
    if fine_tune:
        hyperparameters = {
            "Chronos2": {
                "fine_tune": True,
                "fine_tune_lr": fine_tune_lr,
                "fine_tune_steps": fine_tune_steps,
                "fine_tune_batch_size": DEFAULT_FINE_TUNE_BATCH_SIZE,
            }
        }
        model_suffix = "finetuned"
    else:
        hyperparameters = {"Chronos2": {}}
        model_suffix = "zeroshot"

    # Create predictor
    model_path = Path(f"{output_dir}/chronos2_{model_suffix}_{stock}_h{horizon}")

    # Check if model already exists and skip_if_exists is enabled
    if skip_if_exists and model_path.exists():
        print(f"Loading existing model from {model_path}...")
        predictor = TimeSeriesPredictor.load(str(model_path))
        print(f"Model loaded successfully (skipped training)")
        return predictor

    # Remove existing model directory to avoid warnings
    if model_path.exists():
        print(f"Removing existing model at {model_path}...")
        shutil.rmtree(model_path)

    predictor = TimeSeriesPredictor(
        prediction_length=horizon,
        target="target",
        known_covariates_names=known_covariates if known_covariates else None,
        path=str(model_path),
        eval_metric="MASE",
    )

    # Fit the model
    predictor.fit(
        train_data=train_data,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        enable_ensemble=False,
    )

    return predictor


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
                pred_df = predictor.predict(context_ts, known_covariates=known_covariates_ts)
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
            predictions.append(0.0)
            predicted_probs.append(0.5)

    return np.array(predictions), np.array(predicted_probs)


# ============================================================================
# Main Evaluation Function
# ============================================================================


def run_finetuned_baseline_for_stock(
    stock: str,
    data_dir: str,
    output_dir: str = ".",
    fine_tune: bool = True,
    fine_tune_lr: float = DEFAULT_FINE_TUNE_LR,
    fine_tune_steps: int = DEFAULT_FINE_TUNE_STEPS,
    use_covariates: bool = False,
    covariate_cols: list = None,
    use_multi_stock_training: bool = True,
    time_limit: int = 600,
    strategy: str = STRATEGY_LONG_SHORT,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    context_length: int = 60,
    skip_if_exists: bool = False,
) -> pd.DataFrame:
    """
    Run fine-tuned Chronos-2 baseline for all horizons of a given stock.

    Uses rolling predictions (like zero-shot baseline) for fair comparison.

    Parameters:
    -----------
    skip_if_exists : bool, if True, load existing models instead of re-training
    """
    mode = "Fine-tuned" if fine_tune else "Zero-shot"
    print(f"\n{'='*70}")
    print(f"CHRONOS-2 {mode.upper()}: {stock}")
    if use_covariates:
        print(f"Covariates: {covariate_cols}")
    print(f"{'='*70}")

    # Load data
    df = load_data(stock, data_dir)

    # Split train/test
    train_df = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Year"] == TEST_YEAR].copy()

    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Sort DataFrames by date for rolling predictions
    train_df_sorted = train_df.sort_values("__DateDT__").reset_index(drop=True)
    test_df_sorted = test_df.sort_values("__DateDT__").reset_index(drop=True)

    # Prepare training data for fine-tuning
    if use_multi_stock_training and fine_tune:
        print("Using multi-stock training data for fine-tuning...")
        train_ts = prepare_multi_stock_data(
            STOCKS, data_dir, TRAIN_YEARS, covariate_cols if use_covariates else None
        )
    else:
        train_ts = prepare_autogluon_data(
            train_df, stock, covariate_cols if use_covariates else None
        )

    # Use all horizons with default thresholds
    print(f"Using all horizons with default thresholds")
    horizons_thresholds = [(h, DEFAULT_THRESHOLDS[h]) for h in ALL_HORIZONS]

    results = []

    for horizon, threshold in horizons_thresholds:
        horizon = int(horizon)
        print(f"\n{'-'*50}")
        print(f"Horizon: {horizon} days, Threshold: {threshold:.4f}")
        print(f"{'-'*50}")

        try:
            # Fine-tune the model (or load existing)
            predictor = finetune_model(
                train_ts,
                horizon=horizon,
                stock=stock,
                covariate_cols=covariate_cols if use_covariates else None,
                fine_tune=fine_tune,
                fine_tune_lr=fine_tune_lr,
                fine_tune_steps=fine_tune_steps,
                time_limit=time_limit,
                output_dir=output_dir,
                skip_if_exists=skip_if_exists,
            )

            # Generate rolling predictions (like zero-shot baseline)
            print("Generating rolling predictions...")
            predicted_returns, predicted_probs = generate_rolling_predictions(
                predictor,
                train_df_sorted,
                test_df_sorted,
                horizon=horizon,
                context_length=context_length,
                covariate_cols=covariate_cols if use_covariates else None,
            )

            # Convert to binary predictions
            y_pred = (predicted_returns > threshold).astype(int)

            # Calculate actual target
            test_copy = test_df_sorted.copy()
            test_copy["target"] = calculate_target(test_copy, horizon, threshold)
            test_copy = test_copy.dropna(subset=["target"])

            # Align predictions with targets
            n_valid = min(len(y_pred), len(test_copy) - horizon)
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
                strategy=strategy,
                predicted_probs=predicted_probs,
                confidence_threshold=confidence_threshold,
            )

            result = {
                "Stock": stock,
                "Horizon": horizon,
                "BestThreshold": threshold,
                "Mode": mode,
                "Strategy": strategy,
                "ConfidenceThreshold": confidence_threshold if "confidence" in strategy else None,
                "FineTuneLR": fine_tune_lr if fine_tune else None,
                "FineTuneSteps": fine_tune_steps if fine_tune else None,
                "UseCovariates": use_covariates,
                "Test_Accuracy": test_acc,
                "Test_ROC_AUC": test_auc,
                "Trades": backtest["Trades"],
                "LongTrades": backtest["LongTrades"],
                "ShortTrades": backtest["ShortTrades"],
                "Skipped": backtest["Skipped"],
                "WinRate": backtest["WinRate"],
                "Sharpe": backtest["Sharpe"],
                "TotalReturn": backtest["TotalReturn"],
            }
            results.append(result)

            print(f"\nTest Results:")
            print(f"  Predictions: {len(y_pred)}")
            print(f"  Accuracy: {test_acc:.3f}")
            print(f"  AUC: {test_auc:.3f}")
            print(f"  Trades: {backtest['Trades']}")
            print(f"  Sharpe: {backtest['Sharpe']:.2f}")
            print(f"  Win Rate: {backtest['WinRate']*100:.1f}%")
            print(f"  Total Return: {backtest['TotalReturn']*100:.1f}%")

        except Exception as e:
            print(f"Error for horizon {horizon}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    results_df = pd.DataFrame(results)
    mode_suffix = "finetuned" if fine_tune else "zeroshot"
    cov_suffix = "_cov" if use_covariates else ""
    strategy_suffix = f"_{strategy}" if strategy != STRATEGY_LONG_SHORT else ""
    output_file = f"{output_dir}/Chronos2_{mode_suffix}{cov_suffix}_{stock}_results{strategy_suffix}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df, strategy_suffix


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Chronos-2 for Stock Prediction"
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
        "--data-dir",
        type=str,
        default="./dataset/training_data",
        help="Directory containing *_data_model_training.csv files",
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./results/baselines/finetuned_single", 
        help="Directory to save output results"
    )
    parser.add_argument(
        "--no-fine-tune",
        action="store_true",
        help="Run zero-shot instead of fine-tuning",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=DEFAULT_FINE_TUNE_LR,
        help=f"Learning rate for fine-tuning (default: {DEFAULT_FINE_TUNE_LR})",
    )
    parser.add_argument(
        "--fine-tune-steps",
        type=int,
        default=DEFAULT_FINE_TUNE_STEPS,
        help=f"Number of fine-tuning steps (default: {DEFAULT_FINE_TUNE_STEPS})",
    )
    parser.add_argument(
        "--use-covariates",
        action="store_true",
        help="Use covariate features",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        nargs="+",
        default=["Weighted Sentiment Score"],
        help="Covariate column names to use",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600,
        help="Time limit in seconds for fitting (default: 600)",
    )
    parser.add_argument(
        "--multi-stock-training",
        action="store_true",
        help="Use all stocks for training (default: use only target stock)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=STRATEGY_LONG_SHORT,
        choices=VALID_STRATEGIES,
        help="Trading strategy for backtesting (default: long_short)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for confidence-based strategies (default: 0.6)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=60,
        help="Number of historical days to use as context for rolling predictions (default: 60)",
    )
    parser.add_argument(
        "--overwrite-if-exists",
        action="store_true",
        help="Overwrite existing models and re-train (default: load existing models if available)",
    )

    args = parser.parse_args()

    covariate_cols = args.covariates if args.use_covariates else None
    fine_tune = not args.no_fine_tune
    skip_if_exists = not args.overwrite_if_exists

    if skip_if_exists:
        print(f"Will load existing models if available (use --overwrite-if-exists to force re-training)")
        print(f"Models will be loaded from: {args.output_dir}")
    else:
        print(f"Will overwrite and re-train existing models")
        print(f"Models will be saved to: {args.output_dir}")

    if args.all_stocks:
        all_results = []
        strategy_suffix = ""
        for stock in STOCKS:
            results, strategy_suffix = run_finetuned_baseline_for_stock(
                stock=stock,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                fine_tune=fine_tune,
                fine_tune_lr=args.fine_tune_lr,
                fine_tune_steps=args.fine_tune_steps,
                use_covariates=args.use_covariates,
                covariate_cols=covariate_cols,
                use_multi_stock_training=args.multi_stock_training,
                time_limit=args.time_limit,
                strategy=args.strategy,
                confidence_threshold=args.confidence_threshold,
                context_length=args.context_length,
                skip_if_exists=skip_if_exists,
            )
            all_results.append(results)

        # Combine and save all results
        combined = pd.concat(all_results, ignore_index=True)
        mode_suffix = "finetuned" if fine_tune else "zeroshot"
        cov_suffix = "_cov" if args.use_covariates else ""
        combined.to_csv(
            f"{args.output_dir}/Chronos2_{mode_suffix}{cov_suffix}_all_stocks_results{strategy_suffix}.csv",
            index=False,
        )

        # Print summary
        print(f"\n\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"\nMode: {'Fine-tuned' if fine_tune else 'Zero-shot'}")
        print(f"Strategy: {args.strategy}")
        if args.use_covariates:
            print(f"Covariates: {covariate_cols}")

        print(f"\nOverall Average:")
        print(f"  Accuracy:     {combined['Test_Accuracy'].mean():.3f}")
        print(f"  AUC:          {combined['Test_ROC_AUC'].mean():.3f}")
        print(f"  Sharpe:       {combined['Sharpe'].mean():.2f}")
        print(f"  Win%:         {combined['WinRate'].mean()*100:.1f}%")

    else:
        run_finetuned_baseline_for_stock(
            stock=args.stock,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            fine_tune=fine_tune,
            fine_tune_lr=args.fine_tune_lr,
            fine_tune_steps=args.fine_tune_steps,
            use_covariates=args.use_covariates,
            covariate_cols=covariate_cols,
            use_multi_stock_training=args.multi_stock_training,
            time_limit=args.time_limit,
            strategy=args.strategy,
            confidence_threshold=args.confidence_threshold,
            context_length=args.context_length,
            skip_if_exists=skip_if_exists,
        )


if __name__ == "__main__":
    main()