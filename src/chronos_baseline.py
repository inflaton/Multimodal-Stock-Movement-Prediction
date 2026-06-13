"""
Chronos-2 Baseline for Stock Prediction

This script creates a baseline using Amazon's Chronos-2 foundation model
for time series forecasting. It uses the same evaluation methodology as
the LSTM and other models in the paper.

Methodology:
1. Use Chronos-2 to predict the Close price H days ahead
2. Convert the price prediction to a binary classification:
   - Predict 1 (up) if predicted return > threshold
   - Predict 0 (down) otherwise
3. Trading simulation with configurable strategies:
   - long_short: Go long when pred=1, short when pred=0 (always in market)
   - long_only: Go long when pred=1, stay in cash when pred=0
   - long_short_confidence: Long/short only when p(up) > threshold or < (1-threshold)
   - long_only_confidence: Long only when p(up) > threshold
4. Evaluate using the same metrics: Accuracy, AUC, Sharpe, Win Rate

Installation:
    pip install chronos-forecasting torch

Usage:
    # Default long/short strategy
    python chronos_baseline.py --all-stocks
    python chronos_baseline.py --stock AAPL

    # Long-only strategy
    python chronos_baseline.py --all-stocks --strategy long_only

    # Confidence-filtered strategies
    python chronos_baseline.py --all-stocks --strategy long_short_confidence --confidence-threshold 0.6
    python chronos_baseline.py --all-stocks --strategy long_only_confidence --confidence-threshold 0.7
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import argparse
from pathlib import Path

import torch
from chronos import Chronos2Pipeline, ChronosBoltPipeline, ChronosPipeline

from sklearn.metrics import accuracy_score, roc_auc_score

from _load_features import (
    load_features, add_alpha_news_arg, add_split_args,
    train_val_test_split, SENTIMENT_OUT,
    DEFAULT_TRAIN_YEARS, DEFAULT_VAL_YEAR, DEFAULT_TEST_YEAR, DEFAULT_EMBARGO_DAYS,
)

# ============================================================================
# Configuration
# ============================================================================

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
# Default 2020-22 / 2023 / 2024 split with a 5-day embargo, matching
# configs/default.yaml. Override per run with --train-years / --val-year /
# --test-year / --embargo-days (see add_split_args).
TRAIN_YEARS = DEFAULT_TRAIN_YEARS
VAL_YEAR = DEFAULT_VAL_YEAR
TEST_YEAR = DEFAULT_TEST_YEAR
EMBARGO_DAYS = DEFAULT_EMBARGO_DAYS
FEE_BPS_ROUND_TRIP = 10  # 10 basis points = 0.10%
RANDOM_SEED = 42

# Default Chronos model - can be overridden via command line
DEFAULT_MODEL = "amazon/chronos-2"

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

# Available models:
# Chronos-2 (latest, recommended):
# - amazon/chronos-2 (120M params) - state-of-the-art performance
#
# Chronos-Bolt (fast inference):
# - amazon/chronos-bolt-tiny (9M params)
# - amazon/chronos-bolt-mini (21M params)
# - amazon/chronos-bolt-small (48M params)
# - amazon/chronos-bolt-base (205M params)
#
# Chronos-T5 (original):
# - amazon/chronos-t5-tiny (8M params)
# - amazon/chronos-t5-mini (20M params)
# - amazon/chronos-t5-small (46M params)
# - amazon/chronos-t5-base (200M params)
# - amazon/chronos-t5-large (710M params)

# ============================================================================
# Utility Functions
# ============================================================================


def load_data(stock: str, data_dir: str, alpha_news: float = 0.7) -> pd.DataFrame:
    """Load and preprocess stock data from the training_features layout.

    Adds a `Sentiment_S_t` covariate (per-source-mean fusion of news + social at
    `alpha_news`), so the foundation comparison uses the same sentiment as the
    rest of the pipeline. Pass the learned-α value from results/learned_alpha.json.
    """
    return load_features(stock, data_dir, alpha_news=alpha_news)


def load_results(stock: str, results_dir: str) -> pd.DataFrame:
    """Load hyperparameter tuning results to get thresholds and horizons."""
    return pd.read_csv(
        f"{results_dir}/Filtered_{stock}_hyperparameter_tuned_results.csv"
    )


def calculate_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.Series:
    """
    Calculate binary target variable.

    Target = 1 if price increases by more than threshold over horizon days.
    """
    future_return = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    return (future_return > threshold).astype(int)


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
    dict with Trades, LongTrades, ShortTrades, WinRate, Sharpe, TotalReturn
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
# Chronos Prediction Functions
# ============================================================================


def get_model_type(model_name: str) -> str:
    """
    Determine the model type from the model name.

    Returns:
    --------
    str: 'chronos2', 'bolt', or 't5'
    """
    model_lower = model_name.lower()
    if "chronos-2" in model_lower:
        return "chronos2"
    elif "bolt" in model_lower:
        return "bolt"
    else:
        return "t5"


def load_chronos_model(model_name: str, device: str = None):
    """
    Load Chronos model pipeline.

    Parameters:
    -----------
    model_name : str, Hugging Face model name
    device : str, device to use ('cuda', 'mps', 'cpu', or None for auto-detect)

    Returns:
    --------
    Chronos2Pipeline, ChronosBoltPipeline, or ChronosPipeline instance
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model_type = get_model_type(model_name)
    print(f"Loading Chronos model: {model_name} (type: {model_type}) on {device}")

    if model_type == "chronos2":
        pipeline = Chronos2Pipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
    elif model_type == "bolt":
        pipeline = ChronosBoltPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )
    else:  # t5
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.float32,
        )

    return pipeline


def predict_with_chronos_multivariate(
    pipeline,
    context_df: pd.DataFrame,
    prediction_length: int,
    target_col: str = "Close",
    covariate_cols: list = None,
    model_type: str = "chronos2",
) -> tuple:
    """
    Generate predictions using Chronos-2 with multivariate/covariate support.

    This uses the tensor-based predict() method with multiple variates.
    Chronos-2 expects input shape: (n_series, n_variates, history_length)

    For multivariate forecasting, we stack the target and covariates as
    separate variates. Chronos-2 will forecast all variates, but we only
    use the first one (target) for our predictions.

    Parameters:
    -----------
    pipeline : Chronos2Pipeline instance
    context_df : DataFrame with target and optional covariate columns
    prediction_length : number of steps to predict
    target_col : name of the target column
    covariate_cols : list of covariate column names (optional)
    model_type : str, must be 'chronos2' for multivariate support

    Returns:
    --------
    tuple of (median_prediction, low_quantile, high_quantile)
    """
    if model_type != "chronos2":
        raise ValueError("Multivariate/covariate support only available for Chronos-2")

    # Build list of columns to use as variates
    cols_to_use = [target_col]
    if covariate_cols:
        # Filter to only existing columns and handle NaN
        valid_covariates = [c for c in covariate_cols if c in context_df.columns]
        cols_to_use.extend(valid_covariates)

    # Extract data and handle NaN values
    data = context_df[cols_to_use].copy()
    # Forward-fill and backward-fill NaN values
    data = data.ffill().bfill()
    # If still NaN (e.g., all NaN column), fill with 0
    data = data.fillna(0)

    # Convert to tensor: shape (n_variates, history_length)
    data_np = data.values.T.astype(np.float32)  # Shape: (n_variates, history_length)

    # Add batch dimension: shape (1, n_variates, history_length)
    input_tensor = torch.tensor(data_np, dtype=torch.float32).unsqueeze(0)

    # Predict
    forecast_list = pipeline.predict(input_tensor, prediction_length=prediction_length)

    # Returns list of tensors, each with shape (n_variates, 21_quantiles, prediction_length)
    forecast_np = forecast_list[0].numpy()

    # Extract quantiles for the first variate (target/Close price)
    # Quantiles are evenly spaced from 0.025 to 0.975
    # Index 2 ≈ 0.1, index 10 = 0.5 (median), index 18 ≈ 0.9
    low = forecast_np[0, 2, :]  # ~0.1 quantile for target
    median = forecast_np[0, 10, :]  # 0.5 quantile (median) for target
    high = forecast_np[0, 18, :]  # ~0.9 quantile for target

    return median, low, high


def predict_with_chronos(
    pipeline,
    context: np.ndarray,
    prediction_length: int,
    model_type: str = "chronos2",
    num_samples: int = 20,
) -> tuple:
    """
    Generate predictions using Chronos.

    Parameters:
    -----------
    pipeline : Chronos2Pipeline, ChronosBoltPipeline, or ChronosPipeline instance
    context : array of historical prices
    prediction_length : number of steps to predict
    model_type : str, 'chronos2', 'bolt', or 't5'
    num_samples : number of samples for uncertainty estimation (only for T5 models)

    Returns:
    --------
    tuple of (median_prediction, low_quantile, high_quantile)
    """
    # Convert to torch tensor
    context_tensor = torch.tensor(context, dtype=torch.float32)

    if model_type == "chronos2":
        # Chronos2Pipeline expects shape (n_series, n_variates, history_length)
        # For univariate: (1, 1, history_length)
        input_tensor = context_tensor.reshape(1, 1, -1)
        forecast_list = pipeline.predict(input_tensor, prediction_length=prediction_length)
        # Returns list of tensors, each with shape (n_variates, 21_quantiles, prediction_length)
        forecast_np = forecast_list[0].numpy()
        # For univariate, shape is (1, 21, prediction_length)
        # Quantiles are evenly spaced from 0.025 to 0.975
        # Index 2 â‰ˆ 0.1, index 10 = 0.5 (median), index 18 â‰ˆ 0.9
        low = forecast_np[0, 2, :]  # ~0.1 quantile
        median = forecast_np[0, 10, :]  # 0.5 quantile (median)
        high = forecast_np[0, 18, :]  # ~0.9 quantile
    elif model_type == "bolt":
        # ChronosBoltPipeline returns shape (batch, 9_quantiles, prediction_length)
        # Quantiles are: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        forecast = pipeline.predict(
            context_tensor.unsqueeze(0),  # Add batch dimension
            prediction_length=prediction_length,
        )
        forecast_np = forecast.numpy()
        # Extract quantiles: index 0 = 0.1, index 4 = 0.5, index 8 = 0.9
        low = forecast_np[0, 0, :]  # 0.1 quantile
        median = forecast_np[0, 4, :]  # 0.5 quantile (median)
        high = forecast_np[0, 8, :]  # 0.9 quantile
    else:  # t5
        # ChronosPipeline (T5) returns shape (batch, num_samples, prediction_length)
        forecast = pipeline.predict(
            context_tensor.unsqueeze(0),  # Add batch dimension
            prediction_length=prediction_length,
            num_samples=num_samples,
        )
        forecast_np = forecast.numpy()
        # Get quantiles from samples
        low = np.quantile(forecast_np, 0.1, axis=1)[0]
        median = np.quantile(forecast_np, 0.5, axis=1)[0]
        high = np.quantile(forecast_np, 0.9, axis=1)[0]

    return median, low, high


def generate_predictions_for_test(
    pipeline,
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    horizon: int,
    context_length: int = 60,
    model_type: str = "chronos2",
    num_samples: int = 20,
) -> tuple:
    """
    Generate predictions for all test data points (univariate mode).

    Uses a rolling window approach: for each test point,
    use the last `context_length` prices as context.

    Parameters:
    -----------
    pipeline : Chronos2Pipeline, ChronosBoltPipeline, or ChronosPipeline instance
    train_prices : array of training prices
    test_prices : array of test prices
    horizon : prediction horizon
    context_length : number of historical days to use as context
    model_type : str, 'chronos2', 'bolt', or 't5'
    num_samples : number of samples for uncertainty estimation (only for T5 models)

    Returns:
    --------
    tuple of (predictions, predicted_probs)
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
        median, low, high = predict_with_chronos(
            pipeline,
            context,
            prediction_length=horizon,
            model_type=model_type,
            num_samples=num_samples,
        )

        # Current price
        current_price = all_prices[current_idx]

        # Predicted price at horizon
        predicted_price = median[-1]  # Last prediction in the horizon

        # Predicted return
        predicted_return = (predicted_price - current_price) / current_price

        # Store results
        predictions.append(predicted_return)

        # For probability, use the proportion of samples predicting positive return
        # This is approximated using the quantiles
        # If low quantile > current price, high confidence of increase
        # If high quantile < current price, high confidence of decrease
        if low[-1] > current_price:
            prob = 0.9
        elif high[-1] < current_price:
            prob = 0.1
        else:
            # Linear interpolation based on median
            prob = 0.5 + 0.4 * (predicted_return / (0.05 + abs(predicted_return)))
            prob = np.clip(prob, 0.1, 0.9)

        predicted_probs.append(prob)

    return np.array(predictions), np.array(predicted_probs)


def generate_predictions_with_covariates(
    pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon: int,
    context_length: int = 60,
    covariate_cols: list = None,
) -> tuple:
    """
    Generate predictions for all test data points using covariates (Chronos-2 only).

    Uses a rolling window approach with additional covariate features
    like sentiment scores and technical indicators.

    Parameters:
    -----------
    pipeline : Chronos2Pipeline instance
    train_df : DataFrame with training data (must have Close, __DateDT__, and covariate columns)
    test_df : DataFrame with test data
    horizon : prediction horizon
    context_length : number of historical days to use as context
    covariate_cols : list of column names to use as covariates

    Returns:
    --------
    tuple of (predictions, predicted_probs)
    """
    # Combine train and test for rolling context
    all_df = pd.concat([train_df, test_df], ignore_index=True).sort_values("__DateDT__").reset_index(drop=True)
    train_len = len(train_df)

    predictions = []
    predicted_probs = []

    for i in range(len(test_df) - horizon):
        # Current position in the combined array
        current_idx = train_len + i

        # Get context: last context_length rows before current point
        start_idx = max(0, current_idx - context_length)
        context_df = all_df.iloc[start_idx:current_idx].copy()

        # Predict using multivariate method
        median, low, high = predict_with_chronos_multivariate(
            pipeline,
            context_df,
            prediction_length=horizon,
            target_col="Close",
            covariate_cols=covariate_cols,
            model_type="chronos2",
        )

        # Current price
        current_price = all_df.iloc[current_idx]["Close"]

        # Predicted price at horizon
        predicted_price = median[-1]  # Last prediction in the horizon

        # Predicted return
        predicted_return = (predicted_price - current_price) / current_price

        # Store results
        predictions.append(predicted_return)

        # Probability estimation based on quantiles
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


def run_chronos_baseline_for_stock(
    stock: str,
    data_dir: str,
    results_dir: str,
    model_name: str,
    context_length: int = 60,
    num_samples: int = 20,
    output_dir: str = ".",
    pipeline=None,
    use_covariates: bool = False,
    covariate_cols: list = None,
    use_all_horizons: bool = False,
    strategy: str = STRATEGY_LONG_SHORT,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    alpha_news: float = 0.7,
) -> pd.DataFrame:
    """
    Run Chronos baseline for all horizons of a given stock.

    Parameters:
    -----------
    stock : str, stock symbol
    data_dir : str, directory containing data files
    results_dir : str, directory containing results files
    model_name : str, Chronos model name
    context_length : int, number of historical days to use as context
    num_samples : int, number of samples for uncertainty estimation
    output_dir : str, directory to save output
    pipeline : ChronosPipeline or ChronosBoltPipeline, pre-loaded pipeline (optional)
    use_covariates : bool, whether to use covariate features (Chronos-2 only)
    covariate_cols : list, column names to use as covariates
    strategy : str, trading strategy (long_short, long_only, long_short_confidence, long_only_confidence)
    confidence_threshold : float, threshold for confidence-based strategies (default: 0.6)

    Returns:
    --------
    DataFrame with results
    """
    # Determine model type
    model_type = get_model_type(model_name)

    # Validate covariate usage
    if use_covariates and model_type != "chronos2":
        print(f"Warning: Covariates only supported for Chronos-2. Falling back to univariate mode.")
        use_covariates = False

    print(f"\n{'='*70}")
    print(f"CHRONOS BASELINE: {stock}")
    print(f"Model: {model_name}")
    if use_covariates:
        print(f"Mode: Multivariate with covariates")
        print(f"Covariates: {covariate_cols}")
    else:
        print(f"Mode: Univariate")
    print(f"{'='*70}")

    # Load Chronos model if not provided
    if pipeline is None:
        pipeline = load_chronos_model(model_name)

    # Load data (Sentiment_S_t is fused at alpha_news; pass the learned-α here)
    df = load_data(stock, data_dir, alpha_news=alpha_news)

    # Split train / val / test with embargo (matches configs/default.yaml)
    sp = train_val_test_split(df, TRAIN_YEARS, VAL_YEAR, TEST_YEAR, EMBARGO_DAYS)
    train_df, val_df, test_df = sp.train, sp.val, sp.test
    print(f"Train samples: {len(train_df)}, "
          f"Val samples: {len(val_df)}, "
          f"Test samples (post {EMBARGO_DAYS}-day embargo): {len(test_df)}")

    # Prices for the price-only rolling predictor. The val/test rolling loops
    # use the train tail as context, then walk forward — so val's context is
    # train_prices, test's context is train_prices + val_prices.
    train_prices = train_df["Close"].values
    val_prices = val_df["Close"].values
    test_prices = test_df["Close"].values

    # Load existing results to get horizons and thresholds
    if use_all_horizons:
        print(f"Using all horizons with default thresholds")
        horizons_thresholds = [(h, DEFAULT_THRESHOLDS[h]) for h in ALL_HORIZONS]
    else:
        try:
            existing_results = load_results(stock, results_dir)
            horizons_thresholds = existing_results[
                ["Horizon", "BestThreshold"]
            ].values.tolist()
        except FileNotFoundError:
            print(
                f"Warning: No existing results found for {stock}. Using all horizons with default thresholds."
            )
            horizons_thresholds = [(h, DEFAULT_THRESHOLDS[h]) for h in ALL_HORIZONS]

    def _score_slice(slice_df: pd.DataFrame,
                     context_train_df: pd.DataFrame,
                     context_train_prices: np.ndarray,
                     slice_prices: np.ndarray,
                     horizon: int, threshold: float, prefix: str) -> dict:
        """Run rolling Chronos prediction over `slice_df`, return prefixed metrics.

        `context_train_df` / `context_train_prices` are the data used as the
        rolling-window context for the first step into the slice. For val,
        this is train; for test, this is train+val (no leakage — val labels
        are not used for selection inside this function).
        """
        if use_covariates:
            predicted_returns, predicted_probs = generate_predictions_with_covariates(
                pipeline,
                context_train_df,
                slice_df,
                horizon=horizon,
                context_length=context_length,
                covariate_cols=covariate_cols,
            )
        else:
            predicted_returns, predicted_probs = generate_predictions_for_test(
                pipeline,
                context_train_prices,
                slice_prices,
                horizon=horizon,
                context_length=context_length,
                model_type=model_type,
                num_samples=num_samples,
            )

        y_pred = (predicted_returns > threshold).astype(int)
        slice_copy = slice_df.copy()
        slice_copy["target"] = calculate_target(slice_copy, horizon, threshold)
        slice_copy = slice_copy.dropna(subset=["target"])
        n_valid = min(len(y_pred), len(slice_copy) - horizon)
        if n_valid <= 0:
            return {f"{prefix}Accuracy": float("nan"),
                    f"{prefix}ROC_AUC": float("nan"),
                    f"{prefix}Trades": 0,
                    f"{prefix}LongTrades": 0,
                    f"{prefix}ShortTrades": 0,
                    f"{prefix}Skipped": 0,
                    f"{prefix}WinRate": float("nan"),
                    f"{prefix}Sharpe": float("nan"),
                    f"{prefix}TotalReturn": float("nan")}
        y_pred = y_pred[:n_valid]
        predicted_probs = predicted_probs[:n_valid]
        y_true = slice_copy["target"].values[:n_valid]
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, predicted_probs)
        except ValueError:
            auc = 0.5
        bt = non_overlap_backtest(
            slice_copy, y_pred, horizon,
            strategy=strategy,
            predicted_probs=predicted_probs,
            confidence_threshold=confidence_threshold,
        )
        return {f"{prefix}Accuracy": acc,
                f"{prefix}ROC_AUC": auc,
                f"{prefix}Trades": bt["Trades"],
                f"{prefix}LongTrades": bt["LongTrades"],
                f"{prefix}ShortTrades": bt["ShortTrades"],
                f"{prefix}Skipped": bt["Skipped"],
                f"{prefix}WinRate": bt["WinRate"],
                f"{prefix}Sharpe": bt["Sharpe"],
                f"{prefix}TotalReturn": bt["TotalReturn"]}

    # For test scoring, walk forward through train+val so the rolling window
    # has up-to-2024-Jan context for the first test step (no peek into test).
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_prices = np.concatenate([train_prices, val_prices])

    results = []
    for horizon, threshold in horizons_thresholds:
        horizon = int(horizon)
        print(f"\n{'-'*50}\nHorizon: {horizon} days, Threshold: {threshold:.4f}\n{'-'*50}")
        print("Generating Chronos predictions on val...")
        val_metrics = _score_slice(val_df, train_df, train_prices, val_prices,
                                   horizon, threshold, prefix="Val_")
        print("Generating Chronos predictions on test...")
        test_metrics = _score_slice(test_df, train_val_df, train_val_prices, test_prices,
                                    horizon, threshold, prefix="Test_")
        if val_metrics.get("Val_Trades", 0) == 0 and test_metrics.get("Test_Trades", 0) == 0:
            print(f"Skipping horizon {horizon}: no valid predictions on either slice")
            continue

        result = {
            "Stock": stock,
            "Horizon": horizon,
            "BestThreshold": threshold,
            "Model": model_name,
            "ContextLength": context_length,
            "UseCovariates": use_covariates,
            "Strategy": strategy,
            "ConfidenceThreshold": confidence_threshold if "confidence" in strategy else None,
            **val_metrics,
            **test_metrics,
        }
        results.append(result)
        # Compact summary line; keep verbose per-slice printouts to a minimum
        print(f"\nResults ({strategy}): "
              f"Val AUC={val_metrics.get('Val_ROC_AUC', float('nan')):.3f}, "
              f"Test AUC={test_metrics.get('Test_ROC_AUC', float('nan')):.3f}, "
              f"Test Sharpe={test_metrics.get('Test_Sharpe', float('nan')):.2f}, "
              f"Test Trades={test_metrics.get('Test_Trades', 0)}, "
              f"Test Win Rate={test_metrics.get('Test_WinRate', 0)*100:.1f}%, "
              f"Test Return={test_metrics.get('Test_TotalReturn', 0)*100:.1f}%")

    # Save results
    results_df = pd.DataFrame(results)
    model_short = model_name.split("/")[-1]
    # Include strategy in filename (only if not default long_short)
    strategy_suffix = f"_{strategy}" if strategy != STRATEGY_LONG_SHORT else ""
    if use_covariates:
        output_file = f"{output_dir}/multivariate_{model_short}_{stock}{strategy_suffix}_results.csv"
    else:
        output_file = f"{output_dir}/zero-shot_{model_short}_{stock}{strategy_suffix}_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chronos-2 Baseline for Stock Prediction"
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
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Chronos model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing *_data_model_training.csv files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=".",
        help="Directory containing Filtered_*_hyperparameter_tuned_results.csv files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save output results"
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=60,
        help="Number of historical days to use as context (default: 60)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for torch/numpy (default: {RANDOM_SEED}).",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=None,
        help="Horizons to evaluate (default: script's ALL_HORIZONS = 2..10). "
             "Pass e.g. `--horizons 2 5 10` to match configs/default.yaml.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples for uncertainty estimation (default: 20)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--use-covariates",
        action="store_true",
        help="Use multivariate mode with covariates (Chronos-2 only)",
    )
    parser.add_argument(
        "--covariates",
        type=str,
        nargs="+",
        default=[SENTIMENT_OUT],
        help=f"Covariate column names to use (default: {SENTIMENT_OUT!r}, the "
             f"per-source-mean sentiment computed from --alpha-news).",
    )
    add_alpha_news_arg(parser)
    add_split_args(parser)
    parser.add_argument(
        "--use-all-horizons",
        action="store_true",
        help="Use all horizons (2-10) with default thresholds instead of stock-specific filtered results",
    )
    parser.add_argument(
        "--auto-output-dir",
        action="store_true",
        help="Automatically set output directory to baseline_single/ or baseline_covariates/ based on mode",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=STRATEGY_LONG_SHORT,
        choices=VALID_STRATEGIES,
        help=f"Trading strategy (default: {STRATEGY_LONG_SHORT}). Options: {', '.join(VALID_STRATEGIES)}",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for confidence-based strategies (default: {DEFAULT_CONFIDENCE_THRESHOLD})",
    )

    args = parser.parse_args()

    # Auto-set output directory based on covariate mode
    if args.auto_output_dir:
        if args.use_covariates:
            args.output_dir = "baseline_covariates"
        else:
            args.output_dir = "baseline_single"
        # Create directory if it doesn't exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Auto output directory: {args.output_dir}")

    # Get covariate columns from args if using covariates mode
    covariate_cols = args.covariates if args.use_covariates else None

    # Apply split overrides (CLI > module defaults). The run_*_for_stock function
    # below reads these module-level constants, so reassigning here is sufficient.
    global TRAIN_YEARS, VAL_YEAR, TEST_YEAR, EMBARGO_DAYS
    TRAIN_YEARS = list(args.train_years)
    VAL_YEAR = int(args.val_year)
    TEST_YEAR = int(args.test_year)
    EMBARGO_DAYS = int(args.embargo_days)
    print(f"Split: train={TRAIN_YEARS}, val={VAL_YEAR}, test={TEST_YEAR}, "
          f"embargo={EMBARGO_DAYS} days")

    # Seed RNGs (torch + numpy) so multi-seed sweeps are reproducible.
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch as _torch
        _torch.manual_seed(args.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass
    print(f"Seed: {args.seed}")

    if args.horizons:
        global ALL_HORIZONS
        ALL_HORIZONS = list(args.horizons)
        # Force the ALL_HORIZONS path — otherwise the script reads horizons from
        # the legacy `Filtered_<STOCK>_hyperparameter_tuned_results.csv` under
        # --results-dir (../paper/ by default), which still lists 2-10.
        args.use_all_horizons = True
        print(f"Horizons: {ALL_HORIZONS}")

    # Load model once (shared across all stocks)
    pipeline = load_chronos_model(args.model, device=args.device)

    print(f"Trading strategy: {args.strategy}")
    if "confidence" in args.strategy:
        print(f"Confidence threshold: {args.confidence_threshold}")

    if args.all_stocks:
        all_results = []
        for stock in STOCKS:
            results = run_chronos_baseline_for_stock(
                stock=stock,
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                model_name=args.model,
                context_length=args.context_length,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                pipeline=pipeline,
                use_covariates=args.use_covariates,
                covariate_cols=covariate_cols,
                use_all_horizons=args.use_all_horizons,
                strategy=args.strategy,
                confidence_threshold=args.confidence_threshold,
                alpha_news=args.alpha_news,
            )
            all_results.append(results)

        # Combine and save all results
        combined = pd.concat(all_results, ignore_index=True)
        model_short = args.model.split("/")[-1]
        # Include strategy in filename (only if not default long_short)
        strategy_suffix = f"_{args.strategy}" if args.strategy != STRATEGY_LONG_SHORT else ""
        if args.use_covariates:
            combined_file = f"{args.output_dir}/multivariate_{model_short}_all_stocks{strategy_suffix}_results.csv"
        else:
            combined_file = f"{args.output_dir}/zero-shot_{model_short}_all_stocks{strategy_suffix}_results.csv"
        combined.to_csv(combined_file, index=False)

        # Print summary using "best by criterion per stock" methodology
        # This matches how Tables III-IV in the paper are computed
        print(f"\n\n{'='*70}")
        print("SUMMARY (Best by Criterion per Stock, then Averaged)")
        print(f"{'='*70}")
        print(f"\nModel: {args.model}")
        if args.use_covariates:
            print(f"Mode: Multivariate with covariates: {covariate_cols}")
        
        # Compute "Best by AUC" summary
        auc_selected = []
        for stock in STOCKS:
            stock_data = combined[combined['Stock'] == stock]
            if len(stock_data) > 0:
                best_idx = stock_data['Test_ROC_AUC'].idxmax()
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
            stock_data = combined[combined['Stock'] == stock]
            if len(stock_data) > 0:
                best_idx = stock_data['Sharpe'].idxmax()
                sharpe_selected.append(stock_data.loc[best_idx])
        sharpe_df = pd.DataFrame(sharpe_selected)
        
        print(f"\nBest by Sharpe (per stock, then averaged):")
        print(f"  Accuracy:     {sharpe_df['Test_Accuracy'].mean():.3f}")
        print(f"  AUC:          {sharpe_df['Test_ROC_AUC'].mean():.3f}")
        print(f"  Sharpe:       {sharpe_df['Sharpe'].mean():.2f}")
        print(f"  Win%:         {sharpe_df['WinRate'].mean()*100:.1f}%")
        
        # Print LaTeX table rows for easy copy-paste
        print(f"\n{'='*70}")
        print("LaTeX Table Rows (for Table VI):")
        print(f"{'='*70}")
        print(f"Chronos-2 (Best by AUC) & {auc_df['Test_Accuracy'].mean():.3f} & {auc_df['Test_ROC_AUC'].mean():.3f} & {auc_df['Sharpe'].mean():.2f} & {auc_df['WinRate'].mean()*100:.1f} \\\\")
        print(f"Chronos-2 (Best by Sharpe) & {sharpe_df['Test_Accuracy'].mean():.3f} & {sharpe_df['Test_ROC_AUC'].mean():.3f} & {sharpe_df['Sharpe'].mean():.2f} & {sharpe_df['WinRate'].mean()*100:.1f} \\\\")
        
        # Also print overall average for reference
        print(f"\n{'='*70}")
        print("Overall Average (all configurations):")
        print(f"{'='*70}")
        print(f"  Accuracy:     {combined['Test_Accuracy'].mean():.3f}")
        print(f"  AUC:          {combined['Test_ROC_AUC'].mean():.3f}")
        print(f"  Sharpe:       {combined['Sharpe'].mean():.2f}")
        print(f"  Win%:         {combined['WinRate'].mean()*100:.1f}%")
        
        print(f"\nCombined results saved to: {combined_file}")
    else:
        run_chronos_baseline_for_stock(
            stock=args.stock,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            model_name=args.model,
            context_length=args.context_length,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            pipeline=pipeline,
            use_covariates=args.use_covariates,
            covariate_cols=covariate_cols,
            use_all_horizons=args.use_all_horizons,
            strategy=args.strategy,
            confidence_threshold=args.confidence_threshold,
            alpha_news=args.alpha_news,
        )


if __name__ == "__main__":
    main()