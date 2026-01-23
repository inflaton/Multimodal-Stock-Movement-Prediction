"""
LSTM Hyperparameter Tuning for Stock Prediction

This script adds Bayesian hyperparameter optimization for LSTM models,
which was missing from the original experiments where LSTM used fixed hyperparameters.

Key hyperparameters tuned:
- Learning rate
- Dropout rate
- LSTM units (layer 1 and 2)
- Batch size

Usage:
    python lstm_hyperparameter_tuning.py --stock AAPL --data-dir /path/to/data --n-calls 20
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import random
import argparse
from datetime import datetime

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# ============================================================================
# Configuration
# ============================================================================

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR = 2024
RANDOM_SEED = 42
SEQ_LEN = 30
EPOCHS = 50
PATIENCE = 10
FEE_BPS_ROUND_TRIP = 10

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

# ============================================================================
# LSTM Hyperparameter Search Space
# ============================================================================

lstm_space = [
    Real(1e-4, 1e-2, name="learning_rate", prior="log-uniform"),
    Real(0.1, 0.5, name="dropout_rate"),
    Integer(32, 256, name="lstm_units_1"),
    Integer(16, 128, name="lstm_units_2"),
    Integer(16, 64, name="batch_size"),
]

# ============================================================================
# Utility Functions
# ============================================================================


def set_global_seed(seed=RANDOM_SEED):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def load_data(stock: str, data_dir: str) -> pd.DataFrame:
    """Load and preprocess stock data."""
    df = pd.read_csv(f"{data_dir}/{stock}_data_model_training.csv")
    df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Year"] = df["__DateDT__"].dt.year
    return df


def find_best_threshold_for_horizon(
    df, horizon, thresholds=np.arange(0.0025, 0.05, 0.0025)
):
    """
    Find best threshold for target variable generation to achieve balanced classes.

    Args:
        df: DataFrame with stock data
        horizon: Number of days to look ahead
        thresholds: Array of candidate thresholds to test

    Returns:
        best_threshold: Threshold that creates most balanced dataset (closest to 50/50)
    """
    best_threshold, best_score = 0.01, -np.inf

    for th in thresholds:
        df_copy = df.copy()
        df_copy[f"Y_{horizon}d"] = (
            (df_copy["Close"].shift(-horizon) - df_copy["Close"]) / df_copy["Close"]
            > th
        ).astype(int)
        df_valid = df_copy.dropna(subset=[f"Y_{horizon}d"])

        if df_valid.empty:
            continue

        # Check class balance
        pos_ratio = df_valid[f"Y_{horizon}d"].mean()
        if 0.3 < pos_ratio < 0.7:  # Reasonable class balance
            score = 1 - abs(pos_ratio - 0.5)  # Prefer balanced classes
            if score > best_score:
                best_score = score
                best_threshold = th

    print(f"  Horizon {horizon}d -> Best threshold = {best_threshold:.4f} (balance score: {best_score:.3f})")
    return best_threshold


def create_sequences(X, y, seq_len=SEQ_LEN):
    """Create sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        # Handle both DataFrame and numpy array inputs
        if hasattr(X, "iloc"):
            Xs.append(X.iloc[i : i + seq_len].values)
        else:
            Xs.append(X[i : i + seq_len])
        ys.append(y.iloc[i + seq_len] if hasattr(y, "iloc") else y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_lstm_model(
    seq_len, num_features, learning_rate, dropout_rate, lstm_units_1, lstm_units_2
):
    """Build LSTM model with specified hyperparameters."""
    # Convert numpy integers to native Python int (required by Keras)
    lstm_units_1 = int(lstm_units_1)
    lstm_units_2 = int(lstm_units_2)

    model = Sequential(
        [
            LSTM(
                lstm_units_1, input_shape=(seq_len, num_features), return_sequences=True
            ),
            Dropout(dropout_rate),
            LSTM(lstm_units_2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(dropout_rate / 2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def non_overlap_backtest(
    test_df,
    predictions,
    horizon,
    fee_bps=FEE_BPS_ROUND_TRIP,
    strategy=STRATEGY_LONG_SHORT,
    predicted_probs=None,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
):
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
# Main Tuning Function
# ============================================================================


def tune_lstm_for_horizon(
    X_train, y_train, X_val, y_val, seq_len, num_features, n_calls=20, verbose=True
):
    """
    Tune LSTM hyperparameters using Bayesian optimization.

    Returns:
        best_params: dict of best hyperparameters
        best_auc: best validation AUC achieved
    """

    @use_named_args(lstm_space)
    def objective(learning_rate, dropout_rate, lstm_units_1, lstm_units_2, batch_size):
        # Reset seeds for reproducibility
        set_global_seed(RANDOM_SEED)

        try:
            model = build_lstm_model(
                seq_len=seq_len,
                num_features=num_features,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                lstm_units_1=lstm_units_1,
                lstm_units_2=lstm_units_2,
            )

            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=0,
            )

            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            )

            model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=int(batch_size),
                callbacks=[early_stopping, reduce_lr],
                verbose=0,
            )

            y_pred_proba = model.predict(X_val, verbose=0).flatten()
            auc = roc_auc_score(y_val, y_pred_proba)

            if verbose:
                print(
                    f"  LR={learning_rate:.5f}, dropout={dropout_rate:.2f}, "
                    f"units=({lstm_units_1},{lstm_units_2}), batch={batch_size} -> AUC={auc:.4f}"
                )

            return -auc  # Minimize negative AUC

        except Exception as e:
            print(f"  Error: {e}")
            return 0  # Return worst score on error

    print(f"Starting Bayesian optimization with {n_calls} iterations...")
    result = gp_minimize(
        objective, lstm_space, n_calls=n_calls, random_state=RANDOM_SEED, verbose=False
    )

    best_params = {
        "learning_rate": result.x[0],
        "dropout_rate": result.x[1],
        "lstm_units_1": result.x[2],
        "lstm_units_2": result.x[3],
        "batch_size": result.x[4],
    }
    best_auc = -result.fun

    return best_params, best_auc


def run_tuning_for_stock(
    stock,
    data_dir,
    n_calls=20,
    output_dir=".",
    strategy=STRATEGY_LONG_SHORT,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    horizons=[2, 3, 4, 5, 6, 7, 8, 9, 10],
):
    """
    Run LSTM hyperparameter tuning for all horizons of a given stock.
    """
    print(f"\n{'='*70}")
    print(f"LSTM HYPERPARAMETER TUNING: {stock}")
    print(f"{'='*70}")

    # Load data
    df = load_data(stock, data_dir)

    # Get signal columns
    signal_cols = [c for c in df.columns if "Combi" in c and "Days" not in c]

    # Encode signals
    signal_map = {"Buy": 1, "Sell": -1, "Hold": 0, "": 0}
    for col in signal_cols:
        df[col] = df[col].astype(str).str.strip().map(signal_map).fillna(0).astype(int)

    # Split train/test
    train_df = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Year"] == TEST_YEAR].copy()

    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Find best thresholds for each horizon
    print(f"Finding best thresholds for horizons: {horizons}...")
    horizons_thresholds = []
    for horizon in horizons:
        best_threshold = find_best_threshold_for_horizon(train_df, horizon)
        horizons_thresholds.append((horizon, best_threshold))
    print(f"Found thresholds for {len(horizons_thresholds)} horizons")

    # Features
    feature_cols = signal_cols + ["Close"]
    if "Filtered Sentiment Score" in df.columns:
        feature_cols.append("Filtered Sentiment Score")
    elif "Weighted Sentiment Score" in df.columns:
        feature_cols.append("Weighted Sentiment Score")

    results = []

    for horizon, threshold in horizons_thresholds:
        horizon = int(horizon)
        print(f"\n{'-'*50}")
        print(f"Horizon: {horizon} days, Threshold: {threshold:.4f}")
        print(f"{'-'*50}")

        # Create target
        train_copy = train_df.copy()
        test_copy = test_df.copy()

        train_copy["target"] = (
            (train_copy["Close"].shift(-horizon) - train_copy["Close"])
            / train_copy["Close"]
            > threshold
        ).astype(int)
        test_copy["target"] = (
            (test_copy["Close"].shift(-horizon) - test_copy["Close"])
            / test_copy["Close"]
            > threshold
        ).astype(int)

        train_copy = train_copy.dropna(subset=["target"])
        test_copy = test_copy.dropna(subset=["target"])

        # Prepare features
        X_train = train_copy[feature_cols].values
        y_train = train_copy["target"].values
        X_test = test_copy[feature_cols].values
        y_test = test_copy["target"].values

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LEN)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, SEQ_LEN)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            print(f"Skipping horizon {horizon}: insufficient data for sequences")
            continue

        # Split training into train/val for tuning
        val_size = int(len(X_train_seq) * 0.2)
        X_train_tune = X_train_seq[:-val_size]
        y_train_tune = y_train_seq[:-val_size]
        X_val_tune = X_train_seq[-val_size:]
        y_val_tune = y_train_seq[-val_size:]

        num_features = X_train_seq.shape[2]

        # Tune hyperparameters
        best_params, val_auc = tune_lstm_for_horizon(
            X_train_tune,
            y_train_tune,
            X_val_tune,
            y_val_tune,
            seq_len=SEQ_LEN,
            num_features=num_features,
            n_calls=n_calls,
        )

        print(f"\nBest hyperparameters: {best_params}")
        print(f"Validation AUC: {val_auc:.4f}")

        # Retrain on full training data with best params
        set_global_seed(RANDOM_SEED)
        batch_size = int(best_params.pop("batch_size"))
        final_model = build_lstm_model(
            seq_len=SEQ_LEN, num_features=num_features, **best_params
        )
        best_params["batch_size"] = batch_size  # Add back for results logging

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0,
        )

        # Use last 20% of training as validation for early stopping
        final_model.fit(
            X_train_tune,
            y_train_tune,
            validation_data=(X_val_tune, y_val_tune),
            epochs=EPOCHS,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
        )

        # Evaluate on test set
        y_pred_proba = final_model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)

        test_acc = accuracy_score(y_test_seq, y_pred)
        test_auc = roc_auc_score(y_test_seq, y_pred_proba)

        # Evaluate on training set
        y_train_pred_proba = final_model.predict(X_train_seq, verbose=0).flatten()
        y_train_pred = (y_train_pred_proba > 0.5).astype(int)
        train_acc = accuracy_score(y_train_seq, y_train_pred)
        train_auc = roc_auc_score(y_train_seq, y_train_pred_proba)

        # Backtest
        backtest = non_overlap_backtest(
            test_copy,
            y_pred,
            horizon,
            strategy=strategy,
            predicted_probs=y_pred_proba,
            confidence_threshold=confidence_threshold,
        )

        result = {
            "Stock": stock,
            "Horizon": horizon,
            "BestThreshold": threshold,
            "Strategy": strategy,
            "ConfidenceThreshold": confidence_threshold if "confidence" in strategy else None,
            "learning_rate": best_params["learning_rate"],
            "dropout_rate": best_params["dropout_rate"],
            "lstm_units_1": best_params["lstm_units_1"],
            "lstm_units_2": best_params["lstm_units_2"],
            "batch_size": best_params["batch_size"],
            "Train_Accuracy": train_acc,
            "Train_ROC_AUC": train_auc,
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
        print(f"  Accuracy: {test_acc:.3f}")
        print(f"  AUC: {test_auc:.3f}")
        print(f"  Sharpe: {backtest['Sharpe']:.2f}")
        print(f"  Win Rate: {backtest['WinRate']*100:.1f}%")

    # Save results
    results_df = pd.DataFrame(results)
    strategy_suffix = f"_{strategy}" if strategy != STRATEGY_LONG_SHORT else ""
    output_file = f"{output_dir}/LSTM_tuned_{stock}_results{strategy_suffix}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df, strategy_suffix


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="LSTM Hyperparameter Tuning for Stock Prediction"
    )
    parser.add_argument(
        "--stock", type=str, default="AAPL", choices=STOCKS, help="Stock symbol to tune"
    )
    parser.add_argument(
        "--all-stocks", action="store_true", help="Run tuning for all stocks"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing *_data_model_training.csv files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Directory to save output results"
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=30,
        help="Number of Bayesian optimization iterations (default: 30)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=STRATEGY_LONG_SHORT,
        choices=VALID_STRATEGIES,
        help="Trading strategy for backtesting (default: long_short)",
    )
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Run tuning for all 4 strategies",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for confidence-based strategies (default: 0.6)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Prediction horizons in days (default: 2 3 4 5 6 7 8 9 10)",
    )

    args = parser.parse_args()

    # Determine which strategies to run
    strategies_to_run = VALID_STRATEGIES if args.all_strategies else [args.strategy]

    set_global_seed(RANDOM_SEED)

    for strategy in strategies_to_run:
        print(f"\n{'#'*70}")
        print(f"# STRATEGY: {strategy}")
        print(f"{'#'*70}")

        if args.all_stocks:
            all_results = []
            strategy_suffix = ""
            for stock in STOCKS:
                results, strategy_suffix = run_tuning_for_stock(
                    stock=stock,
                    data_dir=args.data_dir,
                    n_calls=args.n_calls,
                    output_dir=args.output_dir,
                    strategy=strategy,
                    confidence_threshold=args.confidence_threshold,
                    horizons=args.horizons,
                )
                all_results.append(results)

            # Combine and save all results
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(
                f"{args.output_dir}/LSTM_tuned_all_stocks_results{strategy_suffix}.csv", index=False
            )
            print(
                f"\nCombined results saved to: {args.output_dir}/LSTM_tuned_all_stocks_results{strategy_suffix}.csv"
            )
        else:
            run_tuning_for_stock(
                stock=args.stock,
                data_dir=args.data_dir,
                n_calls=args.n_calls,
                output_dir=args.output_dir,
                strategy=strategy,
                confidence_threshold=args.confidence_threshold,
                horizons=args.horizons,
            )


if __name__ == "__main__":
    main()
