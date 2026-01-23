"""
XGBoost Hyperparameter Tuning for Stock Prediction

This script performs Bayesian hyperparameter optimization for XGBoost models.

Key hyperparameters tuned:
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree
- gamma
- reg_alpha
- reg_lambda

Usage:
    python xgboost_hyperparameter_tuning.py --stock AAPL --data-dir /path/to/data --n-calls 30
    python xgboost_hyperparameter_tuning.py --all-stocks --strategy long_only_confidence
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os
import random
import argparse

# ML imports
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# ============================================================================
# Configuration
# ============================================================================

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR = 2024
RANDOM_SEED = 42
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
# XGBoost Hyperparameter Search Space
# ============================================================================

xgb_space = [
    Integer(50, 500, name="n_estimators"),
    Integer(3, 12, name="max_depth"),
    Real(0.01, 0.3, name="learning_rate", prior="log-uniform"),
    Real(0.5, 1.0, name="subsample"),
    Real(0.5, 1.0, name="colsample_bytree"),
    Real(0, 10, name="gamma"),
    Real(0, 10, name="reg_alpha"),
    Real(0, 10, name="reg_lambda"),
]

# ============================================================================
# Utility Functions
# ============================================================================


def set_global_seed(seed=RANDOM_SEED):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
            action = "long" if pred == 1 else "short"
        elif strategy == STRATEGY_LONG_ONLY:
            action = "long" if pred == 1 else None
        elif strategy == STRATEGY_LONG_SHORT_CONFIDENCE:
            if prob is not None:
                if prob > confidence_threshold:
                    action = "long"
                elif prob < (1 - confidence_threshold):
                    action = "short"
        elif strategy == STRATEGY_LONG_ONLY_CONFIDENCE:
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
# XGBoost Tuning Function
# ============================================================================


def tune_xgboost_for_horizon(
    X_train, y_train, X_val, y_val, n_calls=30, verbose=True
):
    """
    Tune XGBoost hyperparameters using Bayesian optimization.

    Returns:
        best_params: dict of best hyperparameters
        best_auc: best validation AUC achieved
    """

    @use_named_args(xgb_space)
    def objective(n_estimators, max_depth, learning_rate, subsample,
                  colsample_bytree, gamma, reg_alpha, reg_lambda):
        set_global_seed(RANDOM_SEED)

        try:
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=RANDOM_SEED,
                use_label_encoder=False,
                eval_metric="logloss",
                n_jobs=-1,
            )

            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)

            if verbose:
                print(f"  AUC: {auc:.4f} | n_est={n_estimators}, depth={max_depth}, lr={learning_rate:.4f}")

            return -auc  # Minimize negative AUC

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            return 0  # Return worst score on error

    # Run optimization
    result = gp_minimize(
        objective,
        xgb_space,
        n_calls=n_calls,
        random_state=RANDOM_SEED,
        verbose=False,
    )

    best_params = {
        "n_estimators": result.x[0],
        "max_depth": result.x[1],
        "learning_rate": result.x[2],
        "subsample": result.x[3],
        "colsample_bytree": result.x[4],
        "gamma": result.x[5],
        "reg_alpha": result.x[6],
        "reg_lambda": result.x[7],
    }
    best_auc = -result.fun

    return best_params, best_auc


def run_tuning_for_stock(
    stock,
    data_dir,
    n_calls=30,
    output_dir=".",
    strategy=STRATEGY_LONG_SHORT,
    confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    horizons=[2, 3, 4, 5, 6, 7, 8, 9, 10],
):
    """
    Run XGBoost hyperparameter tuning for all horizons of a given stock.
    """
    print(f"\n{'='*70}")
    print(f"XGBOOST HYPERPARAMETER TUNING: {stock}")
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

        # Split training into train/val for tuning
        val_size = int(len(X_train) * 0.2)
        X_train_tune = X_train[:-val_size]
        y_train_tune = y_train[:-val_size]
        X_val_tune = X_train[-val_size:]
        y_val_tune = y_train[-val_size:]

        # Tune hyperparameters
        best_params, val_auc = tune_xgboost_for_horizon(
            X_train_tune,
            y_train_tune,
            X_val_tune,
            y_val_tune,
            n_calls=n_calls,
        )

        print(f"\nBest hyperparameters: {best_params}")
        print(f"Validation AUC: {val_auc:.4f}")

        # Retrain on full training data with best params
        set_global_seed(RANDOM_SEED)
        final_model = XGBClassifier(
            **best_params,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        final_model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        test_acc = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred_proba)

        # Evaluate on training set
        y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_pred_proba > 0.5).astype(int)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_pred_proba)

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
            "n_estimators": best_params["n_estimators"],
            "max_depth": best_params["max_depth"],
            "learning_rate": best_params["learning_rate"],
            "subsample": best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "gamma": best_params["gamma"],
            "reg_alpha": best_params["reg_alpha"],
            "reg_lambda": best_params["reg_lambda"],
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
    output_file = f"{output_dir}/XGBoost_tuned_{stock}_results{strategy_suffix}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df, strategy_suffix


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Hyperparameter Tuning for Stock Prediction"
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
                f"{args.output_dir}/XGBoost_tuned_all_stocks_results{strategy_suffix}.csv", index=False
            )
            print(
                f"\nCombined results saved to: {args.output_dir}/XGBoost_tuned_all_stocks_results{strategy_suffix}.csv"
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
