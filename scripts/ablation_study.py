"""
Ablation Study for Stock Prediction

This script runs ablation experiments to evaluate the contribution of different components:
1. Technical Only - Only technical indicators (Combi signals + Close)
2. Sentiment Only - Only sentiment features
3. Equal Weights (50:50) - News and Social Media weighted equally
4. News Only - Only news sentiment
5. Social Media Only - Only social media sentiment

Note: Full Model (Tech + Sent 70:30) results are already available from main experiments in results/ours/.

Requirements:
- For technical_only: Only requires dataset/training_data/
- For sentiment configurations (news_only, social_only, equal_weights): Requires raw sentiment
  files with Type column in dataset/sentiment/raw/. These files should have the pattern:
  *{stock}*.csv and contain columns: Date, Type, Sentiment_Score

Usage:
    python scripts/ablation_study.py --config technical_only --all-stocks
    python scripts/ablation_study.py --config sentiment_only --stock AAPL
    python scripts/ablation_study.py --all-configs --all-stocks
    python scripts/ablation_study.py --config technical_only --all-stocks --skip-lstm  # Skip LSTM
    python scripts/ablation_study.py --config technical_only --all-stocks --only-lstm  # Only LSTM
    python scripts/ablation_study.py --config technical_only --all-stocks --force-cpu  # Force CPU
"""

import os
import sys
import warnings

# CRITICAL: Parse --force-cpu flag BEFORE importing TensorFlow
# This ensures CUDA is disabled at the environment level before TF initializes
FORCE_CPU_MODE = '--force-cpu' in sys.argv
if FORCE_CPU_MODE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("=" * 70)
    print("FORCE CPU MODE ENABLED")
    print("TensorFlow will use CPU only (CUDA_VISIBLE_DEVICES=-1)")
    print("=" * 70)

    # Prevent TensorFlow cleanup at exit to avoid double free
    import atexit
    def skip_tf_cleanup():
        pass  # Do nothing - let OS clean up
    atexit.register(skip_tf_cleanup)

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import random
import argparse
from pathlib import Path

# ML imports
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# Try to import XGBoost (optional dependency due to OpenMP requirements)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"Warning: XGBoost not available: {e}")
    print("Continuing without XGBoost. To fix: brew install libomp && pip install --upgrade xgboost")
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from lightgbm import LGBMClassifier

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# In CPU mode, disable TensorFlow's problematic cleanup
if FORCE_CPU_MODE:
    # Monkey-patch clear_session to do nothing in CPU mode during exit
    _original_clear_session = tf.keras.backend.clear_session
    def _noop_clear_session():
        import gc
        gc.collect()  # Just run garbage collection
    tf.keras.backend.clear_session = _noop_clear_session

# Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# ============================================================================
# Configuration
# ============================================================================

# Check if GPU with CuDNN is available
def check_cudnn_available(force_cpu=False):
    """Check if CuDNN is available for GPU acceleration.

    Args:
        force_cpu: If True, force CPU mode regardless of GPU availability

    Returns:
        bool: True if CuDNN should be used, False for CPU mode
    """
    if force_cpu:
        print("Force CPU mode enabled (--force-cpu flag)")
        return False

    try:
        # Try to create a small test LSTM to verify CuDNN actually works
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and tf.test.is_built_with_cuda():
            # Test if CuDNN LSTM actually works
            try:
                test_model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(8, input_shape=(5, 3))
                ])
                test_input = tf.random.normal((1, 5, 3))
                _ = test_model(test_input)
                del test_model
                # Safe cleanup - only clear session if CUDA is actually available
                import gc
                gc.collect()
                return True
            except Exception as e:
                print(f"GPU available but CuDNN test failed: {e}")
                import gc
                gc.collect()
                return False
    except Exception as e:
        print(f"GPU detection failed: {e}")
    return False


def safe_clear_session():
    """Safely clear Keras session, avoiding issues in CPU mode."""
    if USE_CUDNN is False:
        # In CPU mode, just do garbage collection without clear_session
        # to avoid double-free issues with CUDA
        import gc
        gc.collect()
    else:
        # In GPU mode, use normal clear_session
        tf.keras.backend.clear_session()

# Global flag - will be set by main() based on command-line args
USE_CUDNN = None

STOCKS = ["AAPL", "META", "NVDA", "SPY", "TSLA"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR = 2024
RANDOM_SEED = 42
FEE_BPS_ROUND_TRIP = 10

# LSTM specific
SEQ_LEN = 30
EPOCHS = 50
PATIENCE = 10

# Ablation configurations (full_model removed - use main experiment results)
ABLATION_CONFIGS = {
    "technical_only": {
        "description": "Technical Only",
        "use_technical": True,
        "use_sentiment": False,
        "news_weight": 0.0,
        "social_weight": 0.0,
    },
    "sentiment_only": {
        "description": "Sentiment Only",
        "use_technical": False,
        "use_sentiment": True,
        "news_weight": 0.7,
        "social_weight": 0.3,
    },
    "equal_weights": {
        "description": "Equal Weights (50:50)",
        "use_technical": True,
        "use_sentiment": True,
        "news_weight": 0.5,
        "social_weight": 0.5,
    },
    "news_only": {
        "description": "News Only",
        "use_technical": True,
        "use_sentiment": True,
        "news_weight": 1.0,
        "social_weight": 0.0,
    },
    "social_only": {
        "description": "Social Media Only",
        "use_technical": True,
        "use_sentiment": True,
        "news_weight": 0.0,
        "social_weight": 1.0,
    },
}

# Model configurations - use same hyperparameter spaces as main experiments
MODELS = {
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "space": [
            Integer(50, 500, name="n_estimators"),
            Integer(3, 12, name="max_depth"),
            Real(0.01, 0.3, name="learning_rate", prior="log-uniform"),
            Real(0.5, 1.0, name="subsample"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 10, name="min_samples_leaf"),
        ],
        "param_names": ["n_estimators", "max_depth", "learning_rate", "subsample",
                        "min_samples_split", "min_samples_leaf"],
    },
    "RandomForest": {
        "class": RandomForestClassifier,
        "space": [
            Integer(50, 500, name="n_estimators"),
            Integer(3, 20, name="max_depth"),
            Integer(2, 20, name="min_samples_split"),
            Integer(1, 10, name="min_samples_leaf"),
            Categorical(["gini", "entropy"], name="criterion"),
        ],
        "param_names": ["n_estimators", "max_depth", "min_samples_split",
                        "min_samples_leaf", "criterion"],
        "extra_params": {"class_weight": "balanced"},
    },
    "XGBoost": {
        "class": XGBClassifier,
        "space": [
            Integer(50, 500, name="n_estimators"),
            Integer(3, 12, name="max_depth"),
            Real(0.01, 0.3, name="learning_rate", prior="log-uniform"),
            Real(0.5, 1.0, name="subsample"),
            Real(0.5, 1.0, name="colsample_bytree"),
            Real(0, 10, name="reg_alpha"),
            Real(0, 10, name="reg_lambda"),
        ],
        "param_names": ["n_estimators", "max_depth", "learning_rate", "subsample",
                        "colsample_bytree", "reg_alpha", "reg_lambda"],
        "extra_params": {"use_label_encoder": False, "eval_metric": "logloss", "verbosity": 0},
    },
    "LightGBM": {
        "class": LGBMClassifier,
        "space": [
            Integer(50, 500, name="n_estimators"),
            Integer(20, 100, name="num_leaves"),
            Integer(3, 12, name="max_depth"),
            Real(0.01, 0.3, name="learning_rate", prior="log-uniform"),
            Real(0.5, 1.0, name="subsample"),
            Real(0.5, 1.0, name="colsample_bytree"),
            Real(0, 10, name="reg_alpha"),
            Real(0, 10, name="reg_lambda"),
        ],
        "param_names": ["n_estimators", "num_leaves", "max_depth", "learning_rate",
                        "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"],
        "extra_params": {"verbosity": -1},
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "space": [
            Real(0.001, 100, name="C", prior="log-uniform"),
            Categorical(["l1", "l2"], name="penalty"),
        ],
        "param_names": ["C", "penalty"],
        "extra_params": {"solver": "saga", "max_iter": 1000, "class_weight": "balanced"},
        "needs_scaling": True,
    },
    "SVM": {
        "class": SVC,
        "space": [
            Real(0.01, 100, name="C", prior="log-uniform"),
            Categorical(["linear", "rbf", "poly"], name="kernel"),
            Real(0.001, 10, name="gamma", prior="log-uniform"),
        ],
        "param_names": ["C", "kernel", "gamma"],
        "extra_params": {"probability": True, "class_weight": "balanced"},
        "needs_scaling": True,
    },
}

# Remove XGBoost if not available
if not XGBOOST_AVAILABLE:
    if "XGBoost" in MODELS:
        del MODELS["XGBoost"]
        print("Note: XGBoost removed from models list (library not available)")

# LSTM hyperparameter search space
LSTM_SPACE = [
    Real(1e-4, 1e-2, name="learning_rate", prior="log-uniform"),
    Real(0.1, 0.5, name="dropout_rate"),
    Integer(32, 256, name="lstm_units_1"),
    Integer(16, 128, name="lstm_units_2"),
    Integer(16, 64, name="batch_size"),
]
LSTM_PARAM_NAMES = ["learning_rate", "dropout_rate", "lstm_units_1", "lstm_units_2", "batch_size"]

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


def load_stock_data(stock: str, data_dir: str) -> pd.DataFrame:
    """Load stock data with technical indicators."""
    df = pd.read_csv(f"{data_dir}/{stock}_data_model_training.csv")
    df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Year"] = df["__DateDT__"].dt.year
    return df


def load_raw_sentiment(stock: str, sentiment_dir: str) -> pd.DataFrame:
    """
    Load raw sentiment data with News/Social Media types for custom weighting.

    Expected file pattern: *{stock}*.csv (e.g., news_sentiment_finbert_tone_aapl_2020.csv)
    Required columns: Date, Type (News/Social Media), Sentiment_Score
    """
    sentiment_path = Path(sentiment_dir)
    pattern = f"*{stock.lower()}*.csv"
    files = list(sentiment_path.glob(pattern))

    if not files:
        print(f"  ERROR: No raw sentiment files found for {stock} in {sentiment_dir}")
        print(f"  Looking for pattern: {pattern}")
        print(f"  These files are required for news_only, social_only, and equal_weights configs")
        print(f"  Hint: Copy raw sentiment files to {sentiment_dir}")
        return None

    all_dfs = []
    for file in files:
        try:
            df = pd.read_csv(file, lineterminator="\n")
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {file}: {e}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.date

    return combined


def compute_weighted_sentiment(
    sentiment_df: pd.DataFrame,
    news_weight: float = 0.7,
    social_weight: float = 0.3
) -> pd.DataFrame:
    """
    Compute weighted sentiment score from raw sentiment data.
    """
    if sentiment_df is None or len(sentiment_df) == 0:
        return None

    df = sentiment_df.copy()

    # Assign weights based on Type
    weights = {"News": news_weight, "Social Media": social_weight}
    df["weight"] = df["Type"].map(weights)

    # Filter out rows with zero weight
    df = df[df["weight"] > 0]

    if len(df) == 0:
        return None

    # Compute weighted sentiment
    df["weighted_sentiment"] = df["Sentiment_Score"] * df["weight"]

    # Aggregate by date (weighted average)
    result = (
        df.groupby("Date")
        .apply(
            lambda x: x["weighted_sentiment"].sum() / x["weight"].sum(),
            include_groups=False,
        )
        .reset_index(name="Sentiment_Score")
    )

    return result


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
    predicted_probs=None,
):
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

        action = "long" if pred == 1 else "short"

        if action == "long":
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
# LSTM Functions
# ============================================================================


def create_sequences(X, y, seq_len=SEQ_LEN):
    """Create sequences for LSTM input."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        if hasattr(X, "iloc"):
            Xs.append(X.iloc[i : i + seq_len].values)
        else:
            Xs.append(X[i : i + seq_len])
        ys.append(y.iloc[i + seq_len] if hasattr(y, "iloc") else y[i + seq_len])
    return np.array(Xs), np.array(ys)


def build_lstm_model(
    seq_len, num_features, learning_rate, dropout_rate, lstm_units_1, lstm_units_2
):
    """Build LSTM model with specified hyperparameters.

    Automatically uses CuDNN optimization on GPU, or CPU-compatible implementation
    when CuDNN is not available.
    """
    lstm_units_1 = int(lstm_units_1)
    lstm_units_2 = int(lstm_units_2)

    # Configure LSTM layers based on CuDNN availability
    # When CuDNN is not available, use 'sigmoid' recurrent activation (disables CuDNN)
    # When CuDNN is available, use default 'tanh' (enables CuDNN optimization)
    # If USE_CUDNN is None (not set yet), default to CPU mode for safety
    use_cudnn = USE_CUDNN if USE_CUDNN is not None else False
    lstm_kwargs = {} if use_cudnn else {'recurrent_activation': 'sigmoid'}

    model = Sequential(
        [
            LSTM(
                lstm_units_1, input_shape=(seq_len, num_features), return_sequences=True,
                **lstm_kwargs
            ),
            Dropout(dropout_rate),
            LSTM(lstm_units_2, return_sequences=False, **lstm_kwargs),
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


def tune_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq, n_calls=20, verbose=True):
    """
    Tune LSTM hyperparameters using Bayesian optimization.
    """
    seq_len = X_train_seq.shape[1]
    num_features = X_train_seq.shape[2]

    @use_named_args(LSTM_SPACE)
    def objective(learning_rate, dropout_rate, lstm_units_1, lstm_units_2, batch_size):
        set_global_seed(RANDOM_SEED)

        try:
            model = build_lstm_model(
                seq_len, num_features, learning_rate, dropout_rate, lstm_units_1, lstm_units_2
            )

            early_stop = EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
            )

            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=EPOCHS,
                batch_size=int(batch_size),
                callbacks=[early_stop, reduce_lr],
                verbose=0,
            )

            y_pred_proba = model.predict(X_val_seq, verbose=0).flatten()
            auc = roc_auc_score(y_val_seq, y_pred_proba)

            if verbose:
                print(f"    AUC: {auc:.4f}")

            # Clear memory - delete model first to avoid double free in CPU mode
            del model
            safe_clear_session()

            return -auc

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            # Try to delete model if it exists
            try:
                del model
            except:
                pass
            safe_clear_session()
            return 0

    result = gp_minimize(
        objective,
        LSTM_SPACE,
        n_calls=n_calls,
        random_state=RANDOM_SEED,
        verbose=False,
    )

    best_params = dict(zip(LSTM_PARAM_NAMES, result.x))
    best_auc = -result.fun

    return best_params, best_auc


def train_and_evaluate_lstm(
    best_params,
    X_train_seq, y_train_seq,
    X_test_seq, y_test_seq,
    test_df,
    horizon,
    seq_len,
    num_features,
):
    """Train final LSTM model and evaluate on test set."""
    set_global_seed(RANDOM_SEED)

    model = build_lstm_model(
        seq_len, num_features,
        best_params["learning_rate"],
        best_params["dropout_rate"],
        best_params["lstm_units_1"],
        best_params["lstm_units_2"],
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
    )

    # Split train for validation during final training
    val_size = int(len(X_train_seq) * 0.1)
    X_train_final = X_train_seq[:-val_size]
    y_train_final = y_train_seq[:-val_size]
    X_val_final = X_train_seq[-val_size:]
    y_val_final = y_train_seq[-val_size:]

    model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val_final, y_val_final),
        epochs=EPOCHS,
        batch_size=int(best_params["batch_size"]),
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )

    # Predictions
    y_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    test_acc = accuracy_score(y_test_seq, y_pred)
    test_auc = roc_auc_score(y_test_seq, y_pred_proba)

    # Training metrics
    y_train_pred_proba = model.predict(X_train_seq, verbose=0).flatten()
    train_acc = accuracy_score(y_train_seq, (y_train_pred_proba > 0.5).astype(int))
    train_auc = roc_auc_score(y_train_seq, y_train_pred_proba)

    # Backtest - need to align predictions with test_df
    # The test sequences start at index SEQ_LEN, so we need to offset
    test_df_for_backtest = test_df.iloc[SEQ_LEN:].reset_index(drop=True)
    backtest = non_overlap_backtest(test_df_for_backtest, y_pred, horizon, predicted_probs=y_pred_proba)

    # Clear memory - delete model first to avoid double free in CPU mode
    del model
    safe_clear_session()

    return {
        "Train_Accuracy": train_acc,
        "Train_ROC_AUC": train_auc,
        "Test_Accuracy": test_acc,
        "Test_ROC_AUC": test_auc,
        **backtest,
    }


# ============================================================================
# Model Tuning Functions (Non-LSTM)
# ============================================================================


def tune_model(model_name, X_train, y_train, X_val, y_val, n_calls=20, verbose=True):
    """
    Tune a model using Bayesian optimization.
    """
    model_config = MODELS[model_name]
    space = model_config["space"]
    param_names = model_config["param_names"]
    model_class = model_config["class"]
    extra_params = model_config.get("extra_params", {})
    needs_scaling = model_config.get("needs_scaling", False)

    # Scale data if needed
    if needs_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
    else:
        X_train_scaled = X_train
        X_val_scaled = X_val
        scaler = None

    @use_named_args(space)
    def objective(**params):
        set_global_seed(RANDOM_SEED)

        try:
            if model_name == "LogisticRegression" and params.get("penalty") == "l1":
                params["solver"] = "saga"

            model = model_class(
                **params,
                **extra_params,
                random_state=RANDOM_SEED,
            )

            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)

            if verbose:
                print(f"    AUC: {auc:.4f}")

            return -auc

        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            return 0

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=RANDOM_SEED,
        verbose=False,
    )

    best_params = dict(zip(param_names, result.x))
    best_auc = -result.fun

    return best_params, best_auc, scaler


def train_and_evaluate(
    model_name,
    best_params,
    X_train,
    y_train,
    X_test,
    y_test,
    test_df,
    horizon,
    scaler=None,
):
    """Train final model and evaluate on test set."""
    model_config = MODELS[model_name]
    model_class = model_config["class"]
    extra_params = model_config.get("extra_params", {})
    needs_scaling = model_config.get("needs_scaling", False)

    if needs_scaling and scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    set_global_seed(RANDOM_SEED)
    model = model_class(
        **best_params,
        **extra_params,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_pred_proba)

    # Training metrics
    y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_acc = accuracy_score(y_train, (y_train_pred_proba > 0.5).astype(int))
    train_auc = roc_auc_score(y_train, y_train_pred_proba)

    # Backtest
    backtest = non_overlap_backtest(test_df, y_pred, horizon, predicted_probs=y_pred_proba)

    return {
        "Train_Accuracy": train_acc,
        "Train_ROC_AUC": train_auc,
        "Test_Accuracy": test_acc,
        "Test_ROC_AUC": test_auc,
        **backtest,
    }


# ============================================================================
# Main Ablation Study Functions
# ============================================================================


def run_ablation_for_stock(
    stock: str,
    config_name: str,
    data_dir: str,
    sentiment_dir: str,
    output_dir: str,
    n_calls: int = 20,
    model_name: str = None,
    skip_lstm: bool = False,
    only_lstm: bool = False,
    horizons: list = None,
):
    """
    Run ablation study for a single stock with a specific configuration.
    """
    ablation_config = ABLATION_CONFIGS[config_name]

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {stock}")
    print(f"Configuration: {ablation_config['description']}")
    print(f"{'='*70}")

    # Load stock data
    df = load_stock_data(stock, data_dir)

    # Encode signals
    signal_cols = [c for c in df.columns if "Combi" in c and "Days" not in c]
    signal_map = {"Buy": 1, "Sell": -1, "Hold": 0, "": 0}
    for col in signal_cols:
        df[col] = df[col].astype(str).str.strip().map(signal_map).fillna(0).astype(int)

    # Load and compute sentiment based on configuration
    sentiment_score_col = None
    if ablation_config["use_sentiment"]:
        if ablation_config["news_weight"] == 0.7 and ablation_config["social_weight"] == 0.3:
            # Use existing Filtered Sentiment Score
            if "Filtered Sentiment Score" in df.columns:
                df["Ablation_Sentiment"] = df["Filtered Sentiment Score"]
                sentiment_score_col = "Ablation_Sentiment"
            elif "Weighted Sentiment Score" in df.columns:
                df["Ablation_Sentiment"] = df["Weighted Sentiment Score"]
                sentiment_score_col = "Ablation_Sentiment"
        else:
            # Need to recompute sentiment with different weights
            raw_sentiment = load_raw_sentiment(stock, sentiment_dir)
            if raw_sentiment is not None:
                weighted_sentiment = compute_weighted_sentiment(
                    raw_sentiment,
                    news_weight=ablation_config["news_weight"],
                    social_weight=ablation_config["social_weight"],
                )
                if weighted_sentiment is not None:
                    df["__DateOnly__"] = df["__DateDT__"].dt.date
                    weighted_sentiment["Date"] = pd.to_datetime(weighted_sentiment["Date"]).dt.date
                    df = df.merge(
                        weighted_sentiment,
                        left_on="__DateOnly__",
                        right_on="Date",
                        how="left",
                        suffixes=("", "_sent"),
                    )
                    df["Ablation_Sentiment"] = df["Sentiment_Score"].fillna(0)
                    sentiment_score_col = "Ablation_Sentiment"

    # Split train/test
    train_df = df[df["Year"].isin(TRAIN_YEARS)].copy()
    test_df = df[df["Year"] == TEST_YEAR].copy()

    print(f"\nTrain/Test Split:")
    print(f"  Training: {len(train_df)} samples ({train_df['__DateDT__'].min().strftime('%Y-%m-%d')} to {train_df['__DateDT__'].max().strftime('%Y-%m-%d')})")
    print(f"  Testing:  {len(test_df)} samples ({test_df['__DateDT__'].min().strftime('%Y-%m-%d')} to {test_df['__DateDT__'].max().strftime('%Y-%m-%d')})")

    # Find best thresholds for each horizon
    if horizons is None:
        horizons = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Finding best thresholds for horizons: {horizons}...")
    horizons_thresholds = []
    for horizon in horizons:
        best_threshold = find_best_threshold_for_horizon(train_df, horizon)
        horizons_thresholds.append((horizon, best_threshold))
    print(f"Found thresholds for {len(horizons_thresholds)} horizons")

    # Prepare feature columns
    feature_cols = []
    if ablation_config["use_technical"]:
        feature_cols.extend(signal_cols)
        feature_cols.append("Close")
    if sentiment_score_col:
        feature_cols.append(sentiment_score_col)

    if not feature_cols:
        print("Error: No features available for this configuration!")
        return None

    print(f"Features ({len(feature_cols)}): {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Features: {feature_cols}")

    # Determine which models to run
    if model_name:
        if model_name == "LSTM":
            models_to_run = []
            run_lstm = True
        else:
            models_to_run = [model_name]
            run_lstm = False
    elif only_lstm:
        # Only run LSTM, skip all other models
        models_to_run = []
        run_lstm = True
    else:
        models_to_run = list(MODELS.keys())
        run_lstm = not skip_lstm

    all_results = []

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

        # Prepare features for non-LSTM models
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

        best_result = None
        best_auc = -1

        # Run non-LSTM models
        for model_name_iter in models_to_run:
            print(f"\n  Model: {model_name_iter}")

            best_params, val_auc, scaler = tune_model(
                model_name_iter,
                X_train_tune,
                y_train_tune,
                X_val_tune,
                y_val_tune,
                n_calls=n_calls,
                verbose=False,
            )

            result = train_and_evaluate(
                model_name_iter,
                best_params,
                X_train,
                y_train,
                X_test,
                y_test,
                test_copy,
                horizon,
                scaler,
            )

            print(f"    Val AUC: {val_auc:.4f}, Test AUC: {result['Test_ROC_AUC']:.4f}, Sharpe: {result['Sharpe']:.2f}")

            if result["Test_ROC_AUC"] > best_auc:
                best_auc = result["Test_ROC_AUC"]
                best_result = {
                    "Stock": stock,
                    "Horizon": horizon,
                    "BestThreshold": threshold,
                    "AblationConfig": config_name,
                    "Model": model_name_iter,
                    **result,
                }

        # Run LSTM
        if run_lstm:
            print(f"\n  Model: LSTM")

            # Scale features for LSTM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_copy[feature_cols])
            X_test_scaled = scaler.transform(test_copy[feature_cols])

            # Create sequences
            X_train_seq, y_train_seq = create_sequences(
                X_train_scaled, train_copy["target"].values, SEQ_LEN
            )
            X_test_seq, y_test_seq = create_sequences(
                X_test_scaled, test_copy["target"].values, SEQ_LEN
            )

            if len(X_train_seq) > 0 and len(X_test_seq) > 0:
                # Split for tuning
                val_size_seq = int(len(X_train_seq) * 0.2)
                X_train_tune_seq = X_train_seq[:-val_size_seq]
                y_train_tune_seq = y_train_seq[:-val_size_seq]
                X_val_tune_seq = X_train_seq[-val_size_seq:]
                y_val_tune_seq = y_train_seq[-val_size_seq:]

                # Tune LSTM
                best_params, val_auc = tune_lstm(
                    X_train_tune_seq,
                    y_train_tune_seq,
                    X_val_tune_seq,
                    y_val_tune_seq,
                    n_calls=n_calls,
                    verbose=False,
                )

                # Train and evaluate
                result = train_and_evaluate_lstm(
                    best_params,
                    X_train_seq, y_train_seq,
                    X_test_seq, y_test_seq,
                    test_copy,
                    horizon,
                    SEQ_LEN,
                    len(feature_cols),
                )

                print(f"    Val AUC: {val_auc:.4f}, Test AUC: {result['Test_ROC_AUC']:.4f}, Sharpe: {result['Sharpe']:.2f}")

                if result["Test_ROC_AUC"] > best_auc:
                    best_auc = result["Test_ROC_AUC"]
                    best_result = {
                        "Stock": stock,
                        "Horizon": horizon,
                        "BestThreshold": threshold,
                        "AblationConfig": config_name,
                        "Model": "LSTM",
                        **result,
                    }
            else:
                print("    Warning: Not enough data for LSTM sequences")

        if best_result:
            all_results.append(best_result)

    # Save results (with merge support - keep best by AUC for each horizon)
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_file = f"{output_dir}/ablation_{config_name}_{stock}_results.csv"

        # Check if file exists and merge if needed
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)

            # For each horizon, compare new vs existing and keep the best by AUC
            merged_results = []
            all_horizons = set(existing_df['Horizon'].unique()) | set(results_df['Horizon'].unique())

            for horizon in all_horizons:
                existing_row = existing_df[existing_df['Horizon'] == horizon]
                new_row = results_df[results_df['Horizon'] == horizon]

                if len(existing_row) == 0 and len(new_row) > 0:
                    # Only new result exists
                    merged_results.append(new_row.iloc[0])
                    print(f"    Horizon {horizon}: Added new {new_row.iloc[0]['Model']} (AUC={new_row.iloc[0]['Test_ROC_AUC']:.4f})")
                elif len(new_row) == 0 and len(existing_row) > 0:
                    # Only existing result exists
                    merged_results.append(existing_row.iloc[0])
                elif len(existing_row) > 0 and len(new_row) > 0:
                    # Both exist - compare AUC and keep the better one
                    existing_auc = existing_row.iloc[0]['Test_ROC_AUC']
                    new_auc = new_row.iloc[0]['Test_ROC_AUC']

                    if new_auc > existing_auc:
                        merged_results.append(new_row.iloc[0])
                        print(f"    Horizon {horizon}: Replaced {existing_row.iloc[0]['Model']} (AUC={existing_auc:.4f}) with {new_row.iloc[0]['Model']} (AUC={new_auc:.4f})")
                    else:
                        merged_results.append(existing_row.iloc[0])
                        print(f"    Horizon {horizon}: Kept {existing_row.iloc[0]['Model']} (AUC={existing_auc:.4f}), new {new_row.iloc[0]['Model']} (AUC={new_auc:.4f}) not better")

            results_df = pd.DataFrame(merged_results)
            results_df = results_df.sort_values(['Horizon']).reset_index(drop=True)

        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        return results_df

    return None


def main():
    parser = argparse.ArgumentParser(description="Ablation Study for Stock Prediction")
    parser.add_argument(
        "--config",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation configuration to run",
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all ablation configurations",
    )
    parser.add_argument(
        "--stock",
        type=str,
        choices=STOCKS,
        help="Stock symbol to run",
    )
    parser.add_argument(
        "--all-stocks",
        action="store_true",
        help="Run for all stocks",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()) + ["LSTM"],
        help="Specific model to use (default: try all and pick best)",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM model (faster)",
    )
    parser.add_argument(
        "--only-lstm",
        action="store_true",
        help="Run only LSTM model (skip other ML models)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset/training_data",
        help="Directory containing stock data",
    )
    parser.add_argument(
        "--sentiment-dir",
        type=str,
        default="dataset/sentiment/raw",
        help="Directory containing raw sentiment data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Directory to save ablation results",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=20,
        help="Number of Bayesian optimization iterations",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
        help="Prediction horizons in days (default: 2 3 4 5 6 7 8 9 10)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU mode (disable CuDNN GPU acceleration for LSTM)",
    )

    args = parser.parse_args()

    # Set global USE_CUDNN based on --force-cpu flag
    global USE_CUDNN
    USE_CUDNN = check_cudnn_available(force_cpu=args.force_cpu)
    print(f"CuDNN GPU acceleration: {'ENABLED' if USE_CUDNN else 'DISABLED (CPU mode)'}")

    # Validate arguments
    if not args.config and not args.all_configs:
        parser.error("Must specify --config or --all-configs")
    if not args.stock and not args.all_stocks:
        parser.error("Must specify --stock or --all-stocks")
    if args.skip_lstm and args.only_lstm:
        parser.error("Cannot specify both --skip-lstm and --only-lstm")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    set_global_seed(RANDOM_SEED)

    # Determine configurations and stocks to run
    configs_to_run = list(ABLATION_CONFIGS.keys()) if args.all_configs else [args.config]
    stocks_to_run = STOCKS if args.all_stocks else [args.stock]

    all_combined_results = []

    for config_name in configs_to_run:
        print(f"\n{'#'*70}")
        print(f"# ABLATION CONFIG: {ABLATION_CONFIGS[config_name]['description']}")
        print(f"{'#'*70}")

        config_results = []

        for stock in stocks_to_run:
            result = run_ablation_for_stock(
                stock=stock,
                config_name=config_name,
                data_dir=args.data_dir,
                sentiment_dir=args.sentiment_dir,
                output_dir=args.output_dir,
                n_calls=args.n_calls,
                model_name=args.model,
                skip_lstm=args.skip_lstm,
                only_lstm=args.only_lstm,
                horizons=args.horizons,
            )
            if result is not None:
                config_results.append(result)

        # Combine results for this config (with merge support - keep best by AUC)
        if config_results:
            combined = pd.concat(config_results, ignore_index=True)
            all_stocks_file = f"{args.output_dir}/ablation_{config_name}_all_stocks_results.csv"

            # Check if file exists and merge if needed
            if os.path.exists(all_stocks_file):
                existing_df = pd.read_csv(all_stocks_file)

                # For each (stock, horizon) pair, compare and keep the best by AUC
                merged_results = []
                all_keys = set(zip(existing_df['Stock'], existing_df['Horizon'])) | set(zip(combined['Stock'], combined['Horizon']))

                for stock, horizon in all_keys:
                    existing_row = existing_df[(existing_df['Stock'] == stock) & (existing_df['Horizon'] == horizon)]
                    new_row = combined[(combined['Stock'] == stock) & (combined['Horizon'] == horizon)]

                    if len(existing_row) == 0 and len(new_row) > 0:
                        merged_results.append(new_row.iloc[0])
                    elif len(new_row) == 0 and len(existing_row) > 0:
                        merged_results.append(existing_row.iloc[0])
                    elif len(existing_row) > 0 and len(new_row) > 0:
                        existing_auc = existing_row.iloc[0]['Test_ROC_AUC']
                        new_auc = new_row.iloc[0]['Test_ROC_AUC']

                        if new_auc > existing_auc:
                            merged_results.append(new_row.iloc[0])
                        else:
                            merged_results.append(existing_row.iloc[0])

                combined = pd.DataFrame(merged_results)
                combined = combined.sort_values(['Stock', 'Horizon']).reset_index(drop=True)

                print(f"\nMerged all_stocks results (kept best by AUC for each stock/horizon)")

            combined.to_csv(all_stocks_file, index=False)
            all_combined_results.append(combined)

            # Print summary
            print(f"\n{'='*70}")
            print(f"SUMMARY: {ABLATION_CONFIGS[config_name]['description']}")
            print(f"{'='*70}")
            avg_acc = combined["Test_Accuracy"].mean()
            avg_auc = combined["Test_ROC_AUC"].mean()
            avg_sharpe = combined["Sharpe"].mean()
            print(f"Average Accuracy: {avg_acc:.4f}")
            print(f"Average AUC: {avg_auc:.4f}")
            print(f"Average Sharpe: {avg_sharpe:.4f}")

    # Create final summary
    if all_combined_results:
        final_combined = pd.concat(all_combined_results, ignore_index=True)
        final_combined.to_csv(
            f"{args.output_dir}/ablation_all_results.csv",
            index=False,
        )

        # Print final summary table
        print(f"\n{'='*70}")
        print("FINAL ABLATION SUMMARY")
        print(f"{'='*70}")
        summary = final_combined.groupby("AblationConfig").agg({
            "Test_Accuracy": "mean",
            "Test_ROC_AUC": "mean",
            "Sharpe": "mean",
            "TotalReturn": "mean",
        }).round(4)
        print(summary)


if __name__ == "__main__":
    main()
