"""LSTM HPO with the Section 1.1 compute cut:

  - Run skopt HPO ONCE per (stock, ablation) at the representative horizon
    (default h=5).
  - Apply the resulting hyperparameters to all configured horizons × all seeds
    without re-tuning. `cfg["horizons"]` is currently a 3-point sweep ({2, 5, 10}).

This brings the LSTM grid from ~47k fits to ~500 fits, fitting the 4090 budget.

Requires TensorFlow / Keras. The 2-layer LSTM matches the original paper
(Section III.E LSTM Architecture).
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# TF is imported lazily inside set_seed() so config loading doesn't pay the cost
def _import_tf():
    import tensorflow as tf
    from tensorflow.keras import callbacks as kc
    from tensorflow.keras import layers, models, optimizers
    return tf, kc, layers, models, optimizers

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from . import backtest as bt
from . import config as cfg_mod
from . import data as data_mod
from . import metrics as M

SEQ_LEN = 30  # matches the original paper


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf, *_ = _import_tf()
    tf.random.set_seed(seed)
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


def to_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN
                ) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    return np.asarray(Xs), np.asarray(ys)


def build_lstm(u1: int, u2: int, dropout: float, lr: float, n_features: int):
    tf, kc, layers, models, optimizers = _import_tf()
    # Keras 3 rejects np.int64 with "output_size must be an integer"; cast at
    # the builder boundary so all callers (HPO + cached-hparam broadcast) are safe.
    u1, u2 = int(u1), int(u2)
    m = models.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(u1, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(u2),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout / 2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss="binary_crossentropy", metrics=["AUC"])
    return m


def train_lstm(X_tr, y_tr, X_v, y_v, *, u1, u2, dropout, lr, batch_size, epochs=50, seed=42):
    set_seed(seed)
    tf, kc, *_ = _import_tf()
    m = build_lstm(u1=u1, u2=u2, dropout=dropout, lr=lr, n_features=X_tr.shape[-1])
    cb = [
        kc.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        kc.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    m.fit(X_tr, y_tr, validation_data=(X_v, y_v),
          batch_size=batch_size, epochs=epochs, callbacks=cb,
          verbose=0)
    return m


def evaluate_lstm_cell(stock: str, ablation: Dict[str, Any], horizon: int,
                      threshold: float, hparams: Dict[str, Any], seed: int,
                      cfg: Dict[str, Any],
                      learned_alpha_news: float | None = None) -> Dict[str, Any]:
    set_seed(seed)
    # Normalize np.int64 / np.float64 (from skopt) to Python scalars so json.dumps
    # of HParams and Keras layer constructors both succeed.
    hparams = {k: (int(v) if isinstance(v, np.integer)
                   else float(v) if isinstance(v, np.floating) else v)
               for k, v in hparams.items()}
    feats = data_mod.build_feature_matrix(stock, ablation, cfg,
                                          learned_alpha_news=learned_alpha_news)
    split = data_mod.split_with_embargo(feats, cfg, horizon=horizon, threshold=threshold)

    feat_cols = split.feature_columns
    # Fit on numpy too — otherwise sklearn warns each predict() that the
    # feature_names_in_ recorded at fit don't match the array passed in.
    sc = StandardScaler().fit(split.train[feat_cols].to_numpy())
    Xtr = sc.transform(split.train[feat_cols].to_numpy())
    Xv = sc.transform(split.val[feat_cols].to_numpy())
    Xte = sc.transform(split.test[feat_cols].to_numpy())

    Xtr_s, ytr_s = to_sequences(Xtr, split.train["y"].to_numpy())
    Xv_s,  yv_s  = to_sequences(Xv,  split.val["y"].to_numpy())
    Xte_s, yte_s = to_sequences(Xte, split.test["y"].to_numpy())

    if len(Xtr_s) < 100 or len(Xv_s) < 20:
        return {}  # too little data

    m = train_lstm(Xtr_s, ytr_s, Xv_s, yv_s, seed=seed, **hparams)
    pv = m.predict(Xv_s, verbose=0).ravel()
    pt = m.predict(Xte_s, verbose=0).ravel()

    val_auc = roc_auc_score(yv_s, pv) if len(np.unique(yv_s)) >= 2 else 0.5
    test_auc = roc_auc_score(yte_s, pt) if len(np.unique(yte_s)) >= 2 else 0.5

    # align prices with sequence outputs (shift by SEQ_LEN)
    val_prices = split.val["Close"].to_numpy()[SEQ_LEN:]
    test_prices = split.test["Close"].to_numpy()[SEQ_LEN:]
    val_trades = bt.long_short_backtest(val_prices, pv, horizon=horizon,
                                        fee_bps_round_trip=cfg["backtest"]["fee_bps_round_trip"],
                                        strategy=cfg["backtest"]["strategy"])
    test_trades = bt.long_short_backtest(test_prices, pt, horizon=horizon,
                                         fee_bps_round_trip=cfg["backtest"]["fee_bps_round_trip"],
                                         strategy=cfg["backtest"]["strategy"])
    vs = M.summarize(val_trades, cfg["backtest"]["trading_days_per_year"])
    ts = M.summarize(test_trades, cfg["backtest"]["trading_days_per_year"])

    return {
        "Stock": stock, "Ablation": ablation["name"], "Model": "lstm",
        "Horizon": horizon, "Threshold": threshold, "Seed": seed,
        "HParams": json.dumps(hparams),
        "Val_Accuracy": float(((pv > 0.5).astype(int) == yv_s.astype(int)).mean()),
        "Val_ROC_AUC": val_auc,
        "Val_Trades": vs["n_trades"], "Val_WinRate": vs["win_rate"],
        "Val_Sharpe": vs["sharpe"], "Val_AnnualReturn": vs["annual_return"],
        "Val_MDD": vs["max_drawdown"], "Val_Sortino": vs["sortino"],
        "Test_Accuracy": float(((pt > 0.5).astype(int) == yte_s.astype(int)).mean()),
        "Test_ROC_AUC": test_auc,
        "Test_Trades": ts["n_trades"], "Test_WinRate": ts["win_rate"],
        "Test_Sharpe": ts["sharpe"], "Test_AnnualReturn": ts["annual_return"],
        "Test_MDD": ts["max_drawdown"], "Test_Sortino": ts["sortino"],
        "Test_Calmar": ts["calmar"],
    }


def tune_at_representative_horizon(stock: str, ablation: Dict[str, Any],
                                    cfg: Dict[str, Any], seed: int = 42,
                                    learned_alpha_news: float | None = None
                                    ) -> Dict[str, Any]:
    h = cfg["hpo"]["lstm_representative_horizon"]
    thr = 0.005   # representative threshold; the broadcast loop will redo per-horizon thresholds

    space = [
        Real(1e-4, 1e-2, prior="log-uniform", name="lr"),
        Real(0.1,  0.5, name="dropout"),
        Integer(64, 256, name="u1"),
        Integer(16, 128, name="u2"),
        Integer(16, 64, name="batch_size"),
    ]
    n_calls = cfg["hpo"]["lstm_n_calls"]

    def objective(params):
        lr, drop, u1, u2, bs = params
        # skopt's Integer returns np.int64; Keras 3's LSTM layer rejects anything
        # that isn't a Python int with "output_size must be an integer". Cast at
        # the boundary so downstream code sees clean Python scalars.
        hp = dict(lr=float(lr), dropout=float(drop),
                  u1=int(u1), u2=int(u2), batch_size=int(bs))
        try:
            row = evaluate_lstm_cell(stock, ablation, h, thr, hp, seed, cfg,
                                     learned_alpha_news=learned_alpha_news)
        except Exception as e:
            print(f"     LSTM HPO eval failed: {e}")
            return 0.0
        if not row:
            return 0.0
        return -row["Val_ROC_AUC"]

    res = gp_minimize(objective, space, n_calls=n_calls, random_state=seed, verbose=False)
    best = res.x
    return dict(lr=best[0], dropout=best[1], u1=int(best[2]), u2=int(best[3]),
                batch_size=int(best[4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default=None)
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cache-hparams", default=None,
                        help="JSON path to cache best hparams from rep-horizon HPO")
    args = parser.parse_args()

    cfg = cfg_mod.load_config()
    stocks = [args.stock] if args.stock else cfg_mod.stocks(cfg)
    ablations = ([ab for ab in cfg["ablations"] if ab["name"] == args.ablation]
                 if args.ablation else list(cfg["ablations"]))
    seeds = args.seeds if args.seeds else cfg_mod.ml_seeds(cfg)

    cache_path = (Path(args.cache_hparams) if args.cache_hparams else
                  cfg_mod.results_dir() / "lstm_best_hparams.json")
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    out_path = Path(args.out) if args.out else (cfg_mod.results_dir() / "raw" / "lstm.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load learned-α once (per_stock + global) so the learned_alpha_* ablations
    # have their alpha to feed `compose_sentiment`. Same helper that hpo_traditional
    # uses, so the rich notebook-02 JSON is the source of truth.
    from .hpo_traditional import _load_learned_alpha, _resolve_learned_alpha, pick_threshold_for_balance
    needs_learned = any(ab.get("sentiment") == "learned" for ab in ablations)
    learned = _load_learned_alpha() if needs_learned else None

    rows = []
    for stock in stocks:
        for ab in ablations:
            alpha = _resolve_learned_alpha(ab, stock, learned) if learned else None
            key = f"{stock}::{ab['name']}"
            if key in cache:
                hp = cache[key]
                print(f"  LSTM HPO {stock}/{ab['name']}: cached hparams")
            else:
                print(f"  LSTM HPO {stock}/{ab['name']}: tuning at rep horizon")
                hp = tune_at_representative_horizon(stock, ab, cfg,
                                                    learned_alpha_news=alpha)
                cache[key] = hp
                cache_path.write_text(json.dumps(cache, indent=2))

            # Broadcast across all horizons and seeds
            for h in cfg["horizons"]:
                thr = pick_threshold_for_balance(stock, h, cfg)
                for seed in seeds:
                    print(f"     fit/eval {stock}/{ab['name']} h={h} seed={seed}")
                    row = evaluate_lstm_cell(stock, ab, h, thr, hp, seed, cfg,
                                             learned_alpha_news=alpha)
                    if row:
                        rows.append(row)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows):,} rows to {out_path}")


if __name__ == "__main__":
    main()
