"""Traditional ML HPO runner (LR, SVM, RF, GB, XGB, LGBM) with explicit
val/test split, multi-seed support, and Val_* + Test_* output columns.

This is the consolidated replacement for the 6 per-model HPO scripts under
../paper/{logistic_regression,svm,random_forest,gradient_boosting,xgboost,
lightgbm}_hyperparameter_tuning.py. The two key changes vs. the old code:

  1. Validation slice is the 2023 calendar year (not last 20% of train).
  2. Output CSVs include Val_ROC_AUC, Val_Sharpe, Val_Trades, Val_WinRate
     so select_best_config.py can lock model/horizon by VAL, not test.

The grid is parameterized in configs/default.yaml.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import warnings
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from . import backtest as bt
from . import config as cfg_mod
from . import data as data_mod
from . import metrics as M


# SVC max_iter is a deliberate safety cap (see model registry below); the resulting
# ConvergenceWarnings are informational, not actionable, and flood the log on the
# pathological (C, γ) corners that skopt eventually learns to avoid. Filter once
# at module load so both the HPO objective and the post-HPO refit are covered.
warnings.filterwarnings("ignore", category=ConvergenceWarning,
                        module=r"sklearn\.svm\._base")
# LightGBM auto-assigns feature names ("Column_0", "Column_1", ...) when fit on
# a numpy array; sklearn then warns on every predict() because the predict-time
# array has no names attached. We always pass numpy in and out consistently, so
# this is purely log noise. (19,200 of these in the May 19/20 run.)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module=r"sklearn\.utils\.validation",
)


# -------------------- model registry --------------------

def get_search_space(model_name: str):
    """Return (skopt search space, model factory) for the given model."""
    if model_name == "logistic_regression":
        # sklearn 1.8 deprecated `penalty`; l1_ratio=0 ≡ L2, l1_ratio=1 ≡ L1 (liblinear).
        space = [
            Real(1e-3, 100.0, prior="log-uniform", name="C"),
            Categorical([0.0, 1.0], name="l1_ratio"),
        ]
        def make(C, l1_ratio, seed):
            return LogisticRegression(C=C, l1_ratio=l1_ratio, solver="liblinear",
                                      class_weight="balanced", random_state=seed,
                                      max_iter=1000)
        return space, make
    if model_name == "svm":
        # Narrowed from C ∈ [1e-2, 100], γ ∈ [1e-3, 10]: the wider space lets skopt
        # wander into a libsvm-SMO-pathological corner (large C × small γ × rbf ×
        # probability=True) where a single .fit() can run for hours. max_iter caps
        # the worst-case cell at a known budget; pathological corners hit the cap
        # and return a suboptimal AUC, which trains the GP to avoid them.
        space = [
            Real(0.1, 10.0, prior="log-uniform", name="C"),
            Real(0.01, 1.0, prior="log-uniform", name="gamma"),
            Categorical(["rbf", "linear"], name="kernel"),
        ]
        def make(C, gamma, kernel, seed):
            return SVC(C=C, gamma=gamma, kernel=kernel, probability=True,
                       class_weight="balanced", random_state=seed,
                       max_iter=20000)
        return space, make
    if model_name == "random_forest":
        space = [
            Integer(50, 500, name="n_estimators"),
            Integer(3, 20, name="max_depth"),
            Categorical(["gini", "entropy"], name="criterion"),
        ]
        def make(n_estimators, max_depth, criterion, seed):
            return RandomForestClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth, criterion=criterion,
                                          class_weight="balanced",
                                          random_state=seed, n_jobs=-1)
        return space, make
    if model_name == "gradient_boosting":
        space = [
            Integer(50, 500, name="n_estimators"),
            Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
            Integer(3, 10, name="max_depth"),
        ]
        def make(n_estimators, learning_rate, max_depth, seed):
            return GradientBoostingClassifier(n_estimators=n_estimators,
                                              learning_rate=learning_rate,
                                              max_depth=max_depth,
                                              random_state=seed)
        return space, make
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        space = [
            Integer(50, 500, name="n_estimators"),
            Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
            Integer(3, 12, name="max_depth"),
            Real(0.5, 1.0, name="subsample"),
            Real(0.5, 1.0, name="colsample_bytree"),
        ]
        def make(n_estimators, learning_rate, max_depth, subsample, colsample_bytree, seed):
            return XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                 max_depth=max_depth, subsample=subsample,
                                 colsample_bytree=colsample_bytree,
                                 random_state=seed, tree_method="hist",
                                 eval_metric="logloss",
                                 use_label_encoder=False, verbosity=0)
        return space, make
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        space = [
            Integer(50, 500, name="n_estimators"),
            Real(0.01, 0.3, prior="log-uniform", name="learning_rate"),
            Integer(20, 100, name="num_leaves"),
            Real(0.5, 1.0, name="subsample"),
        ]
        def make(n_estimators, learning_rate, num_leaves, subsample, seed):
            return LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                  num_leaves=num_leaves, subsample=subsample,
                                  random_state=seed, n_jobs=-1, verbose=-1)
        return space, make
    raise ValueError(f"unknown model: {model_name}")


def needs_scaling(model_name: str) -> bool:
    return model_name in {"logistic_regression", "svm"}


# -------------------- one (stock, ablation, horizon, seed) cell --------------------

def evaluate_cell(stock: str,
                  ablation: Dict[str, Any],
                  horizon: int,
                  threshold: float,
                  model_name: str,
                  seed: int,
                  cfg: Dict[str, Any],
                  learned_alpha_news: float | None = None) -> Dict[str, Any]:
    """Tune one (stock, ablation, horizon, model, seed) cell and return both
    val and test metrics. Used by run_hpo() and by the controlled-table generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    feats = data_mod.build_feature_matrix(stock, ablation, cfg,
                                          learned_alpha_news=learned_alpha_news)
    split = data_mod.split_with_embargo(feats, cfg, horizon=horizon, threshold=threshold)

    feat_cols = split.feature_columns
    Xtr = split.train[feat_cols].to_numpy()
    ytr = split.train["y"].to_numpy()
    Xv = split.val[feat_cols].to_numpy()
    yv = split.val["y"].to_numpy()
    Xte = split.test[feat_cols].to_numpy()
    yte = split.test["y"].to_numpy()

    if needs_scaling(model_name):
        sc = StandardScaler().fit(Xtr)
        Xtr, Xv, Xte = sc.transform(Xtr), sc.transform(Xv), sc.transform(Xte)

    space, make = get_search_space(model_name)
    n_calls = cfg["hpo"]["traditional_n_calls"]
    history = []

    def objective(params):
        clf = make(*params, seed=seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xv)[:, 1]
        if len(np.unique(yv)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(yv, p)
        history.append((params, auc))
        return -auc  # skopt minimizes

    # Suppress skopt's "re-proposed already-evaluated point" UserWarning (informational;
    # skopt auto-falls back to a random sample). SVC ConvergenceWarning is filtered at
    # module load (see top of file) because best_clf.fit() below can also hit the cap.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The objective has been evaluated at point",
            category=UserWarning,
            module=r"skopt\.optimizer\.optimizer",
        )
        res = gp_minimize(objective, space, n_calls=n_calls,
                          random_state=seed, verbose=False)
    best_params = res.x
    best_clf = make(*best_params, seed=seed)
    best_clf.fit(Xtr, ytr)

    # VAL metrics
    pv = best_clf.predict_proba(Xv)[:, 1]
    val_auc = roc_auc_score(yv, pv) if len(np.unique(yv)) >= 2 else 0.5
    val_acc = float((pv > 0.5).astype(float).mean() == yv.mean())  # placeholder
    val_acc = float(((pv > 0.5).astype(int) == yv.astype(int)).mean())
    val_prices = split.val["Close"].to_numpy()
    val_trades = bt.long_short_backtest(val_prices, pv, horizon=horizon,
                                        fee_bps_round_trip=cfg["backtest"]["fee_bps_round_trip"],
                                        strategy=cfg["backtest"]["strategy"])
    val_summary = M.summarize(val_trades, cfg["backtest"]["trading_days_per_year"])

    # TEST metrics
    pt = best_clf.predict_proba(Xte)[:, 1]
    test_auc = roc_auc_score(yte, pt) if len(np.unique(yte)) >= 2 else 0.5
    test_acc = float(((pt > 0.5).astype(int) == yte.astype(int)).mean())
    test_prices = split.test["Close"].to_numpy()
    test_trades = bt.long_short_backtest(test_prices, pt, horizon=horizon,
                                         fee_bps_round_trip=cfg["backtest"]["fee_bps_round_trip"],
                                         strategy=cfg["backtest"]["strategy"])
    test_summary = M.summarize(test_trades, cfg["backtest"]["trading_days_per_year"])

    row = {
        "Stock": stock, "Ablation": ablation["name"],
        "Model": model_name, "Horizon": horizon, "Threshold": threshold,
        "Seed": seed,
        "Best_Params": json.dumps({
            n.name if hasattr(n, "name") else f"p{i}": (v if isinstance(v, (int, float, str)) else str(v))
            for i, (n, v) in enumerate(zip(space, best_params))
        }),
        # VAL columns (the ones select_best_config.py reads)
        "Val_Accuracy": val_acc, "Val_ROC_AUC": val_auc,
        "Val_Trades": val_summary["n_trades"], "Val_WinRate": val_summary["win_rate"],
        "Val_Sharpe": val_summary["sharpe"], "Val_AnnualReturn": val_summary["annual_return"],
        "Val_MDD": val_summary["max_drawdown"], "Val_Sortino": val_summary["sortino"],
        # TEST columns (the headline numbers)
        "Test_Accuracy": test_acc, "Test_ROC_AUC": test_auc,
        "Test_Trades": test_summary["n_trades"], "Test_WinRate": test_summary["win_rate"],
        "Test_Sharpe": test_summary["sharpe"], "Test_AnnualReturn": test_summary["annual_return"],
        "Test_MDD": test_summary["max_drawdown"], "Test_Sortino": test_summary["sortino"],
        "Test_Calmar": test_summary["calmar"],
    }
    # also dump trade-level returns for downstream bootstrap CIs
    row["__val_trade_returns__"] = list(map(float, val_trades))
    row["__test_trade_returns__"] = list(map(float, test_trades))
    return row


# -------------------- driver --------------------

def _load_learned_alpha() -> Dict[str, Any]:
    """Load per-stock + global learned α from results/learned_alpha.json.

    Notebook 02 is the canonical writer (rich schema with `per_stock`/`global`).
    Falls back to the pipeline-written `learned_alpha_pipeline.json` (flat array
    of `{stock, alpha_news, scope}` records) so the pipeline is runnable even
    when notebook 02 hasn't been executed.
    """
    rich = cfg_mod.results_dir() / "learned_alpha.json"
    slim = cfg_mod.results_dir() / "learned_alpha_pipeline.json"
    if rich.exists():
        data = json.loads(rich.read_text())
        return {
            "per_stock": {r["stock"]: r["alpha_news"] for r in data["per_stock"]},
            "global": data["global"]["alpha_news"],
        }
    if slim.exists():
        data = json.loads(slim.read_text())
        return {
            "per_stock": {r["stock"]: r["alpha_news"]
                          for r in data if r["scope"] == "per_stock"},
            "global": next(r["alpha_news"] for r in data if r["scope"] == "global"),
        }
    raise FileNotFoundError(
        f"learned-α not available — run notebook 02 (writes {rich.name}) or "
        f"`run_pipeline --steps 1` (writes {slim.name}) before step 2."
    )


def _resolve_learned_alpha(ab: Dict[str, Any], stock: str,
                           learned: Dict[str, Any]) -> float | None:
    if ab.get("sentiment") != "learned":
        return None
    scope = ab.get("learned_scope", "per_stock")
    return learned["per_stock"][stock] if scope == "per_stock" else learned["global"]


def run_hpo(model_name: str,
            stocks: List[str],
            ablations: List[Dict[str, Any]],
            cfg: Dict[str, Any],
            seeds: List[int]) -> pd.DataFrame:
    needs_learned = any(ab.get("sentiment") == "learned" for ab in ablations)
    learned = _load_learned_alpha() if needs_learned else None

    rows = []
    for stock in stocks:
        for ab in ablations:
            alpha = _resolve_learned_alpha(ab, stock, learned) if learned else None
            for h in cfg["horizons"]:
                # one threshold per horizon — pick the one that gives the most
                # balanced training labels (search the small grid)
                thr = pick_threshold_for_balance(stock, h, cfg)
                for seed in seeds:
                    print(f"  HPO {model_name} | {stock} | {ab['name']} | h={h} | seed={seed}")
                    row = evaluate_cell(stock, ab, h, thr, model_name, seed, cfg,
                                        learned_alpha_news=alpha)
                    rows.append({k: v for k, v in row.items()
                                 if not k.startswith("__")})
    return pd.DataFrame(rows)


def pick_threshold_for_balance(stock: str, horizon: int, cfg: Dict[str, Any]) -> float:
    """Find the label threshold in cfg['label_threshold_grid'] that most balances
    the binary labels on the TRAINING window.
    """
    feats = data_mod.load_training_features(stock)
    feats = feats[feats["Year"].isin(cfg["train_years"])]
    best = (None, 1.0)
    for t in cfg["label_threshold_grid"]:
        y = data_mod.make_target(feats, horizon=horizon, threshold=t)
        y = y.dropna()
        if len(y) == 0:
            continue
        imbalance = abs(y.mean() - 0.5)
        if imbalance < best[1]:
            best = (t, imbalance)
    return float(best[0]) if best[0] is not None else 0.005


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["logistic_regression", "svm", "random_forest",
                                 "gradient_boosting", "xgboost", "lightgbm"])
    parser.add_argument("--stock", default=None, help="single stock or omit for all")
    parser.add_argument("--ablation", default=None,
                        help="single ablation name or omit for all")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="override ml_seeds from config")
    parser.add_argument("--out", default=None,
                        help="output CSV path (default: results/raw/{model}.csv)")
    args = parser.parse_args()

    cfg = cfg_mod.load_config()
    stocks = [args.stock] if args.stock else cfg_mod.stocks(cfg)
    ablations = ([ab for ab in cfg["ablations"] if ab["name"] == args.ablation]
                 if args.ablation else list(cfg["ablations"]))
    seeds = args.seeds if args.seeds else cfg_mod.ml_seeds(cfg)

    out_path = Path(args.out) if args.out else (cfg_mod.results_dir() / "raw" / f"{args.model}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = run_hpo(args.model, stocks, ablations, cfg, seeds)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
