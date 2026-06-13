"""Learned-alpha sentiment fusion (Section 3.1 of the revision plan).

Replaces the fixed 70:30 news/social weighting with a learned scalar
`alpha_news in [0, 1]`, optimized on the *validation* set per stock (or
globally, by aggregating per-stock val AUC), with an L2 prior pulling
toward 0.7 so the fixed 70:30 emerges as a special case.

Per-source-mean fusion (matches `src/data.py::compose_sentiment`):

    S_t(alpha) = alpha * News_Sentiment_t + (1 - alpha) * Social_Sentiment_t

The objective is validation ROC-AUC of a simple linear classifier on
the per-stock indicator columns + S_t. Per-stock indicator names differ
(R1 Appendix A: each stock has its own selected pairs) so the global fit
optimizes a *shared* alpha by maximizing the mean of per-stock val AUCs,
rather than pooling DataFrames with mismatched columns.

Optimization uses an exact grid search over alpha in [0, 1] (default 101
points, resolution 0.01). The val-AUC landscape is multimodal on small
noisy datasets, so Brent's method finds inconsistent local optima — the
grid is cheap (~100 logistic-regression fits per stock) and guaranteed
exact on its resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config as cfg_mod
from . import data as data_mod


N_GRID = 101  # alpha grid resolution: 101 points on [0, 1] -> step 0.01


@dataclass
class LearnedAlphaResult:
    stock: Optional[str]      # None for the global fit
    alpha_news: float
    val_auc: float
    prior_alpha_news: float
    prior_l2: float


def _fit_eval(train_df: pd.DataFrame,
              val_df: pd.DataFrame,
              indicator_cols: List[str]) -> Tuple[float, float]:
    """Train a Logistic Regression with the current S_t and return (val_auc, val_loss)."""
    Xtr = train_df[indicator_cols + ["S_t"]].to_numpy()
    ytr = train_df["y"].to_numpy()
    Xv = val_df[indicator_cols + ["S_t"]].to_numpy()
    yv = val_df["y"].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xv = scaler.transform(Xv)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xv)[:, 1]
    if len(np.unique(yv)) < 2:
        return 0.5, 1.0
    auc = roc_auc_score(yv, p)
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    loss = -np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p))
    return float(auc), float(loss)


def _apply_alpha(df: pd.DataFrame, alpha_news: float) -> pd.Series:
    """Per-source-mean S_t for a given alpha. NaNs in either channel -> 0."""
    s_news = df[data_mod.NEWS_COL].fillna(0.0)
    s_social = df[data_mod.SOCIAL_COL].fillna(0.0)
    return alpha_news * s_news + (1.0 - alpha_news) * s_social


def _prepare_stock(stock: str,
                   cfg: Dict,
                   horizon: int,
                   threshold: float
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load + encode features and produce (train_df, val_df, indicator_cols) for one stock."""
    feats = data_mod.load_training_features(stock)
    indicator_cols = data_mod.get_indicator_columns(feats)
    feats = data_mod.encode_indicator_signals(feats, indicator_cols)
    feats["y"] = data_mod.make_target(feats, horizon=horizon, threshold=threshold)
    feats = feats.dropna(subset=["y"]).reset_index(drop=True)

    train_df = feats[feats["Year"].isin(cfg["train_years"])].copy()
    val_df = feats[feats["Year"] == cfg["val_year"]].copy()
    if len(val_df) > 0:
        val_df = val_df.iloc[: max(0, len(val_df) - horizon)].copy()
    return train_df, val_df, indicator_cols


def alpha_curve(stock: str,
                cfg: Dict,
                horizon: int = 5,
                threshold: float = 0.005,
                n_grid: int = N_GRID) -> pd.DataFrame:
    """Per-alpha val-AUC curve for one stock — the building block of every fit/eval.

    Returns a DataFrame with columns `alpha_news` and `val_auc`. The val-AUC
    landscape is multimodal on small noisy datasets; evaluating the full grid
    once and then applying regularization analytically avoids Brent's local-
    optimum traps.
    """
    train_df, val_df, indicator_cols = _prepare_stock(stock, cfg, horizon, threshold)
    alphas = np.linspace(0.0, 1.0, n_grid)
    aucs = np.empty_like(alphas)
    for i, a in enumerate(alphas):
        tr = train_df.copy()
        vl = val_df.copy()
        tr["S_t"] = _apply_alpha(tr, a).values
        vl["S_t"] = _apply_alpha(vl, a).values
        auc, _ = _fit_eval(tr, vl, indicator_cols)
        aucs[i] = auc
    return pd.DataFrame({"alpha_news": alphas, "val_auc": aucs})


def _argmin_on_curve(curve: pd.DataFrame,
                     prior_alpha_news: float,
                     prior_l2: float) -> Tuple[float, float]:
    """Apply the L2 prior analytically to a precomputed curve and return (alpha*, val_auc*)."""
    a = curve["alpha_news"].to_numpy()
    auc = curve["val_auc"].to_numpy()
    obj = -auc + prior_l2 * (a - prior_alpha_news) ** 2
    idx = int(np.argmin(obj))
    return float(a[idx]), float(auc[idx])


def eval_at_alpha(stock: str,
                  cfg: Dict,
                  alpha_news: float,
                  horizon: int = 5,
                  threshold: float = 0.005) -> float:
    """Validation AUC for a single stock at a fixed alpha (no optimization, no prior)."""
    train_df, val_df, indicator_cols = _prepare_stock(stock, cfg, horizon, threshold)
    train_df["S_t"] = _apply_alpha(train_df, alpha_news).values
    val_df["S_t"] = _apply_alpha(val_df, alpha_news).values
    val_auc, _ = _fit_eval(train_df, val_df, indicator_cols)
    return val_auc


def fit_alpha_per_stock(stock: str,
                        cfg: Dict,
                        horizon: int = 5,
                        threshold: float = 0.005,
                        prior_alpha_news: float = 0.7,
                        prior_l2: float = 0.1,
                        curve: Optional[pd.DataFrame] = None) -> LearnedAlphaResult:
    """Fit alpha_news per-stock on the 2023 validation slice.

    Pass a precomputed `curve` (from `alpha_curve(stock, cfg, ...)`) to amortize
    the LogReg fits across multiple lambda values (e.g. for a sensitivity scan).
    """
    if curve is None:
        curve = alpha_curve(stock, cfg, horizon=horizon, threshold=threshold)
    alpha, val_auc = _argmin_on_curve(curve, prior_alpha_news, prior_l2)
    return LearnedAlphaResult(stock=stock, alpha_news=alpha, val_auc=val_auc,
                              prior_alpha_news=prior_alpha_news, prior_l2=prior_l2)


def fit_alpha_global(cfg: Dict,
                     horizon: int = 5,
                     threshold: float = 0.005,
                     prior_alpha_news: float = 0.7,
                     prior_l2: float = 0.1,
                     curves: Optional[Dict[str, pd.DataFrame]] = None) -> LearnedAlphaResult:
    """Fit a single alpha_news shared across all stocks.

    Aggregation objective: mean of per-stock val AUCs at the candidate alpha,
    minus the L2 prior penalty. Pass `curves={stock: alpha_curve(...)}` to
    amortize across lambdas.
    """
    if curves is None:
        curves = {s: alpha_curve(s, cfg, horizon=horizon, threshold=threshold)
                  for s in cfg_mod.stocks(cfg)}

    # All curves share the same alpha grid; mean across stocks at each grid point.
    alphas = next(iter(curves.values()))["alpha_news"].to_numpy()
    mean_auc = np.mean([c["val_auc"].to_numpy() for c in curves.values()], axis=0)
    mean_curve = pd.DataFrame({"alpha_news": alphas, "val_auc": mean_auc})
    alpha, val_auc = _argmin_on_curve(mean_curve, prior_alpha_news, prior_l2)
    return LearnedAlphaResult(stock=None, alpha_news=alpha, val_auc=val_auc,
                              prior_alpha_news=prior_alpha_news, prior_l2=prior_l2)
