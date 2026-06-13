"""Data loading + chronological split with embargo.

Implements Section 2.1 of the revision plan:
  - Train = train_years (default 2020-2022)
  - Val   = val_year    (default 2023)
  - Test  = test_year   (default 2024)
  - Embargo: drop the first `embargo_trading_days` of test so val and test
    are separated by the embargo (mitigates label leakage near the boundary).

Sentiment is fused via per-source-mean:
    S_t = alpha_news * News_t + (1 - alpha_news) * Social_t

News_t and Social_t are the per-trading-day mean FinBERT-Tone scores stored
directly in `training_features/<STOCK>.csv` (columns `News Sentiment Score`
and `Social Sentiment Score`). Days with no articles of a given type get NaN
in the CSV and are imputed as 0 ("no signal = neutral") when composing S_t.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from . import config as cfg_mod


# -------------------- raw loader --------------------

NEWS_COL = "News Sentiment Score"
SOCIAL_COL = "Social Sentiment Score"


def load_training_features(stock: str) -> pd.DataFrame:
    """Load the per-stock training feature CSV.

    Columns: Date, Close, Combi 1..10 (Buy/Sell/Hold), Combi N Days,
             News Sentiment Score, Social Sentiment Score.
    """
    fp = cfg_mod.training_features_dir() / f"{stock}.csv"
    df = pd.read_csv(fp)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    if df["Date"].isna().any():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Year"] = df["Date"].dt.year
    return df


# -------------------- sentiment recomposition --------------------

def compose_sentiment(
    df: pd.DataFrame,
    ablation: Dict[str, Any],
    learned_alpha_news: Optional[float] = None,
) -> pd.Series:
    """Per-source-mean fusion of News and Social sentiment.

    Inputs:
      df: a DataFrame with columns `News Sentiment Score` and `Social Sentiment Score`.
      ablation: ablation dict from configs/default.yaml.
      learned_alpha_news: required if ablation['sentiment'] == 'learned'.

    Returns a pd.Series of S_t indexed by df's index, NaN imputed as 0
    (no articles of that type on the day -> neutral signal).
    """
    rule = ablation["sentiment"]
    s_news = df[NEWS_COL].fillna(0.0)
    s_social = df[SOCIAL_COL].fillna(0.0)

    if rule == "weighted":
        a = float(ablation["alpha_news"])
        out = a * s_news + (1.0 - a) * s_social
    elif rule == "learned":
        if learned_alpha_news is None:
            raise ValueError(
                f"ablation {ablation['name']} requires a learned alpha, but none was passed in"
            )
        a = float(learned_alpha_news)
        out = a * s_news + (1.0 - a) * s_social
    elif rule == "news_only":
        out = s_news
    elif rule == "social_only":
        out = s_social
    elif rule == "none":
        out = pd.Series(0.0, index=df.index)
    else:
        raise ValueError(f"unknown sentiment rule: {rule}")
    out.name = "S_t"
    return out


# -------------------- feature construction --------------------

SIGNAL_ENCODING = {"Buy": 1, "Sell": -1, "Hold": 0}


def get_indicator_columns(df: pd.DataFrame) -> List[str]:
    """Detect the 10 indicator-pair columns in a training_features DataFrame.

    Per-stock indicator selection is part of the legacy pipeline (R1 Appendix A),
    so the column names differ across stocks. TSLA additionally has whitespace
    quirks in its column headers ('Combi1:' with no space, embedded newlines).
    Detect by prefix instead of hardcoding names.
    """
    return [c for c in df.columns if c.lstrip().startswith("Combi") and "Days" not in c]


def encode_indicator_signals(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert Buy/Sell/Hold strings to {+1, -1, 0}. NaNs become 0 (no signal)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0).astype(int)
        else:
            out[c] = out[c].map(SIGNAL_ENCODING).fillna(0).astype(int)
    return out


def build_feature_matrix(
    stock: str,
    ablation: Dict[str, Any],
    cfg: Dict[str, Any],
    learned_alpha_news: Optional[float] = None,
) -> pd.DataFrame:
    """Build the feature matrix for one (stock, ablation).

    Returns a DataFrame with columns:
      Date, Year, Close, [10 indicator cols if drop_technical=False], S_t
    """
    feats = load_training_features(stock)

    if ablation["sentiment"] != "none":
        feats["S_t"] = compose_sentiment(feats, ablation, learned_alpha_news=learned_alpha_news).values
    else:
        feats["S_t"] = 0.0

    keep = ["Date", "Year", "Close"]
    if not ablation.get("drop_technical", False):
        cols = get_indicator_columns(feats)
        keep += cols
        feats = encode_indicator_signals(feats, cols)
    keep += ["S_t"]
    return feats[keep].copy()


# -------------------- target construction --------------------

def make_target(
    df: pd.DataFrame,
    horizon: int,
    threshold: float,
) -> pd.Series:
    """Binary label: 1 if (Close[t+h] - Close[t]) / Close[t] > threshold else 0.

    The last `horizon` rows have NaN labels (no future to look at).
    """
    fut = df["Close"].shift(-horizon)
    ret = (fut - df["Close"]) / df["Close"]
    y = (ret > threshold).astype(float)
    y[ret.isna()] = np.nan
    return y


# -------------------- chronological split --------------------

@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    horizon: int
    threshold: float

    @property
    def feature_columns(self) -> List[str]:
        drop = {"Date", "Year", "Close", "y"}
        return [c for c in self.train.columns if c not in drop]


def split_with_embargo(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    horizon: int,
    threshold: float,
) -> Split:
    """Apply the 2020-22 / 2023 / 2024 calendar split with embargo before test.

    The test split also drops the `horizon` rows at the end of val so that
    the val labels (which look `horizon` days into the future) don't peek
    into the test period.
    """
    train_years = set(cfg["train_years"])
    val_year = cfg["val_year"]
    test_year = cfg["test_year"]
    embargo = int(cfg.get("embargo_trading_days", 5))

    df = df.copy()
    df["y"] = make_target(df, horizon=horizon, threshold=threshold)

    train_df = df[df["Year"].isin(train_years)].copy()
    val_df = df[df["Year"] == val_year].copy()
    test_df = df[df["Year"] == test_year].copy()

    # Drop the tail of val whose labels reach into test
    if len(val_df) > 0:
        val_df = val_df.iloc[: max(0, len(val_df) - horizon)].copy()

    # Embargo: drop the first `embargo` rows of test
    if embargo > 0 and len(test_df) > embargo:
        test_df = test_df.iloc[embargo:].copy()

    # Drop rows with NaN labels (end of train/val/test from the shift)
    for s in (train_df, val_df, test_df):
        s.dropna(subset=["y"], inplace=True)

    return Split(train=train_df, val=val_df, test=test_df,
                 horizon=horizon, threshold=threshold)
