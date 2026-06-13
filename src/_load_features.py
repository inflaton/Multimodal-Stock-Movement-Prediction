"""Shared data loader for the bundled foundation-model scripts.

Reads `<data_dir>/<STOCK>.csv` from `2026_icdm/dataset/training_features/`
(the canonical per-stock feature CSV used by `src.data`) and produces a
DataFrame with the columns the foundation scripts (chronos_baseline.py,
chronos_finetune.py, fincast_baseline.py, fincast_finetune.py) consume:

  Date, Close, Sentiment_S_t, __DateDT__, Year

`Sentiment_S_t` is the per-source-mean fusion
   S_t = alpha_news · News_Sentiment_Score + (1 - alpha_news) · Social_Sentiment_Score
matching `src.data.compose_sentiment`, so the foundation comparison uses the
same sentiment fusion as the LSTM and traditional-ML grids (learned-α from
`results/learned_alpha.json`).

The original `paper/data/{stock}_data_model_training.csv` schema (with
`Weighted Sentiment Score`, `Filtered Sentiment Score`, and `Combi N Days`
columns) is *not* recreated — the legacy `Combi N Days` columns are never
read by the foundation scripts, and `Weighted/Filtered Sentiment Score` are
replaced by the configurable `Sentiment_S_t` covariate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

NEWS_COL = "News Sentiment Score"
SOCIAL_COL = "Social Sentiment Score"
SENTIMENT_OUT = "Sentiment_S_t"

# Default split — keep in sync with 2026_icdm/configs/default.yaml.
# Hard-coded here so the bundled scripts can run without depending on the
# `src.config` module's package layout.
DEFAULT_TRAIN_YEARS: List[int] = [2020, 2021, 2022]
DEFAULT_VAL_YEAR: int = 2023
DEFAULT_TEST_YEAR: int = 2024
DEFAULT_EMBARGO_DAYS: int = 5


def load_features(stock: str, data_dir: str,
                  alpha_news: float = 0.7) -> pd.DataFrame:
    """Load training_features/<STOCK>.csv and add `Sentiment_S_t` at `alpha_news`.

    Args:
        stock: ticker (filename stem in `data_dir`).
        data_dir: path containing `<STOCK>.csv` (e.g. `dataset/training_features`).
        alpha_news: news weight in [0, 1]; 0.7 is the legacy fixed weighting.
            Pass the value from `results/learned_alpha.json` to keep the
            foundation comparison aligned with the rest of the pipeline.

    Returns:
        DataFrame with at minimum: Date (string, dd/mm/YYYY), Close (float),
        Sentiment_S_t (float, NaN-imputed to 0), __DateDT__ (datetime), Year (int).
        Other columns from the source CSV (Combi N indicator strings) are kept
        in case downstream code wants them.
    """
    if not 0.0 <= alpha_news <= 1.0:
        raise ValueError(f"alpha_news must be in [0, 1], got {alpha_news}")
    fp = Path(data_dir) / f"{stock}.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} not found. The foundation scripts now read training_features/; "
            f"pass --data-dir dataset/training_features when running."
        )
    df = pd.read_csv(fp)
    # training_features stores dates as dd/mm/YYYY, same as the legacy CSVs.
    df["__DateDT__"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    if df["__DateDT__"].isna().any():
        df["__DateDT__"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["__DateDT__"].dt.year

    if NEWS_COL not in df.columns or SOCIAL_COL not in df.columns:
        raise KeyError(
            f"Expected {NEWS_COL!r} and {SOCIAL_COL!r} in {fp}; got {list(df.columns)}"
        )
    s_news = df[NEWS_COL].fillna(0.0)
    s_social = df[SOCIAL_COL].fillna(0.0)
    df[SENTIMENT_OUT] = alpha_news * s_news + (1.0 - alpha_news) * s_social
    return df


def add_alpha_news_arg(parser, default: float = 0.7) -> None:
    """Argparse helper — every foundation script gets the same flag."""
    parser.add_argument(
        "--alpha-news", type=float, default=default,
        help=("News/social fusion weight in [0,1]. 0.7 reproduces the legacy "
              "70:30 weighting; pass the value from results/learned_alpha.json "
              "(per-stock or global) to align with the rest of the pipeline."),
    )


def add_split_args(parser) -> None:
    """Argparse helper — train/val/test split + embargo, matches configs/default.yaml."""
    parser.add_argument(
        "--train-years", type=int, nargs="+", default=DEFAULT_TRAIN_YEARS,
        help="Years that constitute the training slice (default: 2020 2021 2022).",
    )
    parser.add_argument(
        "--val-year", type=int, default=DEFAULT_VAL_YEAR,
        help="Year that constitutes the validation slice (default: 2023).",
    )
    parser.add_argument(
        "--test-year", type=int, default=DEFAULT_TEST_YEAR,
        help="Year that constitutes the test slice (default: 2024).",
    )
    parser.add_argument(
        "--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS,
        help="Trading-day gap dropped from the start of the test slice (default: 5).",
    )


@dataclass
class TrainValTestSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def train_val_test_split(df: pd.DataFrame,
                         train_years: Iterable[int] = DEFAULT_TRAIN_YEARS,
                         val_year: int = DEFAULT_VAL_YEAR,
                         test_year: int = DEFAULT_TEST_YEAR,
                         embargo_days: int = DEFAULT_EMBARGO_DAYS,
                         ) -> TrainValTestSplit:
    """Split a feature frame on `Year` into train / val / test with an embargo.

    Matches `src.data.split_with_embargo`:
      - train = rows whose Year is in `train_years`
      - val   = rows whose Year is `val_year`
      - test  = rows whose Year is `test_year`, then drop the first
                `embargo_days` rows so val labels can't leak into test

    Each returned frame is sorted by `__DateDT__` and re-indexed.
    """
    if "Year" not in df.columns or "__DateDT__" not in df.columns:
        raise KeyError("Input frame needs `Year` and `__DateDT__` columns "
                       "(produced by `load_features`).")
    train_years = set(int(y) for y in train_years)
    val_year = int(val_year)
    test_year = int(test_year)
    embargo_days = max(0, int(embargo_days))

    train = (df[df["Year"].isin(train_years)]
             .sort_values("__DateDT__").reset_index(drop=True))
    val = (df[df["Year"] == val_year]
           .sort_values("__DateDT__").reset_index(drop=True))
    test = (df[df["Year"] == test_year]
            .sort_values("__DateDT__").reset_index(drop=True))
    if embargo_days > 0 and len(test) > embargo_days:
        test = test.iloc[embargo_days:].reset_index(drop=True)
    return TrainValTestSplit(train=train, val=val, test=test)
