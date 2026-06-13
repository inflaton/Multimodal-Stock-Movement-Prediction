"""Trade simulator: long/short, non-overlapping h-day positions, transaction costs.

Matches the original paper's evaluation protocol:
  - For each test-row prediction prob_t (or label), decide direction at time t.
  - Hold for `horizon` trading days, then close.
  - Skip the next horizon-1 rows while position is open (non-overlapping).
  - Apply `fee_bps_round_trip` basis points to every executed trade.

Returns a numpy array of realized per-trade returns (net of fees), ready to
feed into src.metrics.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def long_short_backtest(
    prices: np.ndarray,
    predictions: np.ndarray,
    horizon: int,
    fee_bps_round_trip: float = 10.0,
    confidence_threshold: float = 0.5,
    strategy: str = "long_short",
) -> np.ndarray:
    """Run a non-overlapping h-day position backtest.

    Args:
        prices: 1-D array of closing prices in chronological order, len T.
        predictions: 1-D array of predicted probabilities P(up) of length T.
                     If you have hard labels, pass {0.0, 1.0}.
        horizon: holding period h (days).
        fee_bps_round_trip: round-trip transaction cost in bps (e.g. 10 = 0.10%).
        confidence_threshold: prediction prob threshold for going long
                              (1 - threshold for going short in long/short mode).
        strategy: one of:
            'long_only'             : only go long when prob > threshold
            'long_short'            : long if prob > 0.5, short if prob < 0.5
            'long_only_confidence'  : only go long if prob > threshold
            'long_short_confidence' : only trade if |prob - 0.5| > (threshold - 0.5)

    Returns:
        np.ndarray of per-trade *realized* returns (after fees).
    """
    prices = np.asarray(prices, dtype=float)
    preds = np.asarray(predictions, dtype=float)
    assert prices.shape == preds.shape
    fee = fee_bps_round_trip / 10000.0

    trade_returns: List[float] = []
    i = 0
    T = len(prices)
    while i + horizon < T:
        p = preds[i]
        direction = 0  # -1 short, 0 skip, +1 long
        if strategy == "long_only":
            if p > 0.5:
                direction = 1
        elif strategy == "long_short":
            direction = 1 if p > 0.5 else -1
        elif strategy == "long_only_confidence":
            if p > confidence_threshold:
                direction = 1
        elif strategy == "long_short_confidence":
            band = confidence_threshold - 0.5
            if p > 0.5 + band:
                direction = 1
            elif p < 0.5 - band:
                direction = -1
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        if direction == 0:
            i += 1
            continue

        entry = prices[i]
        exit_ = prices[i + horizon]
        gross = (exit_ - entry) / entry * direction
        net = gross - fee   # round-trip fee once per trade
        trade_returns.append(net)
        i += horizon  # non-overlapping

    return np.asarray(trade_returns, dtype=float)


def backtest_from_test_frame(
    test_df: pd.DataFrame,
    pred_col: str,
    horizon: int,
    fee_bps_round_trip: float = 10.0,
    strategy: str = "long_short",
    confidence_threshold: float = 0.5,
) -> np.ndarray:
    """Convenience wrapper that expects a DataFrame with Close + prediction column."""
    prices = test_df["Close"].to_numpy()
    preds = test_df[pred_col].to_numpy()
    return long_short_backtest(
        prices=prices,
        predictions=preds,
        horizon=horizon,
        fee_bps_round_trip=fee_bps_round_trip,
        confidence_threshold=confidence_threshold,
        strategy=strategy,
    )
