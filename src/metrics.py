"""Statistical + trading metrics required by the revision plan.

Specifically (Sections 2.2, 2.4):
  - Classification: accuracy, ROC-AUC
  - Trading: number of trades, win rate, Sharpe, annual return, MDD, Calmar, Sortino
  - Uncertainty: stationary block bootstrap (Politis-Romano) CIs
  - Selection-bias-aware: Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014),
                          Probability of Backtest Overfitting (PBO)

All functions take *trade-level returns* arrays (not daily portfolio NAVs).
A trade-level return is the realized return on each non-overlapping
horizon-h position, net of transaction costs.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


# -------------------- single-run trading metrics --------------------

def sharpe_ratio(returns: np.ndarray) -> float:
    """Sharpe = (mean / std) * sqrt(N). Matches the paper's definition."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2 or r.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=0) * np.sqrt(len(r)))


def annual_return(returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    """Compound return annualized assuming the trades cover one calendar year."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    total = float(np.prod(1.0 + r) - 1.0)
    # We assume trades are non-overlapping h-day positions over ~1 test year;
    # the (1+total)^(252/N) form scales N-trade total to annualized.
    return float((1.0 + total) ** (trading_days_per_year / max(len(r), 1)) - 1.0)


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def sortino_ratio(returns: np.ndarray) -> float:
    """Sharpe variant using only downside deviation."""
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    if len(downside) == 0 or downside.std(ddof=0) == 0:
        return 0.0
    return float(r.mean() / downside.std(ddof=0) * np.sqrt(len(r)))


def calmar_ratio(returns: np.ndarray, trading_days_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annual_return(returns, trading_days_per_year) / abs(mdd))


def win_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    return float((r > 0).mean()) if len(r) else 0.0


def trade_count(returns: np.ndarray) -> int:
    return int(len(np.asarray(returns)))


# -------------------- block bootstrap CIs (Politis-Romano stationary bootstrap) --------------------

def stationary_bootstrap_indices(
    n: int,
    block_avg_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one resampled index sequence of length n using stationary bootstrap.

    Block lengths are geometric with mean `block_avg_len`, and the start of each
    block is a uniform random index. Wraps around at n.
    """
    p = 1.0 / max(int(block_avg_len), 1)
    idx = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = rng.integers(0, n)
        # geometric block length (at least 1)
        L = int(rng.geometric(p))
        end = min(i + L, n)
        for k in range(end - i):
            idx[i + k] = (start + k) % n
        i = end
    return idx


def bootstrap_metric_ci(
    returns: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    block_avg_len: Optional[int] = None,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (point, lower, upper) for the given metric using stationary bootstrap.

    `metric_fn` accepts a 1-D np.ndarray of returns and returns a scalar.
    `block_avg_len` defaults to floor(N**0.5).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    if block_avg_len is None:
        block_avg_len = max(1, int(np.floor(np.sqrt(len(r)))))
    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(len(r), block_avg_len, rng)
        boot_vals[b] = metric_fn(r[idx])
    lo = float(np.quantile(boot_vals, (1.0 - confidence) / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - (1.0 - confidence) / 2.0))
    return float(metric_fn(r)), lo, hi


# -------------------- selection-bias-aware metrics --------------------

def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    n_trades: int,
    skew_trade_returns: float,
    excess_kurtosis_trade_returns: float,
) -> float:
    """Bailey & Lopez de Prado 2014: Deflated Sharpe Ratio.

    Returns the probability that the *observed* Sharpe is greater than the
    Sharpe achievable by the best of `n_trials` independent random strategies,
    given the moments of the trade-return distribution.

    Implementation matches eqs. (8)-(9) of the paper.
    """
    if n_trials <= 1 or n_trades < 4:
        return 0.5  # not enough data to compute
    # Expected maximum Sharpe under the null over N trials (Mertens approximation):
    gamma = 0.5772156649  # Euler-Mascheroni
    E_max = (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials) + \
            gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    # Std of Sharpe estimator with non-normal returns (Mertens / Lo):
    s_var = (
        1.0
        - skew_trade_returns * sharpe_observed
        + ((excess_kurtosis_trade_returns) / 4.0) * (sharpe_observed ** 2)
    )
    s_var = max(s_var, 0.0)
    s_std = np.sqrt(s_var / max(n_trades - 1, 1))
    if s_std == 0:
        return 0.5
    z = (sharpe_observed - E_max) / s_std
    return float(stats.norm.cdf(z))


def probability_of_backtest_overfitting(
    is_oos_pairs: List[Tuple[float, float]],
) -> float:
    """Bailey, Borwein, Lopez de Prado & Zhu PBO (CSCV variant simplified).

    `is_oos_pairs` is a list of (in_sample_metric, out_of_sample_metric) pairs
    from CSCV partitions. Returns the probability that the IS-best configuration
    underperforms the OOS median.
    """
    if len(is_oos_pairs) < 2:
        return 0.5
    rank_pairs = [(rs, ro) for rs, ro in is_oos_pairs]
    # For each pair, see if IS-best is OOS-below-median:
    # Simplified: rank logits.
    is_ranks = stats.rankdata([p[0] for p in rank_pairs])
    oos_ranks = stats.rankdata([p[1] for p in rank_pairs])
    n = len(rank_pairs)
    logits = []
    for i in range(n):
        # logit of the OOS rank of the IS-best (highest IS rank) pair
        if is_ranks[i] == n:
            oos = oos_ranks[i]
            # avoid extreme logits
            frac = max(min(oos / (n + 1), 0.999), 0.001)
            logits.append(np.log(frac / (1 - frac)))
    if not logits:
        return 0.5
    # PBO = fraction of logits below zero, i.e. OOS rank below median
    return float(np.mean(np.array(logits) < 0))


# -------------------- summary helper --------------------

def summarize(returns: np.ndarray, trading_days_per_year: int = 252) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float)
    return {
        "n_trades": trade_count(r),
        "win_rate": win_rate(r),
        "sharpe": sharpe_ratio(r),
        "sortino": sortino_ratio(r),
        "annual_return": annual_return(r, trading_days_per_year),
        "max_drawdown": max_drawdown(r),
        "calmar": calmar_ratio(r, trading_days_per_year),
        "mean_trade_return": float(r.mean()) if len(r) else 0.0,
        "std_trade_return": float(r.std(ddof=0)) if len(r) else 0.0,
    }
