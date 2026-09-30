"""Statistical machinery for separating skill from luck: PSR, Deflated Sharpe Ratio, Sharpe-difference test,
and Monte Carlo resampling.

All Sharpe ratios inside PSR/DSR are PER-PERIOD (not annualised), matching Bailey & Lopez de Prado
(2012, "The Sharpe Ratio Efficient Frontier"; 2014, "The Deflated Sharpe Ratio").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def sr_period(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    s = x.std(ddof=1)
    return float(x.mean() / s) if len(x) > 2 and s > 0 else float("nan")


def psr(sr_hat: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star). `kurt` is NON-excess kurtosis (normal = 3)."""
    denom = 1.0 - skew * sr_hat + (kurt - 1.0) / 4.0 * sr_hat**2
    if n < 3 or denom <= 0 or not np.isfinite(sr_hat):
        return float("nan")
    return float(sps.norm.cdf((sr_hat - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)))


def expected_max_sr(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe among `n_trials` independent trials whose true SR is zero."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    a = (1.0 - EULER_GAMMA) * sps.norm.ppf(1.0 - 1.0 / n_trials)
    b = EULER_GAMMA * sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * (a + b))


def dsr(excess_returns: pd.Series, n_trials: int, sr_variance: float) -> dict:
    """Deflated Sharpe Ratio of a return series (excess of rf), given the number of strategy variants tried and the
    cross-trial variance of per-period Sharpe ratios. Returns probability of skill (must be >= 0.95 to pass)."""
    x = pd.Series(excess_returns).dropna()
    sr = sr_period(x)
    skew, kurt = float(sps.skew(x)), float(sps.kurtosis(x, fisher=False))
    sr0 = expected_max_sr(n_trials, sr_variance)
    return {"sr_period": sr, "sr_annualized_equiv": sr * np.sqrt(252), "n_obs": len(x), "n_trials": n_trials,
            "sr_variance_across_trials": sr_variance, "sr0_expected_max": sr0, "sr0_annualized_equiv": sr0 * np.sqrt(252),
            "psr_vs_zero": psr(sr, 0.0, len(x), skew, kurt), "dsr": psr(sr, sr0, len(x), skew, kurt)}


def sharpe_diff_test(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Jobson-Korkie test with Memmel's correction: H1 = SR(a) > SR(b). Inputs are excess returns on the same dates.
    Returns (z, one-sided p). Assumes iid normal returns (approximation)."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    n = len(df)
    if n < 10:
        return float("nan"), float("nan")
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    sa, sb = x.mean() / x.std(ddof=1), y.mean() / y.std(ddof=1)
    rho = float(np.corrcoef(x, y)[0, 1])
    v = (2 - 2 * rho + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / n
    z = (sa - sb) / np.sqrt(v)
    return float(z), float(1 - sps.norm.cdf(z))


# --------------------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------------------
def _dd_from_equity(eq: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(np.maximum(eq, 1.0), axis=1)
    return (eq / peak - 1.0).min(axis=1)


def mc_trades(trade_pnl_frac, n_sims: int = 10_000, n_trades: int | None = None, seed: int = 0, chunk: int = 2_000) -> dict:
    """Resample trades with replacement. `trade_pnl_frac` = each trade's P&L as a fraction of account equity.
    Equity compounds multiplicatively across the resampled sequence. Reports the 5th percentile outcome."""
    x = np.asarray(trade_pnl_frac, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {"error": f"only {len(x)} trades; Monte Carlo not meaningful"}
    n = n_trades or len(x)
    rng = np.random.default_rng(seed)
    finals, dds = [], []
    for s in range(0, n_sims, chunk):
        m = min(chunk, n_sims - s)
        eq = np.cumprod(1.0 + x[rng.integers(0, len(x), size=(m, n))], axis=1)
        finals.append(eq[:, -1])
        dds.append(_dd_from_equity(eq))
    f, d = np.concatenate(finals), np.concatenate(dds)
    return {"n_sims": n_sims, "n_trades_per_sim": n, "final_equity_p05": float(np.percentile(f, 5)),
            "final_equity_p50": float(np.percentile(f, 50)), "final_equity_p95": float(np.percentile(f, 95)),
            "maxdd_p50": float(np.percentile(d, 50)), "maxdd_p05_worst": float(np.percentile(d, 5)),
            "prob_final_below_1": float((f < 1.0).mean()), "prob_maxdd_worse_than_25pct": float((d < -0.25).mean())}


def mc_block_bootstrap(r: pd.Series, horizon: int | None = None, block: int = 21, n_sims: int = 10_000, seed: int = 0,
                       chunk: int = 1_000) -> dict:
    """Circular block bootstrap of period returns (keeps volatility clustering / short-range dependence)."""
    x = r.dropna().to_numpy()
    T = len(x)
    if T < 2 * block:
        return {"error": f"only {T} periods; block bootstrap not meaningful"}
    h = horizon or T
    nb = int(np.ceil(h / block))
    rng = np.random.default_rng(seed)
    finals, dds = [], []
    offs = np.arange(block)
    for s in range(0, n_sims, chunk):
        m = min(chunk, n_sims - s)
        starts = rng.integers(0, T, size=(m, nb))
        idx = ((starts[:, :, None] + offs) % T).reshape(m, -1)[:, :h]
        eq = np.cumprod(1.0 + x[idx], axis=1)
        finals.append(eq[:, -1])
        dds.append(_dd_from_equity(eq))
    f, d = np.concatenate(finals), np.concatenate(dds)
    return {"n_sims": n_sims, "horizon_periods": h, "block": block, "final_equity_p05": float(np.percentile(f, 5)),
            "final_equity_p50": float(np.percentile(f, 50)), "final_equity_p95": float(np.percentile(f, 95)),
            "maxdd_p50": float(np.percentile(d, 50)), "maxdd_p05_worst": float(np.percentile(d, 5)),
            "prob_final_below_1": float((f < 1.0).mean()), "prob_maxdd_worse_than_25pct": float((d < -0.25).mean())}
