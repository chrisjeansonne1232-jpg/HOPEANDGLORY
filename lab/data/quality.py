"""Data-error detection and cross-source spot checks. Run on every dataset before it is used for anything."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Issue:
    severity: str   # "error" | "warn" | "info"
    check: str
    detail: str

    def __str__(self):
        return f"[{self.severity.upper()}] {self.check}: {self.detail}"


def check_ohlcv(df: pd.DataFrame, name: str = "", max_abs_ret: float = 0.25, max_gap_days: int = 5, tol: float = 1e-6) -> list[Issue]:
    iss: list[Issue] = []
    d = df.copy()
    if d.index.has_duplicates:
        iss.append(Issue("error", "duplicate_dates", f"{d.index.duplicated().sum()} duplicated timestamps"))
    if not d.index.is_monotonic_increasing:
        iss.append(Issue("error", "unsorted", "index not ascending"))
    for c in ("open", "high", "low", "close"):
        if c in d:
            n_nan, n_bad = int(d[c].isna().sum()), int((d[c] <= 0).sum())
            if n_nan:
                iss.append(Issue("warn", f"nan_{c}", f"{n_nan} NaN"))
            if n_bad:
                iss.append(Issue("error", f"nonpositive_{c}", f"{n_bad} rows <= 0"))
    if {"open", "high", "low", "close"} <= set(d.columns):
        hi_bad = d["high"] < d[["open", "close", "low"]].max(axis=1) * (1 - tol)
        lo_bad = d["low"] > d[["open", "close", "high"]].min(axis=1) * (1 + tol)
        if hi_bad.any():
            iss.append(Issue("error", "high_below_ohlc", f"{int(hi_bad.sum())} rows, first {d.index[hi_bad][0].date()}"))
        if lo_bad.any():
            iss.append(Issue("error", "low_above_ohlc", f"{int(lo_bad.sum())} rows, first {d.index[lo_bad][0].date()}"))
    px = d["adj_close"] if "adj_close" in d else d["close"]
    ret = px.pct_change()
    big = ret.abs() > max_abs_ret
    if big.any():
        worst = ret[big].abs().sort_values(ascending=False).head(3)
        iss.append(Issue("warn", "extreme_returns",
                         f"{int(big.sum())} moves > {max_abs_ret:.0%} (bad tick / unadjusted split?): " +
                         ", ".join(f"{i.date()} {ret[i]:+.1%}" for i in worst.index)))
    gaps = d.index.to_series().diff().dt.days
    if (gaps > max_gap_days).any():
        g = gaps[gaps > max_gap_days]
        iss.append(Issue("warn", "calendar_gaps", f"{len(g)} gaps > {max_gap_days} days, largest {int(g.max())}d ending {g.idxmax().date()}"))
    if "volume" in d and d["volume"].notna().any():
        stale = (d["close"].diff() == 0) & (d["volume"] == 0)
        if stale.rolling(5).sum().ge(5).any():
            iss.append(Issue("warn", "stale_prices", "5+ consecutive unchanged closes with zero volume"))
    if "open" in d:
        eq = (d["open"] - d["close"].shift(1)).abs() < 1e-9 * d["close"].shift(1)
        frac = float(eq.mean())
        if frac > 0.2:
            iss.append(Issue("warn", "open_equals_prev_close",
                             f"{frac:.0%} of opens equal the prior close: opens are not independent prints (typical of "
                             "older index history). Prefer exec_mode='next_close' for this dataset."))
    return iss


def spot_check(a: pd.DataFrame, b: pd.DataFrame, n: int = 40, seed: int = 0, ret_tol: float = 0.002) -> dict:
    """Cross-source check on DAILY RETURNS (levels differ by adjustment basis). Compares the closes of two sources on the
    common dates: correlation, share of days within tolerance, worst disagreements, plus a random sample of dates."""
    ra, rb = a["close"].pct_change(), b["close"].pct_change()
    df = pd.concat([ra, rb], axis=1, join="inner").dropna()
    df.columns = ["a", "b"]
    diff = (df["a"] - df["b"]).abs()
    rng = np.random.default_rng(seed)
    sample = df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)] if len(df) else df
    ratio = (a["close"] / b["close"]).dropna()
    return {"n_common_days": len(df), "corr_returns": float(df["a"].corr(df["b"])) if len(df) > 2 else np.nan,
            "pct_within_tol": float((diff <= ret_tol).mean()) if len(df) else np.nan,
            "sample_n": len(sample), "sample_max_abs_diff": float((sample["a"] - sample["b"]).abs().max()) if len(sample) else np.nan,
            "worst_days": [(str(i.date()), float(diff[i])) for i in diff.sort_values(ascending=False).head(5).index],
            "level_ratio_min": float(ratio.min()) if len(ratio) else np.nan, "level_ratio_max": float(ratio.max()) if len(ratio) else np.nan}


def assert_clean(df: pd.DataFrame, name: str = "") -> None:
    errs = [i for i in check_ohlcv(df, name) if i.severity == "error"]
    if errs:
        raise ValueError(f"data quality errors in {name}: " + "; ".join(map(str, errs)))
