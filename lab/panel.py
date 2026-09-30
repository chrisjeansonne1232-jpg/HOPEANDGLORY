"""Aligned price panel + provenance/caveats that travel with every result."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLCV = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass
class Panel:
    """Wide price panel (date x asset). `close` is raw; `adj_close` is dividend/split adjusted (total return).

    `caveats` is a list of human-readable data limitations (survivorship, price-only, synthetic, ...). They are
    copied into every BacktestResult and printed in every summary, so a result can never be separated from them.
    """

    open: pd.DataFrame | None
    high: pd.DataFrame | None
    low: pd.DataFrame | None
    close: pd.DataFrame
    adj_close: pd.DataFrame
    volume: pd.DataFrame | None = None
    ppy: int = 252
    source: str = ""
    caveats: list[str] = field(default_factory=list)
    holdout_open: bool = False  # set only by lab.splits.open_holdout

    # ---- construction --------------------------------------------------------------
    @classmethod
    def from_frames(cls, frames: dict[str, pd.DataFrame], ppy: int = 252, source: str = "", caveats=None) -> "Panel":
        """frames: {asset: DataFrame[open, high, low, close, (adj_close), (volume)]} with a DatetimeIndex."""
        cols: dict[str, dict[str, pd.Series]] = {k: {} for k in OHLCV}
        for asset, df in frames.items():
            df = df.copy()
            df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
            if "adj_close" not in df:
                df["adj_close"] = df["close"]
            if df.index.has_duplicates:
                raise ValueError(f"{asset}: duplicate dates in index")
            if not df.index.is_monotonic_increasing:
                raise ValueError(f"{asset}: index not sorted ascending")
            for k in OHLCV:
                if k in df:
                    cols[k][asset] = df[k]
        wide = {k: (pd.DataFrame(v).sort_index() if v else None) for k, v in cols.items()}
        idx = wide["close"].index
        for k, v in wide.items():
            if v is not None:
                wide[k] = v.reindex(idx)
        return cls(ppy=ppy, source=source, caveats=list(caveats or []), **wide)

    # ---- views ---------------------------------------------------------------------
    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def assets(self) -> list[str]:
        return list(self.close.columns)

    @property
    def adj_open(self) -> pd.DataFrame:
        """Open adjusted by the same factor as the close, so overnight returns include dividends."""
        if self.open is None:
            raise ValueError("panel has no open prices")
        return self.open * (self.adj_close / self.close)

    def slice(self, start=None, end=None) -> "Panel":
        sl = slice(pd.Timestamp(start) if start is not None else None, pd.Timestamp(end) if end is not None else None)
        kw = {k: (getattr(self, k).loc[sl] if getattr(self, k) is not None else None) for k in OHLCV}
        return Panel(ppy=self.ppy, source=self.source, caveats=list(self.caveats), holdout_open=self.holdout_open, **kw)

    def truncate(self, end) -> "Panel":
        """Data available at decision time `end` (inclusive). Used by the look-ahead truncation test."""
        return self.slice(end=end)

    def subset(self, assets: list[str]) -> "Panel":
        kw = {k: (getattr(self, k)[assets] if getattr(self, k) is not None else None) for k in OHLCV}
        return Panel(ppy=self.ppy, source=self.source, caveats=list(self.caveats), holdout_open=self.holdout_open, **kw)


def daily_rf_from_monthly_percent(monthly_pct: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """Spread a monthly risk-free rate (percent per month, e.g. Ken French RF) evenly (geometrically)
    across the trading days of that month. Returns per-period (daily) decimal returns aligned to daily_index."""
    m = monthly_pct.copy()
    m.index = pd.PeriodIndex(m.index, freq="M")
    per = pd.PeriodIndex(daily_index, freq="M")
    ndays = pd.Series(1, index=daily_index).groupby(per).transform("sum")
    monthly = pd.Series(m.reindex(per).to_numpy(), index=daily_index) / 100.0
    if monthly.isna().any():
        missing = monthly.index[monthly.isna()]
        raise ValueError(f"no monthly RF for {len(missing)} days, first {missing[0].date()}")
    return (1.0 + monthly) ** (1.0 / ndays) - 1.0


def rf_from_annual(annual: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Annualised decimal risk-free rate (e.g. FRED DGS3MO/100, forward-filled) -> per-period simple return."""
    a = annual.reindex(annual.index.union(index)).ffill().reindex(index)
    dt = pd.Series(index, index=index).diff().dt.days.fillna(1).clip(lower=1)
    return a * dt / 365.0
