"""SYNTHETIC data generators used ONLY to unit-test the machinery. Nothing produced from these is evidence of any edge."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.config import CostModel
from lab.panel import Panel

ZERO_COST = CostModel(half_spread_bps=0.0, slippage_bps=0.0, fee_bps=0.0, sec_fee_per_million_usd=0.0, taf_per_share_usd=0.0,
                      borrow_bps_annual=0.0, margin_rate_annual=0.0, cash_haircut_annual=0.0)


def synth_frame(n=400, seed=0, start="2005-01-03", vol=0.006, drift=0.0003, gap=0.002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    cc = rng.normal(drift, vol, n)
    close = 100 * np.cumprod(1 + cc)
    prev = np.concatenate([[100.0], close[:-1]])
    open_ = prev * (1 + rng.normal(0, gap, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1e6}, index=idx)


def synth_panel(assets=("X",), n=400, seed=0, **kw) -> Panel:
    frames = {a: synth_frame(n=n, seed=seed + i, **kw) for i, a in enumerate(assets)}
    return Panel.from_frames(frames, source="SYNTHETIC (unit tests only)", caveats=["SYNTHETIC DATA - not evidence"])


def flat_panel(assets=("X",), n=30, price=100.0, freq="B", start="2010-01-04") -> Panel:
    idx = pd.date_range(start, periods=n, freq=freq)
    frames = {a: pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1e6}, index=idx) for a in assets}
    return Panel.from_frames(frames, source="SYNTHETIC flat", caveats=["SYNTHETIC DATA"])
