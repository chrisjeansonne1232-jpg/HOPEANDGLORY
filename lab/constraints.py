"""Small-account feasibility: whole/fractional shares, PDT rule, options capital, prediction-market liquidity."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def shares_for_weight(weight: float, price: float, equity: float, fractional: bool = True) -> float:
    """Shares held for a target weight. Whole shares round DOWN (never exceed the target)."""
    raw = weight * equity / price
    return raw if fractional else math.floor(raw + 1e-12)


def rounding_error(weights: dict[str, float], prices: dict[str, float], equity: float, fractional: bool = False) -> dict:
    """How far whole-share rounding pushes a target portfolio from its weights at a given account size."""
    out, invested = {}, 0.0
    for a, w in weights.items():
        sh = shares_for_weight(w, prices[a], equity, fractional)
        out[a] = sh * prices[a] / equity
        invested += sh * prices[a]
    return {"realized_weights": out, "max_abs_weight_error": max(abs(out[a] - weights[a]) for a in weights),
            "cash_left": equity - invested}


def count_day_trades(trades: pd.DataFrame) -> pd.Series:
    """Round trips opened and closed on the same calendar day, counted per business day (needs entry/exit timestamps)."""
    if not len(trades):
        return pd.Series(dtype=int)
    same = trades[trades.entry.dt.normalize() == trades.exit.dt.normalize()]
    return same.groupby(same.entry.dt.normalize()).size()


def pdt_check(trades: pd.DataFrame, equity: float, mode: str = "none") -> dict:
    """mode='legacy': margin account under $25k may make at most 3 day trades in any 5 business days.
    mode='none'   : rule replaced by intraday-margin standards (Robinhood removed PDT flags 2026-06-04; NEEDS-VERIFY)."""
    if mode == "none" or equity >= 25_000:
        return {"ok": True, "mode": mode, "max_in_5d": None}
    dts = count_day_trades(trades)
    if not len(dts):
        return {"ok": True, "mode": mode, "max_in_5d": 0}
    daily = dts.reindex(pd.bdate_range(dts.index.min(), dts.index.max()), fill_value=0)
    mx = int(daily.rolling(5, min_periods=1).sum().max())
    return {"ok": mx <= 3, "mode": mode, "max_in_5d": mx}


def cash_secured_put_min_account(strike: float, contracts: int = 1) -> float:
    return strike * 100 * contracts


def fits_account(required_usd: float, equity: float, max_fraction_at_risk: float = 0.25) -> bool:
    """A single position's capital requirement must fit AND its worst-case loss must respect the 25% single-event limit
    (checked by the caller with the position's max loss)."""
    return required_usd <= equity


def single_event_loss_ok(max_loss_usd: float, equity: float, limit: float = 0.25) -> bool:
    return max_loss_usd <= limit * equity
