"""Option fill/payoff accounting. Requires historical option QUOTES (bid/ask) to be useful: none are available from the
free sources, so any options strategy result is 'needs data' until quotes are supplied (e.g. Massive/Polygon options
plan, CBOE DataShop, OptionMetrics, ThetaData). This module only defines HOW fills and costs are charged.

RULES: buys fill at the ASK, sells fill at the BID. Mid is never used. A sell against a zero bid does not fill.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MULT = 100  # shares per contract

# Robinhood approval levels (2nd-hand, NEEDS-VERIFY): L1 covered calls / cash-secured puts; L2 long calls/puts (+ L1);
# L3 debit/credit spreads, iron condors, iron butterflies (+ L2). Undefined-risk (naked) writing is not assumed available.
REQUIRED_LEVEL = {"covered_call": 1, "cash_secured_put": 1, "long_call": 2, "long_put": 2, "long_straddle": 2, "long_strangle": 2,
                  "debit_spread": 3, "credit_spread": 3, "iron_condor": 3, "iron_butterfly": 3, "calendar_spread": 3}


class NoFill(Exception):
    pass


@dataclass(frozen=True)
class OptionQuote:
    right: str          # "C" or "P"
    strike: float
    bid: float
    ask: float
    underlying: float | None = None
    expiry: str | None = None

    def __post_init__(self):
        if self.right not in ("C", "P"):
            raise ValueError("right must be 'C' or 'P'")
        if self.ask < self.bid:
            raise ValueError(f"crossed quote bid={self.bid} ask={self.ask}")

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass(frozen=True)
class OptionCosts:
    commission_per_contract: float = 0.0        # Robinhood: $0 commission
    passthrough_per_contract: float = 0.04      # regulatory + clearing pass-throughs per contract per side. NEEDS-VERIFY
                                                # (options TAF ~$0.0033 on sells per 2026 schedule; OCC/ORF amounts unconfirmed)
    cost_multiplier: float = 1.0

    def per_contract(self) -> float:
        return (self.commission_per_contract + self.passthrough_per_contract) * self.cost_multiplier


Leg = tuple  # (side: "buy"|"sell", qty: int, quote: OptionQuote)


def fill_price(side: str, q: OptionQuote) -> float:
    if side == "buy":
        if not (q.ask > 0):
            raise NoFill("no ask")
        return q.ask
    if side == "sell":
        if not (q.bid > 0):
            raise NoFill("no bid: cannot sell")
        return q.bid
    raise ValueError("side must be 'buy' or 'sell'")


def open_position(legs: list[Leg], costs: OptionCosts = OptionCosts()) -> dict:
    """Cash flows of opening a (multi-leg) position, each leg crossing the spread. net_premium > 0 is a credit."""
    net = fees = 0.0
    for side, qty, q in legs:
        px = fill_price(side, q)
        net += (px if side == "sell" else -px) * MULT * qty
        fees += qty * costs.per_contract()
    return {"net_premium": net, "fees": fees, "cash_after_fees": net - fees}


def intrinsic(q: OptionQuote, spot: float) -> float:
    return max(spot - q.strike, 0.0) if q.right == "C" else max(q.strike - spot, 0.0)


def expiry_value(legs: list[Leg], spot: float) -> float:
    """Value of the position at expiry (positive = we receive), before fees."""
    return sum((1 if side == "buy" else -1) * qty * MULT * intrinsic(q, spot) for side, qty, q in legs)


def pnl_at_expiry(legs: list[Leg], spot: float, costs: OptionCosts = OptionCosts(), close_fees: bool = False) -> float:
    o = open_position(legs, costs)
    fees = o["fees"] + (sum(qty for _, qty, _ in legs) * costs.per_contract() if close_fees else 0.0)
    return o["net_premium"] + expiry_value(legs, spot) - fees


def max_loss(legs: list[Leg], costs: OptionCosts = OptionCosts()) -> float:
    """Worst-case loss at expiry (positive number); math.inf if unbounded (naked short). Piecewise-linear payoff, so
    checking spot=0, every strike and a very high spot is exact."""
    ks = sorted({q.strike for _, _, q in legs})
    hi = max(ks) * 10 + 1000
    pts = [0.0] + ks + [hi, hi * 2]
    pnl = [pnl_at_expiry(legs, s, costs) for s in pts]
    if pnl[-1] < pnl[-2] - 1e-9:      # still falling at very high spot -> unbounded
        return math.inf
    return float(max(0.0, -min(pnl)))


def buying_power_required(kind: str, legs: list[Leg], costs: OptionCosts = OptionCosts()) -> float:
    """Cash needed at entry (USD) under a cash-account style rule: defined-risk = worst-case loss; cash-secured put =
    strike*100 minus premium; long options = debit paid."""
    o = open_position(legs, costs)
    if kind in ("long_call", "long_put", "long_straddle", "long_strangle", "debit_spread"):
        return -o["cash_after_fees"]
    if kind == "cash_secured_put":
        (_, qty, q), = legs
        return q.strike * MULT * qty - o["cash_after_fees"]
    return max_loss(legs, costs)


def round_trip_spread_cost(legs: list[Leg]) -> float:
    """USD lost to crossing the spread on entry AND exit vs mid (informational; fills themselves never use mid)."""
    return sum(qty * MULT * q.spread for _, qty, q in legs)
