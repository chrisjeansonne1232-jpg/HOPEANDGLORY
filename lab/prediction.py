"""Binary prediction-market contract accounting (Kalshi / Polymarket US). Prices are dollars in (0, 1).

Fills never use mid: buying YES pays the YES ask; buying NO pays (1 - YES bid); selling crosses the other way.

Fee formulas:
  Kalshi (2nd-hand but consistent across several sources, NEEDS-VERIFY against kalshi.com/docs/kalshi-fee-schedule.pdf):
      taker = ceil_to_cent( multiplier * 0.07   * C * P * (1-P) );  maker = ceil_to_cent( multiplier * 0.0175 * C * P * (1-P) )
  Polymarket US (UNVERIFIED single search snippet: taker coefficient 0.0695, maker rebate coefficient -0.0125 as of 2026-09;
      an older filing said flat 0.30% taker / 0.20% maker rebate). Treat any Polymarket-US result as provisional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _ceil_cent(x: float) -> float:
    return math.ceil(round(x * 100.0, 9)) / 100.0


def kalshi_fee(contracts: int, price: float, maker: bool = False, multiplier: float = 1.0) -> float:
    coeff = 0.0175 if maker else 0.07
    return _ceil_cent(multiplier * coeff * contracts * price * (1.0 - price))


def polymarket_us_fee(contracts: int, price: float, maker: bool = False, taker_coeff: float = 0.0695,
                      maker_rebate_coeff: float = -0.0125) -> float:
    """Provisional. Negative result = rebate."""
    coeff = maker_rebate_coeff if maker else taker_coeff
    raw = coeff * contracts * price * (1.0 - price)
    return -_ceil_cent(-raw) if raw < 0 else _ceil_cent(raw)


@dataclass(frozen=True)
class BinaryQuote:
    yes_bid: float
    yes_ask: float

    def __post_init__(self):
        if not (0 <= self.yes_bid <= self.yes_ask <= 1):
            raise ValueError(f"bad quote {self.yes_bid}/{self.yes_ask}")

    def buy_price(self, side: str) -> float:
        return self.yes_ask if side == "yes" else 1.0 - self.yes_bid

    def sell_price(self, side: str) -> float:
        return self.yes_bid if side == "yes" else 1.0 - self.yes_ask


def settle_pnl(side: str, contracts: int, entry_price: float, resolved_yes: bool, entry_fee: float = 0.0) -> float:
    """P&L in USD of a position held to resolution (Kalshi charges no separate settlement fee; NEEDS-VERIFY)."""
    win = (side == "yes") == resolved_yes
    return (contracts if win else 0.0) - contracts * entry_price - entry_fee


def fee_per_contract(price: float, contracts: int = 100, maker: bool = False, fee_fn=kalshi_fee) -> float:
    return fee_fn(contracts, price, maker) / contracts


def expected_profit_per_contract(p_win: float, price: float, contracts: int = 100, maker: bool = False, fee_fn=kalshi_fee) -> float:
    """Expected USD profit per contract bought at `price` if the true win probability is p_win, after the entry fee."""
    return p_win - price - fee_per_contract(price, contracts, maker, fee_fn)


def kelly_fraction(p_win: float, price: float, fee_per: float = 0.0) -> float:
    """Kelly fraction of bankroll for a binary contract costing c = price + fee: f* = (p - c) / (1 - c); floored at 0."""
    c = price + fee_per
    return max(0.0, (p_win - c) / (1.0 - c)) if c < 1 else 0.0
