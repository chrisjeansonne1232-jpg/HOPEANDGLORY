"""Central, pre-registered configuration.

Every assumption that can change a result lives here so it can be audited and stress-tested.
The date splits were frozen BEFORE any strategy was evaluated (see RESEARCH_LOG.md, "Pre-registration").
Values marked NEEDS-VERIFY come from second-hand sources (search summaries); primary fee pages were
unreachable from the research container. They are deliberately conservative and covered by the 2x cost stress.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass, field

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRIALS_CSV = ROOT / "trials.csv"
HOLDOUT_LEDGER = ROOT / "holdout_ledger.csv"
DATA_RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"


# --------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    """Development / validation / final-holdout boundaries (inclusive end dates).

    Development : start of data .. dev_end     -> design and tune here
    Validation  : dev_end  .. val_end          -> test tuned strategies here (walk-forward allowed)
    Holdout     : val_end  .. latest data      -> ONE look per finalist, via lab.splits.open_holdout()
    """

    dev_end: str = "2013-12-31"
    val_end: str = "2023-12-31"
    name: str = "equity"

    def label(self, ts: pd.Timestamp) -> str:
        ts = pd.Timestamp(ts)
        if ts <= pd.Timestamp(self.dev_end):
            return "DEVELOPMENT"
        if ts <= pd.Timestamp(self.val_end):
            return "VALIDATION"
        return "HOLDOUT"

    def bounds(self, split: str, first: pd.Timestamp, last: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        dev_end, val_end = pd.Timestamp(self.dev_end), pd.Timestamp(self.val_end)
        if split == "DEVELOPMENT":
            return first, dev_end
        if split == "VALIDATION":
            return dev_end + pd.Timedelta(days=1), val_end
        if split == "HOLDOUT":
            return val_end + pd.Timedelta(days=1), last
        if split in ("RESEARCH", "DEV+VAL"):  # everything a researcher is allowed to see
            return first, val_end
        raise ValueError(f"unknown split {split!r}")


# Equity/ETF/macro data: dev contains 2000-02 and 2008; validation contains 2015-16, 2018, 2020, 2022;
# holdout (2024-01-01 .. latest, ~2.75y as of 2026-09) contains Aug-2024 and Apr-2025 vol shocks.
EQUITY_SPLITS = SplitConfig(dev_end="2013-12-31", val_end="2023-12-31", name="equity")
# Crypto history is shorter (spot BTC-USD from ~2015): dev 2015-2019, validation 2020-2023.
CRYPTO_SPLITS = SplitConfig(dev_end="2019-12-31", val_end="2023-12-31", name="crypto")
# Prediction markets (Kalshi weather history starts 2021-08). REVISED 2026-09-30, before any prediction-market outcome was viewed
# (original: dev<=2022-12-31, val<=2023-12-31, which left only ~2.4y for dev+val): dev <= 2023-03-31, val 2023-04..2024-09,
# HOLDOUT 2024-10-01..latest (~2.0y).
PREDICTION_SPLITS = SplitConfig(dev_end="2023-03-31", val_end="2024-09-30", name="prediction")
SPLITS_BY_MARKET = {"equity": EQUITY_SPLITS, "crypto": CRYPTO_SPLITS, "prediction": PREDICTION_SPLITS}


# --------------------------------------------------------------------------------------
# Costs (spot equities / ETFs / spot crypto)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CostModel:
    """Per-side proportional costs in basis points of traded notional, plus financing.

    All spread / slippage / fee / borrow / financing terms are multiplied by `cost_multiplier`
    (set to 2.0 for the Phase 3 "survives 2x costs" test).
    """

    fee_bps: float = 0.0                      # proportional broker/exchange fee per side (crypto: taker fee)
    half_spread_bps: float = 1.0              # default half bid/ask spread; liquid ETFs are ~0.1-1 bp, so this is conservative
    half_spread_bps_by_asset: dict = field(default_factory=dict)
    slippage_bps: float = 1.0                 # market-order slippage vs. reference price (open print / close)
    commission_per_trade_usd: float = 0.0     # Robinhood equities: $0
    sec_fee_per_million_usd: float = 27.80    # on sales. NEEDS-VERIFY current rate (tiny either way)
    taf_per_share_usd: float = 0.000166       # FINRA TAF on sales, per share. NEEDS-VERIFY (tiny either way)
    borrow_bps_annual: float = 50.0           # easy-to-borrow ETF short; hard-to-borrow names need far more. NEEDS-VERIFY
    margin_rate_annual: float = 0.06          # broker interest on borrowed cash (gross long > 1). NEEDS-VERIFY
    cash_haircut_annual: float = 0.0025       # uninvested cash earns risk-free minus this
    cost_multiplier: float = 1.0

    def half_spread_for(self, asset: str) -> float:
        return self.half_spread_bps_by_asset.get(asset, self.half_spread_bps)

    def scaled(self, k: float) -> "CostModel":
        return dataclasses.replace(self, cost_multiplier=self.cost_multiplier * k)


# Retail spot crypto: proportional fees dominate (second-hand, NEEDS-VERIFY):
#   Coinbase Advanced entry tier taker ~0.6% (one source says 0.9% from 2026-09), Robinhood ~0.35-0.95% all-in.
CRYPTO_COST = CostModel(fee_bps=60.0, half_spread_bps=2.0, slippage_bps=5.0, margin_rate_annual=0.10)


# --------------------------------------------------------------------------------------
# Lab-wide config
# --------------------------------------------------------------------------------------
@dataclass
class LabConfig:
    account_size: float = 5_000.0             # USD; user range $2,000 - $10,000
    account_sizes_to_test: tuple = (2_000.0, 5_000.0, 10_000.0)
    short_term_tax_rate: float = 0.22         # federal+state marginal on short-term gains; sensitivity shown at 0/12/22/32%
    max_drawdown_tolerance: float = 0.25      # user's stated limit; also the single-event loss limit
    fractional_shares: bool = True            # Robinhood supports fractional shares on many stocks/ETFs (NEEDS-VERIFY per symbol)
    pdt_mode: str = "none"                    # "none": FINRA rule 4210 amendment effective 2026-06-04, Robinhood day-one rollout (2nd-hand, NEEDS-VERIFY)
                                              # "legacy": <$25k margin accounts limited to 3 day trades / 5 business days (conservative fallback)
    robinhood_options_level: int = 3          # assumed approvable level; L1 CSP/CC, L2 long options, L3 spreads/condors
    mc_sims: int = 10_000
    benchmark: str = "SPY"
    cost: CostModel = field(default_factory=CostModel)
    splits: SplitConfig = EQUITY_SPLITS

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)


DEFAULT = LabConfig()
