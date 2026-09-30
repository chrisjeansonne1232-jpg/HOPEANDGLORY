import math

import pandas as pd
import pytest

from lab import constraints as C
from lab import options as O
from lab import prediction as P
from lab import taxes as X


# ---------------------------------------------------------------- options
def _condor():
    # SPY-like $5-wide iron condor, quotes are illustrative test numbers
    sp_l = O.OptionQuote("P", 590, bid=0.50, ask=0.60)
    sp_s = O.OptionQuote("P", 595, bid=1.20, ask=1.30)
    sc_s = O.OptionQuote("C", 605, bid=1.10, ask=1.20)
    sc_l = O.OptionQuote("C", 610, bid=0.40, ask=0.50)
    return [("buy", 1, sp_l), ("sell", 1, sp_s), ("sell", 1, sc_s), ("buy", 1, sc_l)]


def test_buys_at_ask_sells_at_bid_never_mid():
    q = O.OptionQuote("C", 100, bid=1.00, ask=1.20)
    assert O.fill_price("buy", q) == 1.20 and O.fill_price("sell", q) == 1.00
    with pytest.raises(O.NoFill):
        O.fill_price("sell", O.OptionQuote("C", 100, bid=0.0, ask=0.05))
    with pytest.raises(ValueError):
        O.OptionQuote("C", 100, bid=1.3, ask=1.2)


def test_iron_condor_credit_fees_and_max_loss():
    costs = O.OptionCosts(commission_per_contract=0.0, passthrough_per_contract=0.04)
    o = O.open_position(_condor(), costs)
    # credit = (1.20 + 1.10 - 0.60 - 0.50) * 100 = 120 ; fees = 4 legs * 0.04
    assert o["net_premium"] == pytest.approx(120.0)
    assert o["fees"] == pytest.approx(0.16)
    assert O.max_loss(_condor(), costs) == pytest.approx(500 - 120 + 0.16)
    assert O.pnl_at_expiry(_condor(), 600, costs) == pytest.approx(120 - 0.16)          # inside the wings: keep credit
    assert O.pnl_at_expiry(_condor(), 580, costs) == pytest.approx(120 - 500 - 0.16)    # through the put wing
    assert O.pnl_at_expiry(_condor(), 620, costs) == pytest.approx(120 - 500 - 0.16)    # through the call wing


def test_naked_short_is_unbounded_and_csp_capital():
    naked = [("sell", 1, O.OptionQuote("C", 100, bid=1.0, ask=1.1))]
    assert O.max_loss(naked) == math.inf
    put = O.OptionQuote("P", 50, bid=1.0, ask=1.1)
    need = O.buying_power_required("cash_secured_put", [("sell", 1, put)], O.OptionCosts(passthrough_per_contract=0.0))
    assert need == pytest.approx(5000 - 100)
    assert C.cash_secured_put_min_account(590) == 59_000        # SPY-sized CSP is impossible on a $2-10k account


def test_2x_costs_double_fees():
    a = O.open_position(_condor(), O.OptionCosts())["fees"]
    b = O.open_position(_condor(), O.OptionCosts(cost_multiplier=2.0))["fees"]
    assert b == pytest.approx(2 * a)


# ---------------------------------------------------------------- prediction markets
def test_kalshi_fee_known_values():
    assert P.kalshi_fee(100, 0.50) == pytest.approx(1.75)
    assert P.kalshi_fee(1, 0.50) == pytest.approx(0.02)           # 1.75 cents rounds UP to 2 cents
    assert P.kalshi_fee(100, 0.05) == pytest.approx(0.34)         # 0.3325 -> 0.34
    assert P.kalshi_fee(100, 0.50, maker=True) == pytest.approx(0.44)  # 0.4375 -> 0.44
    assert P.kalshi_fee(100, 0.99) == pytest.approx(0.07)         # 0.0693 -> 0.07


def test_binary_fills_and_settlement():
    q = P.BinaryQuote(yes_bid=0.40, yes_ask=0.44)
    assert q.buy_price("yes") == 0.44 and q.buy_price("no") == pytest.approx(0.60)
    assert q.sell_price("yes") == 0.40 and q.sell_price("no") == pytest.approx(0.56)
    fee = P.kalshi_fee(10, 0.44)
    assert P.settle_pnl("yes", 10, 0.44, True, fee) == pytest.approx(10 - 4.4 - fee)
    assert P.settle_pnl("yes", 10, 0.44, False, fee) == pytest.approx(-4.4 - fee)
    assert P.settle_pnl("no", 10, 0.60, False, 0.0) == pytest.approx(4.0)


def test_edge_and_kelly():
    assert P.expected_profit_per_contract(0.50, 0.50) < 0          # a fair coin at 50c loses to the fee
    assert P.kelly_fraction(0.50, 0.50, 0.0175) == 0.0
    assert P.kelly_fraction(0.60, 0.50) == pytest.approx(0.2)


# ---------------------------------------------------------------- constraints / taxes
def test_whole_share_rounding_at_small_account():
    r = C.rounding_error({"SPY": 1.0}, {"SPY": 700.0}, equity=2000.0, fractional=False)
    assert r["realized_weights"]["SPY"] == pytest.approx(1400 / 2000)   # 2 shares only
    assert C.shares_for_weight(1.0, 700.0, 2000.0, fractional=True) == pytest.approx(2000 / 700)


def test_pdt_legacy_vs_none():
    e = pd.to_datetime(["2026-01-05 10:00", "2026-01-05 11:00", "2026-01-06 10:00", "2026-01-07 10:00"])
    tr = pd.DataFrame({"entry": e, "exit": e + pd.Timedelta(hours=1)})
    assert C.pdt_check(tr, 5000, "legacy")["ok"] is False        # 4 day trades in 3 days
    assert C.pdt_check(tr, 5000, "none")["ok"] is True
    assert C.pdt_check(tr, 30_000, "legacy")["ok"] is True


def test_tax_drag_and_loss_carryforward():
    idx = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-12-31"])
    r = pd.Series([0.10, -0.20, 0.30], index=idx)
    c = X.after_tax_curve(r, 0.25)
    # hand calc: 2020 gain 0.10 -> tax 0.025 -> equity 1.075; 2021 loss 1.075*0.8-1.075 = -0.215 (carried), equity 0.86;
    # 2022 gain 0.86*0.3 = 0.258, less carried 0.215 = 0.043 -> tax 0.01075
    assert c.loc[2020, "tax"] == pytest.approx(0.025)
    assert c.loc[2021, "tax"] == 0.0 and c.loc[2021, "equity_end"] == pytest.approx(0.86)
    assert c.loc[2022, "tax"] == pytest.approx(0.25 * 0.043)
