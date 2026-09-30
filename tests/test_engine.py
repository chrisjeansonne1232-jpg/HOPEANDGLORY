"""Known-answer tests for the backtest engine. Synthetic data; validates arithmetic/timing, not any strategy."""
import numpy as np
import pandas as pd
import pytest

from lab.engine import run_backtest, buy_and_hold
from lab.lookahead import LookAheadError, check_lookahead
from lab.config import CostModel, EQUITY_SPLITS
from lab.panel import Panel
from lab.splits import HoldoutViolation, open_holdout
from tests.helpers import ZERO_COST, flat_panel, synth_panel, synth_frame


def test_buy_and_hold_zero_cost_matches_open_to_close_path():
    p = synth_panel(n=300)
    w = pd.DataFrame(1.0, index=p.index, columns=["X"])
    r = run_backtest(p, w, cost=ZERO_COST, exec_mode="next_open")
    # decided at close of bar 0, executed at open of bar 1, held to the end
    expected = p.close["X"].iloc[-1] / p.open["X"].iloc[1] - 1
    assert (1 + r.returns).prod() - 1 == pytest.approx(expected, rel=1e-9)
    assert r.returns.iloc[0] == 0.0  # nothing held during bar 0


def test_same_close_and_next_close_timing():
    p = synth_panel(n=200)
    w = pd.DataFrame(1.0, index=p.index, columns=["X"])
    r_same = run_backtest(p, w, cost=ZERO_COST, exec_mode="same_close")
    r_next = run_backtest(p, w, cost=ZERO_COST, exec_mode="next_close")
    c = p.close["X"]
    assert (1 + r_same.returns).prod() - 1 == pytest.approx(c.iloc[-1] / c.iloc[0] - 1, rel=1e-9)   # earns from close of bar 0
    assert (1 + r_next.returns).prod() - 1 == pytest.approx(c.iloc[-1] / c.iloc[1] - 1, rel=1e-9)   # one bar later


def test_round_trip_cost_is_exactly_two_sides():
    p = synth_panel(n=60)
    cost = CostModel(half_spread_bps=5.0, slippage_bps=5.0, sec_fee_per_million_usd=0.0, taf_per_share_usd=0.0,
                     cash_haircut_annual=0.0, borrow_bps_annual=0.0)
    w = pd.DataFrame(0.0, index=p.index, columns=["X"])
    w.iloc[10:20] = 1.0
    r = run_backtest(p, w, cost=cost)
    assert len(r.trades) == 1
    assert r.costs["trading"].sum() == pytest.approx(0.0010 + 0.0010, rel=1e-6)        # 10 bps in + 10 bps out
    t = r.trades.iloc[0]
    assert t.entry == p.index[11] and t.exit == p.index[21] and t.periods == 10
    gross_move = p.open["X"].iloc[21] / p.open["X"].iloc[11] - 1
    assert t.pnl_frac == pytest.approx(gross_move - 0.002, abs=1.5e-3)                 # additive vs compounded daily
    # doubling costs doubles the drag
    r2 = run_backtest(p, w, cost=cost.scaled(2.0))
    assert r2.costs["trading"].sum() == pytest.approx(2 * r.costs["trading"].sum(), rel=1e-9)


def test_weight_drift_creates_rebalancing_turnover():
    idx = pd.bdate_range("2010-01-04", periods=6)
    a = pd.DataFrame({"open": 100.0, "high": 110.0, "low": 100.0, "close": [100, 110, 110, 110, 110, 110.0], "volume": 1}, index=idx)
    b = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1}, index=idx)
    p = Panel.from_frames({"A": a, "B": b}, source="SYNTHETIC")
    w = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    r = run_backtest(p, w, cost=ZERO_COST, exec_mode="same_close")
    assert r.turnover.iloc[0] == pytest.approx(1.0)
    # A drifts to 0.5*1.1/1.05 = 0.52381, B to 0.47619; rebalancing back to 50/50 trades 2 * 0.02381
    assert r.turnover.iloc[1] == pytest.approx(2 * abs(0.5 - 0.5 * 1.1 / 1.05), rel=1e-9)
    assert r.turnover.iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_short_borrow_cost_accrues_per_calendar_day():
    p = flat_panel(n=366, freq="D")
    cost = CostModel(half_spread_bps=0, slippage_bps=0, sec_fee_per_million_usd=0, taf_per_share_usd=0, borrow_bps_annual=100.0,
                     cash_haircut_annual=0.0)
    w = pd.DataFrame(-0.5, index=p.index, columns=["X"])
    r = run_backtest(p, w, cost=cost, allow_short=True)
    assert r.costs["borrow"].sum() == pytest.approx(0.5 * 0.01 * (365.5 / 365), rel=0.01)
    with pytest.raises(ValueError):
        run_backtest(p, w, cost=cost, allow_short=False)


def test_cash_earns_rf_minus_haircut():
    p = flat_panel(n=20)
    rf = pd.Series(0.0002, index=p.index)
    r = run_backtest(p, pd.DataFrame(0.0, index=p.index, columns=["X"]), cost=ZERO_COST, rf_period=rf)
    assert (r.returns == 0.0002).all()


def test_leverage_and_holdout_guards():
    p = synth_panel(n=100)
    with pytest.raises(ValueError):
        run_backtest(p, pd.DataFrame(1.5, index=p.index, columns=["X"]), cost=ZERO_COST)
    late = synth_panel(n=100, start="2024-06-03")
    with pytest.raises(HoldoutViolation):
        run_backtest(late, pd.DataFrame(1.0, index=late.index, columns=["X"]), cost=ZERO_COST)


def test_holdout_can_only_be_opened_once_per_strategy(tmp_path):
    late = synth_panel(n=100, start="2023-10-02")
    led = tmp_path / "ledger.csv"
    opened = open_holdout(late, "stratA", {"n": 200}, "unit test", ledger_path=led)
    assert opened.holdout_open
    run_backtest(opened, pd.DataFrame(1.0, index=opened.index, columns=["X"]), cost=ZERO_COST)  # allowed now
    with pytest.raises(HoldoutViolation):
        open_holdout(late, "stratA", {"n": 100}, "second look", ledger_path=led)
    open_holdout(late, "stratB", {"n": 100}, "different finalist", ledger_path=led)


def _sma_strategy(n):
    def f(panel):
        c = panel.adj_close
        return (c > c.rolling(n).mean()).astype(float)
    return f


def test_lookahead_detector_accepts_honest_and_rejects_cheaters():
    p = synth_panel(n=400)
    assert check_lookahead(_sma_strategy(50), p)

    def peek_future(panel):
        c = panel.adj_close
        return (c.shift(-1) > c).astype(float)

    def centered(panel):
        c = panel.adj_close
        return (c > c.rolling(21, center=True).mean()).astype(float)

    def fullsample_z(panel):
        c = panel.adj_close
        return ((c - c.mean()) / c.std() > 0).astype(float)

    for bad in (peek_future, centered, fullsample_z):
        with pytest.raises(LookAheadError):
            check_lookahead(bad, p)


def test_cheater_would_look_amazing_if_undetected():
    """Why the detector matters: a one-bar peek produces a huge fake Sharpe under same_close execution."""
    p = synth_panel(n=500, drift=0.0)
    c = p.adj_close
    w = (c.shift(-1) > c).astype(float).iloc[:-1]
    r = run_backtest(p, w, cost=ZERO_COST, exec_mode="same_close")
    from lab.metrics import sharpe
    assert sharpe(r.returns, None, 252) > 5


def test_trade_extraction_handles_flip_and_open_trade():
    p = synth_panel(n=80)
    w = pd.DataFrame(0.0, index=p.index, columns=["X"])
    w.iloc[5:15] = 1.0
    w.iloc[15:30] = -1.0
    w.iloc[60:] = 1.0
    r = run_backtest(p, w, cost=ZERO_COST, allow_short=True)
    assert list(r.trades.side) == ["long", "short", "long"]
    assert r.trades.still_open.tolist() == [False, False, True]
    # sum of trade P&L equals total additive P&L within rounding of attribution
    assert r.trades.pnl_frac.sum() == pytest.approx(r.returns.sum(), abs=1e-9)
