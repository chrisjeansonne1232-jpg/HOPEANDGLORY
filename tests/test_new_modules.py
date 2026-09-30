import numpy as np
import pandas as pd
import pytest

from lab import stats as S
from lab.config import CostModel
from lab.data import kalshi as K
from lab.engine import run_session_only
from lab.prediction import kalshi_fee
from research import weather as W
from tests.helpers import ZERO_COST, synth_panel


def test_session_only_matches_leg_compounding_and_charges_both_sides():
    p = synth_panel(n=100)
    w = pd.DataFrame(1.0, index=p.index, columns=["X"])
    ov = p.open["X"] / p.close["X"].shift(1) - 1
    intr = p.close["X"] / p.open["X"] - 1
    assert (1 + run_session_only(p, w, "overnight", cost=ZERO_COST).returns).prod() == pytest.approx((1 + ov.iloc[1:]).prod())
    assert (1 + run_session_only(p, w, "intraday", cost=ZERO_COST).returns).prod() == pytest.approx((1 + intr.iloc[1:]).prod())
    c = CostModel(half_spread_bps=1, slippage_bps=1, sec_fee_per_million_usd=0, taf_per_share_usd=0)
    assert run_session_only(p, w, "overnight", cost=c).costs.trading.iloc[5] * 1e4 == pytest.approx(4.0)   # 2 sides x (1+1) bp


def test_hurdle_variance_is_robust_to_cost_doomed_trials():
    rng = np.random.default_rng(0)
    good = list(rng.normal(0.0, 0.016, 200))
    junk = [-0.66, -0.55, -0.54]
    plain = S.hurdle_variance(good + junk, 4000)
    assert plain["plain"] > 5 * plain["robust_mad"]                      # junk explodes the plain variance only
    assert plain["used"] == pytest.approx(max(plain["robust_mad"], 1 / 4000))
    assert S.hurdle_variance(good, 100)["used"] == pytest.approx(max(S.hurdle_variance(good, 100)["robust_mad"], 1 / 100))


def test_bucket_probabilities_partition_the_outcome_space():
    mu, sig = 70.3, 2.2
    tot = W.outcome_prob("less", None, 66, mu, sig) + W.outcome_prob("between", 66, 67, mu, sig) + W.outcome_prob("between", 68, 69, mu, sig) \
        + W.outcome_prob("between", 70, 71, mu, sig) + W.outcome_prob("between", 72, 73, mu, sig) + W.outcome_prob("greater", 73, None, mu, sig)
    assert tot == pytest.approx(1.0, abs=1e-12)                          # "<66", 66-67, ..., ">73" cover every integer exactly once


def _df(rows):
    d = pd.DataFrame(rows)
    d["D"] = pd.Timestamp("2024-01-05")
    d["dt_name"] = "DT1"
    return d


def test_simulate_fills_at_ask_and_no_ask_with_slippage_and_kalshi_fees():
    # YES ask 0.40 bid 0.38; model says 0.70 -> buys YES at 0.41 (ask + 1c slip), wins -> pnl = n*(1-0.41) - fee
    d = _df([{"ticker": "a", "ask_close": 0.40, "bid_close": 0.38, "volume": 1000.0, "y": 1}])
    tr = W.simulate(d, np.array([0.70]), theta=0.03, frac=0.01, bankroll=5000.0, topk=8)
    assert len(tr) == 1 and tr.side_yes.iloc[0] and tr.price.iloc[0] == pytest.approx(0.41)
    n = int(tr.n.iloc[0])
    assert n == int(np.floor(50 / 0.41))
    assert tr.pnl.iloc[0] == pytest.approx(n * 0.59 - kalshi_fee(n, 0.41))
    # model says 0.10 -> buys NO at 1 - bid + 1c = 0.63 ; outcome y=0 means NO wins
    d2 = _df([{"ticker": "b", "ask_close": 0.40, "bid_close": 0.38, "volume": 1000.0, "y": 0}])
    tr2 = W.simulate(d2, np.array([0.10]), theta=0.03, frac=0.01, bankroll=5000.0, topk=8)
    assert not tr2.side_yes.iloc[0] and tr2.price.iloc[0] == pytest.approx(0.63) and tr2.win.iloc[0]
    # empty book (ask 1.0) and zero volume are never traded
    d3 = _df([{"ticker": "c", "ask_close": 1.0, "bid_close": 0.0, "volume": 500.0, "y": 1}, {"ticker": "d", "ask_close": 0.3, "bid_close": 0.28, "volume": 0.0, "y": 1}])
    assert len(W.simulate(d3, np.array([0.9, 0.9]), theta=0.03)) == 0


def test_favourite_rule_buys_the_expensive_side_only():
    d = _df([{"ticker": "a", "ask_close": 0.04, "bid_close": 0.03, "volume": 100.0, "y": 0}, {"ticker": "b", "ask_close": 0.50, "bid_close": 0.48, "volume": 100.0, "y": 1}])
    tr = W.simulate(d, np.array([0.5, 0.5]), theta=0.0, mode="fav", longshot=0.10)
    assert list(tr.ticker) == ["a"] and not tr.side_yes.iloc[0]          # yes is a longshot at 4c -> buy NO at 1-0.03+0.01=0.98


def test_kalshi_candle_parser_handles_live_and_archived_field_names(monkeypatch):
    live = {"candlesticks": [{"end_period_ts": 100, "open_interest_fp": "9.5", "volume_fp": "3.0", "price": {"close_dollars": "0.2"},
                              "yes_bid": {"close_dollars": "0.18", "open_dollars": "0.10"}, "yes_ask": {"close_dollars": "0.22", "open_dollars": "1.00"}}]}
    old = {"candlesticks": [{"end_period_ts": 200, "open_interest": "7.0", "volume": "2.0", "price": {"close": None},
                             "yes_bid": {"close": "0.66", "open": "0.35"}, "yes_ask": {"close": "0.68", "open": "1.0000"}}]}
    monkeypatch.setattr(K, "_cutoff_ts", lambda: 10**12)
    for payload, bid, ask, vol in ((live, 0.18, 0.22, 3.0), (old, 0.66, 0.68, 2.0)):
        monkeypatch.setattr(K, "_get", lambda path, params, _p=payload: _p)
        r = K._candles_one("S", "S-1", 0, 1)[0]
        assert (r["bid_close"], r["ask_close"], r["volume"]) == (bid, ask, vol)
