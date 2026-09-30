"""Parser tests use hand-written FORMAT fixtures (not real market data) - they verify parsing, not that live APIs still
match. Quality-check tests inject known errors into synthetic frames."""
import numpy as np
import pandas as pd
import pytest

from lab.data import loaders as L
from lab.data import quality as Q
from tests.helpers import synth_frame


def test_parse_stooq_fixture():
    txt = "Date,Open,High,Low,Close,Volume\n2020-01-02,10,11,9,10.5,1000\n2020-01-03,10.5,12,10,11,2000\n"
    df = L.parse_stooq_csv(txt)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2020-01-02", "2020-01-03"] and df.close.iloc[1] == 11
    with pytest.raises(ValueError):
        L.parse_stooq_csv("No data")


def test_parse_yahoo_fixture():
    js = {"chart": {"result": [{"meta": {"gmtoffset": -18000}, "timestamp": [1577975400, 1578061800],
                                "indicators": {"quote": [{"open": [1, 2], "high": [2, 3], "low": [0.5, 1.5], "close": [1.5, 2.5], "volume": [10, 20]}],
                                               "adjclose": [{"adjclose": [1.4, 2.4]}]}}]}}
    df = L.parse_yahoo_chart(js)
    assert len(df) == 2 and df.adj_close.iloc[0] == 1.4 and df.index[0] == pd.Timestamp("2020-01-02")


def test_parse_fred_fixture():
    s = L.parse_fred_csv("observation_date,DGS3MO\n2020-01-01,.\n2020-01-02,1.55\n", "DGS3MO")
    assert np.isnan(s.iloc[0]) and s.iloc[1] == 1.55


def test_parse_coinbase_and_binance_fixtures():
    cb = L.parse_coinbase_candles([[1577923200, 1, 3, 2, 2.5, 100], [1577836800, 1, 2, 1.5, 1.8, 50]])
    assert cb.index.is_monotonic_increasing and cb.close.iloc[0] == 1.8 and cb.high.iloc[1] == 3
    ms = L.parse_binance_klines([[1577836800000, "1", "2", "0.5", "1.5", "10", 0, 0, 0, 0, 0, 0]])
    us = L.parse_binance_klines([[1735689600000000, "1", "2", "0.5", "1.5", "10", 0, 0, 0, 0, 0, 0]])
    assert ms.index[0] == pd.Timestamp("2020-01-01") and us.index[0] == pd.Timestamp("2025-01-01")


def test_quality_catches_injected_errors():
    df = synth_frame(n=300)
    assert [i for i in Q.check_ohlcv(df) if i.severity == "error"] == []
    bad = df.copy()
    bad.iloc[50, bad.columns.get_loc("high")] = bad.iloc[50]["low"] * 0.5          # high below low
    bad.iloc[100, bad.columns.get_loc("close")] *= 3                               # 200% bad tick
    bad.iloc[150, bad.columns.get_loc("close")] = -1.0
    checks = {i.check for i in Q.check_ohlcv(bad)}
    assert {"high_below_ohlc", "extreme_returns", "nonpositive_close"} <= checks
    dup = pd.concat([df, df.iloc[[5]]]).sort_index()
    assert "duplicate_dates" in {i.check for i in Q.check_ohlcv(dup)}
    gap = df.drop(df.index[100:120])
    assert "calendar_gaps" in {i.check for i in Q.check_ohlcv(gap)}
    with pytest.raises(ValueError):
        Q.assert_clean(bad, "bad")


def test_open_equals_prev_close_artifact_is_flagged():
    df = synth_frame(n=300)
    df["open"] = df["close"].shift(1).fillna(df["open"])
    assert "open_equals_prev_close" in {i.check for i in Q.check_ohlcv(df)}


def test_spot_check_agrees_on_identical_and_flags_corruption():
    a = synth_frame(n=300)
    same = Q.spot_check(a, a.copy())
    assert same["pct_within_tol"] == 1.0 and same["corr_returns"] == pytest.approx(1.0)
    b = a.copy()
    b.iloc[200:, b.columns.get_loc("close")] *= 1.05                               # step error in one source
    res = Q.spot_check(a, b)
    assert res["pct_within_tol"] < 1.0 and res["worst_days"][0][0] == str(a.index[200].date())
