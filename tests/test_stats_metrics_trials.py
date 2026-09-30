import numpy as np
import pandas as pd
import pytest

from lab import metrics as M
from lab import stats as S
from lab import trials as T


def test_expected_max_sr_matches_closed_form():
    # (1-g)*Phi^-1(1-1/N) + g*Phi^-1(1-1/(N e)) for N=10, unit variance = 1.5749 (hand calc)
    assert S.expected_max_sr(10, 1.0) == pytest.approx(1.5749, abs=2e-3)
    assert S.expected_max_sr(1, 1.0) == 0.0


def test_dsr_reduces_to_psr_for_one_trial_and_falls_with_more_trials():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(0.0006, 0.01, 2500))
    one = S.dsr(x, 1, 1e-4)
    assert one["dsr"] == pytest.approx(one["psr_vs_zero"])
    d10, d100, d1000 = (S.dsr(x, n, 1e-4)["dsr"] for n in (10, 100, 1000))
    assert d10 > d100 > d1000


def test_psr_increases_with_sample_length():
    a = S.psr(0.05, 0.0, 250, 0.0, 3.0)
    b = S.psr(0.05, 0.0, 2500, 0.0, 3.0)
    assert b > a and 0.5 < a < 1


def test_noise_strategies_do_not_survive_deflation():
    """Best of 200 pure-noise strategies has a flattering Sharpe; DSR (given N=200) must reject it."""
    rng = np.random.default_rng(7)
    R = pd.DataFrame(rng.normal(0, 0.01, (1250, 200)))
    srs = R.apply(S.sr_period)
    best = R[srs.idxmax()]
    res = S.dsr(best, 200, float(srs.var(ddof=1)))
    assert res["sr_annualized_equiv"] > 1.0          # looks like a great strategy...
    assert res["psr_vs_zero"] > 0.95                 # ...and would pass a naive test...
    assert res["dsr"] < 0.95                         # ...but not after deflating for 200 trials


def test_sharpe_diff_test_detects_clear_difference():
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(0.0002, 0.01, 3000))
    a = b + 0.0006 + rng.normal(0, 0.002, 3000)
    z, p = S.sharpe_diff_test(a, b)
    assert z > 3 and p < 0.01
    z2, p2 = S.sharpe_diff_test(b, a)
    assert p2 > 0.99


def test_mc_trades_known_answer():
    res = S.mc_trades([0.01] * 50, n_sims=500)
    assert res["final_equity_p05"] == pytest.approx(1.01**50)
    assert res["prob_final_below_1"] == 0.0
    losers = S.mc_trades([-0.02] * 30, n_sims=500)
    assert losers["prob_final_below_1"] == 1.0 and losers["maxdd_p05_worst"] < -0.4


def test_mc_block_bootstrap_shapes():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0004, 0.01, 1500), index=pd.bdate_range("2010-01-04", periods=1500))
    res = S.mc_block_bootstrap(r, n_sims=1000)
    assert res["final_equity_p05"] < res["final_equity_p50"] < res["final_equity_p95"]
    assert -1 < res["maxdd_p05_worst"] < 0


def test_drawdown_and_extremes():
    idx = pd.bdate_range("2020-01-01", periods=6)
    r = pd.Series([0.10, -0.50, 0.20, 0.0, 0.0, 0.0], index=idx)
    assert M.max_drawdown(r) == pytest.approx(-0.5)
    assert M.worst_day(r) == -0.5
    assert M.max_dd_duration(r) == 5
    assert M.worst_month(r) == pytest.approx(1.1 * 0.5 * 1.2 - 1)   # whole series is in one month


def test_sharpe_known_value():
    r = pd.Series([0.01, -0.01] * 100)
    assert M.sharpe(r, None, 252) == pytest.approx(0.0, abs=1e-12)
    r2 = pd.Series([0.02, 0.0] * 100)
    assert M.sharpe(r2, None, 252) == pytest.approx(0.01 / r2.std(ddof=1) * np.sqrt(252))


def test_trials_ledger_dedupe_and_counting(tmp_path):
    f = tmp_path / "trials.csv"
    m = {"n_periods": 100, "sharpe": 1.0, "sr_period": 0.06, "cagr": 0.1, "max_dd": -0.1, "skew": 0.0, "kurt": 3.0}
    kw = dict(family="trend", universe="SPY", data_source="unit", data_range="a..b", split="DEV+VAL", metrics=m, path=f)
    a = T.log_trial(variant="sma200", params={"n": 200}, **kw)
    a2 = T.log_trial(variant="sma200", params={"n": 200}, **kw)          # duplicate evaluation -> not re-appended
    b = T.log_trial(variant="sma160", params={"n": 160}, **kw)
    T.log_trial(variant="idea", params={}, status="needs_data", notes="needs data: OPRA quotes", **{**kw, "metrics": None})
    assert a == a2 and a != b
    assert len(T.load(f)) == 3
    assert T.n_trials(f) == 2                                             # needs_data is not a tested trial
