"""Phase-3 battery for the BTC trend x vol-target candidate (the best equity/crypto DSR). The candidate is fixed BEFORE running:
rule = BTC-USD close > SMA(150); size = min(1, 0.40 / realised 30d vol) x SCALE, with SCALE chosen only so full-history dev+val max drawdown <= 25% (risk sizing, not tuning of edge).
Robustness variants (+-20% on every parameter) and related asset (ETH) are logged as trials. Holdout NOT loaded here."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from lab import metrics as M
from lab.config import CRYPTO_COST, CRYPTO_SPLITS
from research import signals as S
from research.common import Runner
from research import phase3

RAT = ("economic: retail/flow-driven trends in a frictional market with few arbitrageurs; counterparty = late momentum chasers; "
       "vol-targeted sizing to control tail risk.")


def scaled(fn, k):
    def f(panel):
        return fn(panel) * k
    return f


def series_by_split(res, runner):
    return {s: runner._slice(res, s).returns for s in ("DEVELOPMENT", "VALIDATION", "DEV+VAL")}


def main(scale=None, verbose=True):
    sym = "BTC-USD_86400"
    r = Runner([sym], "2016-02-15", "BTC", splits=CRYPTO_SPLITS, bench=sym, cost=CRYPTO_COST, ppy=365, src="coinbase", verbose=verbose)
    base = r.run("crypto_finalist", "BTC_sma150_vt40_x1", {"n": 150, "target": 0.4, "lookback": 30, "scale": 1.0}, S.trend_voltarget_crypto(sym, 150, 0.4, 30), RAT, log=False)
    full_dd = M.max_drawdown(r._slice(base["res"], "DEV+VAL").returns)
    k = scale or min(1.0, 0.25 / abs(full_dd) * 0.9)          # 10% safety margin below the 25% limit
    print(f"full-size dev+val max DD {full_dd:.1%} -> scale {k:.2f}")
    fn = lambda n=150, tgt=0.4, lb=30: scaled(S.trend_voltarget_crypto(sym, n, tgt, lb), k)
    v = {}
    v["base"] = r.run("crypto_finalist", f"BTC_sma150_vt40_x{k:.2f}", {"n": 150, "target": 0.4, "lookback": 30, "scale": round(k, 2)}, fn(), RAT)
    nud = {}
    for lab, kw in (("n120", dict(n=120)), ("n180", dict(n=180)), ("vt32", dict(tgt=0.32)), ("vt48", dict(tgt=0.48)), ("lb24", dict(lb=24)), ("lb36", dict(lb=36))):
        o = r.run("crypto_finalist", f"BTC_{lab}", {**{"n": 150, "target": 0.4, "lookback": 30, "scale": round(k, 2)}, "nudge": lab}, fn(**kw), RAT, notes="+-20% robustness")
        nud[lab] = series_by_split(o["res"], r)
    c2 = r.run("crypto_finalist", f"BTC_sma150_vt40_x{k:.2f}", {"n": 150, "target": 0.4, "lookback": 30, "scale": round(k, 2)}, fn(), RAT, cost_mult=2.0)
    e = Runner(["ETH-USD_86400"], "2017-01-01", "ETH", splits=CRYPTO_SPLITS, bench="ETH-USD_86400", cost=CRYPTO_COST, ppy=365, src="coinbase", verbose=verbose)
    eo = e.run("crypto_finalist", f"ETH_sma150_vt40_x{k:.2f}", {"n": 150, "target": 0.4, "lookback": 30, "scale": round(k, 2), "asset": "ETH"},
               scaled(S.trend_voltarget_crypto("ETH-USD_86400", 150, 0.4, 30), k), RAT, notes="related asset")
    ret = series_by_split(v["base"]["res"], r)
    bench = series_by_split(r.bench, r)["DEV+VAL"]
    tr = v["base"]["res"].trades
    oos = int((tr.entry > pd.Timestamp(CRYPTO_SPLITS.dev_end)).sum())
    out = phase3.evaluate("BTC trend x vol-target (SMA150, 40% vol, scaled x%.2f)" % k, ret, bench, r.rf, 365, tr.pnl_frac.to_numpy(), oos,
                          nudges=nud, cost2x=series_by_split(c2["res"], r), related={"ETH": series_by_split(eo["res"], e)})
    return out, r


if __name__ == "__main__":
    out, _ = main()
    print(phase3.to_markdown(out))
