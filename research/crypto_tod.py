"""BTC time-of-day: hold only during a UTC session window each day, paying both legs' costs daily. Pre-registered variants: three sessions."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from lab import metrics as M, trials as T
from lab.config import CRYPTO_COST, CRYPTO_SPLITS
from lab.data import cache
from lab.stats import sr_period

WINDOWS = {"asia_00_08": (0, 8), "europe_08_16": (8, 16), "us_14_22": (14, 22)}
RAT = ("economic: crypto flow follows regional trading-session liquidity (equity-hours correlation, Asian retail); counterparty = "
       "session-agnostic holders. Weak prior; retail fees ~1.2%/day round trip make it near-impossible.")


def run(verbose=True):
    h = cache.load("coinbase", "BTC-USD_3600")
    r = (h["close"] / h["close"].shift(1) - 1.0)
    idx_dev, idx_val = pd.Timestamp(CRYPTO_SPLITS.dev_end), pd.Timestamp(CRYPTO_SPLITS.val_end)
    r = r[r.index >= "2016-02-15"]
    r = r[r.index <= idx_val + pd.Timedelta(days=1)]                      # holdout never loaded
    day = r.index.normalize()
    by_hour = r.groupby(r.index.hour).mean() * 1e4
    if verbose:
        print("gross mean return by UTC hour (bps), dev+val:", by_hour.round(1).to_dict())
    per_side_bps = CRYPTO_COST.fee_bps + CRYPTO_COST.half_spread_bps + CRYPTO_COST.slippage_bps
    btc = cache.load("coinbase", "BTC-USD_86400")["close"].pct_change()
    rf = (cache.load("yahoo", "^IRX")["close"].ffill() / 100 / 365)
    out = {}
    for name, (a, b) in WINDOWS.items():
        m = (r.index.hour >= a) & (r.index.hour < b)
        gross = (1 + r[m]).groupby(day[m]).prod() - 1
        net = gross - 2 * per_side_bps / 1e4
        days = pd.date_range(gross.index.min(), idx_val, freq="D")
        g, n = gross.reindex(days).fillna(0.0), net.reindex(days).fillna(0.0)
        rfd = rf.reindex(days.union(rf.index)).ffill().reindex(days)
        for lab, s, (lo, hi) in (("DEVELOPMENT", None, (days[0], idx_dev)), ("VALIDATION", None, (idx_dev + pd.Timedelta(days=1), idx_val)), ("DEV+VAL", None, (days[0], idx_val))):
            sn = n.loc[lo:hi]
            summ = M.summarize(sn, rfd.loc[lo:hi], 365)
            summ["sr_period"] = sr_period(M.excess(sn, rfd.loc[lo:hi]))
            summ["n_trades"] = int((g.loc[lo:hi] != 0).sum())
            summ["gross_sharpe"] = M.sharpe(g.loc[lo:hi], rfd.loc[lo:hi], 365)
            bsh = M.sharpe(btc.loc[lo:hi], rfd.loc[lo:hi], 365)
            T.log_trial(family="crypto_time_of_day", variant=name, params={"window_utc": [a, b], "fee_bps_side": per_side_bps}, universe="BTC-USD hourly (Coinbase)",
                        data_source="coinbase exchange candles 3600s", data_range=f"{lo.date()}..{hi.date()}", split=lab, metrics=summ, n_trades=summ["n_trades"],
                        bench_sharpe=bsh, rationale=RAT)
            if lab == "DEV+VAL" and verbose:
                print(f"  {name:<14} gross Sharpe {summ['gross_sharpe']:+.2f} -> net Sharpe {summ['sharpe']:+.2f} | net CAGR {summ['cagr']:+.1%} | mean gross/day {g.loc[lo:hi].mean()*1e4:+.1f}bps "
                      f"vs cost {2*per_side_bps:.0f}bps/day | BTC B&H Sharpe {bsh:+.2f}")
        out[name] = (g, n)
    return out


if __name__ == "__main__":
    run()
