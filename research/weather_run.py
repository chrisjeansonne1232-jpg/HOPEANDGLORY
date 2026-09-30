"""Run the pre-registered Kalshi-weather grid (RESEARCH_LOG.md section 9 + addendum). Holdout rows (D > val_end) are dropped on load."""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from lab import metrics as M
from lab import trials as T
from lab.config import DATA_RAW, PREDICTION_SPLITS
from lab.data import cache
from research import weather as W

SPL = PREDICTION_SPLITS
RATIONALE = ("economic: retail bettors anchor on headline temperatures/stale forecasts while NBM gives a calibrated distribution; counterparty = "
             "those bettors. Fees peak at mid prices, so only large model edges survive.")
RATIONALE_FAV = ("economic: favourite-longshot bias - retail overpays for low-probability outcomes (lottery preference); counterparty = longshot buyers; "
                 "fees are tiny at extreme prices. Model-free.")
OUT = DATA_RAW.parent / "processed"


def load_all(series_list) -> pd.DataFrame:
    parts = []
    for ser in series_list:
        d = W.build_city(ser)
        d = d[d.D <= pd.Timestamp(SPL.val_end)]                      # the holdout is never loaded for research
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = W.add_rolling_bias(df)
    return df


def spy_calendar_returns(index: pd.DatetimeIndex) -> pd.Series:
    px = cache.load("yahoo", "SPY")["adj_close"]
    r = px.pct_change()
    return r.reindex(index).fillna(0.0)                              # weekends/holidays: no move


def rf_calendar(index) -> pd.Series:
    irx = cache.load("yahoo", "^IRX")["close"].ffill() / 100.0
    return (irx.reindex(index.union(irx.index)).ffill().reindex(index) / 365.0)


def split_bounds(first: pd.Timestamp):
    return {"DEVELOPMENT": (first, pd.Timestamp(SPL.dev_end)), "VALIDATION": (pd.Timestamp(SPL.dev_end) + pd.Timedelta(days=1), pd.Timestamp(SPL.val_end)),
            "DEV+VAL": (first, pd.Timestamp(SPL.val_end))}


def run(series_list, universe: str, verbose=True):
    df = load_all(series_list)
    df = df.dropna(subset=["txn", "bias"])                          # burn-in / missing forecast rows excluded from ALL evaluation
    first = df.D.min()
    sig = {dtn: W.fit_sigma(df, dtn) for dtn in W.DT_SPEC}
    if verbose:
        print("sigma fits (dev only):", {k: {a: round(b, 3) for a, b in v.items()} for k, v in sig.items()})
    df["p"] = np.nan
    for dtn in W.DT_SPEC:
        m = df.dt_name == dtn
        df.loc[m, "p"] = W.p_yes(df[m], sig[dtn]["a"], sig[dtn]["c"])
    bounds = split_bounds(first)
    idx_all = pd.date_range(first, pd.Timestamp(SPL.val_end), freq="D")
    bench = spy_calendar_returns(idx_all)
    rf = rf_calendar(idx_all)
    bsh = {s: M.sharpe(bench.loc[a:b], rf.loc[a:b], 365) for s, (a, b) in bounds.items()}
    results = []
    configs = []
    for dtn in W.DT_SPEC:
        for th in (0.03, 0.05, 0.08):
            configs.append(("kalshi_weather_nbm", f"{dtn}_theta{th}", {"dt": dtn, "theta": th, "topk": 8, "frac": 0.01}, dtn, "model", th, None))
        for L in (0.05, 0.10):
            configs.append(("kalshi_weather_fav", f"{dtn}_fav{L}", {"dt": dtn, "longshot": L, "frac": 0.0025}, dtn, "fav", 0.0, L))
    for fam, var, params, dtn, mode, th, L in configs:
        sub = df[df.dt_name == dtn]
        for mult in (1.0, 2.0):
            tr = W.simulate(sub, sub.p.to_numpy(), th, slip=W.SLIP * mult, fee_mult=mult, mode=mode, longshot=L, frac=params["frac"],
                            topk=params.get("topk", 0) or 0)
            summ = {}
            for s, (a, b) in bounds.items():
                t_s = tr[(tr.D >= a) & (tr.D <= b)]
                summ[s] = W.summarize_trades(t_s, a, b)
            if mult == 1.0:
                tr.to_parquet(OUT / f"weather_trades_{var}_{universe}.parquet") if OUT.exists() else None
            for s in ("DEVELOPMENT", "VALIDATION", "DEV+VAL"):
                a, b = bounds[s]
                if not summ[s]:
                    continue
                T.log_trial(family=fam, variant=var, params={**params, "cities": len(series_list)}, universe=universe,
                            data_source="Kalshi hourly candles + IEM NBM(NBS) archive", data_range=f"{a.date()}..{b.date()}", split=s, metrics=summ[s],
                            n_trades=summ[s]["n_trades"], cost_mult=mult, bench_sharpe=bsh[s], rationale=RATIONALE if mode == "model" else RATIONALE_FAV,
                            notes=f"win_rate={summ[s]['win_rate']:.3f}; avg_pnl_per_trade=${summ[s]['avg_pnl_per_trade']:.2f}")
            if verbose and mult == 1.0:
                d, v, dv = summ["DEVELOPMENT"], summ["VALIDATION"], summ["DEV+VAL"]
                g = lambda x, k, f: (f.format(x[k]) if x and k in x else "  n/a")
                print(f"  {var:<18} DEV sh {g(d,'sharpe','{:+.2f}')} n={g(d,'n_trades','{:.0f}')} win {g(d,'win_rate','{:.2f}')} | VAL sh {g(v,'sharpe','{:+.2f}')} "
                      f"n={g(v,'n_trades','{:.0f}')} win {g(v,'win_rate','{:.2f}')} | DV sh {g(dv,'sharpe','{:+.2f}')} cagr {g(dv,'cagr','{:+.1%}')} dd {g(dv,'max_dd','{:.0%}')} "
                      f"(SPY {bsh['DEV+VAL']:+.2f})")
            results.append((var, mult, summ, tr))
    return df, sig, results


if __name__ == "__main__":
    ser = sys.argv[1].split(",") if len(sys.argv) > 1 else list(W.CITY_STATION)
    run(ser, universe=f"Kalshi KXHIGH x{len(ser)} cities")
