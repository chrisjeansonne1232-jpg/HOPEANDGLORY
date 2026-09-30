"""Diagnostics for the weather dataset (coverage, forecast error by year, calibration, model-vs-market Brier). No trading rules, no trials."""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from research import weather as W, weather_run as R

def main(series):
    df = R.load_all(series)
    print("rows", len(df), "| series", df.series.nunique(), "| D", df.D.min().date(), df.D.max().date())
    for dtn in W.DT_SPEC:
        g = df[df.dt_name == dtn]
        print(f"{dtn}: forecast matched {g.txn.notna().mean():.3f} | quote present {g.ask_close.notna().mean():.3f} | with bias {g.bias.notna().mean():.3f}")
    ok = df.dropna(subset=["txn", "bias"])
    print("\nforecast error (actual - NBM txn) by series/year, DT1 (before bias correction):")
    print(W.forecast_error_report(df).to_string())
    sig = {d: W.fit_sigma(ok, d) for d in W.DT_SPEC}
    print("\nsigma fits (dev only):", {k: {a: round(b, 3) for a, b in v.items()} for k, v in sig.items()})
    ok = ok.copy(); ok["p"] = np.nan
    for d in W.DT_SPEC:
        m = ok.dt_name == d
        ok.loc[m, "p"] = W.p_yes(ok[m], sig[d]["a"], sig[d]["c"])
    q = ok.dropna(subset=["bid_close", "ask_close"]).copy()
    q = q[(q.ask_close < 0.99) & (q.bid_close > 0.01)]
    q["mid"] = (q.bid_close + q.ask_close) / 2
    for dtn in W.DT_SPEC:
        for nm, lo, hi in (("DEV", pd.Timestamp("2000-01-01"), pd.Timestamp(R.SPL.dev_end)), ("VAL", pd.Timestamp(R.SPL.dev_end) + pd.Timedelta(days=1), pd.Timestamp(R.SPL.val_end))):
            g = q[(q.dt_name == dtn) & (q.D >= lo) & (q.D <= hi)]
            if len(g):
                print(f"{dtn} {nm}: n={len(g):>6} Brier model {((g.p-g.y)**2).mean():.4f} | Brier market mid {((g.mid-g.y)**2).mean():.4f} | mean spread {(g.ask_close-g.bid_close).mean():.3f}")
    g = q[q.dt_name == "DT1"].copy(); g["bucket"] = pd.cut(g.mid, [0, .05, .1, .2, .35, .5, .65, .8, .9, .95, 1])
    print("\nMARKET calibration at DT1 (all dev+val): mid bucket -> n, mean mid, realised YES freq, mean model p")
    print(g.groupby("bucket", observed=True).agg(n=("y", "size"), mid=("mid", "mean"), freq=("y", "mean"), model=("p", "mean")).round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else list(W.CITY_STATION))
