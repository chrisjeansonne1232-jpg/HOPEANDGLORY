"""Rank all logged trials by Deflated Sharpe using ONLY trials.csv (selection sample = DEV+VAL rows)."""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats as sps
from lab import trials as T, stats as S

df = T.load()
t = df[(df.status == "tested") & (df.split == "DEV+VAL")].drop_duplicates(["trial_id", "cost_mult"])
t = t[t.cost_mult == 1.0].drop_duplicates("trial_id")
N, var = T.n_trials(), T.sr_variance()
sr0 = S.expected_max_sr(N, var)
print(f"N tested trials = {N} | cross-trial variance of per-period Sharpe = {var:.3e} (sd {np.sqrt(var):.4f} = {np.sqrt(var*252):.2f} annualised) | "
      f"deflation hurdle SR0 = {sr0:.5f}/period = {sr0*np.sqrt(252):.2f} annualised (equity ppy)")
rows = []
for _, r in t.iterrows():
    sr, n = float(r.sr_period), int(r.n_periods)
    d = S.psr(sr, sr0, n, float(r["skew"]), float(r["kurt"]))
    rows.append({"variant": f"{r.family}/{r.variant}", "sharpe_ann": r.sharpe_ann, "bench_sharpe": r.bench_sharpe, "cagr": r.cagr,
                 "max_dd": r.max_dd, "n": n, "trades": r.n_trades, "PSR0": S.psr(sr, 0.0, n, float(r["skew"]), float(r["kurt"])), "DSR": d})
lb = pd.DataFrame(rows).sort_values("DSR", ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 44)
print(lb.head(int(sys.argv[1]) if len(sys.argv) > 1 else 25).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\ntrials with DSR >= 0.95: {(lb.DSR >= 0.95).sum()} of {len(lb)}")
