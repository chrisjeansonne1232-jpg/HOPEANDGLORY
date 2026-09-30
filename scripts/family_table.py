"""Markdown table of every strategy family in trials.csv: variants tested, best/median DEV+VAL Sharpe vs benchmark, best DSR, verdict."""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from lab import trials as T, stats as S

VERDICT = {}     # family -> one-line reason; filled in by the report author (kept in research/verdicts.py)
try:
    from research.verdicts import VERDICT as V
    VERDICT.update(V)
except Exception:
    pass

def table() -> str:
    df = T.load()
    N = T.n_trials()
    srs = T.trial_srs()
    t = df[(df.status == "tested") & (df.split == "DEV+VAL") & (df.cost_mult == 1.0)].drop_duplicates("trial_id").copy()
    t["dsr"] = [S.psr(float(r.sr_period), S.expected_max_sr(N, S.hurdle_variance(srs, int(r.n_periods))["used"]), int(r.n_periods), float(r["skew"]), float(r["kurt"])) for _, r in t.iterrows()]
    rows = ["| family | variants | best Sharpe (bench) | median Sharpe | best max DD | best DSR | verdict |", "|---|---|---|---|---|---|---|"]
    for fam, g in t.groupby("family"):
        b = g.sort_values("dsr", ascending=False).iloc[0]
        rows.append(f"| {fam} | {len(g)} | {g.sharpe_ann.max():.2f} ({b.bench_sharpe:.2f}) | {g.sharpe_ann.median():.2f} | {b.max_dd:.0%} | {g.dsr.max():.2f} | {VERDICT.get(fam, '')} |")
    nd = df[df.status == "needs_data"]
    rows += ["", "**Not testable with obtainable data (logged `needs_data`):**", ""] + [f"- {r.family} / {r.variant}: {r.notes}" for r in nd.itertuples()]
    return "\n".join(rows)

if __name__ == "__main__":
    print(table())
