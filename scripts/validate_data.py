"""Data validation for every cached dataset -> reports/data_validation.md. Run after any fetch."""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from lab.data import cache, quality as Q
from lab.config import DATA_RAW

man = json.load(open("data/manifest.json"))
lines = ["# Data validation report", "", "| dataset | rows | first | last | errors | notable warnings |", "|---|---|---|---|---|---|"]
opens = {}
for key, m in sorted(man.items()):
    src, name = key.split("/", 1)
    df = cache.load(src, name)
    if df is None or "close" not in df: continue
    iss = Q.check_ohlcv(df, name, max_abs_ret=0.35)
    errs = [i for i in iss if i.severity == "error"]
    warns = [i for i in iss if i.severity == "warn" and i.check not in ("calendar_gaps",)]
    if "open" in df:
        eq = ((df.open - df.close.shift(1)).abs() < 1e-9 * df.close.shift(1))
        opens[name] = float(eq.mean())
    lines.append(f"| {key} | {len(df)} | {df.index.min().date()} | {df.index.max().date()} | {len(errs)} | " + ("; ".join(f"{w.check}" for w in warns) or "-") + " |")
    for i in errs: lines.append(f"|  | | | | ERROR | {i} |")
spy, gspc = cache.load("yahoo", "SPY"), cache.load("yahoo", "^GSPC")
c = pd.concat([spy.close.pct_change(), gspc.close.pct_change()], axis=1, join="inner").dropna(); c.columns = ["spy", "gspc"]
tr = (spy.adj_close.pct_change() - gspc.close.pct_change()).dropna()
lines += ["", "## Cross-checks", f"- SPY vs ^GSPC daily price-return correlation {c.corr().iloc[0,1]:.5f} over {len(c)} days; max |diff| {abs(c.spy-c.gspc).max():.4f} on {abs(c.spy-c.gspc).idxmax().date()}",
          f"- SPY total-return (adj close) CAGR {(spy.adj_close.iloc[-1]/spy.adj_close.iloc[0])**(365.25/(spy.index[-1]-spy.index[0]).days)-1:.2%} vs price-only CAGR {(spy.close.iloc[-1]/spy.close.iloc[0])**(365.25/(spy.index[-1]-spy.index[0]).days)-1:.2%} (gap = dividends, ~1.5-2% expected)",
          f"- SPY adj-return minus index price-return: mean {tr.mean()*252:.2%}/yr (dividend yield less fee, ~+1.3-1.9% expected)",
          "", "## Fraction of opens equal to prior close (unreliable-open indicator)", "", "| symbol | fraction |", "|---|---|"]
lines += [f"| {k} | {v:.2f} |" for k, v in sorted(opens.items(), key=lambda x: -x[1])[:12]]
open("reports/data_validation.md", "w").write("\n".join(lines))
print("\n".join(lines))
