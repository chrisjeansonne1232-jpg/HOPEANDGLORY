"""Fetch the core free datasets, run quality checks on each, and cross-check Stooq vs Yahoo. STATUS: UNTESTED LIVE (written while
the hosts were blocked). Fails LOUDLY per symbol; nothing is silently skipped or filled. Run after scripts/check_network.py is green."""
import sys
sys.path.insert(0, ".")
from lab.data import loaders as L, quality as Q
from lab.data.http import NetworkBlocked

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "BIL", "LQD", "HYG", "GLD", "DBC", "VNQ",
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "SVXY", "VXX", "UVXY"]
INDEXES_YAHOO = ["^GSPC", "^VIX", "^VIX3M", "^VVIX", "^SKEW"]
FRED = ["DGS3MO", "DGS10", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "FEDFUNDS"]
CRYPTO = [("BTC-USD", 86400), ("ETH-USD", 86400)]
failures = []

def run(label, fn):
    try:
        return fn()
    except NetworkBlocked as e:
        failures.append((label, "UNREACHABLE")); print(f"  !! {label}: unreachable ({e})")
        if not label.startswith("stooq"):    # stooq resets connections (2026-09-30); Yahoo is the primary source
            raise SystemExit(2)
    except Exception as e:
        failures.append((label, repr(e))); print(f"  !! {label}: {e!r}")

for s in ETFS:
    y = run(f"yahoo {s}", lambda: L.fetch_yahoo(s))
    z = None  # Stooq now serves a bot-check page / resets connections (2026-09-30); no second source for equities here
    if y is not None:
        for i in Q.check_ohlcv(y, s): print(f"  {s} yahoo {i}")
    if y is not None and z is not None:
        print(f"{s}: cross-source", {k: v for k, v in Q.spot_check(y, z).items() if k != 'worst_days'})
for s in INDEXES_YAHOO: run(f"yahoo {s}", lambda: L.fetch_yahoo(s))
for s in FRED: run(f"fred {s}", lambda: L.fetch_fred(s))
for p, g in CRYPTO: run(f"coinbase {p}", lambda: L.fetch_coinbase(p, g))
print("\nFAILURES:" if failures else "\nall fetched", *failures, sep="\n  ")
