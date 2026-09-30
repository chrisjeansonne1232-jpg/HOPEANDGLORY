"""Which data hosts can this environment reach? Run this first in any new session / after changing network settings.
Any HTTP response (even 403/404 from the site itself) counts as reachable; a proxy CONNECT refusal counts as BLOCKED."""
import json, sys, datetime as dt
import requests

HOSTS = {
    "stooq (daily equities/ETFs)": "https://stooq.com/q/d/l/?s=spy.us&i=d",
    "yahoo chart (daily, adj close)": "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d",
    "fred (macro, rates)": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO",
    "coinbase candles": "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=86400",
    "binance.us klines": "https://api.binance.us/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=2",
    "binance public bulk data": "https://data.binance.vision/",
    "kalshi api": "https://api.elections.kalshi.com/trade-api/v2/markets?limit=1",
    "polymarket gamma": "https://gamma-api.polymarket.com/markets?limit=1",
    "polymarket clob": "https://clob.polymarket.com/",
    "noaa weather.gov": "https://api.weather.gov/",
    "noaa ncei (GHCN history)": "https://www.ncei.noaa.gov/",
    "iowa mesonet (forecast/MOS archive)": "https://mesonet.agron.iastate.edu/",
    "sec edgar": "https://www.sec.gov/",
    "cboe (VIX futures history)": "https://www.cboe.com/",
    "massive/polygon api": "https://api.massive.com/v2/aggs/ticker/SPY/prev",
    "ntfy.sh (final heads-up)": "https://ntfy.sh/",
    "pypi (packages)": "https://pypi.org/simple/pandas/",
}
out = {}
for name, url in HOSTS.items():
    try:
        r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        out[name] = f"reachable (HTTP {r.status_code})"
    except requests.exceptions.ProxyError as e:
        out[name] = "BLOCKED by network policy (proxy refused CONNECT)"
    except Exception as e:
        out[name] = f"ERROR {type(e).__name__}"
w = max(map(len, out))
for k, v in out.items():
    print(f"{k:<{w}}  {v}")
json.dump({"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "hosts": out},
          open("reports/network_status.json", "w"), indent=1)
sys.exit(0 if not any(v.startswith(("BLOCKED", "ERROR")) for k, v in out.items() if "pypi" not in k) else 1)
