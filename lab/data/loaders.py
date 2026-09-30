"""Free-source loaders. Parsing is separated from HTTP so parsers can be unit-tested against format fixtures.

STATUS: UNTESTED LIVE. The research container's network policy blocked every one of these hosts (HTTP 403 on CONNECT) when
this file was written, so only the parsers are tested (against hand-written fixtures in the documented formats). The first
thing to do once the hosts are allowed: run scripts/check_network.py, then scripts/fetch_core_data.py, and fix whatever the
real responses reveal. Never trust a loader's output until lab.data.quality checks and a cross-source spot check pass.

Not written yet (need real responses to build against): Kalshi, Polymarket, NOAA/NCEI weather.
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile

import numpy as np
import pandas as pd

from . import cache
from .http import get

OHLCV_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


# ---------------------------------------------------------------- Stooq (daily equities/ETFs/indices)
def parse_stooq_csv(text: str) -> pd.DataFrame:
    if not text.strip() or text.strip().lower().startswith("no data") or "Date" not in text.splitlines()[0]:
        raise ValueError(f"stooq returned no usable data: {text[:80]!r}")
    df = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date").sort_index()
    df.columns = [c.lower() for c in df.columns]
    df.index.name = "date"
    df["adj_close"] = df["close"]           # adjustment basis of Stooq closes is unverified -> cross-check vs Yahoo adj_close
    if "volume" not in df:
        df["volume"] = np.nan
    return df[OHLCV_COLS]


def fetch_stooq(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """symbol like 'spy.us' or '^spx'."""
    if use_cache and (c := cache.load("stooq", symbol)) is not None:
        return c
    url = "https://stooq.com/q/d/l/"
    df = parse_stooq_csv(get(url, params={"s": symbol.lower(), "i": "d"}))
    cache.save(df, "stooq", symbol, url=f"{url}?s={symbol}&i=d", note="adjustment basis unverified")
    return df


# ---------------------------------------------------------------- Yahoo chart API (daily, with adjusted close)
def parse_yahoo_chart(js: dict) -> pd.DataFrame:
    res = js["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s") + pd.to_timedelta(res["meta"].get("gmtoffset", 0), unit="s")
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "adj_close": adj,
                       "volume": q["volume"]}, index=ts.normalize())
    df.index.name = "date"
    return df.dropna(subset=["close"])[OHLCV_COLS]


def fetch_yahoo(symbol: str, start: str = "1990-01-01", end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    if use_cache and (c := cache.load("yahoo", symbol)) is not None:
        return c
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end or dt.date.today()).timestamp()) + 86400
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    df = parse_yahoo_chart(get(url, params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"}, as_json=True))
    cache.save(df, "yahoo", symbol, url=url, note="unofficial API; adj_close is dividend+split adjusted")
    return df


# ---------------------------------------------------------------- FRED (macro, risk-free rate)
def parse_fred_csv(text: str, series: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    date_col = df.columns[0]
    s = pd.to_numeric(df[series], errors="coerce")
    s.index = pd.to_datetime(df[date_col])
    s.name = series
    s.index.name = "date"
    return s


def fetch_fred(series: str, use_cache: bool = True) -> pd.Series:
    if use_cache and (c := cache.load("fred", series)) is not None:
        return c[series]
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    s = parse_fred_csv(get(url, params={"id": series}), series)
    cache.save(s.to_frame(), "fred", series, url=f"{url}?id={series}")
    return s


# ---------------------------------------------------------------- Coinbase Exchange candles (spot crypto)
def parse_coinbase_candles(rows: list) -> pd.DataFrame:
    """Each row: [time, low, high, open, close, volume] (descending time)."""
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df.pop("t"), unit="s")
    df.index.name = "date"
    df = df.sort_index()
    df["adj_close"] = df["close"]
    return df[OHLCV_COLS]


def fetch_coinbase(product: str = "BTC-USD", granularity: int = 86400, start: str = "2015-01-01", end: str | None = None,
                   use_cache: bool = True) -> pd.DataFrame:
    key = f"{product}_{granularity}"
    if use_cache and (c := cache.load("coinbase", key)) is not None:
        return c
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    t, end_ts, span, out = pd.Timestamp(start), pd.Timestamp(end or dt.datetime.utcnow()), pd.Timedelta(seconds=granularity * 299), []
    while t < end_ts:
        e = min(t + span, end_ts)
        rows = get(url, params={"granularity": granularity, "start": t.isoformat(), "end": e.isoformat()}, as_json=True)
        if rows:
            out.append(parse_coinbase_candles(rows))
        t = e
    df = pd.concat(out).sort_index()
    df = df[~df.index.duplicated()]
    cache.save(df, "coinbase", key, url=url)
    return df


# ---------------------------------------------------------------- Binance public klines (data only; Binance.com is not tradable from the US)
def parse_binance_klines(rows: list) -> pd.DataFrame:
    """12-field klines: open_time, o, h, l, c, volume, close_time, quote_vol, trades, taker_base, taker_quote, ignore.
    Spot files switched from millisecond to microsecond timestamps in 2025; both handled."""
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["t", "open", "high", "low", "close", "volume"]
    t = pd.to_numeric(df.pop("t"))
    unit = "us" if t.iloc[0] > 1e14 else "ms"
    df.index = pd.to_datetime(t, unit=unit)
    df.index.name = "date"
    df = df.astype(float)
    df["adj_close"] = df["close"]
    return df[OHLCV_COLS]


def parse_binance_vision_zip(payload: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        with z.open(z.namelist()[0]) as f:
            return parse_binance_klines(pd.read_csv(f, header=None).values.tolist())


def fetch_binance_vision(symbol: str = "BTCUSDT", interval: str = "1d", months: list[str] | None = None) -> pd.DataFrame:
    """Bulk monthly kline zips from data.binance.vision (months like '2021-03')."""
    frames = []
    for m in months or []:
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{m}.zip"
        frames.append(parse_binance_vision_zip(get(url, as_bytes=True)))
    df = pd.concat(frames).sort_index()
    cache.save(df, "binance", f"{symbol}_{interval}", url="https://data.binance.vision")
    return df
