"""Kalshi public market data (no auth): settled-market listings (archived + live) and hourly bid/ask candlesticks.
Verified live 2026-09-30: /historical/markets (archive, older than the cutoff) + /markets?status=settled (recent) +
/series/{series}/markets/{ticker}/candlesticks (serves both eras). The batch /markets/candlesticks endpoint returns nothing for archived markets."""
from __future__ import annotations

import datetime as dt
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from ..config import DATA_RAW

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT = DATA_RAW / "kalshi"
_lock, _last = threading.Lock(), [0.0]


def _get(path: str, params: dict, retries: int = 6, min_interval: float = 0.07):
    for i in range(retries):
        with _lock:                                   # global pacing (~12 req/s max across threads)
            wait = _last[0] + min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = requests.get(BASE + path, params=params, timeout=60)
        except requests.exceptions.RequestException:          # dropped connection / SSL EOF / timeout: back off and retry
            time.sleep(min(2 ** i, 30))
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** i, 30))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
    raise RuntimeError(f"{path} failed after {retries} tries")


def _ts(x: str) -> int:
    return int(dt.datetime.fromisoformat(x.replace("Z", "+00:00")).timestamp())


MARKET_COLS = ["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "result", "expiration_value", "open_time",
               "close_time", "settlement_ts", "volume_fp", "open_interest_fp", "status", "yes_sub_title"]


def list_markets(series: str) -> pd.DataFrame:
    rows = []
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra, **({"cursor": cursor} if cursor else {})}
            d = _get(path, p)
            if not d or not d.get("markets"):
                break
            rows += d["markets"]
            cursor = d.get("cursor")
            if not cursor:
                break
    if not rows:
        return pd.DataFrame(columns=MARKET_COLS)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[[c for c in MARKET_COLS if c in df.columns]].copy()
    for c in ("floor_strike", "cap_strike", "volume_fp", "open_interest_fp"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["expiration_value"] = pd.to_numeric(df["expiration_value"], errors="coerce")
    for c in ("open_time", "close_time", "settlement_ts"):
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601")
    return df.sort_values("close_time").reset_index(drop=True)


_CUTOFF = [None]


def _cutoff_ts() -> int:
    if _CUTOFF[0] is None:
        _CUTOFF[0] = _ts(_get("/historical/cutoff", {})["market_settled_ts"])
    return _CUTOFF[0]


def _candles_one(series: str, ticker: str, o: int, c: int, period: int = 60) -> list[dict]:
    """Archived markets (closed before the historical cutoff) live behind /historical/...; recent ones behind /series/..."""
    path = f"/historical/markets/{ticker}/candlesticks" if c < _cutoff_ts() else f"/series/{ticker.split('-')[0]}/markets/{ticker}/candlesticks"
    d = _get(path, {"start_ts": o, "end_ts": c, "period_interval": period})
    out = []
    for k in (d or {}).get("candlesticks", []):
        def f(blk, fld):                                   # live format: "close_dollars"; archived format: "close"
            b = k.get(blk) or {}
            v = b.get(f"{fld}_dollars", b.get(fld))
            return float(v) if v not in (None, "") else None
        vol = k.get("volume_fp", k.get("volume"))
        oi = k.get("open_interest_fp", k.get("open_interest"))
        out.append({"ticker": ticker, "end_ts": k["end_period_ts"], "bid_close": f("yes_bid", "close"), "ask_close": f("yes_ask", "close"),
                    "bid_open": f("yes_bid", "open"), "ask_open": f("yes_ask", "open"), "price_close": f("price", "close"),
                    "volume": float(vol or 0), "oi": float(oi or 0)})
    return out


def fetch_series(series: str, workers: int = 8, min_volume: float = 1.0, max_markets: int | None = None, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (resumable) markets + hourly candles for one series; returns (markets, candles)."""
    OUT.mkdir(parents=True, exist_ok=True)
    mk_path, cd_path = OUT / f"{series}_markets.parquet", OUT / f"{series}_candles.parquet"
    mk = list_markets(series)
    mk.to_parquet(mk_path)
    todo = mk[(mk.volume_fp.fillna(0) >= min_volume)]
    have = pd.read_parquet(cd_path) if cd_path.exists() else pd.DataFrame()
    done = set(have.ticker.unique()) if len(have) else set()
    todo = todo[~todo.ticker.isin(done)]
    if max_markets:
        todo = todo.head(max_markets)
    log(f"{series}: {len(mk)} markets ({(mk.volume_fp.fillna(0)>=min_volume).sum()} with volume); {len(done)} already have candles; fetching {len(todo)}")
    rows, n_done, t0 = [], 0, time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_candles_one, series, r.ticker, _ts(r.open_time.isoformat()), _ts(r.close_time.isoformat())): r.ticker for r in todo.itertuples()}
        for f in as_completed(futs):
            rows += f.result()
            n_done += 1
            if n_done % 500 == 0:
                log(f"  {series}: {n_done}/{len(todo)} markets ({time.time()-t0:.0f}s)")
                have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True); rows = []
                have.to_parquet(cd_path)
    have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True) if rows else have
    have.to_parquet(cd_path)
    log(f"{series}: done, {len(have)} candle rows")
    return mk, have
