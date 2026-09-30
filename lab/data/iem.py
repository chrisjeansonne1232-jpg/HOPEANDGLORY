"""Iowa Environmental Mesonet archive of NWS model output statistics (MOS): GFS MOS (MAV) `n_x` and NBM (NBS) `txn`/`xnd`.
Bulk CSV service: cgi-bin/request/mos.py. `runtime` = model cycle time (UTC); `ftime` = valid time."""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from ..config import DATA_RAW

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
OUT = DATA_RAW / "iem"


def fetch_mos(station: str, model: str, start: str, end: str) -> pd.DataFrame:
    """Monthly chunks; keeps only rows carrying a max/min-temperature forecast (n_x for GFS, txn/xnd for NBS)."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model}_{station}.parquet"
    have = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    months = pd.period_range(start, end, freq="M")
    got = set(have["_month"].unique()) if len(have) else set()
    parts = [have] if len(have) else []
    for m in months:
        key = str(m)
        if key in got:
            continue
        s, e = m.start_time, (m + 1).start_time + pd.Timedelta(days=2)
        for attempt in range(5):
            r = requests.get(URL, params={"station": station, "model": model, "sts": s.strftime("%Y-%m-%dT%H:%MZ"), "ets": e.strftime("%Y-%m-%dT%H:%MZ"), "format": "csv"}, timeout=180)
            if r.status_code == 200:
                break
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"IEM {station} {model} {key}: HTTP {r.status_code}")
        d = pd.read_csv(io.StringIO(r.text))
        keep = ["runtime", "ftime", "station", "model"] + [c for c in ("n_x", "txn", "xnd", "tmp") if c in d.columns]
        d = d[keep]
        d = d[d["n_x"].notna()] if "n_x" in d else d[d["txn"].notna()]
        d["runtime"], d["ftime"] = pd.to_datetime(d["runtime"], utc=True), pd.to_datetime(d["ftime"], utc=True)
        d["_month"] = key
        parts.append(d)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["runtime", "ftime"]) if parts else pd.DataFrame()
    out.to_parquet(path)
    return out
