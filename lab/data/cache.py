"""Cache with provenance. Raw files live in data/raw/<source>/<key>.parquet (git-ignored); a small manifest
(data/manifest.json, tracked) records source, url, fetch time, rows, date range and sha256 so results are traceable and a
future session can re-fetch and verify it got the same bytes."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib

import pandas as pd

from ..config import DATA_RAW, ROOT

MANIFEST = ROOT / "data" / "manifest.json"


def _p(source: str, key: str) -> pathlib.Path:
    safe = key.replace("/", "_").replace("^", "_")
    return DATA_RAW / source / f"{safe}.parquet"


def save(df: pd.DataFrame, source: str, key: str, url: str = "", note: str = "") -> pathlib.Path:
    p = _p(source, key)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    meta = {"source": source, "key": key, "url": url, "note": note, "rows": int(len(df)),
            "first": str(df.index.min()), "last": str(df.index.max()),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    man[f"{source}/{key}"] = meta
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True))
    return p


def load(source: str, key: str) -> pd.DataFrame | None:
    p = _p(source, key)
    return pd.read_parquet(p) if p.exists() else None
