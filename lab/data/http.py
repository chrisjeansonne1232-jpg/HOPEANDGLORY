"""HTTP helper with polite retries and an explicit, actionable error when the sandbox blocks a host."""
from __future__ import annotations

import time

import requests


class NetworkBlocked(RuntimeError):
    """Host unreachable (typically the cloud sandbox's egress policy answering 403 to CONNECT)."""


HEADERS = {"User-Agent": "Mozilla/5.0 (research-lab; personal use)"}


def get(url: str, params: dict | None = None, headers: dict | None = None, retries: int = 3, timeout: int = 30,
        as_json: bool = False, as_bytes: bool = False):
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=timeout)
        except (requests.exceptions.ProxyError, requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            raise NetworkBlocked(f"cannot reach {url}: {type(e).__name__}. In the Claude cloud sandbox add this host to the "
                                 "environment's allowed domains (Network access setting).") from e
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            time.sleep(2 ** i)
            continue
        r.raise_for_status()
        return r.json() if as_json else (r.content if as_bytes else r.text)
    raise RuntimeError(f"{url}: gave up after {retries} tries ({last})")
