from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class DiskTTLCache:
    """
    Small JSON-backed TTL cache.
    Stores values on disk so cache survives process restarts.
    """

    def __init__(self, cache_dir: str, ttl_s: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_s = max(0, int(ttl_s))

    @staticmethod
    def make_key(prefix: str, *parts: Any) -> str:
        raw = "|".join([str(prefix)] + [str(x) for x in parts]).encode("utf-8", errors="ignore")
        return hashlib.sha1(raw).hexdigest()[:24]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            ts = int(payload.get("ts", 0))
            if ts <= 0:
                return None
            if self.ttl_s > 0 and (int(time.time()) - ts) > self.ttl_s:
                return None
            return payload.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        payload = {"ts": int(time.time()), "value": value}
        try:
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return

