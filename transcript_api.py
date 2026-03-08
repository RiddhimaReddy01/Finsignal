from __future__ import annotations

import json
import os
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

ALPHAVANTAGE_BASE = "https://www.alphavantage.co/query"
FMP_BASE = "https://financialmodelingprep.com/stable"


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _extract_text(payload: Any) -> Optional[str]:
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    for k in ("transcript", "content", "text"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


class _DiskTTLCache:
    def __init__(self, cache_dir: str, ttl_s: int):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_s = max(0, int(ttl_s))

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            ts = int(obj.get("ts", 0))
            if ts <= 0:
                return None
            if self.ttl_s > 0 and (int(time.time()) - ts) > self.ttl_s:
                return None
            return obj.get("value")
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        p = self._path(key)
        payload = {"ts": int(time.time()), "value": value}
        try:
            p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            return


def _cache_key(prefix: str, *parts: Any) -> str:
    raw = "|".join([prefix] + [str(x) for x in parts]).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:24]


class AlphaVantageTranscriptAPI:
    """
    Free-tier friendly transcript client.
    Uses built-in pacing to avoid hitting minute limits.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_interval_s: float = 12.0,
        cache_dir: str = "data/cache/transcripts",
        transcript_ttl_s: int = 60 * 60 * 24 * 30,
        latest_ttl_s: int = 60 * 60 * 6,
    ):
        self.api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("AV_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY missing")
        self.min_interval_s = max(0.0, float(min_interval_s))
        self._last_call_ts = 0.0
        self._transcript_cache = _DiskTTLCache(cache_dir=f"{cache_dir}/alphavantage/transcript", ttl_s=transcript_ttl_s)
        self._latest_cache = _DiskTTLCache(cache_dir=f"{cache_dir}/alphavantage/latest", ttl_s=latest_ttl_s)

    def _pace(self) -> None:
        now = time.time()
        wait_s = self.min_interval_s - (now - self._last_call_ts)
        if wait_s > 0:
            time.sleep(wait_s)

    def _get(self, params: Dict[str, Any]) -> Any:
        self._pace()
        p = dict(params)
        p["apikey"] = self.api_key
        resp = requests.get(ALPHAVANTAGE_BASE, params=p, timeout=30)
        self._last_call_ts = time.time()
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            if data.get("Note"):
                raise RuntimeError(f"Alpha Vantage throttle: {data.get('Note')}")
            if data.get("Error Message"):
                raise RuntimeError(f"Alpha Vantage error: {data.get('Error Message')}")
        return data

    @staticmethod
    def _quarter_from_date(date_str: str) -> Optional[Tuple[int, int]]:
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except Exception:
            return None
        q = ((dt.month - 1) // 3) + 1
        return dt.year, q

    def _latest_quarters(self, ticker: str, n: int = 2) -> List[Tuple[int, int]]:
        data = self._get({"function": "EARNINGS", "symbol": ticker})
        rows = []
        if isinstance(data, dict):
            rows = data.get("quarterlyEarnings") or []
        out: List[Tuple[int, int]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            fy = self._quarter_from_date(str(r.get("fiscalDateEnding", "")))
            if fy is None:
                continue
            out.append(fy)
        out = sorted(set(out), reverse=True)
        return out[:n]

    def get_transcript(self, ticker: str, year: int, quarter: int) -> Optional[str]:
        q = _safe_int(quarter)
        y = _safe_int(year)
        if q is None or y is None:
            raise ValueError("year and quarter must be integers")
        if q < 1 or q > 4:
            raise ValueError("quarter must be 1..4")

        key = _cache_key("av_transcript", str(ticker).upper(), int(y), int(q))
        cached = self._transcript_cache.get(key)
        if isinstance(cached, str):
            return cached

        data = self._get({
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": ticker,
            "quarter": f"{y}Q{q}",
        })
        text = _extract_text(data)
        if text is not None:
            self._transcript_cache.set(key, text)
        return text

    def get_latest_transcripts(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        latest_key = _cache_key("av_latest_pair", str(ticker).upper())
        cached = self._latest_cache.get(latest_key)
        if isinstance(cached, dict):
            cur = cached.get("current")
            prev = cached.get("prior")
            if (cur is None or isinstance(cur, str)) and (prev is None or isinstance(prev, str)):
                return cur, prev

        quarters = self._latest_quarters(ticker, n=2)
        if len(quarters) < 2:
            return None, None
        (y0, q0), (y1, q1) = quarters[0], quarters[1]
        pair = (
            self.get_transcript(ticker, y0, q0),
            self.get_transcript(ticker, y1, q1),
        )
        self._latest_cache.set(latest_key, {"current": pair[0], "prior": pair[1]})
        return pair


class FMPTranscriptAPI:
    """
    Optional paid fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache/transcripts",
        transcript_ttl_s: int = 60 * 60 * 24 * 30,
        latest_ttl_s: int = 60 * 60 * 6,
    ):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY missing")
        self._transcript_cache = _DiskTTLCache(cache_dir=f"{cache_dir}/fmp/transcript", ttl_s=transcript_ttl_s)
        self._latest_cache = _DiskTTLCache(cache_dir=f"{cache_dir}/fmp/latest", ttl_s=latest_ttl_s)

    def get_latest_transcripts(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        latest_key = _cache_key("fmp_latest_pair", str(ticker).upper())
        cached = self._latest_cache.get(latest_key)
        if isinstance(cached, dict):
            cur = cached.get("current")
            prev = cached.get("prior")
            if (cur is None or isinstance(cur, str)) and (prev is None or isinstance(prev, str)):
                return cur, prev

        url = f"{FMP_BASE}/earning-call-transcript-dates"
        r = requests.get(url, params={"symbol": ticker, "apikey": self.api_key}, timeout=30)
        r.raise_for_status()
        dates = r.json() or []
        if len(dates) < 2:
            return None, None
        dates = sorted(dates, key=lambda x: (x.get("year", 0), x.get("quarter", 0)), reverse=True)
        latest, prior = dates[0], dates[1]
        pair = (
            self.get_transcript(ticker, int(latest["year"]), int(latest["quarter"])),
            self.get_transcript(ticker, int(prior["year"]), int(prior["quarter"])),
        )
        self._latest_cache.set(latest_key, {"current": pair[0], "prior": pair[1]})
        return pair

    def get_transcript(self, ticker: str, year: int, quarter: int) -> Optional[str]:
        key = _cache_key("fmp_transcript", str(ticker).upper(), int(year), int(quarter))
        cached = self._transcript_cache.get(key)
        if isinstance(cached, str):
            return cached

        url = f"{FMP_BASE}/earning-call-transcript"
        r = requests.get(
            url,
            params={"symbol": ticker, "year": year, "quarter": quarter, "apikey": self.api_key},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json() or []
        if not data:
            return None
        text = _extract_text(data)
        if text is not None:
            self._transcript_cache.set(key, text)
        return text


class FreeTranscriptAPI:
    """
    Free-tier-first wrapper:
      1) Alpha Vantage
      2) FMP fallback if explicitly configured and AV fails
    """

    def __init__(self, prefer: str = "alphavantage", allow_fmp_fallback: bool = True):
        self.prefer = (prefer or "alphavantage").strip().lower()
        self.allow_fmp_fallback = bool(allow_fmp_fallback)

    def _av(self) -> AlphaVantageTranscriptAPI:
        return AlphaVantageTranscriptAPI()

    def _fmp(self) -> FMPTranscriptAPI:
        return FMPTranscriptAPI()

    def get_latest_transcripts(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        errs: List[str] = []

        if self.prefer in ("alphavantage", "alpha_vantage", "av"):
            try:
                return self._av().get_latest_transcripts(ticker)
            except Exception as e:
                errs.append(f"alphavantage:{type(e).__name__}:{e}")
            if self.allow_fmp_fallback:
                try:
                    return self._fmp().get_latest_transcripts(ticker)
                except Exception as e:
                    errs.append(f"fmp:{type(e).__name__}:{e}")
        else:
            try:
                return self._fmp().get_latest_transcripts(ticker)
            except Exception as e:
                errs.append(f"fmp:{type(e).__name__}:{e}")
            try:
                return self._av().get_latest_transcripts(ticker)
            except Exception as e:
                errs.append(f"alphavantage:{type(e).__name__}:{e}")

        raise RuntimeError("all transcript providers failed: " + " | ".join(errs))
