from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: str
    ticker: str
    fiscal_period: str
    speaker: str
    role: Optional[str]
    text: str
    order_idx: int


@dataclass(frozen=True)
class TranscriptDoc:
    ticker: str
    fiscal_period: str
    transcript_date: Optional[str]
    source: str
    raw_text: str
    segments: List[TranscriptSegment]


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def _make_segment_id(ticker: str, fiscal_period: str, idx: int) -> str:
    return f"{ticker}_{fiscal_period}_{idx:04d}"


def split_transcript_by_speaker(
    *,
    ticker: str,
    fiscal_period: str,
    raw_text: str,
) -> List[TranscriptSegment]:
    """
    Expects patterns like:
      Jensen Huang -- Chief Executive Officer
      Operator
      Analyst Name -- Firm
    """
    lines = [ln.strip() for ln in str(raw_text or "").splitlines() if ln.strip()]
    segments: List[TranscriptSegment] = []

    current_speaker = "Unknown"
    current_role: Optional[str] = None
    current_buf: List[str] = []
    order_idx = 0

    speaker_pat = re.compile(r"^([A-Za-z .,'\-()]+?)(?:\s+--\s+(.+))?$")

    def flush():
        nonlocal order_idx, current_buf
        text = _norm_text(" ".join(current_buf))
        if text:
            segments.append(
                TranscriptSegment(
                    segment_id=_make_segment_id(ticker, fiscal_period, order_idx),
                    ticker=ticker,
                    fiscal_period=fiscal_period,
                    speaker=current_speaker,
                    role=current_role,
                    text=text,
                    order_idx=order_idx,
                )
            )
            order_idx += 1
        current_buf = []

    for line in lines:
        m = speaker_pat.match(line)
        looks_like_speaker = (
            m is not None
            and len(line.split()) <= 10
            and not line.endswith(".")
            and len(line) < 120
        )

        if looks_like_speaker:
            flush()
            current_speaker = _norm_text(m.group(1)) or "Unknown"
            current_role = _norm_text(m.group(2)) or None
        else:
            current_buf.append(line)

    flush()
    return segments


class TranscriptIngestionClient:
    """
    Stub-friendly transcript loader.

    For the hackathon, you can:
    1. load local transcript files, or
    2. later replace fetch_transcript() with a real API provider.
    """

    def __init__(self, transcript_dir: str = "data/transcripts"):
        self.transcript_dir = Path(transcript_dir)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

    def load_local_transcript(
        self,
        *,
        ticker: str,
        fiscal_period: str,
        path: str,
        transcript_date: Optional[str] = None,
        source: str = "local_file",
    ) -> TranscriptDoc:
        raw_text = Path(path).read_text(encoding="utf-8")
        segments = split_transcript_by_speaker(
            ticker=ticker,
            fiscal_period=fiscal_period,
            raw_text=raw_text,
        )
        return TranscriptDoc(
            ticker=ticker,
            fiscal_period=fiscal_period,
            transcript_date=transcript_date,
            source=source,
            raw_text=raw_text,
            segments=segments,
        )

    def save_json(self, doc: TranscriptDoc) -> Path:
        out_path = self.transcript_dir / f"{doc.ticker}_{doc.fiscal_period}.json"
        payload = asdict(doc)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return out_path

    def load_json(self, ticker: str, fiscal_period: str) -> Optional[TranscriptDoc]:
        path = self.transcript_dir / f"{ticker}_{fiscal_period}.json"
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        segs = [TranscriptSegment(**x) for x in obj.get("segments", [])]
        return TranscriptDoc(
            ticker=obj["ticker"],
            fiscal_period=obj["fiscal_period"],
            transcript_date=obj.get("transcript_date"),
            source=obj.get("source", "local_json"),
            raw_text=obj.get("raw_text", ""),
            segments=segs,
        )

    def get_current_and_prior_text(
        self,
        *,
        ticker: str,
        current_period: str,
        prior_period: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        cur = self.load_json(ticker, current_period)
        prev = self.load_json(ticker, prior_period)
        return (
            cur.raw_text if cur else None,
            prev.raw_text if prev else None,
        )


# ──────────────────────────────────────────────────────────────
# Alpha Vantage earnings-call transcript client
# ──────────────────────────────────────────────────────────────

_AV_BASE = "https://www.alphavantage.co/query"
_AV_CACHE_TTL_S = 86_400  # 24 h


class AlphaVantageTranscriptClient:
    """
    Fetches earnings-call transcripts from Alpha Vantage.

    Endpoint:
        GET https://www.alphavantage.co/query
            ?function=EARNINGS_CALL_TRANSCRIPT
            &symbol=AAPL
            &quarter=2024Q4
            &apikey=<ALPHAVANTAGE_API_KEY>

    Requires:
        ALPHAVANTAGE_API_KEY environment variable.

    Disk cache:
        data/cache/transcripts/<TICKER>_<QUARTER>.json  (TTL = 24 h)

    Usage:
        client = AlphaVantageTranscriptClient()
        doc = client.fetch("AAPL", "2024Q4")          # TranscriptDoc or None
        cur, prev = client.get_current_and_prior_text(
            ticker="AAPL",
            current_quarter="2024Q4",
            prior_quarter="2024Q3",
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache/transcripts",
        timeout_s: int = 30,
    ):
        self.api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ALPHAVANTAGE_API_KEY is not set. "
                "Export it or pass api_key= to AlphaVantageTranscriptClient()."
            )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, quarter: str) -> Path:
        return self.cache_dir / f"{ticker.upper()}_{quarter}.json"

    def _load_cache(self, ticker: str, quarter: str) -> Optional[Dict[str, Any]]:
        p = self._cache_path(ticker, quarter)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - float(obj.get("_cached_at", 0)) < _AV_CACHE_TTL_S:
                return obj
        except Exception:
            pass
        return None

    def _save_cache(self, ticker: str, quarter: str, data: Dict[str, Any]) -> None:
        p = self._cache_path(ticker, quarter)
        data["_cached_at"] = time.time()
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _fetch_raw(self, ticker: str, quarter: str) -> Optional[Dict[str, Any]]:
        """Call Alpha Vantage and return the JSON response, or None on failure."""
        params = {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": ticker.upper(),
            "quarter": quarter,
            "apikey": self.api_key,
        }
        try:
            resp = requests.get(_AV_BASE, params=params, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            # AV returns {"Information": "..."} on rate-limit / bad key
            if "Information" in data or "Note" in data:
                msg = data.get("Information") or data.get("Note", "")
                logger.warning("Alpha Vantage rate-limit or key issue: %s", msg)
                return None
            return data
        except Exception as exc:
            logger.warning("AlphaVantage transcript fetch failed (%s %s): %s", ticker, quarter, exc)
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, ticker: str, quarter: str) -> Optional[TranscriptDoc]:
        """
        Fetch and cache a transcript for `ticker` / `quarter` (e.g. "2024Q4").
        Returns a TranscriptDoc, or None if unavailable.
        """
        cached = self._load_cache(ticker, quarter)
        if cached:
            raw_data = cached
        else:
            raw_data = self._fetch_raw(ticker, quarter)
            if raw_data:
                self._save_cache(ticker, quarter, raw_data)

        if not raw_data:
            return None

        return self._parse_response(ticker, quarter, raw_data)

    def _parse_response(
        self,
        ticker: str,
        quarter: str,
        data: Dict[str, Any],
    ) -> Optional[TranscriptDoc]:
        """
        Alpha Vantage transcript JSON shape (as of 2025):
        {
          "symbol": "AAPL",
          "quarter": "2024Q4",
          "transcript": [
            {"speaker": "Tim Cook", "title": "CEO", "content": "..."},
            ...
          ]
        }
        Falls back to a flat string if the shape differs.
        """
        transcript_entries = data.get("transcript") or []
        if not transcript_entries:
            # Try flat-text fallback
            raw_text = str(data.get("content") or data.get("text") or "")
            if not raw_text:
                return None
        else:
            lines: List[str] = []
            for entry in transcript_entries:
                speaker = str(entry.get("speaker") or "Unknown").strip()
                title = str(entry.get("title") or "").strip()
                content = str(entry.get("content") or "").strip()
                header = f"{speaker} -- {title}" if title else speaker
                lines.append(header)
                lines.append(content)
            raw_text = "\n".join(lines)

        segments = split_transcript_by_speaker(
            ticker=ticker,
            fiscal_period=quarter,
            raw_text=raw_text,
        )
        transcript_date = str(data.get("date") or "")

        return TranscriptDoc(
            ticker=ticker.upper(),
            fiscal_period=quarter,
            transcript_date=transcript_date or None,
            source="alpha_vantage",
            raw_text=raw_text,
            segments=segments,
        )

    def get_current_and_prior_text(
        self,
        *,
        ticker: str,
        current_quarter: str,
        prior_quarter: str,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Returns (current_raw_text, prior_raw_text).
        Either may be None if the transcript is unavailable.
        """
        cur = self.fetch(ticker, current_quarter)
        prev = self.fetch(ticker, prior_quarter)
        return (cur.raw_text if cur else None, prev.raw_text if prev else None)