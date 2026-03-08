from __future__ import annotations

import json
import logging
import os
import re
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