# audit.py
# ============================================================
# Observability & Audit Logging (JSONL)
# - Retrieval candidates + scores
# - Reranker scores
# - Context chunk/table IDs + optional offsets
# - Model versions and prompts (hashed)
# - Gate decisions + routing path
# - Latency and cost breakdown (if provided)
# ============================================================

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _safe_json_default(obj: Any) -> Any:
    """Fallback serializer for types json.dumps can't handle natively."""
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


@dataclass
class AuditLogger:
    path: str
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)

    def new_run_id(self) -> str:
        return uuid.uuid4().hex

    def log(self, record: Dict[str, Any]) -> None:
        rec = dict(record)
        rec.setdefault("ts_ms", _now_ms())
        try:
            line = json.dumps(rec, ensure_ascii=False, default=_safe_json_default)
        except Exception:
            logger.exception("audit: failed to serialize record, falling back to repr")
            try:
                line = json.dumps({"_serialization_error": repr(rec)[:2000], "ts_ms": rec.get("ts_ms")})
            except Exception:
                return

        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            logger.exception("audit: failed to write to %s", self.path)

    # ---- helpers to standardize event schemas ----

    def log_retrieval(
        self,
        run_id: str,
        question: str,
        rewrites: List[str],
        hard_filters: dict,
        soft_boosts: List[dict],
        candidates: Optional[List[dict]] = None,
        reranker: Optional[dict] = None,
        selected: Optional[List[dict]] = None,
        packed_ids: Optional[List[str]] = None,
        latency_ms: Optional[float] = None,
        debug: Optional[dict] = None,
    ) -> None:
        self.log({
            "run_id": run_id,
            "event": "retrieval",
            "question": question,
            "rewrites": rewrites,
            "retrieval_plan": {"hard_filters": hard_filters, "soft_boosts": soft_boosts},
            "candidates": candidates or [],
            "reranker": reranker or {},
            "selected": selected or [],
            "packed_ids": packed_ids or [],
            "latency_ms": latency_ms,
            "debug": debug or {},
        })

    def log_gate(
        self,
        run_id: str,
        plan: dict,
        req: dict,
        gate: dict,
        routing: dict,
        latency_ms: Optional[float] = None,
    ) -> None:
        self.log({
            "run_id": run_id,
            "event": "gate_routing",
            "plan": plan,
            "requirements": req,
            "gate": gate,
            "routing": routing,
            "latency_ms": latency_ms,
        })

    def log_generation(
        self,
        run_id: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
        latency_ms: Optional[float] = None,
        token_usage: Optional[dict] = None,
    ) -> None:
        self.log({
            "run_id": run_id,
            "event": "generation",
            "model": model_name,
            "system_hash": _sha256_text(system_prompt),
            "user_hash": _sha256_text(user_prompt),
            "output_hash": _sha256_text(output_text),
            "output_preview": (output_text or "")[:800],
            "latency_ms": latency_ms,
            "usage": token_usage or {},
        })

    def log_validation(
        self,
        run_id: str,
        ok: bool,
        errors: List[str],
        signals: Optional[dict] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        self.log({
            "run_id": run_id,
            "event": "validation",
            "ok": bool(ok),
            "errors": errors,
            "signals": signals or {},
            "latency_ms": latency_ms,
        })
