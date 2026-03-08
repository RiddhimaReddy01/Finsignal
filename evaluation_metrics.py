# evaluation_metrics.py
# ============================================================
# Offline evaluation utilities (run on audit logs + labeled eval set)
# Metrics:
# - Retrieval Recall@k
# - Section hit rate
# - Numeric exactness + unit correctness
# - Unsupported claim rate
# - p50/p95 latency and cost per query
# - Fallback rate vs quality improvement
#
# This file is deliberately "offline": it reads audit JSONL.
# ============================================================

from __future__ import annotations

import json
import logging
import os
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# -----------------------------
# Data loading
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audit log not found: {path}")
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("read_jsonl: skipping corrupt line %d in %s: %s", lineno, path, e)
    return rows


def group_by_run(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rid = r.get("run_id")
        if not rid:
            continue
        by.setdefault(rid, []).append(r)
    return by


def get_event(run_events: List[Dict[str, Any]], event_name: str) -> Optional[Dict[str, Any]]:
    for e in run_events:
        if e.get("event") == event_name:
            return e
    return None


# -----------------------------
# Retrieval metrics
# -----------------------------

def recall_at_k(retrieved_ids: List[str], gold_ids: List[str], k: int) -> float:
    r = retrieved_ids[:k] if retrieved_ids else []
    g = set(gold_ids or [])
    if not g:
        return 1.0
    return len(set(r) & g) / len(g)


def section_hit_rate(selected_meta: List[Dict[str, Any]], desired_section_prefix: str = "Item 1A") -> float:
    if not selected_meta:
        return 0.0
    hit = 0
    for s in selected_meta:
        item = (s.get("item") or "")
        if item.startswith(desired_section_prefix):
            hit += 1
    return hit / len(selected_meta)


# -----------------------------
# Faithfulness metrics
# -----------------------------

def unsupported_claim_rate(validation_event: Dict[str, Any]) -> float:
    if not validation_event:
        return 1.0
    if validation_event.get("ok") is True:
        return 0.0
    errs = validation_event.get("errors") or []
    if not errs:
        return 1.0
    bad = 0
    for e in errs:
        if "not_allowed" in e or "missing_citations" in e or "citation_invalid" in e:
            bad += 1
    return min(1.0, bad / max(1, len(errs)))


def numeric_exactness(pred_value: float, gold_value: float, *, rel_tol: float = 1e-3, abs_tol: float = 0.0) -> float:
    if gold_value == 0:
        return 1.0 if abs(pred_value - gold_value) <= abs_tol else 0.0
    rel = abs(pred_value - gold_value) / abs(gold_value)
    if rel <= rel_tol:
        return 1.0
    if abs_tol and abs(pred_value - gold_value) <= abs_tol:
        return 1.0
    return 0.0


def unit_correctness(pred_unit: str, gold_unit: str) -> float:
    return 1.0 if (pred_unit or "").upper() == (gold_unit or "").upper() else 0.0


# -----------------------------
# Latency / cost metrics
# -----------------------------

def quantiles(values: List[float], qs=(0.5, 0.95)) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    vals.sort()
    out = {}
    n = len(vals)
    for q in qs:
        idx = int(round((n - 1) * q))
        out[f"p{int(q*100)}"] = vals[idx]
    return out


# -----------------------------
# Fallback rate vs quality
# -----------------------------

def fallback_rate(runs: Dict[str, List[Dict[str, Any]]]) -> float:
    total = 0
    fallback = 0
    for rid, evs in runs.items():
        total += 1
        gate = get_event(evs, "gate_routing") or {}
        gen = get_event(evs, "generation") or {}
        val = get_event(evs, "validation") or {}

        route = (gate.get("routing") or {})
        action = route.get("action")
        model = route.get("model")

        if action in ("clarify", "abstain"):
            fallback += 1
            continue
        if model == "large":
            fallback += 1
            continue
        if val and val.get("ok") is False:
            fallback += 1

    return fallback / max(1, total)


# -----------------------------
# End-to-end report builder
# -----------------------------

@dataclass
class EvalReport:
    recall_at_5: float
    recall_at_10: float
    item1a_hit_rate: float
    unsupported_claim_rate: float
    numeric_exactness_mean: float
    unit_correctness_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    cost_usd_p50: float
    cost_usd_p95: float
    fallback_rate: float


def compute_report(
    audit_jsonl_path: str,
    *,
    labeled_gold: Optional[Dict[str, Dict[str, Any]]] = None,
) -> EvalReport:
    """
    labeled_gold keyed by run_id:
      {
        run_id: {
          "gold_ids": [...],
          "desired_section": "Item 1A",
          "gold_numeric": {"value":..., "unit":...}   # optional
        }
      }
    """
    rows = read_jsonl(audit_jsonl_path)
    runs = group_by_run(rows)

    r5s, r10s, shrs, ucrs = [], [], [], []
    num_exacts, unit_corrs = [], []
    latencies, costs = [], []

    for rid, evs in runs.items():
        retr = get_event(evs, "retrieval") or {}
        val = get_event(evs, "validation") or {}
        gen = get_event(evs, "generation") or {}

        selected = retr.get("packed_ids") or []
        selected_meta = retr.get("selected") or []

        gold = (labeled_gold or {}).get(rid, {})
        gold_ids = gold.get("gold_ids") or []
        r5s.append(recall_at_k(selected, gold_ids, 5))
        r10s.append(recall_at_k(selected, gold_ids, 10))

        desired_section = gold.get("desired_section") or "Item 1A"
        shrs.append(section_hit_rate(selected_meta, desired_section_prefix=desired_section))

        ucrs.append(unsupported_claim_rate(val))

        # Numeric exactness + unit correctness (when gold labels available)
        gold_num = gold.get("gold_numeric")
        if gold_num and isinstance(gold_num, dict):
            gold_val = gold_num.get("value")
            gold_unit = gold_num.get("unit")

            pred_val = None
            pred_unit = None
            # Try to extract predicted numeric from validation output_preview or generation
            output_preview = gen.get("output_preview") or ""
            try:
                parsed = json.loads(output_preview)
                num_section = parsed.get("numeric") or parsed.get("computed") or {}
                pred_val = num_section.get("value")
                pred_unit = num_section.get("unit")
            except (json.JSONDecodeError, AttributeError):
                pass

            if pred_val is not None and gold_val is not None:
                num_exacts.append(numeric_exactness(float(pred_val), float(gold_val)))
            if gold_unit is not None:
                unit_corrs.append(unit_correctness(pred_unit or "", gold_unit))

        # Latency
        total_lat = 0.0
        for e in evs:
            lm = e.get("latency_ms")
            if lm is not None:
                total_lat += float(lm)
        if total_lat > 0:
            latencies.append(total_lat)

        # Cost
        usage = gen.get("usage") or {}
        if "cost_usd" in usage and usage["cost_usd"] is not None:
            costs.append(float(usage["cost_usd"]))

    lat_q = quantiles(latencies, (0.5, 0.95))
    cost_q = quantiles(costs, (0.5, 0.95))

    report = EvalReport(
        recall_at_5=float(statistics.mean(r5s)) if r5s else float("nan"),
        recall_at_10=float(statistics.mean(r10s)) if r10s else float("nan"),
        item1a_hit_rate=float(statistics.mean(shrs)) if shrs else float("nan"),
        unsupported_claim_rate=float(statistics.mean(ucrs)) if ucrs else float("nan"),
        numeric_exactness_mean=float(statistics.mean(num_exacts)) if num_exacts else float("nan"),
        unit_correctness_mean=float(statistics.mean(unit_corrs)) if unit_corrs else float("nan"),
        latency_ms_p50=lat_q["p50"],
        latency_ms_p95=lat_q["p95"],
        cost_usd_p50=cost_q["p50"],
        cost_usd_p95=cost_q["p95"],
        fallback_rate=float(fallback_rate(runs)),
    )

    logger.info(
        "eval_report: runs=%d recall@5=%.3f ucr=%.3f fallback=%.3f",
        len(runs), report.recall_at_5, report.unsupported_claim_rate, report.fallback_rate,
    )
    return report
