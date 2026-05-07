# routing.py
# ============================================================
# Cost-Aware Routing
# - Pre-generation risk scoring (retrieval agreement, margin, coverage)
# - small model for high-confidence
# - large model for medium confidence or gate failures
# - abstain/clarify when evidence missing (gate provides this)
#
# Input assumptions:
# - You pass in:
#     plan (from verification.build_task_plan)
#     gate (GateResult as dict or object)
#     retrieval_debug (dict) that includes:
#         - "candidates": [{"id", "score", "kind", "item", ...}, ...]  (pre-rerank)
#         - "reranked":   [{"id", "score"}, ...]                      (post-rerank)
#         - "selected_ids": ["c123","t45", ...]
#         - "bm25_ids": [...], "dense_ids": [...]                     (optional for overlap)
# - If fields are missing, the scorer degrades gracefully.
# ============================================================

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Configurable thresholds (override via env vars)
# -----------------------------------------------------------
COVERAGE_FULL_AT = float(os.environ.get("ROUTING_COVERAGE_FULL_AT", "8"))
SMALL_MODEL_MAX_RISK = float(os.environ.get("ROUTING_SMALL_MAX_RISK", "0.35"))
ABSTAIN_MIN_RISK = float(os.environ.get("ROUTING_ABSTAIN_MIN_RISK", "0.90"))
HARD_MODE_PENALTY = float(os.environ.get("ROUTING_HARD_MODE_PENALTY", "0.10"))

W_COVERAGE = 0.50
W_AGREEMENT = 0.20
W_MARGIN = 0.20
W_ANSWER = 0.10

HARD_MODES = frozenset({
    "lookup_numeric", "compute_metric", "valuation", "relative_valuation",
})


@dataclass
class RoutingDecision:
    action: str   # "answer" | "clarify" | "abstain"
    mode: str     # from plan.mode
    model: str    # "small" | "large" | "none"
    risk: float
    reasons: List[str]
    signals: Dict[str, Any]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def compute_retrieval_signals(debug: Dict[str, Any]) -> Dict[str, Any]:
    sig: Dict[str, Any] = {}

    selected = debug.get("selected_ids") or []
    sig["selected_n"] = int(len(selected))
    sig["coverage"] = _clip01(min(1.0, len(selected) / COVERAGE_FULL_AT))

    reranked = debug.get("reranked") or []
    if isinstance(reranked, list) and len(reranked) >= 2:
        try:
            top1 = float(reranked[0].get("score", 0.0))
            top2 = float(reranked[1].get("score", 0.0))
            denom = max(1e-9, abs(top1))
            sig["rr_margin"] = _clip01((top1 - top2) / denom)
        except Exception:
            sig["rr_margin"] = 0.0
    else:
        sig["rr_margin"] = 0.0

    bm25_ids = debug.get("bm25_ids") or []
    dense_ids = debug.get("dense_ids") or []
    if bm25_ids or dense_ids:
        sig["retrieval_agreement"] = _clip01(jaccard(bm25_ids[:20], dense_ids[:20]))
    else:
        sig["retrieval_agreement"] = 0.5

    a_small = debug.get("answer_small_norm")
    a_large = debug.get("answer_large_norm")
    if a_small is not None and a_large is not None:
        sig["answer_agreement"] = 1.0 if a_small == a_large else 0.0
    else:
        sig["answer_agreement"] = 0.5

    return sig


def risk_score(signals: Dict[str, Any], *, mode: str) -> Tuple[float, List[str]]:
    reasons: List[str] = []

    cov = float(signals.get("coverage", 0.0))
    mar = float(signals.get("rr_margin", 0.0))
    agr = float(signals.get("retrieval_agreement", 0.5))
    ans = float(signals.get("answer_agreement", 0.5))

    bad_cov = 1.0 - _clip01(cov)
    bad_mar = 1.0 - _clip01(mar)
    bad_agr = 1.0 - _clip01(agr)
    bad_ans = 1.0 - _clip01(ans)

    r = W_COVERAGE * bad_cov + W_AGREEMENT * bad_agr + W_MARGIN * bad_mar + W_ANSWER * bad_ans

    if mode in HARD_MODES:
        r = min(1.0, r + HARD_MODE_PENALTY)
        reasons.append("hard_mode_penalty")

    if cov < 0.35:
        reasons.append("low_coverage")
    if agr < 0.25:
        reasons.append("low_retrieval_agreement")
    if mar < 0.05:
        reasons.append("low_margin")
    if ans < 0.5:
        reasons.append("low_answer_agreement")

    return _clip01(r), reasons


def choose_model_from_risk(risk: float) -> str:
    r = _clip01(risk)
    return "small" if r <= SMALL_MODEL_MAX_RISK else "large"


def decide(
    *,
    plan_mode: str,
    gate_action: str,
    gate_ok: bool,
    retrieval_debug: Dict[str, Any],
) -> RoutingDecision:
    if not isinstance(plan_mode, str) or not plan_mode:
        raise ValueError("plan_mode must be a non-empty string")
    if not isinstance(gate_action, str):
        raise TypeError("gate_action must be a string")

    if not gate_ok:
        action = "clarify" if gate_action in ("clarify", "needs_market_data") else "abstain"
        logger.info("routing: %s (gate_%s) mode=%s", action, gate_action, plan_mode)
        return RoutingDecision(
            action=action,
            mode=plan_mode,
            model="none",
            risk=1.0,
            reasons=[f"gate_{gate_action}"],
            signals={"gate_action": gate_action},
        )

    sig = compute_retrieval_signals(retrieval_debug or {})
    r, rs = risk_score(sig, mode=plan_mode)
    model = choose_model_from_risk(r)

    if r > ABSTAIN_MIN_RISK:
        logger.warning("routing: abstain risk=%.3f mode=%s reasons=%s", r, plan_mode, rs)
        return RoutingDecision(
            action="abstain",
            mode=plan_mode,
            model="none",
            risk=r,
            reasons=rs + ["risk_too_high"],
            signals=sig,
        )

    logger.info("routing: answer model=%s risk=%.3f mode=%s", model, r, plan_mode)
    return RoutingDecision(
        action="answer",
        mode=plan_mode,
        model=model,
        risk=r,
        reasons=rs,
        signals=sig,
    )
