from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _regime_from_result(base_result: Dict[str, Any], has_transcript: bool) -> Dict[str, Any]:
    market = base_result.get("market_inputs") or {}
    beta = _safe_float(market.get("beta"))
    if beta is None:
        vol = "unknown"
    elif beta < 0.95:
        vol = "low_vol"
    elif beta > 1.35:
        vol = "high_vol"
    else:
        vol = "mid_vol"
    return {
        "volatility_regime": vol,
        "near_earnings_window": bool(has_transcript),
        "beta": beta,
    }


def _signal(name: str, score: Optional[float], confidence: Optional[float], rationale: str) -> Dict[str, Any]:
    if score is None or confidence is None:
        return {"name": name, "available": False, "score": 0.0, "confidence": 0.0, "rationale": "not_available"}
    return {
        "name": name,
        "available": True,
        "score": _clip(score, -1.0, 1.0),
        "confidence": _clip(confidence, 0.0, 1.0),
        "rationale": rationale,
    }


def _build_independent_signals(
    *,
    risk_severity_avg: float,
    tone_delta: Optional[float],
    valuation_gap_pct: Optional[float],
    revenue_growth_yoy: Optional[float],
    news_direction_score: Optional[float],
    news_items: List[Dict[str, Any]],
    peer_premium_pct: Optional[float],
    scenario_analysis: Optional[Dict[str, Any]],
    evidence_count: int,
) -> List[Dict[str, Any]]:
    sigs: List[Dict[str, Any]] = []

    risk_score = -_clip(risk_severity_avg, 0.0, 1.0)
    risk_conf = _clip(0.55 + 0.03 * min(max(evidence_count, 0), 10), 0.0, 0.9)
    sigs.append(_signal("risk", risk_score, risk_conf, "filing risk language severity"))

    if tone_delta is not None:
        sigs.append(_signal("tone", _clip(tone_delta, -1.0, 1.0), 0.65, "management tone delta"))
    else:
        sigs.append(_signal("tone", None, None, "no transcript pair"))

    if valuation_gap_pct is not None:
        sigs.append(_signal("valuation", _clip(valuation_gap_pct, -1.0, 1.0), 0.72, "valuation gap vs fair value"))
    else:
        sigs.append(_signal("valuation", None, None, "valuation not available"))

    if revenue_growth_yoy is not None:
        sigs.append(_signal("growth", _clip(revenue_growth_yoy / 0.4, -1.0, 1.0), 0.68, "revenue growth normalization"))
    else:
        sigs.append(_signal("growth", None, None, "growth not available"))

    if news_direction_score is not None:
        nconf = _clip(0.35 + 0.1 * min(len(news_items), 5), 0.35, 0.85)
        sigs.append(_signal("news", _clip(news_direction_score, -1.0, 1.0), nconf, "recent catalyst direction"))
    else:
        sigs.append(_signal("news", None, None, "news not available"))

    if peer_premium_pct is not None:
        # Positive premium means expensive vs peers => negative signal.
        sigs.append(_signal("peer_valuation", _clip(-peer_premium_pct, -1.0, 1.0), 0.65, "premium/discount vs peer median"))
    else:
        sigs.append(_signal("peer_valuation", None, None, "peer median not available"))

    scen_score = None
    scen_conf = None
    if isinstance(scenario_analysis, dict):
        comps = scenario_analysis.get("comparisons") or []
        if isinstance(comps, list) and comps:
            ev_deltas = [float(c.get("ev_delta_pct", 0.0)) for c in comps if isinstance(c, dict)]
            if ev_deltas:
                upside = max(ev_deltas)
                downside = abs(min(ev_deltas))
                scen_score = _clip(upside - downside, -1.0, 1.0)
                scen_conf = _clip(0.45 + 0.1 * min(len(ev_deltas), 4), 0.45, 0.85)
    sigs.append(_signal("scenario_resilience", scen_score, scen_conf, "bull/bear/stress EV asymmetry"))

    return sigs


def _regime_multiplier(signal_name: str, regime: Dict[str, Any]) -> float:
    vol = regime.get("volatility_regime", "unknown")
    near_earnings = bool(regime.get("near_earnings_window"))
    m = 1.0
    if signal_name == "valuation":
        if vol == "low_vol":
            m *= 1.35
        elif vol == "high_vol":
            m *= 0.85
    if signal_name == "news":
        if vol == "high_vol":
            m *= 1.25
        if near_earnings:
            m *= 1.35
    if signal_name == "tone" and near_earnings:
        m *= 1.20
    return m


def _detect_contradictions(signals: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    active = [s for s in signals if s.get("available") and s.get("confidence", 0) >= 0.6 and abs(s.get("score", 0.0)) >= 0.2]
    contradictions: List[Dict[str, Any]] = []
    penalty = 0.0
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            a = active[i]
            b = active[j]
            if a["score"] * b["score"] < 0:
                contradictions.append({
                    "a": a["name"],
                    "b": b["name"],
                    "a_score": a["score"],
                    "b_score": b["score"],
                })
                penalty += 0.03
    return contradictions, _clip(penalty, 0.0, 0.18)


def build_quant_decision(
    *,
    base_result: Dict[str, Any],
    risk_severity_avg: float,
    tone_delta: Optional[float],
    valuation_gap_pct: Optional[float],
    revenue_growth_yoy: Optional[float],
    news_direction_score: Optional[float],
    news_items: List[Dict[str, Any]],
    evidence_count: int,
) -> Dict[str, Any]:
    result_obj = base_result.get("result") or {}
    rel = result_obj.get("relative_valuation") if isinstance(result_obj, dict) else None
    peer_premium_pct = None
    if isinstance(rel, dict):
        peer_premium_pct = _safe_float(rel.get("peer_premium_pct"))
    scenario_analysis = result_obj.get("scenario_analysis") if isinstance(result_obj, dict) else None

    signals = _build_independent_signals(
        risk_severity_avg=risk_severity_avg,
        tone_delta=tone_delta,
        valuation_gap_pct=valuation_gap_pct,
        revenue_growth_yoy=revenue_growth_yoy,
        news_direction_score=news_direction_score,
        news_items=news_items,
        peer_premium_pct=peer_premium_pct,
        scenario_analysis=scenario_analysis if isinstance(scenario_analysis, dict) else None,
        evidence_count=evidence_count,
    )

    regime = _regime_from_result(base_result, has_transcript=(tone_delta is not None))
    weighted_rows: List[Dict[str, Any]] = []
    for s in signals:
        if not s.get("available"):
            continue
        rm = _regime_multiplier(str(s["name"]), regime)
        eff_w = _clip(float(s["confidence"]) * rm, 0.0, 1.5)
        weighted_rows.append({
            "name": s["name"],
            "score": float(s["score"]),
            "confidence": float(s["confidence"]),
            "regime_mult": rm,
            "effective_weight": eff_w,
            "weighted_contribution": eff_w * float(s["score"]),
            "rationale": s["rationale"],
        })

    contradictions, contradiction_penalty = _detect_contradictions(signals)
    total_w = sum(r["effective_weight"] for r in weighted_rows)
    raw_score = (sum(r["weighted_contribution"] for r in weighted_rows) / total_w) if total_w > 0 else 0.0
    final_score = _clip(raw_score - contradiction_penalty, -1.0, 1.0)
    agg_conf = _clip((sum(r["confidence"] * r["effective_weight"] for r in weighted_rows) / total_w) if total_w > 0 else 0.0, 0.0, 1.0)

    if agg_conf < 0.45:
        action = "WATCH"
        reason = "low_aggregate_confidence"
    elif final_score >= 0.25:
        action = "ACT"
        reason = "strong_positive_weighted_signal"
    elif final_score >= 0.05:
        action = "WATCH"
        reason = "mixed_or_moderate_signal"
    else:
        action = "NO_ACT"
        reason = "insufficient_positive_edge"

    ranked = sorted(weighted_rows, key=lambda r: abs(r["weighted_contribution"]), reverse=True)
    trace: List[str] = []
    trace.append(f"Regime: vol={regime.get('volatility_regime')} beta={regime.get('beta')} near_earnings={regime.get('near_earnings_window')}")
    for r in ranked[:5]:
        trace.append(
            f"{r['name']}: score={r['score']:+.3f} conf={r['confidence']:.2f} "
            f"regime_mult={r['regime_mult']:.2f} contribution={r['weighted_contribution']:+.3f}"
        )
    if contradictions:
        trace.append(f"Contradictions detected: {len(contradictions)}; penalty={contradiction_penalty:.3f}")
        for c in contradictions[:5]:
            trace.append(f"  - {c['a']} ({c['a_score']:+.2f}) vs {c['b']} ({c['b_score']:+.2f})")
    trace.append(f"Final weighted score={final_score:+.3f}, aggregate_confidence={agg_conf:.2f} => action={action}")

    return {
        "action": action,
        "reason_code": reason,
        "score": round(final_score, 4),
        "aggregate_confidence": round(agg_conf, 4),
        "contradiction_penalty": round(contradiction_penalty, 4),
        "regime": regime,
        "signals": signals,
        "weighted_signals": ranked,
        "contradictions": contradictions,
        "decision_tree_trace": trace,
        "policy": {
            "act_threshold": 0.25,
            "watch_threshold": 0.05,
            "min_conf_for_non_watch": 0.45,
            "weighting": "confidence_weighted * regime_multiplier",
        },
    }
