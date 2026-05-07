from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ToolSignal:
    """Individual tool signal with confidence and evidence."""
    name: str                    # risk, tone, valuation, growth, news
    score: float                 # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0
    evidence_count: int          # number of evidence blocks
    contradictions: List[str]    # list of detected contradictions


@dataclass
class SignalScore:
    signal_score: float          # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0
    label: str                   # BUY / HOLD / CAUTIOUS / AVOID
    component_scores: Dict[str, float]
    component_confidences: Dict[str, float]  # Per-tool confidences
    key_findings: List[str]
    risk_flags: List[str]
    tool_details: Dict[str, Any]  # Raw tool signals for transparency


def signal_action_from_score(
    *,
    signal_score: float,
    confidence: float,
    act_threshold: float = 0.35,
    watch_threshold: float = 0.10,
    min_confidence: float = 0.55,
) -> str:
    """
    Deterministic sponsor-track decision rule.
      - ACT only on strong positive score with minimum confidence
      - WATCH for moderate/uncertain positive signal
      - NO_ACT otherwise
    """
    s = float(signal_score)
    c = float(confidence)
    if c >= float(min_confidence) and s >= float(act_threshold):
        return "ACT"
    if s >= float(watch_threshold):
        return "WATCH"
    return "NO_ACT"


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _detect_tool_contradiction(
    score: float,
    magnitude_threshold: float = 0.20,
) -> Optional[str]:
    """
    Detects if a signal is weak (not confident enough in its own magnitude).
    Returns contradiction reason if score magnitude is below threshold.
    """
    if abs(score) < magnitude_threshold:
        return "weak_signal_magnitude"
    return None


def _adjust_confidence_for_contradictions(
    base_confidence: float,
    contradictions: List[str],
    penalty_per_contradiction: float = 0.08,
) -> float:
    """
    Reduces tool confidence based on detected contradictions.
    Each contradiction penalty: 8 percentage points
    """
    penalty = len(contradictions) * penalty_per_contradiction
    return _clip(base_confidence - penalty, 0.0, 1.0)


def _compute_base_tool_confidence(
    evidence_count: int,
    has_data: bool = True,
    base_conf: float = 0.35,
) -> float:
    """
    Computes base confidence for a tool based on evidence quantity.
    - base_conf: minimum confidence (0.35)
    - +0.08 per evidence block, capped at 0.95
    """
    if not has_data:
        return 0.0  # No data = no confidence
    return min(base_conf + 0.08 * max(evidence_count, 0), 0.95)


def _normalize_risk(risk_severity_avg: float) -> float:
    # risk severity 0..1 -> contribution -1..0
    return -_clip(float(risk_severity_avg), 0.0, 1.0)


def _normalize_tone(tone_delta: float) -> float:
    return _clip(float(tone_delta), -1.0, 1.0)


def _normalize_valuation(valuation_gap_pct: Optional[float]) -> float:
    # positive = undervalued, negative = overvalued
    if valuation_gap_pct is None:
        return 0.0
    return _clip(float(valuation_gap_pct), -1.0, 1.0)


def _normalize_growth(revenue_growth_yoy: Optional[float]) -> float:
    if revenue_growth_yoy is None:
        return 0.0
    # 40% YoY -> +1 cap
    return _clip(float(revenue_growth_yoy) / 0.40, -1.0, 1.0)


def _normalize_news(news_direction_score: Optional[float]) -> float:
    if news_direction_score is None:
        return 0.0
    return _clip(float(news_direction_score), -1.0, 1.0)


def compute_final_signal(
    *,
    risk_severity_avg: float,
    tone_delta: float,
    valuation_gap_pct: Optional[float] = None,
    revenue_growth_yoy: Optional[float] = None,
    news_direction_score: Optional[float] = None,
    evidence_count: int = 0,
    contradiction_penalty: float = 0.0,
) -> SignalScore:
    """
    Weighted final signal for hackathon demo.

    Components:
      risk        35%
      tone        20%
      valuation   25%
      growth      10%
      news        10%
    """

    comp = {
        "risk": _normalize_risk(risk_severity_avg),
        "tone": _normalize_tone(tone_delta),
        "valuation": _normalize_valuation(valuation_gap_pct),
        "growth": _normalize_growth(revenue_growth_yoy),
        "news": _normalize_news(news_direction_score),
    }

    weights = {
        "risk": 0.35,
        "tone": 0.20,
        "valuation": 0.25,
        "growth": 0.10,
        "news": 0.10,
    }

    raw = sum(comp[k] * weights[k] for k in weights)
    raw -= _clip(float(contradiction_penalty), 0.0, 0.5)
    signal_score = round(_clip(raw, -1.0, 1.0), 4)

    # confidence scales with evidence count, reduced by contradiction penalty
    base_conf = min(0.35 + 0.08 * int(evidence_count), 0.95)
    confidence = round(_clip(base_conf - float(contradiction_penalty), 0.0, 1.0), 4)

    if signal_score >= 0.30:
        label = "BUY"
    elif signal_score >= 0.05:
        label = "HOLD"
    elif signal_score >= -0.25:
        label = "CAUTIOUS"
    else:
        label = "AVOID"

    key_findings: List[str] = []
    risk_flags: List[str] = []

    if comp["risk"] < -0.45:
        risk_flags.append("elevated_risk_language")
        key_findings.append("Risk language is materially elevated in the retrieved evidence.")

    if comp["tone"] > 0.10:
        key_findings.append("Management tone improved relative to the prior comparison period.")
    elif comp["tone"] < -0.10:
        key_findings.append("Management tone worsened relative to the prior comparison period.")

    if comp["valuation"] > 0.15:
        key_findings.append("Valuation appears attractive under the current assumptions.")
    elif comp["valuation"] < -0.15:
        key_findings.append("Valuation appears stretched under the current assumptions.")

    if comp["growth"] > 0.20:
        key_findings.append("Revenue growth remains a positive supporting signal.")
    elif comp["growth"] < -0.20:
        key_findings.append("Revenue growth is weak enough to be a negative signal.")

    if comp["news"] < -0.15:
        risk_flags.append("negative_recent_catalysts")
        key_findings.append("Recent news flow is skewing negative.")
    elif comp["news"] > 0.15:
        key_findings.append("Recent news flow is directionally supportive.")

    if not key_findings:
        key_findings.append("Signals are mixed and do not point to a dominant directional edge.")

    return SignalScore(
        signal_score=signal_score,
        confidence=confidence,
        label=label,
        component_scores={k: round(v, 4) for k, v in comp.items()},
        component_confidences={k: 0.7 for k in comp},  # Legacy: fixed confidences for old approach
        key_findings=key_findings,
        risk_flags=risk_flags,
        tool_details={"aggregation_method": "fixed_weights_legacy"},
    )


def compute_final_signal_dynamic(
    *,
    tools: Dict[str, ToolSignal],
    base_weights: Optional[Dict[str, float]] = None,
) -> SignalScore:
    """
    Dynamically weighted signal aggregation based on per-tool confidence.

    Key features:
    1. Weights scale with tool confidence: effective_weight = base_weight × tool_confidence
    2. Contradictions reduce tool confidence before aggregation
    3. Final score and confidence are both confidence-weighted aggregates
    4. Provides transparency on per-tool scores and confidences

    Args:
        tools: Dict of tool_name -> ToolSignal (contains score, confidence, evidence, contradictions)
        base_weights: Dict of tool_name -> base_weight. Defaults to domain-driven weights.
    """

    if base_weights is None:
        # Domain-driven base weights (Risk most reliable, News least)
        base_weights = {
            "risk": 0.35,
            "tone": 0.20,
            "valuation": 0.25,
            "growth": 0.10,
            "news": 0.10,
        }

    # Step 1: Adjust confidences for contradictions within each tool
    adjusted_tools = {}
    for tool_name, tool in tools.items():
        adjusted_conf = _adjust_confidence_for_contradictions(
            tool.confidence,
            tool.contradictions,
            penalty_per_contradiction=0.08,
        )
        adjusted_tools[tool_name] = {
            "score": tool.score,
            "base_confidence": tool.confidence,
            "adjusted_confidence": adjusted_conf,
            "contradictions": tool.contradictions,
            "evidence_count": tool.evidence_count,
        }

    # Step 2: Compute dynamic weights = base_weight × adjusted_confidence
    weighted_rows = []
    total_effective_weight = 0.0

    for tool_name, tool_data in adjusted_tools.items():
        base_w = base_weights.get(tool_name, 0.10)  # Default 10% if not specified
        tool_conf = tool_data["adjusted_confidence"]

        # Effective weight: base weight scaled by tool confidence
        effective_w = base_w * tool_conf

        weighted_contribution = effective_w * tool_data["score"]
        total_effective_weight += effective_w

        weighted_rows.append({
            "name": tool_name,
            "score": tool_data["score"],
            "base_weight": base_w,
            "base_confidence": tool_data["base_confidence"],
            "adjusted_confidence": tool_conf,
            "effective_weight": effective_w,
            "weighted_contribution": weighted_contribution,
        })

    # Step 3: Aggregate weighted score
    if total_effective_weight > 0:
        raw_score = sum(r["weighted_contribution"] for r in weighted_rows) / total_effective_weight
    else:
        raw_score = 0.0

    signal_score = round(_clip(raw_score, -1.0, 1.0), 4)

    # Step 4: Aggregate confidence using effective weights
    # Confidence = weighted average of adjusted per-tool confidences
    if total_effective_weight > 0:
        agg_confidence = sum(
            r["effective_weight"] * r["adjusted_confidence"]
            for r in weighted_rows
        ) / total_effective_weight
    else:
        agg_confidence = 0.0

    confidence = round(_clip(agg_confidence, 0.0, 1.0), 4)

    # Step 5: Generate label based on score
    if signal_score >= 0.30:
        label = "BUY"
    elif signal_score >= 0.05:
        label = "HOLD"
    elif signal_score >= -0.25:
        label = "CAUTIOUS"
    else:
        label = "AVOID"

    # Step 6: Extract component scores and confidences
    component_scores = {r["name"]: r["score"] for r in weighted_rows}
    component_confidences = {r["name"]: r["adjusted_confidence"] for r in weighted_rows}

    # Step 7: Generate key findings and risk flags
    key_findings: List[str] = []
    risk_flags: List[str] = []

    risk_score = component_scores.get("risk", 0.0)
    if risk_score < -0.45:
        risk_flags.append("elevated_risk_language")
        key_findings.append("Risk language is materially elevated in the retrieved evidence.")

    tone_score = component_scores.get("tone", 0.0)
    if tone_score > 0.10:
        key_findings.append("Management tone improved relative to the prior comparison period.")
    elif tone_score < -0.10:
        key_findings.append("Management tone worsened relative to the prior comparison period.")

    val_score = component_scores.get("valuation", 0.0)
    if val_score > 0.15:
        key_findings.append("Valuation appears attractive under the current assumptions.")
    elif val_score < -0.15:
        key_findings.append("Valuation appears stretched under the current assumptions.")

    growth_score = component_scores.get("growth", 0.0)
    if growth_score > 0.20:
        key_findings.append("Revenue growth remains a positive supporting signal.")
    elif growth_score < -0.20:
        key_findings.append("Revenue growth is weak enough to be a negative signal.")

    news_score = component_scores.get("news", 0.0)
    if news_score < -0.15:
        risk_flags.append("negative_recent_catalysts")
        key_findings.append("Recent news flow is skewing negative.")
    elif news_score > 0.15:
        key_findings.append("Recent news flow is directionally supportive.")

    # Check for tool confidence warnings
    for tool_name, tool_conf in component_confidences.items():
        if tool_conf < 0.45:
            key_findings.append(f"{tool_name.capitalize()} signal has low confidence ({tool_conf:.0%}) due to limited evidence or contradictions.")

    if not key_findings:
        key_findings.append("Signals are mixed and do not point to a dominant directional edge.")

    return SignalScore(
        signal_score=signal_score,
        confidence=confidence,
        label=label,
        component_scores=component_scores,
        component_confidences=component_confidences,
        key_findings=key_findings,
        risk_flags=risk_flags,
        tool_details={
            "weighted_rows": weighted_rows,
            "total_effective_weight": round(total_effective_weight, 4),
            "aggregation_method": "confidence_weighted_dynamic",
        },
    )


def build_tool_signals_from_components(
    *,
    risk_avg: float,
    risk_evidence_count: int,
    tone_delta: float,
    tone_evidence_count: int,
    valuation_gap_pct: Optional[float],
    valuation_evidence_count: int,
    revenue_growth_yoy: Optional[float],
    growth_evidence_count: int,
    news_direction_score: Optional[float],
    news_evidence_count: int,
    contradiction_map: Optional[Dict[str, List[str]]] = None,
    risk_quality_confidence: Optional[float] = None,
    tone_quality_confidence: Optional[float] = None,
    valuation_quality_confidence: Optional[float] = None,
    growth_quality_confidence: Optional[float] = None,
    news_quality_confidence: Optional[float] = None,
) -> Dict[str, ToolSignal]:
    """
    Builds ToolSignal objects for dynamic weighting.

    Detects contradictions and computes per-tool confidence:
    - Base confidence: 0.35 + 0.08 × evidence_count (capped at 0.95) OR quality-based if provided
    - Adjusted confidence: base - 0.08 × num_contradictions

    Args:
        *_avg: normalized score for tool
        *_evidence_count: number of evidence blocks supporting tool
        contradiction_map: Dict[tool_name -> List[contradiction_reasons]]
        *_quality_confidence: Optional quality-based confidence (overrides simple formula)

    Returns:
        Dict of tool_name -> ToolSignal
    """
    if contradiction_map is None:
        contradiction_map = {}

    tools = {}

    # Risk tool
    if risk_quality_confidence is not None:
        risk_conf = risk_quality_confidence
    else:
        risk_conf = _compute_base_tool_confidence(risk_evidence_count, has_data=True)
    risk_contradictions = contradiction_map.get("risk", [])
    risk_conf = _adjust_confidence_for_contradictions(risk_conf, risk_contradictions)
    tools["risk"] = ToolSignal(
        name="risk",
        score=_normalize_risk(risk_avg),
        confidence=round(risk_conf, 4),
        evidence_count=risk_evidence_count,
        contradictions=risk_contradictions,
    )

    # Tone tool
    tone_has_data = abs(tone_delta) > 0.0 or tone_evidence_count > 0
    if tone_quality_confidence is not None:
        tone_conf = tone_quality_confidence
    else:
        tone_conf = _compute_base_tool_confidence(tone_evidence_count, has_data=tone_has_data)
    tone_contradictions = contradiction_map.get("tone", [])
    tone_conf = _adjust_confidence_for_contradictions(tone_conf, tone_contradictions)
    tools["tone"] = ToolSignal(
        name="tone",
        score=_normalize_tone(tone_delta),
        confidence=round(tone_conf, 4),
        evidence_count=tone_evidence_count,
        contradictions=tone_contradictions,
    )

    # Valuation tool
    val_has_data = valuation_gap_pct is not None and valuation_evidence_count > 0
    if valuation_quality_confidence is not None:
        val_conf = valuation_quality_confidence
    else:
        val_conf = _compute_base_tool_confidence(valuation_evidence_count, has_data=val_has_data)
    val_contradictions = contradiction_map.get("valuation", [])
    val_conf = _adjust_confidence_for_contradictions(val_conf, val_contradictions)
    tools["valuation"] = ToolSignal(
        name="valuation",
        score=_normalize_valuation(valuation_gap_pct),
        confidence=round(val_conf, 4),
        evidence_count=valuation_evidence_count,
        contradictions=val_contradictions,
    )

    # Growth tool
    growth_has_data = revenue_growth_yoy is not None and growth_evidence_count > 0
    if growth_quality_confidence is not None:
        growth_conf = growth_quality_confidence
    else:
        growth_conf = _compute_base_tool_confidence(growth_evidence_count, has_data=growth_has_data)
    growth_contradictions = contradiction_map.get("growth", [])
    growth_conf = _adjust_confidence_for_contradictions(growth_conf, growth_contradictions)
    tools["growth"] = ToolSignal(
        name="growth",
        score=_normalize_growth(revenue_growth_yoy),
        confidence=round(growth_conf, 4),
        evidence_count=growth_evidence_count,
        contradictions=growth_contradictions,
    )

    # News tool
    news_has_data = news_direction_score is not None and news_evidence_count > 0
    if news_quality_confidence is not None:
        news_conf = news_quality_confidence
    else:
        news_conf = _compute_base_tool_confidence(news_evidence_count, has_data=news_has_data)
    news_contradictions = contradiction_map.get("news", [])
    news_conf = _adjust_confidence_for_contradictions(news_conf, news_contradictions)
    tools["news"] = ToolSignal(
        name="news",
        score=_normalize_news(news_direction_score),
        confidence=round(news_conf, 4),
        evidence_count=news_evidence_count,
        contradictions=news_contradictions,
    )

    return tools


def to_dict(signal: SignalScore) -> Dict[str, Any]:
    return asdict(signal)
