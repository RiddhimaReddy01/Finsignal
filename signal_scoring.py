from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SignalScore:
    signal_score: float          # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0
    label: str                   # BUY / HOLD / CAUTIOUS / AVOID
    component_scores: Dict[str, float]
    key_findings: List[str]
    risk_flags: List[str]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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
        key_findings=key_findings,
        risk_flags=risk_flags,
    )


def to_dict(signal: SignalScore) -> Dict[str, Any]:
    return asdict(signal)