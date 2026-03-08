from __future__ import annotations

from typing import Any, Dict, List, Optional

from demo_reporting import build_demo_signal_report
from news_ingestion import NewsIngestionClient
from nlp_signals import (
    classify_news_catalysts,
    compare_tone,
    extract_risk_signals,
)
from signal_scoring import compute_final_signal, to_dict


def _extract_item_1a_text_from_context(packed_context: str) -> str:
    """
    Pull a best-effort Item 1A slice from the packed SEC context.
    Your retriever already annotates context headers with ticker/FY/item metadata.
    """
    lines = []
    keep = False

    for line in str(packed_context or "").splitlines():
        ll = line.lower()

        # new block header
        if line.startswith("[") and line.endswith("]"):
            keep = False

        if "item 1a" in ll:
            keep = True

        if keep:
            lines.append(line)

    return "\n".join(lines)


def _collect_citations_from_evidence(evidence: Dict[str, Any]) -> List[str]:
    out: List[str] = []

    narrative = (evidence or {}).get("narrative") or {}
    tables = (evidence or {}).get("tables") or {}

    for cid in narrative.get("selected_chunk_ids", []) or []:
        out.append(str(cid))
    for tid in tables.get("selected_table_ids", []) or []:
        out.append(str(tid))

    return out[:40]


def _extract_valuation_signals(base_result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    valuation_gap_pct = None
    revenue_growth_yoy = None

    result_obj = base_result.get("result")
    if not isinstance(result_obj, dict):
        return {
            "valuation_gap_pct": None,
            "revenue_growth_yoy": None,
        }

    # Adapt this to whatever your valuation outputs currently look like
    if isinstance(result_obj.get("valuation"), dict):
        valuation_gap_pct = result_obj["valuation"].get("valuation_gap_pct")
        revenue_growth_yoy = result_obj["valuation"].get("revenue_growth_yoy")

    if isinstance(result_obj.get("relative_valuation"), dict):
        valuation_gap_pct = result_obj["relative_valuation"].get(
            "valuation_gap_pct",
            valuation_gap_pct,
        )

    # fallback: sometimes the top-level result may directly store values
    if valuation_gap_pct is None:
        valuation_gap_pct = result_obj.get("valuation_gap_pct")
    if revenue_growth_yoy is None:
        revenue_growth_yoy = result_obj.get("revenue_growth_yoy")

    return {
        "valuation_gap_pct": valuation_gap_pct,
        "revenue_growth_yoy": revenue_growth_yoy,
    }


def run_hackathon_signal_layer(
    *,
    base_result: Dict[str, Any],
    ticker: str,
    fiscal_year: Optional[int],
    company: Optional[str] = None,
    current_transcript_text: Optional[str] = None,
    prior_transcript_text: Optional[str] = None,
    recent_news_enabled: bool = True,
) -> Dict[str, Any]:
    packed_context = str(base_result.get("packed_context") or "")
    evidence = base_result.get("evidence") or {}

    # 1) Risk extraction from SEC context (Item 1A best-effort slice)
    item1a_text = _extract_item_1a_text_from_context(packed_context)
    risk_signals = extract_risk_signals(item1a_text)
    risk_avg = (
        sum(r.severity for r in risk_signals[:3]) / max(min(len(risk_signals), 3), 1)
        if risk_signals else 0.0
    )

    # 2) Tone delta from transcripts (if available)
    tone_trend: Dict[str, Any] = {}
    tone_delta = 0.0
    if current_transcript_text and prior_transcript_text:
        tone_trend = compare_tone(current_transcript_text, prior_transcript_text)
        tone_delta = float(tone_trend.get("delta", 0.0))

    # 3) Recent news catalysts
    news_summary: List[Dict[str, Any]] = []
    avg_news_score: Optional[float] = None

    if recent_news_enabled:
        try:
            client = NewsIngestionClient()
            articles = client.fetch_recent_news(
                ticker=ticker,
                company=company,
                max_results=10,
            )
            catalysts = classify_news_catalysts([a.__dict__ for a in articles])
            news_summary = [c.__dict__ for c in catalysts[:5]]

            if catalysts:
                top = catalysts[:5]
                avg_news_score = sum(c.score for c in top) / len(top)
        except Exception as exc:
            news_summary = [{
                "direction": "neutral",
                "title": f"news_fetch_failed: {exc}",
                "source_name": "system",
            }]

    # 4) Valuation / growth features
    valuation_bits = _extract_valuation_signals(base_result)
    valuation_gap_pct = valuation_bits.get("valuation_gap_pct")
    revenue_growth_yoy = valuation_bits.get("revenue_growth_yoy")

    valuation_summary = {
        "valuation_gap_pct": valuation_gap_pct,
        "revenue_growth_yoy": revenue_growth_yoy,
    }

    citations = _collect_citations_from_evidence(evidence)

    # 5) Final score
    score = compute_final_signal(
        risk_severity_avg=risk_avg,
        tone_delta=tone_delta,
        valuation_gap_pct=valuation_gap_pct,
        revenue_growth_yoy=revenue_growth_yoy,
        news_direction_score=avg_news_score,
        evidence_count=len(citations),
        contradiction_penalty=0.0,
    )

    report = build_demo_signal_report(
        ticker=ticker,
        fiscal_year=fiscal_year,
        company=company,
        score_obj=to_dict(score),
        top_risks=[r.__dict__ for r in risk_signals[:5]],
        tone_trend=tone_trend,
        valuation_summary=valuation_summary,
        news_summary=news_summary,
        citations=citations,
    )

    enriched = dict(base_result)
    enriched["hackathon_signal_score"] = to_dict(score)
    enriched["hackathon_signal_report"] = report.to_dict()
    enriched["hackathon_signal_markdown"] = report.to_markdown()
    return enriched