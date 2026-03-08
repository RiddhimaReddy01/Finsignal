from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DemoSignalReport:
    ticker: str
    fiscal_year: Optional[int]
    company: Optional[str]
    signal_strength: float
    confidence: float
    recommendation: str

    key_findings: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    top_risks: List[Dict[str, Any]] = field(default_factory=list)
    tone_trend: Dict[str, Any] = field(default_factory=dict)
    valuation_summary: Dict[str, Any] = field(default_factory=dict)
    news_summary: List[Dict[str, Any]] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"## {self.ticker} — Signal Report")
        lines.append("")
        lines.append(f"**Recommendation:** {self.recommendation}")
        lines.append(f"**Signal Strength:** {self.signal_strength:.2f}")
        lines.append(f"**Confidence:** {self.confidence:.2f}")
        lines.append("")

        if self.key_findings:
            lines.append("### Key Findings")
            for x in self.key_findings:
                lines.append(f"- {x}")
            lines.append("")

        if self.top_risks:
            lines.append("### Top Risks")
            for r in self.top_risks[:5]:
                lines.append(
                    f"- {r.get('category')}: severity={r.get('severity')} count={r.get('count')}"
                )
            lines.append("")

        if self.tone_trend:
            lines.append("### Tone Trend")
            lines.append(
                f"- Direction: {self.tone_trend.get('direction')}, "
                f"delta={self.tone_trend.get('delta')}"
            )
            lines.append("")

        if self.valuation_summary:
            lines.append("### Valuation")
            for k, v in self.valuation_summary.items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if self.news_summary:
            lines.append("### Recent Catalysts")
            for item in self.news_summary[:5]:
                lines.append(
                    f"- [{item.get('direction')}] {item.get('title')} ({item.get('source_name')})"
                )
            lines.append("")

        if self.citations:
            lines.append("### Evidence IDs")
            lines.append(", ".join(self.citations[:20]))

        return "\n".join(lines)


def build_demo_signal_report(
    *,
    ticker: str,
    fiscal_year: Optional[int],
    company: Optional[str],
    score_obj: Dict[str, Any],
    top_risks: List[Dict[str, Any]],
    tone_trend: Dict[str, Any],
    valuation_summary: Dict[str, Any],
    news_summary: List[Dict[str, Any]],
    citations: List[str],
) -> DemoSignalReport:
    return DemoSignalReport(
        ticker=ticker,
        fiscal_year=fiscal_year,
        company=company,
        signal_strength=float(score_obj.get("signal_score", 0.0)),
        confidence=float(score_obj.get("confidence", 0.0)),
        recommendation=str(score_obj.get("label", "HOLD")),
        key_findings=list(score_obj.get("key_findings", [])),
        risk_flags=list(score_obj.get("risk_flags", [])),
        top_risks=top_risks,
        tone_trend=tone_trend,
        valuation_summary=valuation_summary,
        news_summary=news_summary,
        citations=citations,
    )