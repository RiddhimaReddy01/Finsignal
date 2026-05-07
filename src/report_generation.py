from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from pydantic_ai import Agent  # type: ignore
except Exception:
    Agent = None  # type: ignore

try:
    from jinja2 import Template
except Exception:
    Template = None  # type: ignore

try:
    from weasyprint import HTML  # type: ignore
except Exception:
    HTML = None  # type: ignore


class ReportClaim(BaseModel):
    title: str
    detail: str
    citations: List[str] = Field(default_factory=list)


class StructuredDecisionReport(BaseModel):
    ticker: str
    fiscal_year: Optional[int] = None
    generated_at: str
    action: str
    signal_score: float
    confidence: float
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    top_risks: List[str] = Field(default_factory=list)
    valuation_notes: List[str] = Field(default_factory=list)
    scenario_notes: List[str] = Field(default_factory=list)
    claims: List[ReportClaim] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)


class StructuredResearchReport(BaseModel):
    user_query: str = ""
    mode: str
    ticker: str
    fiscal_year: Optional[int] = None
    generated_at: str
    verdict: str
    confidence: float
    evidence_score: float
    executive_summary: str
    key_findings: List[str] = Field(default_factory=list)
    analysis_body: str
    evidence_ids: List[str] = Field(default_factory=list)


_REPORT_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ r.ticker }} Decision Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1 { margin-bottom: 6px; }
    .meta { color: #444; margin-bottom: 18px; }
    .kpi { display: flex; gap: 24px; margin: 12px 0 18px 0; }
    .kpi div { padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px; min-width: 160px; }
    .lbl { font-size: 12px; color: #666; text-transform: uppercase; }
    .val { font-size: 20px; font-weight: 700; margin-top: 4px; }
    h2 { margin-top: 18px; font-size: 18px; }
    ul { margin-top: 6px; }
    .claim { border-left: 3px solid #ccc; padding-left: 10px; margin: 8px 0; }
    .cit { font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <h1>{{ r.ticker }} Investment Decision Report</h1>
  <div class="meta">FY{{ r.fiscal_year if r.fiscal_year else "N/A" }} · Generated {{ r.generated_at }}</div>
  <div class="kpi">
    <div><div class="lbl">Action</div><div class="val">{{ r.action }}</div></div>
    <div><div class="lbl">Signal Score</div><div class="val">{{ "%+.4f"|format(r.signal_score) }}</div></div>
    <div><div class="lbl">Confidence</div><div class="val">{{ "{:.1%}".format(r.confidence) }}</div></div>
  </div>

  <h2>Executive Summary</h2>
  <p>{{ r.executive_summary }}</p>

  <h2>Key Findings</h2>
  <ul>{% for x in r.key_findings %}<li>{{ x }}</li>{% endfor %}</ul>

  <h2>Top Risks</h2>
  <ul>{% for x in r.top_risks %}<li>{{ x }}</li>{% endfor %}</ul>

  <h2>Valuation Notes</h2>
  <ul>{% for x in r.valuation_notes %}<li>{{ x }}</li>{% endfor %}</ul>

  <h2>Scenario Notes</h2>
  <ul>{% for x in r.scenario_notes %}<li>{{ x }}</li>{% endfor %}</ul>

  <h2>Claims & Evidence</h2>
  {% for c in r.claims %}
    <div class="claim">
      <div><b>{{ c.title }}</b></div>
      <div>{{ c.detail }}</div>
      <div class="cit">Citations: {{ ", ".join(c.citations) if c.citations else "N/A" }}</div>
    </div>
  {% endfor %}
</body>
</html>
"""


_RESEARCH_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ r.ticker }} Research Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #111; }
    h1 { margin-bottom: 6px; }
    .meta { color: #444; margin-bottom: 18px; }
    .kpi { display: flex; gap: 16px; margin: 12px 0 18px 0; }
    .kpi div { padding: 10px 12px; border: 1px solid #ddd; border-radius: 8px; min-width: 160px; }
    .lbl { font-size: 12px; color: #666; text-transform: uppercase; }
    .val { font-size: 18px; font-weight: 700; margin-top: 4px; }
    h2 { margin-top: 18px; font-size: 18px; }
    ul { margin-top: 6px; }
    .body { white-space: pre-wrap; line-height: 1.55; border: 1px solid #eee; padding: 12px; border-radius: 8px; background:#fafafa; }
    .foot { margin-top: 16px; font-size: 12px; color: #666; }
  </style>
</head>
<body>
  <h1>{{ r.ticker }} Research Report</h1>
  <div class="meta">Mode: {{ r.mode }} · FY{{ r.fiscal_year if r.fiscal_year else "N/A" }} · Generated {{ r.generated_at }}</div>
  {% if r.user_query %}
  <h2>User Query</h2>
  <div class="body">{{ r.user_query }}</div>
  {% endif %}
  <div class="kpi">
    <div><div class="lbl">Verdict</div><div class="val">{{ r.verdict }}</div></div>
    <div><div class="lbl">Confidence</div><div class="val">{{ "{:.1%}".format(r.confidence) }}</div></div>
    <div><div class="lbl">Evidence Score</div><div class="val">{{ "{:.1%}".format(r.evidence_score) }}</div></div>
  </div>
  <h2>Executive Summary</h2>
  <p>{{ r.executive_summary }}</p>
  <h2>Key Findings</h2>
  <ul>{% for x in r.key_findings %}<li>{{ x }}</li>{% endfor %}</ul>
  <h2>Analysis</h2>
  <div class="body">{{ r.analysis_body }}</div>
  <div class="foot">Evidence IDs: {{ ", ".join(r.evidence_ids) if r.evidence_ids else "N/A" }}</div>
</body>
</html>
"""


def _build_base_report_payload(decision_payload: Dict[str, Any]) -> Dict[str, Any]:
    score = decision_payload.get("hackathon_signal_score", {}) or {}
    decision = decision_payload.get("hackathon_signal_decision", {}) or {}
    tools = decision_payload.get("tools_used", {}) or {}
    report = decision_payload.get("hackathon_signal_report", {}) or {}
    ev = decision_payload.get("evidence", {}) or {}
    chunks = ev.get("chunks", []) or []
    ev_ids = [str(c.get("id")) for c in chunks if isinstance(c, dict) and c.get("id")]
    top_risks = []
    for r in (tools.get("risk", {}) or {}).get("factors", [])[:5]:
        if isinstance(r, dict):
            top_risks.append(f"{r.get('display_name', r.get('category', 'risk'))}: severity={r.get('severity')}")
    valuation = (tools.get("valuation", {}) or {}).get("factors", {}) or {}
    scenarios = decision_payload.get("scenarios", {}) or {}
    return {
        "ticker": str(report.get("ticker") or "N/A"),
        "fiscal_year": report.get("fiscal_year"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "action": str(decision.get("action") or "NO_ACT"),
        "signal_score": float(score.get("signal_score", 0.0) or 0.0),
        "confidence": float(score.get("confidence", 0.0) or 0.0),
        "executive_summary": (
            f"Decision action is {decision.get('action', 'NO_ACT')} with signal "
            f"{float(score.get('signal_score', 0.0) or 0.0):+.4f} and confidence "
            f"{float(score.get('confidence', 0.0) or 0.0):.1%}."
        ),
        "key_findings": [
            "Decision synthesized from risk, tone, valuation, growth, news, scenario, and peer tools.",
            f"Policy: {decision.get('policy', 'N/A')}",
        ],
        "top_risks": top_risks,
        "valuation_notes": [
            f"Intrinsic value per share: {valuation.get('intrinsic_value')}",
            f"Current price: {valuation.get('current_price')}",
            f"Valuation gap: {valuation.get('valuation_gap_pct')}",
        ],
        "scenario_notes": [
            f"Bull IV: {(scenarios.get('bull', {}) or {}).get('intrinsic_value')}",
            f"Base IV: {(scenarios.get('base', {}) or {}).get('intrinsic_value')}",
            f"Bear IV: {(scenarios.get('bear', {}) or {}).get('intrinsic_value')}",
        ],
        "claims": [
            {
                "title": "Decision Outcome",
                "detail": f"Action={decision.get('action')} Score={score.get('signal_score')} Confidence={score.get('confidence')}",
                "citations": ev_ids[:5],
            },
            {
                "title": "Valuation Snapshot",
                "detail": f"IV={valuation.get('intrinsic_value')} vs Price={valuation.get('current_price')}",
                "citations": ev_ids[5:10],
            },
        ],
        "evidence_ids": ev_ids[:30],
    }


def generate_structured_decision_report(
    decision_payload: Dict[str, Any],
    *,
    use_pydanticai: bool = True,
) -> StructuredDecisionReport:
    base = _build_base_report_payload(decision_payload)
    if use_pydanticai and Agent is not None:
        try:
            # Uses pydanticai schema validation flow; falls back if model/provider not configured.
            agent = Agent(
                "openai:gpt-4o-mini",
                output_type=StructuredDecisionReport,
                system_prompt=(
                    "Convert the provided decision payload JSON into a structured "
                    "investment report JSON that conforms to the output schema."
                ),
            )
            prompt = json.dumps(base, ensure_ascii=False)
            result = agent.run_sync(prompt)
            data = getattr(result, "output", None)
            if isinstance(data, StructuredDecisionReport):
                return data
        except Exception:
            pass
    return StructuredDecisionReport.model_validate(base)


def _build_base_research_payload(research_payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(research_payload.get("mode") or "research")
    result = research_payload.get("result", {}) or {}
    verification = research_payload.get("verification", {}) or {}
    gate = verification.get("gate", {}) or {}
    answer = str(result.get("final_answer") or "")
    ticker = "N/A"
    try:
        if "provenance" in result and isinstance(result["provenance"], dict):
            ticker = str(result["provenance"].get("ticker") or ticker)
    except Exception:
        pass
    if ticker == "N/A":
        ticker = str(
            research_payload.get("ticker")
            or (research_payload.get("request_context", {}) or {}).get("ticker")
            or "N/A"
        )
    fy = (result.get("provenance", {}) or {}).get("fiscal_year")
    if fy is None:
        fy = (
            research_payload.get("fiscal_year")
            or (research_payload.get("request_context", {}) or {}).get("fiscal_year")
        )
    action_raw = str(research_payload.get("action") or "abstain").upper()
    reason_codes = verification.get("reason_codes") or []
    reason_text = ", ".join([str(x) for x in reason_codes[:3]]) if reason_codes else "insufficient evidence"
    if not answer.strip():
        answer = f"No answer generated. System action: {action_raw}. Primary reason: {reason_text}."
    ev = research_payload.get("evidence_hydrated", []) or []
    ev_ids = [str(x.get("id")) for x in ev if isinstance(x, dict) and x.get("id")]
    key_findings = []
    for ln in answer.splitlines():
        s = ln.strip()
        if s.startswith("- ") or s.startswith("* "):
            key_findings.append(s[2:].strip())
        if len(key_findings) >= 6:
            break
    if not key_findings:
        if action_raw in {"ABSTAIN", "CLARIFY"}:
            key_findings = [
                f"Analysis returned {action_raw} due to verification/evidence gating.",
                f"Primary rationale: {reason_text}.",
            ]
        else:
            key_findings = [
                "Evidence-based response generated from selected mode.",
                "Output includes verification and source traceability.",
            ]
    executive_summary = answer.splitlines()[0][:240] if answer else "No answer generated."
    if action_raw in {"ABSTAIN", "CLARIFY"}:
        executive_summary = f"{action_raw}: {reason_text}."
    return {
        "user_query": str(research_payload.get("query") or ""),
        "mode": mode,
        "ticker": ticker,
        "fiscal_year": fy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": action_raw,
        "confidence": float(result.get("confidence", gate.get("confidence", 0.0)) or 0.0),
        "evidence_score": float(gate.get("score", gate.get("evidence_score", 0.0)) or 0.0),
        "executive_summary": executive_summary,
        "key_findings": key_findings,
        "analysis_body": answer[:8000] if answer else "No analysis text available.",
        "evidence_ids": ev_ids[:40],
    }


def generate_structured_research_report(
    research_payload: Dict[str, Any],
    *,
    use_pydanticai: bool = True,
) -> StructuredResearchReport:
    base = _build_base_research_payload(research_payload)
    if use_pydanticai and Agent is not None:
        try:
            agent = Agent(
                "openai:gpt-4o-mini",
                output_type=StructuredResearchReport,
                system_prompt=(
                    "Convert the provided research payload JSON into a structured "
                    "research report JSON that conforms to the output schema."
                ),
            )
            result = agent.run_sync(json.dumps(base, ensure_ascii=False))
            data = getattr(result, "output", None)
            if isinstance(data, StructuredResearchReport):
                return data
        except Exception:
            pass
    return StructuredResearchReport.model_validate(base)


def render_report_html(report: StructuredDecisionReport) -> str:
    if Template is None:
        # Minimal fallback HTML if jinja2 is unavailable.
        return (
            f"<html><body><h1>{report.ticker} Investment Decision Report</h1>"
            f"<p>Action: {report.action}</p><p>Score: {report.signal_score:+.4f}</p>"
            f"<p>Confidence: {report.confidence:.1%}</p></body></html>"
        )
    tpl = Template(_REPORT_TEMPLATE)
    return tpl.render(r=report.model_dump())


def render_research_report_html(report: StructuredResearchReport) -> str:
    if Template is None:
        return (
            f"<html><body><h1>{report.ticker} Research Report</h1>"
            f"<p>Verdict: {report.verdict}</p><p>Confidence: {report.confidence:.1%}</p></body></html>"
        )
    tpl = Template(_RESEARCH_TEMPLATE)
    return tpl.render(r=report.model_dump())


def export_report_pdf(html: str, out_path: Path) -> Optional[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if HTML is None:
        return None
    try:
        HTML(string=html).write_pdf(str(out_path))
        return out_path
    except Exception:
        return None

