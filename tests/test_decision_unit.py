import pytest
from report_generation import generate_structured_decision_report, render_report_html, StructuredDecisionReport

def test_generate_structured_decision_report():
    mock_payload = {
        "hackathon_signal_score": {
            "signal_score": 0.85,
            "confidence": 0.92
        },
        "hackathon_signal_decision": {
            "action": "BUY",
            "policy": "High conviction based on strong fundamentals."
        },
        "tools_used": {
            "risk": {"factors": [{"category": "macro", "severity": 0.8, "display_name": "Macroeconomic Risk"}]},
            "valuation": {"factors": {"intrinsic_value": 200, "current_price": 150, "valuation_gap_pct": 33.3}}
        },
        "hackathon_signal_report": {
            "ticker": "AAPL",
            "fiscal_year": 2024
        },
        "scenarios": {
            "bull": {"intrinsic_value": 250},
            "base": {"intrinsic_value": 200},
            "bear": {"intrinsic_value": 150}
        },
        "evidence": {
            "chunks": [{"id": "ev1", "text": "Earnings beat expectations."}]
        }
    }
    
    report = generate_structured_decision_report(mock_payload, use_pydanticai=False)
    assert isinstance(report, StructuredDecisionReport)
    assert report.ticker == "AAPL"
    assert report.fiscal_year == 2024
    assert report.action == "BUY"
    assert report.signal_score == 0.85
    assert report.confidence == 0.92
    assert any("Macroeconomic Risk" in r for r in report.top_risks)
    assert any("200" in v for v in report.valuation_notes)
    assert report.evidence_ids == ["ev1"]
    
def test_render_decision_report_html():
    report = StructuredDecisionReport(
        ticker="AAPL",
        fiscal_year=2024,
        generated_at="2024-01-01T00:00:00Z",
        action="BUY",
        signal_score=0.85,
        confidence=0.92,
        executive_summary="Summary",
        key_findings=["Finding 1"],
        top_risks=["Risk 1"],
        valuation_notes=["Valuation 1"],
        scenario_notes=["Scenario 1"],
        evidence_ids=["ev1"]
    )
    html = render_report_html(report)
    assert "AAPL Investment Decision Report" in html
    assert "BUY" in html
    assert "0.85" in html
