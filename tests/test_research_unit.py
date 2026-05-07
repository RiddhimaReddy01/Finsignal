import pytest
from report_generation import generate_structured_research_report, render_research_report_html, StructuredResearchReport

def test_generate_structured_research_report():
    mock_payload = {
        "query": "What is AAPL revenue?",
        "mode": "lookup_numeric",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "action": "ANSWER",
        "result": {
            "final_answer": "- Apple's revenue is $100B.\n- It grew by 5%.",
            "confidence": 0.95,
            "provenance": {"ticker": "AAPL", "fiscal_year": 2024}
        },
        "verification": {
            "gate": {"score": 0.9}
        },
        "evidence_hydrated": [
            {"id": "ev1", "text": "Apple's revenue is $100B."}
        ]
    }
    
    report = generate_structured_research_report(mock_payload, use_pydanticai=False)
    assert isinstance(report, StructuredResearchReport)
    assert report.ticker == "AAPL"
    assert report.fiscal_year == 2024
    assert report.mode == "lookup_numeric"
    assert "Apple's revenue is $100B." in report.key_findings
    assert report.evidence_ids == ["ev1"]
    
def test_render_research_report_html():
    report = StructuredResearchReport(
        mode="lookup_numeric",
        ticker="AAPL",
        generated_at="2024-01-01T00:00:00Z",
        verdict="ANSWER",
        confidence=0.95,
        evidence_score=0.9,
        executive_summary="Summary",
        key_findings=["Finding 1"],
        analysis_body="Body",
        evidence_ids=["ev1"]
    )
    html = render_research_report_html(report)
    assert "AAPL Research Report" in html
    assert "Summary" in html
    assert "Finding 1" in html
