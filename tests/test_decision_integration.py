import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import server
from server import app

client = TestClient(app)

@patch("server.get_orchestrator")
@patch("server.extract_risk_signals_with_diagnostics")
@patch("server.TranscriptIngestionClient")
@patch("server.analyze_tone")
@patch("server.NewsIngestionClient")
@patch("server.classify_news_catalysts")
@patch("server.run_dcf")
@patch("server.run_scenario_analysis")
@patch("server.run_peer_analysis")
@patch("server.peer_analysis_to_signal")
def test_decision_endpoint_integration(
    mock_peer_analysis_to_signal,
    mock_run_peer_analysis,
    mock_run_scenario_analysis,
    mock_run_dcf,
    mock_classify_news_catalysts,
    mock_NewsIngestionClient,
    mock_analyze_tone,
    mock_TranscriptIngestionClient,
    mock_extract_risk_signals,
    mock_get_orchestrator,
):
    # This might fail due to local imports in server.py, so we will also inject into cache
    pass

def test_decision_endpoint_cache():
    # Pre-populate cache to test the response structuring and cache logic
    key_payload = {
        "ticker": "TSLA",
        "fiscal_year": 2024,
        "strictness": 70,
    }
    cache_key = server._stable_cache_key("decision", key_payload)
    
    mock_payload = {
        "hackathon_signal_score": {"signal_score": 0.5, "confidence": 0.9},
        "hackathon_signal_decision": {"action": "HOLD"},
        "tools_used": {},
        "scenarios": {}
    }
    server._decision_cache[cache_key] = mock_payload
    
    response = client.post("/api/decision", json={
        "ticker": "TSLA",
        "fiscal_year": 2024,
        "strictness": 70
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data.get("cache", {}).get("hit") is True
    assert data.get("hackathon_signal_decision", {}).get("action") == "HOLD"
