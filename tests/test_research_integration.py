from fastapi.testclient import TestClient
import pytest
from unittest.mock import MagicMock, patch
from server import app, get_orchestrator

client = TestClient(app)

@pytest.fixture
def mock_orchestrator():
    with patch("server.get_orchestrator") as mock_get:
        mock_orch = MagicMock()
        mock_orch.retrieval.narrative.chunk_row = {}
        mock_orch.retrieval.table_retriever.table_row = {}
        
        mock_orch.answer.return_value = {
            "action": "ANSWER",
            "mode": "auto",
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "query": "What is AAPL's revenue?",
            "result": {
                "final_answer": "AAPL reported $383B revenue in FY2023.",
                "confidence": 0.95
            },
            "verification": {
                "gate": {"score": 0.9}
            },
            "evidence": {
                "narrative": {"selected_chunk_ids": []},
                "tables": {"selected_table_ids": []},
                "xbrl": {"hits": []}
            }
        }
        mock_get.return_value = mock_orch
        yield mock_get

def test_analyze_endpoint_success(mock_orchestrator):
    response = client.post("/api/analyze", json={
        "query": "What is AAPL's revenue?",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "mode": "auto",
        "strictness": 70
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is AAPL's revenue?"
    assert data["ticker"] == "AAPL"
    assert "result" in data
    assert "AAPL reported $383B" in data["result"]["final_answer"]
    assert "generated_report" in data
    assert data["generated_report"]["html"] is not None

def test_analyze_endpoint_cache(mock_orchestrator):
    # First request
    res1 = client.post("/api/analyze", json={
        "query": "What is AAPL's cash?",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "mode": "auto",
        "strictness": 70
    })
    assert res1.status_code == 200
    assert "cache" not in res1.json() or res1.json().get("cache", {}).get("hit") is not True

    # Second request (should be cached in memory)
    res2 = client.post("/api/analyze", json={
        "query": "What is AAPL's cash?",
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "mode": "auto",
        "strictness": 70
    })
    assert res2.status_code == 200
    assert res2.json().get("cache", {}).get("hit") is True
    assert res2.json().get("cache", {}).get("layer") == "memory"
