import requests
import json

def test_research():
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": "What are the recent risk factors discussed by Apple management in their 10-K filings?",
        "ticker": "AAPL",
        "mode": "risk_analysis",
        "strictness": 50 # Lowering strictness to ensure we get results even if index is sparse
    }
    
    print("Running research query for AAPL...")
    try:
        resp = requests.post(url, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", {})
            answer = result.get("final_answer", "")
            
            print("\n--- REFINED RESEARCH REPORT ---\n")
            print(answer[:500] + "...")
            
            # Key for frontend evidence is raw.evidence.chunks? 
            # Let's see where evidence is.
            evidence_keys = list(data.get("evidence", {}).keys())
            print(f"Evidence Keys: {evidence_keys}")
            
            chunks = data.get("evidence", {}).get("narrative", {}).get("reranked", [])
            print(f"Narrative Chunks (reranked): {len(chunks)}")
            
            # The frontend specifically expects raw.evidence.chunks? 
            # Wait, line 236 in FinSightTerminal.jsx: const evidence = raw.evidence?.chunks || [];
            # But the backend returns a more nested structure?
            
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_research()
