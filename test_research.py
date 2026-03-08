import requests
import json

def test_research():
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": "What are the primary growth drivers and risks for Apple in 2024 based on recent filings and news?",
        "ticker": "AAPL",
        "mode": "risk_analysis",
        "strictness": 70
    }
    
    print("Running research query...")
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            print("\n--- RESEARCH ANSWER ---\n")
            print(data.get("answer", "No answer found"))
            print("\n--- EVIDENCE PREVIEW (TOP 2) ---\n")
            ev = data.get("evidence", [])
            for i, e in enumerate(ev[:2]):
                print(f"[{i}] Source: {e.get('source')} | Score: {e.get('score')}")
                print(f"Text Snippet: {e.get('text', '')[:100]}...")
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_research()
