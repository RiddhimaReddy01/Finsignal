import requests
import json

def test_final_research():
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": "Discuss Apple's recent performance in its services segment and the regulatory challenges it faces in the EU.",
        "ticker": "AAPL",
        "mode": "risk_analysis",
        "strictness": 50
    }
    
    print("Running final research query for AAPL...")
    try:
        resp = requests.post(url, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", {})
            answer = result.get("final_answer", "")
            
            print("\n--- FINAL RESEARCH REPORT ---\n")
            print(answer)
            
            hydrated = data.get("evidence_hydrated", [])
            print(f"\nHydrated Evidence Found: {len(hydrated)}")
            for e in hydrated[:3]:
                print(f"- Source: {e.get('source')} | Icon: {e.get('icon')} | Type: {e.get('source_type')}")
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_final_research()
