import requests
import json

def test_research():
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": "Provide a comprehensive report on Nvidia's data center growth and AI market share prospects for 2024.",
        "ticker": "NVDA",
        "mode": "risk_analysis", # Using a mode that triggers LLM narrative
        "strictness": 70
    }
    
    print("Running research query for NVDA...")
    try:
        resp = requests.post(url, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            result = data.get("result", {})
            answer = result.get("final_answer", "")
            
            print("\n--- REFINED RESEARCH REPORT ---\n")
            print(answer)
            
            print("\n--- VERIFICATION ---\n")
            print(f"Contains headers (###): {'###' in answer}")
            print(f"Contains bolding (**): {'**' in answer}")
            print(f"Contains bullets (-): {'-' in answer}")
            
            ev_chunks = data.get("evidence", {}).get("chunks", [])
            print(f"\nEvidence Chunks Found: {len(ev_chunks)}")
            if ev_chunks:
                first_ev = ev_chunks[0]
                print(f"First Evidence Source: {first_ev.get('source')} | Type: {first_ev.get('source_type')}")
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_research()
