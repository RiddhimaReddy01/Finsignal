import requests
import json

def debug_mode(mode, ticker, query):
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": query,
        "ticker": ticker,
        "mode": mode,
        "strictness": 50
    }
    
    print(f"--- Debugging Mode: {mode} ---")
    resp = requests.post(url, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error {resp.status_code}: {resp.text}")

if __name__ == "__main__":
    debug_mode("mba_framework", "NVDA", "Conduct a SWOT analysis for Nvidia.")
