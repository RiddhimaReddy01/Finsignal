import requests
import json
import sys

def debug_mode(mode, ticker, query, output_file):
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": query,
        "ticker": ticker,
        "mode": mode,
        "strictness": 50
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"SUCCESS: Result written to {output_file}")
        else:
            print(f"ERROR {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        # Default for convenience
        debug_mode("mba_framework", "NVDA", "Conduct a SWOT analysis for Nvidia.", "swot_debug.json")
    else:
        debug_mode(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
