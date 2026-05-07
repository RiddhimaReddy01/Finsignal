import requests
import json
import time

def test_decision_e2e():
    print("--- Running End-to-End Test for Decision Mode ---")
    url = "http://localhost:8000/api/decision"
    payload = {
        "ticker": "AAPL",
        "fiscal_year": 2024,
        "strictness": 70
    }
    
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=120)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            print(f"Status: PASS ({resp.status_code}) | Latency: {elapsed:.2f}s")
            
            decision = data.get("hackathon_signal_decision", {})
            action = decision.get("action", "UNKNOWN")
            print(f"Decision Action: {action}")
            
            score = data.get("hackathon_signal_score", {}).get("signal_score", "N/A")
            print(f"Decision Score: {score}")
            
            if "tools_used" in data:
                print(f"Tools Used: {list(data['tools_used'].keys())}")
                
            return True
        else:
            print(f"Status: FAIL ({resp.status_code}) | Error: {resp.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("Backend is not running on localhost:8000. Start it via `python server.py`.")
        return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    test_decision_e2e()
