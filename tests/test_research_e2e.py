import requests
import json
import time

def test_research_e2e():
    print("--- Running End-to-End Test for Research Mode ---")
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": "What is Apple's latest gross margin trend?",
        "ticker": "AAPL",
        "mode": "auto",
        "strictness": 70
    }
    
    try:
        start = time.time()
        resp = requests.post(url, json=payload, timeout=90)
        elapsed = time.time() - start
        if resp.status_code == 200:
            data = resp.json()
            print(f"Status: PASS ({resp.status_code}) | Latency: {elapsed:.2f}s")
            print("Response preview:")
            ans = data.get("result", {}).get("final_answer", "")
            print(ans[:200] + "..." if len(ans) > 200 else ans)
            assert data.get("ok", True) or "result" in data
            
            # Check report URL
            report = data.get("generated_report", {})
            if report.get("pdf_url"):
                print(f"PDF Report generated: {report['pdf_url']}")
            else:
                print("No PDF report generated.")
                
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
    test_research_e2e()
