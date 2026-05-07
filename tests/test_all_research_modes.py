import requests
import json
import time

API_BASE = "http://localhost:8000/api/analyze"

TEST_CASES = [
    {"mode": "auto", "ticker": "AAPL", "query": "What is the latest investment signal and why?"},
    {"mode": "lookup_numeric", "ticker": "NVDA", "query": "What was the total revenue for NVDA in FY2024?"},
    {"mode": "lookup_text", "ticker": "META", "query": "Describe the business strategy and core segments of Meta Platforms."},
    {"mode": "lookup_text_filing", "ticker": "TSLA", "query": "What are the specific risk factors mentioned in Tesla's most recent 10-K?"},
    {"mode": "lookup_text_management", "ticker": "GOOGL", "query": "How does Google management describe their AI infrastructure investments?"},
    {"mode": "lookup_text_news", "ticker": "AAPL", "query": "What are the latest news headlines regarding Apple's product launches?"},
    {"mode": "compute_metric", "ticker": "TSLA", "query": "What is the current ratio for Tesla based on their latest balance sheet?"},
    {"mode": "comparative_analysis", "ticker": "NVDA", "query": "Compare Nvidia's data center revenue growth to its overall revenue growth."},
    {"mode": "risk_analysis", "ticker": "META", "query": "Perform a deep dive into the regulatory risks facing Meta in Europe."},
    {"mode": "valuation", "ticker": "AAPL", "query": "Show a detailed DCF valuation for Apple with explicit assumptions."},
    {"mode": "relative_valuation", "ticker": "TSLA", "query": "Evaluate Tesla's valuation using P/E and P/S multiples relative to its history."},
    {"mode": "explanatory_reasoning", "ticker": "GOOGL", "query": "Explain the drivers behind Google's Cloud segment profitability."},
    {"mode": "mba_framework", "ticker": "NVDA", "query": "Conduct a SWOT analysis for Nvidia's competitive position in the AI chip market."},
    {"mode": "multi_period_analysis", "ticker": "AAPL", "query": "Analyze Apple's gross margin trend over the last three fiscal years."},
    {"mode": "scenario_analysis", "ticker": "TSLA", "query": "What happens to Tesla's valuation in a scenario where FCF growth stays below 5%?"},
    {"mode": "peer_analysis", "ticker": "NVDA", "query": "Benchmark Nvidia against its key industry peers in terms of efficiency and margins."},
]

def run_tests():
    results = []
    for test in TEST_CASES:
        print(f"\n--- Testing Mode: {test['mode']} | Ticker: {test['ticker']} ---")
        payload = {
            "query": test["query"],
            "ticker": test["ticker"],
            "mode": test["mode"],
            "strictness": 30
        }
        
        start_time = time.time()
        try:
            resp = requests.post(API_BASE, json=payload, timeout=90)
            latency = time.time() - start_time
            
            if resp.status_code == 200:
                data = resp.json()
                ok = data.get("ok", False)
                answer = data.get("result", {}).get("final_answer", "")
                evidence_count = len(data.get("evidence_hydrated", []))
                
                status = "PASS" if ok and answer and evidence_count > 0 else "PARTIAL"
                if not answer: status = "FAIL (No Answer)"
                if evidence_count == 0: status = "FAIL (No Evidence)"
                
                print(f"Status: {status} | Latency: {latency:.2f}s | Evidence: {evidence_count}")
                results.append({
                    "mode": test["mode"],
                    "status": status,
                    "latency": f"{latency:.2f}s",
                    "evidence": evidence_count,
                    "answer_preview": answer[:100] + "..." if answer else "N/A"
                })
            else:
                print(f"Status: ERROR ({resp.status_code}) | Latency: {latency:.2f}s")
                results.append({
                    "mode": test["mode"],
                    "status": f"ERROR ({resp.status_code})",
                    "latency": f"{latency:.2f}s",
                    "evidence": 0,
                    "answer_preview": resp.text[:100]
                })
        except Exception as e:
            print(f"Status: EXCEPTION ({str(e)})")
            results.append({
                "mode": test["mode"],
                "status": f"EXCEPTION",
                "latency": "N/A",
                "evidence": 0,
                "answer_preview": str(e)
            })
            
    print("\n\n" + "="*50)
    print("FINAL TEST SUMMARY")
    print("="*50)
    print(f"{'Mode':<25} | {'Status':<15} | {'Evidence':<10} | {'Latency':<10}")
    print("-" * 65)
    for r in results:
        print(f"{r['mode']:<25} | {r['status']:<15} | {r['evidence']:<10} | {r['latency']:<10}")

if __name__ == "__main__":
    run_tests()
