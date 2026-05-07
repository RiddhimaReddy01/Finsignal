import subprocess
import json
import time

QUERIES = [
    {
        "mode": "valuation",
        "q": "Run a discounted cash flow valuation for Apple (AAPL) for FY 2023.",
        "tickers": "AAPL"
    },
    {
        "mode": "relative_valuation",
        "q": "What is Apple's (AAPL) current EV/EBITDA multiple?",
        "tickers": "AAPL"
    },
    {
        "mode": "compute_metric",
        "q": "What was Google's (GOOGL) operating margin in FY 2023?",
        "tickers": "GOOGL"
    },
    {
        "mode": "lookup_numeric",
        "q": "What was Apple's total net sales in 2023?",
        "tickers": "AAPL"
    },
    {
        "mode": "risk_analysis",
        "q": "What are the primary supply chain risks for Apple (AAPL)?",
        "tickers": "AAPL"
    },
    {
        "mode": "lookup_text_management",
        "q": "What is Tesla's forward guidance according to their recent call?",
        "tickers": "TSLA"
    },
    {
        "mode": "lookup_text_news",
        "q": "What is the latest news today regarding Google's antitrust lawsuits?",
        "tickers": "GOOGL"
    },
    {
        "mode": "explanatory_reasoning",
        "q": "Why did Apple's hardware revenue drop in 2023?",
        "tickers": "AAPL"
    },
    {
        "mode": "comparative_analysis",
        "q": "Compare the top risk factors of Microsoft vs Google regarding artificial intelligence.",
        "tickers": "MSFT,GOOGL"
    },
    {
        "mode": "mba_framework",
        "q": "Generate a SWOT analysis for Apple (AAPL).",
        "tickers": "AAPL"
    }
]

def run_tests():
    results = []
    for test in QUERIES:
        print(f"Testing mode: {test['mode']}...")
        start_time = time.time()
        cmd = ["python", "main.py", "-q", test["q"], "--known-tickers", test["tickers"]]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            success = result.returncode == 0
            # Some queries might legitimately abstain or get confused, but a returncode=0 means it ran.
            # We'll consider it a pass if it ran without hard crashing.
            
            output_lower = result.stdout.lower() + result.stderr.lower()
            if "traceback" in output_lower or "exception" in output_lower:
                success = False

            results.append({
                "mode": test["mode"],
                "success": success,
                "time_taken": time.time() - start_time,
                "rc": result.returncode,
                "stderr": result.stderr if not success else ""
            })
            print(f"  -> {'PASS' if success else 'FAIL'} ({time.time() - start_time:.2f}s)")
        except subprocess.TimeoutExpired:
            print(f"  -> TIMEOUT")
            results.append({
                "mode": test["mode"],
                "success": False,
                "time_taken": 120,
                "error": "Timeout"
            })
    
    print("\n--- Summary ---")
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"[{status}] {r['mode']} - {r['time_taken']:.2f}s")
        if not r["success"]:
            print(f"   Error: {r.get('stderr', r.get('error', 'Unknown'))}")
            
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_tests()
