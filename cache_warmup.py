import requests
import json
import time

TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL"]
FISCAL_YEAR = 2023
STRICTNESS = 70
URL = "http://localhost:8000/api/decision"

def warmup_cache():
    print(f"Warming up cache for {len(TICKERS)} tickers...")
    
    for ticker in TICKERS:
        print(f"\\n[{ticker}] Starting processing...")
        start_time = time.time()
        
        payload = {
            "ticker": ticker,
            "fiscal_year": FISCAL_YEAR,
            "strictness": STRICTNESS
        }
        
        try:
            response = requests.post(URL, json=payload, timeout=120)
            
            if response.status_code == 200:
                elapsed = time.time() - start_time
                data = response.json()
                
                # Check if it was cached or freshly computed based on time
                # Normally the server would tell us, but we can guess by speed
                print(f"[{ticker}] SUCCESS! Completed in {elapsed:.2f} seconds.")
                print(f"[{ticker}] Overall Risk Score: {data.get('risk', {}).get('score', 'N/A')}")
                print(f"[{ticker}] Valuation factors: {list(data.get('valuation', {}).get('factors', {}).keys())}")
            else:
                print(f"[{ticker}] FAILED with status code {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"[{ticker}] FAILED with exception: {str(e)}")
            
    print("\\nCache warmup completed!")

if __name__ == "__main__":
    warmup_cache()
