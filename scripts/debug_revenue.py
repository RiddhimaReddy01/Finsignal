import requests
import json

def test_nvda_revenue():
    url = "http://localhost:8000/api/analyze"
    payload = {
        "query": "What was the total revenue for NVDA in FY2024?",
        "ticker": "NVDA",
        "fiscal_year": 2024,
        "mode": "lookup_numeric",
        "strictness": 30 # Use lower strictness to ensure we get an answer
    }
    
    print(f"Testing NVDA Revenue FY2024...")
    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        print(f"Status: {result.get('ok')}")
        print(f"Final Answer: {result.get('final_answer')}")
        print(f"Reason: {result.get('reason')}")
        
        if 'debug' in result:
            debug = result['debug']
            if 'candidates' in debug:
                print("\nCandidates:")
                for c in debug['candidates']:
                    print(f"- {c['evidence_id']}: {c['value_scaled']} (Score: {c['score']}) Line: {c['line']}")
            if 'candidate_errors' in debug:
                print("\nCandidate Errors:")
                for e in debug['candidate_errors']:
                    print(f"- {e['evidence_id']}: {e['errors']}")
                    
        # Also check evidence_hydrated for details
        if 'evidence_hydrated' in result:
            print("\nEvidence Hydrated (First 5):")
            for i, ev in enumerate(result['evidence_hydrated'][:5]):
                print(f"{i+1}. {ev['id']} - {ev['text'][:100]}...")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_nvda_revenue()
