import requests
import json

payload = {
    "query": "What was the total revenue for NVDA for FY2024?",
    "ticker": "NVDA",
    "mode": "lookup_numeric",
    "strictness": 30
}
resp = requests.post("http://localhost:8000/api/analyze", json=payload)
print(json.dumps(resp.json(), indent=2))
