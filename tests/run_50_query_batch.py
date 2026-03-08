from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from local_llm import (
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    build_local_llm_client,
)
from market_api import YahooFinanceMarketDataProvider
from orchestrator import FinancialOrchestrator, OrchestratorConfig


def build_queries() -> list[str]:
    return [
        "What was AAPL revenue in 2024?",
        "What was META net income in 2024?",
        "What was NVDA operating income in 2025?",
        "What was TSLA gross profit in 2024?",
        "What was GOOGL capex in 2025?",
        "What was AAPL EPS in 2024?",
        "What was META cash provided by operating activities in 2024?",
        "What was NVDA revenue in 2024?",
        "What was TSLA net income in 2025?",
        "What was GOOGL operating income in 2024?",
        "What are the key risk factors for AAPL in 2024?",
        "What are the main risk factors for META in 2025?",
        "What uncertainty does NVDA mention in Item 1A for 2024?",
        "Summarize TSLA risk exposures in FY2025.",
        "What competitive risks does GOOGL disclose in 2024?",
        "Explain why AAPL revenue changed in 2024.",
        "Explain the business model of META based on the filing.",
        "Why did NVDA margins improve in 2025?",
        "Explain TSLA management discussion for 2024.",
        "What is the rationale for GOOGL profitability changes in 2025?",
        "Compare AAPL and META revenue in 2024.",
        "Compare NVDA vs TSLA gross margin in 2025.",
        "Compare GOOGL and AAPL risk factors in 2024.",
        "AAPL vs META operating income 2025 comparison.",
        "Compare TSLA and GOOGL cash flow trends.",
        "What is AAPL free cash flow in 2024?",
        "Compute META operating margin in 2025.",
        "Compute NVDA gross margin in 2024.",
        "Calculate TSLA yoy revenue growth for 2025.",
        "What is GOOGL operating margin in 2024?",
        "Estimate AAPL valuation for FY2024.",
        "Estimate META intrinsic value for FY2025.",
        "What is NVDA DCF valuation in 2025?",
        "Provide relative valuation for TSLA in 2024.",
        "What is GOOGL valuation multiple in 2025?",
        "Estimate AAPL valuation for FY2024 with WACC 10% and terminal growth 3%.",
        "Estimate META valuation for FY2025 with WACC 9% and terminal growth 2.5%.",
        "Give NVDA DCF with discount rate 11% and terminal growth 3%.",
        "Run comps valuation for TSLA using EV/EBITDA in 2024.",
        "Relative valuation of GOOGL versus peers in 2025.",
        "",
        "revenue",
        "What was revenue?",
        "Compare valuation of apple vs meta",
        "change in leadership at AAPL",
        "risk of change in market for META",
        "SWOT analysis of NVDA",
        "Porter 5 forces for TSLA",
        "What is market cap of GOOGL?",
        "How much did AAPL earn?",
    ]


def main() -> int:
    base = BASE
    llm = build_local_llm_client(
        primary_model=os.environ.get("LOCAL_SMALL_MODEL", DEFAULT_PRIMARY_MODEL),
        fallback_model=os.environ.get("LOCAL_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
        base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    )
    orch = FinancialOrchestrator(
        cfg=OrchestratorConfig(
            base_dir=base,
            audit_log_path=base / "logs" / "audit.jsonl",
            small_model_name=os.environ.get("LOCAL_SMALL_MODEL", DEFAULT_PRIMARY_MODEL),
            large_model_name=os.environ.get("LOCAL_LARGE_MODEL", DEFAULT_PRIMARY_MODEL),
            known_tickers={"AAPL", "META", "NVDA", "TSLA", "GOOGL"},
            market_provider=YahooFinanceMarketDataProvider(),
        ),
        llm_client=llm,
    )

    queries = build_queries()
    details = []
    action_counter = Counter()
    mode_counter = Counter()
    t0 = time.time()

    for i, q in enumerate(queries, start=1):
        q0 = time.time()
        market_inputs = None
        if "WACC" in q or "discount rate" in q or "terminal growth" in q:
            # Basic defaults for valuation tests that include assumptions text.
            market_inputs = {"wacc": 0.10, "terminal_growth": 0.03}
        try:
            out = orch.answer(q, market_inputs=market_inputs)
            status = "ok" if out.get("ok") else "not_ok"
            action = out.get("action", "unknown")
            mode = out.get("mode", "unknown")
            action_counter[action] += 1
            mode_counter[mode] += 1
            details.append(
                {
                    "idx": i,
                    "question": q,
                    "status": status,
                    "action": action,
                    "mode": mode,
                    "reason": out.get("reason"),
                    "validation_errors": out.get("validation_errors"),
                    "elapsed_ms": int((time.time() - q0) * 1000),
                }
            )
        except Exception as e:
            details.append(
                {
                    "idx": i,
                    "question": q,
                    "status": "exception",
                    "action": "exception",
                    "mode": "unknown",
                    "reason": str(e),
                    "elapsed_ms": int((time.time() - q0) * 1000),
                }
            )
            action_counter["exception"] += 1

    summary = {
        "total_queries": len(queries),
        "total_elapsed_s": round(time.time() - t0, 2),
        "actions": dict(action_counter),
        "modes": dict(mode_counter),
        "exceptions": sum(1 for d in details if d["status"] == "exception"),
    }

    out_path = base / "tests" / "batch_50_results.json"
    out_path.write_text(json.dumps({"summary": summary, "details": details}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved detailed report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

