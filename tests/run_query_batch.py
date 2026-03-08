"""
tests/run_query_batch.py

Runs a fixed set of benchmark queries through the FinancialOrchestrator and
prints a structured log for each query showing the fields requested:

  plan.mode
  plan.retrieval_plan.source_route
  verification.status
  verification.confidence
  verification.reason_codes
  verification.source_coverage
  result.claims
  validation_errors
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from local_llm import build_local_llm_client, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL
from market_api import YahooFinanceMarketDataProvider
from orchestrator import FinancialOrchestrator, OrchestratorConfig
from knowledge_base import TICKERS as KB_TICKERS

logging.basicConfig(
    level=logging.WARNING,          # suppress retrieval noise
    format="%(levelname)s %(name)s %(message)s",
)

# ─────────────────────────────────────────────
# Test queries
# ─────────────────────────────────────────────
QUERIES: List[str] = [
    "What was AAPL revenue in 2024?",
    "What was NVDA net income in 2024?",
    "What is Apple free cash flow in 2024?",
    "What is Apple gross margin in 2024?",
    "Compare AAPL and MSFT revenue growth in 2024.",
    "What risks does Nvidia mention in FY2024?",
    "What did management say about AI demand?",
    "What is the latest news on Tesla in China?",
    "What is Apple's EV/EBITDA?",
    "Do a SWOT for Apple.",
]

SEP = "-" * 72


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _safe(obj: Any) -> Any:
    """Make an object JSON-serialisable for display."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _safe(asdict(obj))
    if hasattr(obj, "__dict__"):
        return _safe(obj.__dict__)
    return str(obj)


def _pp(label: str, value: Any) -> None:
    """Pretty-print a labelled field."""
    serialised = json.dumps(_safe(value), indent=4, ensure_ascii=False)
    print(f"  {label}:")
    for line in serialised.splitlines():
        print(f"    {line}")


def _log_result(n: int, query: str, result: Dict[str, Any]) -> None:
    verification = result.get("verification") or {}
    result_obj    = result.get("result") or {}

    # plan fields come back through verification dict which mirrors plan
    # The orchestrator doesn't return plan directly, but we capture mode.
    mode             = result.get("mode", "unknown")
    # source_route is logged via retrieval debug
    source_route_raw = (
        result.get("evidence", {}).get("narrative", {})  # best effort
        or {}
    )
    # The orchestrator stores source_route inside verification signals
    signals          = verification.get("signals") or {}
    source_route     = signals.get("source_route") or result.get("_source_route") or "see_evidence"

    status           = verification.get("status", "unknown")
    confidence       = verification.get("confidence", None)
    reason_codes     = verification.get("reason_codes", [])
    source_coverage  = verification.get("source_coverage", {})
    claims           = result_obj.get("claims", []) if isinstance(result_obj, dict) else []
    validation_errs  = result.get("validation_errors", [])

    print(SEP)
    print(f"  Q{n:02d}: {query}")
    print(SEP)
    _pp("plan.mode",                              mode)
    _pp("plan.retrieval_plan.source_route",       source_route)
    _pp("verification.status",                    status)
    _pp("verification.confidence",                confidence)
    _pp("verification.reason_codes",              reason_codes)
    _pp("verification.source_coverage",           source_coverage)
    _pp("result.claims",                          claims)
    _pp("validation_errors",                      validation_errs)
    print()


# ─────────────────────────────────────────────
# Build orchestrator
# ─────────────────────────────────────────────

def build_orchestrator() -> FinancialOrchestrator:
    idx = BASE / "index"
    audit_log = BASE / "logs" / "run_query_batch_audit.jsonl"
    audit_log.parent.mkdir(parents=True, exist_ok=True)

    llm_client = build_local_llm_client(
        primary_model=DEFAULT_PRIMARY_MODEL,
        fallback_model=DEFAULT_FALLBACK_MODEL,
        base_url="http://localhost:11434",
    )

    cfg = OrchestratorConfig(
        base_dir=BASE,
        audit_log_path=audit_log,
        small_model_name="small",
        large_model_name="large",
        known_tickers={t.upper() for t in KB_TICKERS if isinstance(t, str)},
        market_provider=YahooFinanceMarketDataProvider(),
    )

    return FinancialOrchestrator(cfg=cfg, llm_client=llm_client)


# ─────────────────────────────────────────────
# We also need to capture source_route from the plan.
# Patch orchestrator.answer to expose it.
# ─────────────────────────────────────────────

def _run_with_source_route(
    orchestrator: FinancialOrchestrator,
    query: str,
) -> Dict[str, Any]:
    """
    Calls orchestrator.answer and back-patches `_source_route` into the
    result dict so the logger can display plan.retrieval_plan.source_route.
    We do this by temporarily instrumenting build_task_plan.
    """
    import verification as _ver
    _orig_build = _ver.build_task_plan
    captured: Dict[str, Any] = {}

    def _patched_build(*args, **kwargs):
        plan = _orig_build(*args, **kwargs)
        captured["source_route"] = _safe(plan.retrieval_plan.source_route)
        captured["mode"]         = plan.mode
        return plan

    _ver.build_task_plan = _patched_build
    try:
        result = orchestrator.answer(query, auto_fetch_market=False)
    finally:
        _ver.build_task_plan = _orig_build

    result["_source_route"] = captured.get("source_route", "unknown")
    return result


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    print("\n" + SEP)
    print("  Financial Analysis Tool — Query Batch Test Runner")
    print(SEP + "\n")

    orchestrator = build_orchestrator()

    for n, query in enumerate(QUERIES, start=1):
        try:
            result = _run_with_source_route(orchestrator, query)
        except Exception as exc:
            print(SEP)
            print(f"  Q{n:02d}: {query}")
            print(SEP)
            print(f"  ERROR: {exc}\n")
            continue

        _log_result(n, query, result)

    print(SEP)
    print("  Done.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
