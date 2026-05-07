# main.py
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_dotenv()

from local_llm import build_local_llm_client, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL
from market_api import YahooFinanceMarketDataProvider
from news_client_adapter import build_optional_news_client
from orchestrator import FinancialOrchestrator, OrchestratorConfig

try:
    from knowledge_base import TICKERS as _KB_TICKERS
except Exception:
    _KB_TICKERS = []

_DEFAULT_KNOWN_TICKERS: set[str] = {t.upper() for t in _KB_TICKERS if isinstance(t, str) and t.strip()}

logger = logging.getLogger(__name__)


# ============================================================
# CLI helpers
# ============================================================

def _parse_known_tickers(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    vals = [x.strip().upper() for x in raw.split(",") if x.strip()]
    return set(vals) if vals else None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the financial analysis orchestrator end-to-end.")
    p.add_argument("-q", "--question", type=str, help="Question to answer. If omitted, interactive prompt is used.")
    p.add_argument("--base-dir", type=str, default=str(Path(__file__).resolve().parent), help="Project base directory.")
    p.add_argument("--audit-log", type=str, default=None, help="Audit JSONL path (default: <base-dir>/logs/audit.jsonl).")
    p.add_argument("--small-model", type=str, default=os.environ.get("LOCAL_SMALL_MODEL", "small"))
    p.add_argument("--large-model", type=str, default=os.environ.get("LOCAL_LARGE_MODEL", "large"))
    p.add_argument("--fallback-model", type=str, default=os.environ.get("LOCAL_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL))
    p.add_argument("--gemini-url", type=str, default=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"))
    p.add_argument("--ollama-url", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--market-inputs", type=str, default=None, help='Optional JSON string: \'{"wacc":0.1,"terminal_growth":0.03}\'')
    p.add_argument("--known-tickers", type=str, default=os.environ.get("KNOWN_TICKERS"), help="Comma-separated known tickers.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--disable-market-autofetch", action="store_true", help="Disable auto market fetch for valuation modes.")
    p.add_argument("--decision-time", type=str, default=None, help="ISO timestamp cutoff (UTC recommended) for replay-safe runs.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    question = (args.question or input("Question: ")).strip()
    if not question:
        print("Question is required.")
        return 1

    base_dir = Path(args.base_dir).resolve()
    audit_log = Path(args.audit_log).resolve() if args.audit_log else (base_dir / "logs" / "audit.jsonl")
    audit_log.parent.mkdir(parents=True, exist_ok=True)

    market_inputs: Optional[Dict[str, Any]] = None
    if args.market_inputs:
        try:
            parsed = json.loads(args.market_inputs)
            if not isinstance(parsed, dict):
                raise ValueError("market_inputs must be a JSON object")
            market_inputs = parsed
        except Exception as e:
            print(f"Invalid --market-inputs JSON: {e}")
            return 1

    llm_client = build_local_llm_client(
        primary_model=os.environ.get("GEMINI_SMALL_MODEL", DEFAULT_PRIMARY_MODEL),
        fallback_model=os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
        base_url=args.gemini_url,
    )

    cfg = OrchestratorConfig(
        base_dir=base_dir,
        audit_log_path=audit_log,
        small_model_name=args.small_model,
        large_model_name=args.large_model,
        known_tickers=_parse_known_tickers(args.known_tickers) or _DEFAULT_KNOWN_TICKERS or None,
        market_provider=YahooFinanceMarketDataProvider(),
        news_client=build_optional_news_client(),
    )

    orchestrator = FinancialOrchestrator(cfg=cfg, llm_client=llm_client)

    result = orchestrator.answer(
        question,
        market_inputs=market_inputs,
        auto_fetch_market=not args.disable_market_autofetch,
        decision_time=args.decision_time,
    )

    _print_result(result)
    return 0


def _print_result(result: Dict[str, Any]) -> None:
    action = result.get("action", "abstain")
    mode = result.get("mode", "unknown")
    run_id = result.get("run_id", "?")
    timing = result.get("timing_ms", {}) or {}

    print(f"\n{'=' * 60}")
    print(f"  Run: {run_id}  |  Mode: {mode}  |  Action: {action}")
    if timing:
        print(f"  Timing — total: {timing.get('total_ms', '?')}ms, "
              f"retrieval: {timing.get('retrieval_ms', '?')}ms, "
              f"generation: {timing.get('generation_ms', '?')}ms")
    print(f"{'=' * 60}\n")

    result_obj = result.get("result") or {}
    final_answer = result_obj.get("final_answer") if isinstance(result_obj, dict) else None

    if action != "answer" or not final_answer:
        reason = result.get("reason", "No answer generated")
        print(f"  [{action.upper()}] {reason}\n")
        print(json.dumps(result, indent=2))
        return

    print(final_answer)
    print()

    errors = result.get("validation_errors", [])
    if errors:
        print(f"  Validation warnings: {', '.join(errors)}")

    assumptions = result.get("assumptions", [])
    if assumptions:
        print(f"  Assumptions: {'; '.join(str(a) for a in assumptions)}")

    print()


if __name__ == "__main__":
    raise SystemExit(main())
