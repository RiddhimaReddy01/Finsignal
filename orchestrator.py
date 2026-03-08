# orchestra.py
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from assumptions_policy import build_assumptions
from api_cache import DiskTTLCache
from audit import AuditLogger
from market_api import (
    MarketDataProvider,
    TTLCache,
    fetch_min_market_inputs,
    merge_market_inputs,
)
from retrieval_tool import FinancialRetrievalTool, RetrievalConfig
from routing import decide as routing_decide
from relative_valuation_engine import compute_multiple
from mba_frameworks import build_framework_prompt, detect_framework_type, validate_framework_output
from multi_period_analysis import build_multi_period_prompt
from peer_analysis import peer_analysis_to_signal, run_peer_analysis
from scenario_analysis import ScenarioSpec, is_whatif_query, parse_whatif, run_scenario_analysis
from transcript_api import FreeTranscriptAPI
from verification import (
    Target,
    TaskPlan,
    VerificationResult,
    build_compute_metric_answer,
    build_json_answer_prompt,
    build_lookup_numeric_answer,
    build_task_plan,
    choose_best_numeric_with_gate,
    evidence_requirements,
    gate_evidence,
    split_context_into_blocks,
    validate_answer_json,
    with_strictness,
)
from valuation_engine import run_dcf
from transcript_ingestion import TranscriptIngestionClient
from hackathon_pipeline import run_hackathon_signal_layer

logger = logging.getLogger(__name__)


# ============================================================
# Interfaces / config
# ============================================================

class LLMClient(Protocol):
    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        ...


class NewsClient(Protocol):
    def fetch_company_news(
        self,
        *,
        ticker: str,
        limit: int = 5,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        ...


@dataclass(frozen=True)
class OrchestratorConfig:
    base_dir: Path
    audit_log_path: Path
    small_model_name: str = "small"
    large_model_name: str = "large"
    known_tickers: Optional[set[str]] = None
    market_provider: Optional[MarketDataProvider] = None
    market_cache_ttl_s: int = 300
    news_client: Optional[NewsClient] = None
    api_cache_dir: str = "data/cache/api"
    news_cache_ttl_s: int = 1800
    transcript_cache_ttl_s: int = 3600


# ============================================================
# Helpers
# ============================================================

def _obj_to_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(_obj_to_dict(obj), ensure_ascii=False)


def _append_inferences(result_obj: Dict[str, Any], new_items: List[str]) -> Dict[str, Any]:
    result_obj.setdefault("inferences", [])
    if isinstance(result_obj["inferences"], list):
        for x in new_items:
            if x not in result_obj["inferences"]:
                result_obj["inferences"].append(x)
    return result_obj


def _build_llm_fallback_result(
    *,
    plan: TaskPlan,
    full_context: str,
    assumptions: List[str],
    verification: VerificationResult,
    error_text: str,
) -> Dict[str, Any]:
    blocks = split_context_into_blocks(full_context)[:4]
    claims: List[Dict[str, Any]] = []
    for b in blocks:
        snippet = (b.text or "").strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220] + "..."
        claims.append({
            "claim_type": "summary",
            "entity": b.ticker,
            "metric_or_topic": plan.targets[0].metric if plan.targets else None,
            "period": f"FY{b.fiscal_year}" if b.fiscal_year is not None else None,
            "unit": None,
            "value_or_summary": snippet,
            "citations": [b.evid],
            "formula": None,
            "inputs": [],
        })

    result_obj: Dict[str, Any] = {
        "final_answer": "LLM unavailable; returned deterministic evidence summary fallback.",
        "claims": claims,
        "tables_used": [b.evid for b in blocks if b.kind == "table"],
        "provenance": {
            "ticker": plan.targets[0].ticker if plan.targets else None,
            "fiscal_year": plan.targets[0].fiscal_year if plan.targets else None,
        },
        "inferences": assumptions + verification.warnings + [f"llm_fallback:{error_text}"],
        "confidence": float(max(0.0, min(verification.confidence, 0.7))),
    }

    if plan.mode == "risk_analysis":
        result_obj["risks"] = [
            {"risk": c["value_or_summary"], "mechanism": "Evidence snippet fallback", "citations": c["citations"]}
            for c in claims
        ]
    elif plan.mode == "mba_framework":
        result_obj["framework"] = {
            "type": "SWOT",
            "bullets": [{"bucket": "Summary", "text": c["value_or_summary"], "citations": c["citations"]} for c in claims],
        }
    elif plan.mode == "comparative_analysis":
        result_obj["comparison"] = {
            "targets": [{"ticker": t.ticker, "fiscal_year": t.fiscal_year} for t in plan.targets],
            "facts": [{"target": 0, "text": c["value_or_summary"], "citations": c["citations"]} for c in claims],
            "summary": result_obj["final_answer"],
        }
    elif plan.mode == "multi_period_analysis":
        result_obj["multi_period_analysis"] = {
            "periods": [{"fiscal_year": b.fiscal_year, "summary": (b.text or "")[:180], "citations": [b.evid]} for b in blocks if b.fiscal_year is not None],
            "trend_summary": result_obj["final_answer"],
        }

    return result_obj


# ============================================================
# Orchestrator
# ============================================================

class FinancialOrchestrator:
    """
    End-to-end flow:
      Question
        -> Planner
        -> Source Routing
            -> SEC filings retrieval
            -> market news API
            -> transcript API
            -> market data API
        -> Verification Layer
        -> Deterministic answer or LLM JSON answer
        -> Final response + citations + confidence
    """

    def __init__(
        self,
        *,
        cfg: OrchestratorConfig,
        retrieval: Optional[FinancialRetrievalTool] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self.cfg = cfg
        self.llm_client = llm_client
        self.audit = AuditLogger(str(cfg.audit_log_path))
        self._mkt_cache = TTLCache(ttl_s=int(cfg.market_cache_ttl_s))
        self._news_cache = DiskTTLCache(
            cache_dir=f"{cfg.api_cache_dir}/news_context",
            ttl_s=int(cfg.news_cache_ttl_s),
        )
        self._transcript_cache = DiskTTLCache(
            cache_dir=f"{cfg.api_cache_dir}/transcript_context",
            ttl_s=int(cfg.transcript_cache_ttl_s),
        )
        self._transcript_api: Optional[FreeTranscriptAPI] = None
        if (os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("AV_API_KEY") or os.environ.get("FMP_API_KEY")):
            self._transcript_api = FreeTranscriptAPI()

        if retrieval is not None:
            self.retrieval = retrieval
        else:
            idx = cfg.base_dir / "index"
            self.retrieval = FinancialRetrievalTool(
                narrative_chunks_path=idx / "chunks.parquet",
                narrative_bm25_path=idx / "bm25.pkl",
                narrative_faiss_path=idx / "faiss.index",
                table_docs_path=idx / "tables.parquet",
                table_bm25_path=idx / "table_bm25.pkl",
                table_faiss_path=idx / "table_faiss.index",
                companyfacts_dir=cfg.base_dir / "data" / "xbrl_companyfacts",
                config=RetrievalConfig(use_rerank=True, rerank_only_if_broad=True),
            )

    # ----------------------------
    # Filters
    # ----------------------------

    def _filters_from_target(self, target: Any) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if target is None:
            return filters
        if getattr(target, "ticker", None):
            filters["ticker"] = target.ticker
        if getattr(target, "fiscal_year", None) is not None:
            filters["fiscal_year"] = target.fiscal_year
        if getattr(target, "item_hint", None):
            filters["item"] = target.item_hint
        return filters

    def _filters_from_plan(self, plan: TaskPlan) -> Dict[str, Any]:
        t0 = plan.targets[0] if plan.targets else None
        return self._filters_from_target(t0)

    # ----------------------------
    # Routing debug
    # ----------------------------

    def _routing_debug_from_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        narrative = evidence.get("narrative") or {}
        tables = evidence.get("tables") or {}
        reranked = narrative.get("reranked") or []
        bm25 = narrative.get("bm25") or []
        dense = narrative.get("dense") or []

        selected_ids = list(narrative.get("selected_chunk_ids") or []) + list(tables.get("selected_table_ids") or [])
        bm25_ids = [cid for cid, _ in bm25 if isinstance(cid, str)]
        dense_ids = [cid for cid, _ in dense if isinstance(cid, str)]

        return {
            "selected_ids": selected_ids,
            "bm25_ids": bm25_ids,
            "dense_ids": dense_ids,
            "reranked": [{"id": cid, "score": score} for cid, score in reranked] if reranked else [],
        }

    # ----------------------------
    # Default FY inference
    # ----------------------------

    def _latest_fiscal_year_for_ticker(self, ticker: str) -> Optional[int]:
        chunks_df = getattr(self.retrieval, "chunks_df", None)
        if chunks_df is None:
            return None
        try:
            subset = chunks_df[chunks_df["ticker"] == ticker]
            if subset.empty:
                return None
            years = subset["fiscal_year"].dropna().astype(int)
            if years.empty:
                return None
            return int(years.max())
        except Exception:
            return None

    def _apply_default_fiscal_year(self, plan: TaskPlan) -> Tuple[TaskPlan, List[str]]:
        if not plan.targets:
            return plan, []

        assumptions: List[str] = []
        new_targets: List[Target] = []

        for t in plan.targets:
            if t.ticker and t.fiscal_year is None:
                latest = self._latest_fiscal_year_for_ticker(t.ticker)
                if latest is not None:
                    assumptions.append(f"Fiscal year not provided; assumed latest available FY{latest} for {t.ticker}.")
                    new_targets.append(replace(t, fiscal_year=latest))
                else:
                    new_targets.append(t)
            else:
                new_targets.append(t)

        if assumptions:
            plan = replace(plan, targets=new_targets)
            hard_filters = dict(plan.retrieval_plan.hard_filters or {})
            hard_filters["ticker"] = plan.targets[0].ticker
            hard_filters["fiscal_year"] = plan.targets[0].fiscal_year
            plan = replace(plan, retrieval_plan=replace(plan.retrieval_plan, hard_filters=hard_filters))

        return plan, assumptions

    # ----------------------------
    # Source routing
    # ----------------------------

    def _maybe_fetch_market_inputs(
        self,
        plan: TaskPlan,
        market_inputs: Optional[Dict[str, Any]],
        *,
        auto_fetch_market: bool,
    ) -> Dict[str, Any]:
        user_inputs = market_inputs if isinstance(market_inputs, dict) else {}

        if not auto_fetch_market:
            return user_inputs
        if self.cfg.market_provider is None:
            return user_inputs
        if not plan.targets or not plan.targets[0].ticker:
            return user_inputs
        if not plan.retrieval_plan.source_route.market_data:
            return user_inputs

        fetched = fetch_min_market_inputs(
            self.cfg.market_provider,
            ticker=plan.targets[0].ticker,
            cache=self._mkt_cache,
        )
        return merge_market_inputs(user_inputs, fetched)

    def _fetch_news_context(self, plan: TaskPlan, *, decision_time: Optional[str] = None) -> str:
        if not self.cfg.news_client or not plan.targets or not plan.targets[0].ticker:
            return ""

        ticker = plan.targets[0].ticker
        fy = plan.targets[0].fiscal_year
        cache_key = DiskTTLCache.make_key("news_context", ticker, fy or "NA", 5, decision_time or "")
        cached = self._news_cache.get(cache_key)
        if isinstance(cached, str):
            return cached

        try:
            rows = self.cfg.news_client.fetch_company_news(
                ticker=ticker,
                limit=5,
                as_of=decision_time,
            )
        except Exception:
            logger.exception("News fetch failed for %s", ticker)
            return ""

        chunks: List[str] = []
        for idx, row in enumerate(rows):
            evid = row.get("id") or f"news_{ticker}_{idx+1}"
            title = row.get("title", "")
            summary = row.get("summary", "") or row.get("text", "")
            ts = row.get("published_at", "")
            header = f"[NEWS {ticker} "
            if fy is not None:
                header += f"FY{fy} "
            header += f"{evid}]"
            body = f"Title: {title}\nPublished: {ts}\nSummary: {summary}".strip()
            chunks.append(f"{header}\n{body}")
        context = "\n\n".join(chunks)
        self._news_cache.set(cache_key, context)
        return context

    def _fetch_transcript_context(self, plan: TaskPlan) -> str:
        if not plan.targets or not plan.targets[0].ticker:
            return ""

        ticker = plan.targets[0].ticker
        fy = plan.targets[0].fiscal_year
        cache_key = DiskTTLCache.make_key("transcript_context", ticker, fy or "NA")
        cached = self._transcript_cache.get(cache_key)
        if isinstance(cached, str):
            return cached

        local_current_text: Optional[str] = None
        local_prior_text: Optional[str] = None
        try:
            tc = TranscriptIngestionClient()
            if fy is not None:
                current_period = f"FY{fy}"
                prior_period = f"FY{fy - 1}"
                local_current_text, local_prior_text = tc.get_current_and_prior_text(
                    ticker=ticker,
                    current_period=current_period,
                    prior_period=prior_period,
                )
        except Exception:
            logger.exception("Local transcript fetch failed for %s", ticker)

        current_text = local_current_text
        prior_text = local_prior_text

        # API fallback (cache-backed providers) when local transcripts are unavailable.
        if (not current_text and not prior_text) and self._transcript_api is not None:
            try:
                current_text, prior_text = self._transcript_api.get_latest_transcripts(ticker)
            except Exception:
                logger.exception("Transcript API fetch failed for %s", ticker)

        chunks: List[str] = []
        if current_text:
            fy_tag = f"FY{fy}" if fy is not None else "FYNA"
            chunks.append(f"[TRANSCRIPT {ticker} {fy_tag} transcript_{ticker}_current]\n{current_text}")
        if prior_text:
            prior_fy = (fy - 1) if fy is not None else None
            fy_tag = f"FY{prior_fy}" if prior_fy is not None else "FYNA"
            chunks.append(f"[TRANSCRIPT {ticker} {fy_tag} transcript_{ticker}_prior]\n{prior_text}")
        context = "\n\n".join(chunks)
        self._transcript_cache.set(cache_key, context)
        return context

    def _relative_multiple_type(self, question: str) -> str:
        q = (question or "").lower()
        if "ev/ebitda" in q or "ev ebitda" in q:
            return "EV_EBITDA"
        if "ev/sales" in q or "ev sales" in q:
            return "EV_SALES"
        if "p/s" in q or "price to sales" in q:
            return "P_S"
        return "P_E"

    def _compose_context(
        self,
        filing_context: str,
        *,
        news_context: str = "",
        transcript_context: str = "",
    ) -> str:
        parts = [x for x in [filing_context, news_context, transcript_context] if x and x.strip()]
        return "\n\n".join(parts).strip()

    # ----------------------------
    # Confidence-aware action mapping
    # ----------------------------

    def _final_action_from_verification(self, verification: VerificationResult) -> str:
        # Legacy helper retained for compatibility with older call-sites.
        return verification.status

    # ============================================================
    # Main entry
    # ============================================================

    def answer(
        self,
        question: str,
        *,
        market_inputs: Optional[Dict[str, Any]] = None,
        auto_fetch_market: bool = True,
        forced_mode: Optional[str] = None,
        ui_intent: Optional[str] = None,
        ui_ticker: Optional[str] = None,
        ui_fiscal_year: Optional[int] = None,
        evidence_strictness: Optional[int] = None,
        decision_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        run_id = self.audit.new_run_id()
        start_t = time.time()

        # 1) Plan
        plan = build_task_plan(
            question.strip(),
            known_tickers=self.cfg.known_tickers,
            forced_mode=forced_mode,
            ui_intent=ui_intent,
            forced_ticker=ui_ticker,
            forced_fiscal_year=ui_fiscal_year,
        )
        plan, assumptions = self._apply_default_fiscal_year(plan)

        req = evidence_requirements(plan)
        if evidence_strictness is not None:
            req = with_strictness(req, int(evidence_strictness))

        # 2) Source routing
        merged_market_inputs = self._maybe_fetch_market_inputs(
            plan,
            market_inputs,
            auto_fetch_market=auto_fetch_market,
        )

        query = plan.retrieval_plan.rewrites[0] if plan.retrieval_plan.rewrites else plan.normalized_question

        # 3) SEC filing retrieval
        r0 = time.time()
        filing_context = ""
        retrieval_debug: Dict[str, Any] = {}
        evidence: Dict[str, Any] = {}
        routing_debug: Dict[str, Any] = {}
        target_runs: List[Dict[str, Any]] = []

        if plan.mode == "comparative_analysis" and len(plan.targets) > 1:
            packed_parts: List[str] = []
            for idx, target in enumerate(plan.targets):
                t_query = query
                if target.ticker:
                    t_query = f"{query} {target.ticker}"
                filters = self._filters_from_target(target)
                t_ctx, t_debug, t_evidence = self.retrieval.retrieve(t_query, filters=filters)
                packed_parts.append(f"[TARGET {idx+1}]\n{t_ctx}")
                target_runs.append({
                    "target_index": idx,
                    "target": target,
                    "filters": filters,
                    "packed_context": t_ctx,
                    "retrieve_debug": t_debug,
                    "evidence": t_evidence,
                })
            filing_context = "\n\n".join(packed_parts).strip()
            retrieval_debug = {"comparative": True}
            evidence = {"targets": [tr["evidence"] for tr in target_runs]}
            routing_debug = {"comparative": True}
        else:
            filters = self._filters_from_plan(plan)
            filing_context, retrieval_debug, evidence = self.retrieval.retrieve(query, filters=filters)
            routing_debug = self._routing_debug_from_evidence(evidence)

        retrieval_ms = int((time.time() - r0) * 1000)

        # 4) External sources
        news_context = (
            self._fetch_news_context(plan, decision_time=decision_time)
            if plan.retrieval_plan.source_route.news else ""
        )
        transcript_context = self._fetch_transcript_context(plan) if plan.retrieval_plan.source_route.transcript else ""

        full_context = self._compose_context(
            filing_context,
            news_context=news_context,
            transcript_context=transcript_context,
        )

        # 5) Verification layer
        g0 = time.time()
        verification = gate_evidence(
            plan,
            req,
            full_context,
            market_inputs=merged_market_inputs,
        )
        gate_ms = int((time.time() - g0) * 1000)

        self.audit.log_retrieval(
            run_id=run_id,
            question=question,
            rewrites=list(plan.retrieval_plan.rewrites),
            hard_filters=dict(plan.retrieval_plan.hard_filters),
            soft_boosts=list(plan.retrieval_plan.soft_boosts),
            candidates=[],
            reranker={"top": routing_debug.get("reranked", []) if isinstance(routing_debug, dict) else []},
            selected=[],
            packed_ids=[],
            latency_ms=retrieval_ms,
            debug={
                "retrieval_debug": retrieval_debug,
                "source_route": _obj_to_dict(plan.retrieval_plan.source_route),
                "news_context_present": bool(news_context),
                "transcript_context_present": bool(transcript_context),
            },
        )

        route_decision = routing_decide(
            plan_mode=plan.mode,
            gate_action=verification.status,
            gate_ok=verification.status in ("answer", "answer_with_warning"),
            retrieval_debug=routing_debug,
        )

        self.audit.log_gate(
            run_id=run_id,
            plan=_obj_to_dict(plan),
            req=_obj_to_dict(req),
            gate=_obj_to_dict(verification),
            routing=_obj_to_dict(route_decision),
            latency_ms=gate_ms,
        )

        action = route_decision.action
        if action not in ("answer", "answer_with_warning"):
            return {
                "run_id": run_id,
                "ok": False,
                "action": action,
                "mode": plan.mode,
                "reason": verification.reason_codes,
                "verification": _obj_to_dict(verification),
                "routing": _obj_to_dict(route_decision),
                "packed_context": full_context,
                "evidence": evidence,
                "target_runs": [{**tr, "target": _obj_to_dict(tr["target"])} for tr in target_runs] if target_runs else None,
                "assumptions": assumptions,
                "market_inputs": merged_market_inputs,
            }

        # 6) Answer generation
        result_obj: Optional[Dict[str, Any]] = None
        validation_ok = True
        validation_errors: List[str] = []
        generation_ms: Optional[int] = None
        usage: Dict[str, Any] = {}

        if plan.mode == "lookup_numeric":
            try:
                payload, det_debug = build_lookup_numeric_answer(full_context, plan.targets[0], req)
            except Exception as e:
                logger.exception("lookup_numeric build failed")
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "error",
                    "mode": plan.mode,
                    "reason": f"lookup_numeric_exception:{type(e).__name__}",
                    "verification": _obj_to_dict(VerificationResult(
                        status="error",
                        confidence=0.0,
                        mode=plan.mode,
                        reason_codes=[f"lookup_numeric_exception:{type(e).__name__}"],
                        errors=[str(e)],
                    )),
                }
            if payload is None:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "abstain",
                    "mode": plan.mode,
                    "reason": "deterministic_numeric_failed",
                    "debug": det_debug,
                    "verification": _obj_to_dict(verification),
                }

            result_obj = {
                "final_answer": f"{payload['metric']}: {payload['value']} {payload['unit']}",
                "claims": [{
                    "claim_type": "fact",
                    "entity": plan.targets[0].ticker,
                    "metric_or_topic": payload["metric"],
                    "period": f"FY{plan.targets[0].fiscal_year}" if plan.targets[0].fiscal_year is not None else None,
                    "unit": payload["unit"],
                    "value_or_summary": payload["value"],
                    "citations": [payload["citation"]],
                    "formula": None,
                    "inputs": [],
                }],
                "tables_used": [payload["citation"]] if str(payload["citation"]).lower().startswith("t") or "_t" in str(payload["citation"]).lower() else [],
                "provenance": {"ticker": plan.targets[0].ticker, "fiscal_year": plan.targets[0].fiscal_year},
                "inferences": assumptions[:],
                "confidence": float(verification.confidence),
                "numeric": payload,
            }

        elif plan.mode == "compute_metric":
            metric_name = plan.targets[0].metric or ""
            try:
                payload, det_debug = build_compute_metric_answer(full_context, plan.targets[0], metric_name, req)
            except Exception as e:
                logger.exception("compute_metric build failed")
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "error",
                    "mode": plan.mode,
                    "reason": f"compute_metric_exception:{type(e).__name__}",
                    "verification": _obj_to_dict(VerificationResult(
                        status="error",
                        confidence=0.0,
                        mode=plan.mode,
                        reason_codes=[f"compute_metric_exception:{type(e).__name__}"],
                        errors=[str(e)],
                    )),
                }
            if payload is None:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "abstain",
                    "mode": plan.mode,
                    "reason": "deterministic_compute_failed",
                    "debug": det_debug,
                    "verification": _obj_to_dict(verification),
                }

            citations = [x.get("citation") for x in payload.get("inputs", []) if isinstance(x, dict) and x.get("citation")]
            result_obj = {
                "final_answer": f"{payload['metric']}: {payload['value']} {payload['unit']}",
                "claims": [{
                    "claim_type": "ratio" if payload["unit"] in ("RATIO", "PERCENT") else "fact",
                    "entity": plan.targets[0].ticker,
                    "metric_or_topic": payload["metric"],
                    "period": f"FY{plan.targets[0].fiscal_year}" if plan.targets[0].fiscal_year is not None else None,
                    "unit": payload["unit"],
                    "value_or_summary": payload["value"],
                    "citations": citations[:4],
                    "formula": payload.get("formula"),
                    "inputs": payload.get("inputs", []),
                }],
                "tables_used": [c for c in citations if isinstance(c, str) and (c.lower().startswith("t") or "_t" in c.lower())],
                "provenance": {"ticker": plan.targets[0].ticker, "fiscal_year": plan.targets[0].fiscal_year},
                "inferences": assumptions[:],
                "confidence": float(verification.confidence),
                "computed": payload,
            }

        elif plan.mode in ("valuation", "scenario_analysis"):
            valuation_context = full_context
            try:
                fcf_target = replace(plan.targets[0], metric="fcf", item_hint="Item 8")
                fcf_payload, fcf_debug = build_compute_metric_answer(valuation_context, fcf_target, "fcf", req)
            except Exception as e:
                logger.exception("valuation fcf build failed")
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "error",
                    "mode": plan.mode,
                    "reason": f"valuation_fcf_exception:{type(e).__name__}",
                    "verification": _obj_to_dict(VerificationResult(
                        status="error",
                        confidence=0.0,
                        mode=plan.mode,
                        reason_codes=[f"valuation_fcf_exception:{type(e).__name__}"],
                        errors=[str(e)],
                    )),
                }
            if fcf_payload is None:
                # Retry with targeted retrieval for cash flow lines.
                t = plan.targets[0]
                qbits = [x for x in [t.ticker, f"FY{t.fiscal_year}" if t.fiscal_year else None] if x]
                targeted_query = " ".join(qbits + [
                    "Item 8",
                    "cash provided by operating activities",
                    "capital expenditures",
                    "free cash flow",
                ]).strip()
                targeted_ctx, targeted_dbg, _ = self.retrieval.retrieve(
                    targeted_query,
                    filters=self._filters_from_target(t),
                )
                if targeted_ctx and targeted_ctx.strip():
                    valuation_context = (valuation_context + "\n\n" + targeted_ctx).strip()
                    fcf_payload, retry_fcf_debug = build_compute_metric_answer(valuation_context, fcf_target, "fcf", req)
                    fcf_debug = {
                        "initial": fcf_debug,
                        "targeted_retrieval": {"query": targeted_query, "debug": targeted_dbg},
                        "retry": retry_fcf_debug,
                    }

            if fcf_payload is None:
                # Fallback: some issuers explicitly report FCF directly, while
                # CFO/CapEx components can be noisy to extract.
                direct_fcf_target = replace(plan.targets[0], metric="fcf", item_hint="Item 8")
                direct_fcf_payload, direct_fcf_debug = build_lookup_numeric_answer(valuation_context, direct_fcf_target, req)
                if direct_fcf_payload is not None:
                    fcf_payload = {
                        "metric": "fcf",
                        "value": float(direct_fcf_payload["value"]),
                        "unit": str(direct_fcf_payload.get("unit") or "USD"),
                        "formula": "direct_fcf_extraction",
                        "inputs": [{
                            "name": "fcf_direct",
                            "value": float(direct_fcf_payload["value"]),
                            "unit": str(direct_fcf_payload.get("unit") or "USD"),
                            "citation": direct_fcf_payload.get("citation"),
                        }],
                    }
                    fcf_debug = {"computed_fcf_debug": fcf_debug, "direct_fcf_debug": direct_fcf_debug}
                else:
                    fcf_debug = {"computed_fcf_debug": fcf_debug, "direct_fcf_debug": direct_fcf_debug}
            if fcf_payload is None:
                # Last-resort deterministic proxy: FCF ~= revenue * margin.
                # Keeps valuation mode operational when cash-flow lines are unavailable.
                revenue_target = replace(plan.targets[0], metric="revenue", item_hint="Item 8")
                revenue_best, revenue_debug = choose_best_numeric_with_gate(valuation_context, revenue_target, req, topn=6)
                if revenue_best is not None:
                    revenue_val = float(revenue_best.value_scaled)
                    fcf_proxy_margin = 0.12
                    fcf_proxy = revenue_val * fcf_proxy_margin
                    fcf_payload = {
                        "metric": "fcf",
                        "value": float(fcf_proxy),
                        "unit": "USD",
                        "formula": "fcf_proxy = revenue * 0.12",
                        "inputs": [{
                            "name": "revenue",
                            "value": revenue_val,
                            "unit": "USD",
                            "citation": revenue_best.evidence_id,
                        }],
                    }
                    assumptions.append("valuation_proxy_fcf_from_revenue_12pct")
                    fcf_debug = {
                        "initial": fcf_debug,
                        "direct_fcf_debug": direct_fcf_debug if 'direct_fcf_debug' in locals() else None,
                        "revenue_proxy_debug": revenue_debug,
                    }
            if fcf_payload is None:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "abstain",
                    "mode": plan.mode,
                    "reason": "deterministic_valuation_missing_fcf",
                    "debug": fcf_debug,
                    "verification": _obj_to_dict(verification),
                }

            overrides: Dict[str, Any] = {}
            if merged_market_inputs.get("wacc") is not None:
                overrides["wacc_base"] = float(merged_market_inputs["wacc"])
            if merged_market_inputs.get("terminal_growth") is not None:
                overrides["terminal_growth_base"] = float(merged_market_inputs["terminal_growth"])
            if merged_market_inputs.get("horizon_years") is not None:
                overrides["horizon_years"] = int(merged_market_inputs["horizon_years"])

            policy_assumptions = build_assumptions(
                strictness=req.evidence_strictness,
                overrides=overrides or None,
            )
            fcf_citations = [x.get("citation") for x in fcf_payload.get("inputs", []) if isinstance(x, dict) and x.get("citation")]
            if not fcf_citations:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "abstain",
                    "mode": plan.mode,
                    "reason": "deterministic_valuation_missing_citations",
                    "debug": fcf_debug,
                    "verification": _obj_to_dict(verification),
                }
            net_debt_val = merged_market_inputs.get("net_debt")
            if net_debt_val is None:
                net_debt_val = 0.0  # Fallback if net debt is missing

            shares_out_val = merged_market_inputs.get("shares_outstanding")
            if shares_out_val is None and merged_market_inputs.get("market_cap") and merged_market_inputs.get("price"):
                mkt_cap = float(merged_market_inputs["market_cap"])
                price_val = float(merged_market_inputs["price"])
                if price_val > 0:
                    shares_out_val = mkt_cap / price_val

            dcf = run_dcf(
                last_fcf=float(fcf_payload["value"]),
                currency=str(merged_market_inputs.get("currency") or "USD"),
                assumptions=policy_assumptions,
                net_debt=float(net_debt_val) if net_debt_val is not None else 0.0,
                shares_outstanding=float(shares_out_val) if shares_out_val is not None else None,
            )

            output_value = dcf.intrinsic_value_per_share if dcf.intrinsic_value_per_share is not None else dcf.enterprise_value
            output_name = "intrinsic_value_per_share" if dcf.intrinsic_value_per_share is not None else "enterprise_value"

            valuation_gap_pct = None
            if dcf.intrinsic_value_per_share is not None and merged_market_inputs.get("price") is not None:
                mkt_price = float(merged_market_inputs["price"])
                if mkt_price > 0:
                    valuation_gap_pct = (float(dcf.intrinsic_value_per_share) - mkt_price) / mkt_price
            elif dcf.enterprise_value is not None and merged_market_inputs.get("enterprise_value") is not None:
                mkt_ev = float(merged_market_inputs["enterprise_value"])
                if mkt_ev > 0:
                    valuation_gap_pct = (float(dcf.enterprise_value) - mkt_ev) / mkt_ev

            result_obj = {
                "final_answer": f"DCF {output_name}: {output_value:.4f} {dcf.currency}",
                "claims": [{
                    "claim_type": "valuation",
                    "entity": plan.targets[0].ticker,
                    "metric_or_topic": "dcf_valuation",
                    "period": f"FY{plan.targets[0].fiscal_year}" if plan.targets[0].fiscal_year is not None else None,
                    "unit": dcf.currency,
                    "value_or_summary": float(output_value),
                    "citations": fcf_citations[:4],
                    "formula": "DCF using projected FCF and Gordon terminal value",
                    "inputs": [{
                        "name": "fcf",
                        "value": float(fcf_payload["value"]),
                        "unit": "USD",
                        "citation": fcf_citations[0] if fcf_citations else None,
                    }],
                }],
                "tables_used": [c for c in fcf_citations if isinstance(c, str) and (c.lower().startswith("t") or "_t" in c.lower())],
                "provenance": {"ticker": plan.targets[0].ticker, "fiscal_year": plan.targets[0].fiscal_year},
                "inferences": assumptions[:],
                "confidence": float(verification.confidence),
                "valuation": {
                    "type": "DCF",
                    "valuation_gap_pct": valuation_gap_pct,
                    "verified_inputs": [{
                        "name": "fcf",
                        "value": float(fcf_payload["value"]),
                        "unit": "USD",
                        "citation": fcf_citations[0],
                    }],
                    "assumptions": [
                        {"name": "wacc_base", "value": policy_assumptions.get("wacc_base"), "source": "policy"},
                        {"name": "terminal_growth_base", "value": policy_assumptions.get("terminal_growth_base"), "source": "policy"},
                        {"name": "horizon_years", "value": policy_assumptions.get("horizon_years"), "source": "policy"},
                    ],
                    "outputs": [
                        {"name": "enterprise_value", "value": float(dcf.enterprise_value), "unit": dcf.currency},
                        {"name": "equity_value", "value": float(dcf.equity_value) if dcf.equity_value is not None else 0.0, "unit": dcf.currency},
                        {"name": output_name, "value": float(output_value), "unit": dcf.currency},
                    ],
                    "sensitivity": dcf.sensitivity,
                },
            }
            try:
                custom_scenarios: List[ScenarioSpec] = []
                if is_whatif_query(plan.raw_question):
                    whatif_overrides = parse_whatif(plan.raw_question, policy_assumptions)
                    if whatif_overrides:
                        custom_scenarios.append(
                            ScenarioSpec(
                                name="User What-If",
                                scenario_type="custom",
                                description=f"User scenario: {plan.raw_question[:100]}",
                                overrides=whatif_overrides,
                            )
                        )
                scenario_names = ["bull", "bear", "stress_recession"]
                scenario_result = run_scenario_analysis(
                    ticker=plan.targets[0].ticker or "",
                    last_fcf=float(fcf_payload["value"]),
                    currency=str(merged_market_inputs.get("currency") or "USD"),
                    scenario_names=scenario_names,
                    custom_scenarios=custom_scenarios or None,
                    net_debt=float(merged_market_inputs["net_debt"]) if merged_market_inputs.get("net_debt") is not None else None,
                    shares_outstanding=float(merged_market_inputs["shares_outstanding"]) if merged_market_inputs.get("shares_outstanding") is not None else None,
                    strictness=req.evidence_strictness,
                    run_monte_carlo=("monte carlo" in plan.raw_question.lower()),
                )
                result_obj["scenario_analysis"] = {
                    "base_ev": float(scenario_result.base_result.enterprise_value),
                    "comparisons": [
                        {
                            "name": c.scenario_name,
                            "ev": float(c.scenario_ev),
                            "ev_delta_pct": float(c.ev_delta_pct),
                            "ivps": float(c.scenario_ivps) if c.scenario_ivps is not None else None,
                            "ivps_delta_pct": float(c.ivps_delta_pct) if c.ivps_delta_pct is not None else None,
                            "parameter_changes": c.parameter_changes,
                        }
                        for c in scenario_result.comparisons
                    ],
                    "monte_carlo": scenario_result.monte_carlo,
                }
            except Exception:
                logger.exception("scenario analysis failed")

        elif plan.mode in ("relative_valuation", "peer_analysis"):
            multiple_type = self._relative_multiple_type(plan.raw_question)
            metric_candidates_map: Dict[str, List[Tuple[str, str]]] = {
                "P_E": [("eps", "primary_eps")],
                "P_S": [("revenue", "primary_revenue")],
                "EV_SALES": [("revenue", "primary_revenue")],
                "EV_EBITDA": [
                    ("ebitda", "primary_ebitda"),
                    ("operating_income", "fallback_operating_income_proxy"),
                ],
            }
            metric_candidates = metric_candidates_map[multiple_type]
            denom_metric = metric_candidates[0][0]
            denom_best = None
            denom_debug: Dict[str, Any] = {"attempts": []}
            denom_source_tag = metric_candidates[0][1]
            relative_context = full_context
            for cand_metric, cand_tag in metric_candidates:
                cand_target = replace(plan.targets[0], metric=cand_metric, item_hint="Item 8")
                cand_best, cand_debug = choose_best_numeric_with_gate(relative_context, cand_target, req, topn=6)
                denom_debug["attempts"].append({"metric": cand_metric, "tag": cand_tag, "debug": cand_debug})
                if cand_best is not None:
                    denom_metric = cand_metric
                    denom_best = cand_best
                    denom_source_tag = cand_tag
                    break

            if denom_best is None:
                # Retry with targeted denominator retrieval.
                t = plan.targets[0]
                qbits = [x for x in [t.ticker, f"FY{t.fiscal_year}" if t.fiscal_year else None] if x]
                targeted_ctx_parts: List[str] = []
                targeted_debug: List[Dict[str, Any]] = []
                for cand_metric, cand_tag in metric_candidates:
                    targeted_query = " ".join(qbits + ["Item 8", cand_metric.replace("_", " ")]).strip()
                    add_ctx, add_dbg, _ = self.retrieval.retrieve(
                        targeted_query,
                        filters=self._filters_from_target(t),
                    )
                    targeted_debug.append({"metric": cand_metric, "tag": cand_tag, "query": targeted_query, "debug": add_dbg})
                    if add_ctx and add_ctx.strip():
                        targeted_ctx_parts.append(add_ctx)
                if targeted_ctx_parts:
                    relative_context = (relative_context + "\n\n" + "\n\n".join(targeted_ctx_parts)).strip()
                    for cand_metric, cand_tag in metric_candidates:
                        cand_target = replace(plan.targets[0], metric=cand_metric, item_hint="Item 8")
                        cand_best, cand_dbg = choose_best_numeric_with_gate(relative_context, cand_target, req, topn=6)
                        denom_debug["attempts"].append({"metric": cand_metric, "tag": f"{cand_tag}_targeted_retry", "debug": cand_dbg})
                        if cand_best is not None:
                            denom_metric = cand_metric
                            denom_best = cand_best
                            denom_source_tag = cand_tag
                            break
                denom_debug["targeted_retrieval"] = targeted_debug

            if denom_best is None:
                # Last-resort fallback: compute EV/Sales if requested denominator is unavailable.
                # This avoids hard-failing relative valuation mode for sparse filings.
                revenue_target = replace(plan.targets[0], metric="revenue", item_hint="Item 8")
                revenue_best, revenue_debug = choose_best_numeric_with_gate(relative_context, revenue_target, req, topn=6)
                if revenue_best is not None:
                    denom_metric = "revenue"
                    denom_best = revenue_best
                    denom_source_tag = "fallback_revenue_for_ev_sales"
                    if multiple_type != "EV_SALES":
                        assumptions.append("relative_valuation_fallback_to_ev_sales")
                        multiple_type = "EV_SALES"
                    denom_debug["revenue_fallback"] = revenue_debug

            if denom_best is None:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "abstain",
                    "mode": plan.mode,
                    "reason": "deterministic_relative_missing_denominator",
                    "debug": denom_debug,
                    "verification": _obj_to_dict(verification),
                }

            denom_value = float(denom_best.value_raw if denom_metric == "eps" else denom_best.value_scaled)
            peer_signal: Optional[Dict[str, Any]] = None
            peer_median: Optional[float] = None
            if self.cfg.market_provider is not None and plan.targets[0].ticker:
                try:
                    peer_result = run_peer_analysis(
                        ticker=plan.targets[0].ticker,
                        market_provider=self.cfg.market_provider,
                        multiples_to_compute=[multiple_type],
                    )
                    peer_signal = peer_analysis_to_signal(peer_result)
                    pm = peer_signal.get("peer_median")
                    if isinstance(pm, (int, float)):
                        peer_median = float(pm)
                except Exception:
                    logger.exception("peer analysis failed for %s", plan.targets[0].ticker)
            try:
                rv = compute_multiple(
                    multiple_type=multiple_type,  # type: ignore[arg-type]
                    currency=str(merged_market_inputs.get("currency") or "USD"),
                    price=float(merged_market_inputs["price"]) if merged_market_inputs.get("price") is not None else None,
                    market_cap=float(merged_market_inputs["market_cap"]) if merged_market_inputs.get("market_cap") is not None else None,
                    enterprise_value=float(merged_market_inputs["enterprise_value"]) if merged_market_inputs.get("enterprise_value") is not None else None,
                    shares_outstanding=float(merged_market_inputs["shares_outstanding"]) if merged_market_inputs.get("shares_outstanding") is not None else None,
                    eps=denom_value if multiple_type == "P_E" else None,
                    revenue=denom_value if multiple_type in ("P_S", "EV_SALES") else None,
                    ebitda=denom_value if multiple_type == "EV_EBITDA" else None,
                    sources={"denominator_citation": denom_best.evidence_id, "denominator_source": denom_source_tag},
                )
            except Exception as e:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "abstain",
                    "mode": plan.mode,
                    "reason": f"deterministic_relative_exception:{type(e).__name__}",
                    "debug": {"multiple_type": multiple_type, "error": str(e)},
                    "verification": _obj_to_dict(verification),
                }
            if multiple_type == "EV_EBITDA" and denom_metric == "operating_income":
                rv.notes = f"{rv.notes} Proxy used: operating_income substituted for EBITDA due missing EBITDA evidence."
                assumptions.append("relative_valuation_proxy_operating_income_for_ebitda")

            result_obj = {
                "final_answer": f"{rv.multiple_type}: {rv.multiple:.4f}x",
                "claims": [{
                    "claim_type": "valuation",
                    "entity": plan.targets[0].ticker,
                    "metric_or_topic": rv.multiple_type,
                    "period": f"FY{plan.targets[0].fiscal_year}" if plan.targets[0].fiscal_year is not None else None,
                    "unit": "RATIO",
                    "value_or_summary": float(rv.multiple),
                    "citations": [denom_best.evidence_id],
                    "formula": rv.notes,
                    "inputs": [{
                        "name": "denominator",
                        "value": denom_value,
                        "unit": denom_best.unit,
                        "citation": denom_best.evidence_id,
                    }],
                }],
                "tables_used": [denom_best.evidence_id] if denom_best.evidence_id.lower().startswith("t") or "_t" in denom_best.evidence_id.lower() else [],
                "provenance": {"ticker": plan.targets[0].ticker, "fiscal_year": plan.targets[0].fiscal_year},
                "inferences": assumptions[:],
                "confidence": float(verification.confidence),
                "relative_valuation": {
                    "multiple": rv.multiple_type,
                    "numerator": {"name": "market_input", "value": float(rv.numerator), "citation": "market"},
                    "denominator": {"name": denom_metric, "value": float(rv.denominator), "citation": denom_best.evidence_id},
                    "value": float(rv.multiple),
                    "peer_median": peer_median,
                    "peer_premium_pct": ((float(rv.multiple) - peer_median) / peer_median) if peer_median not in (None, 0.0) else None,
                },
                "peer_analysis": peer_signal,
            }

        else:
            if self.llm_client is None:
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "error",
                    "mode": plan.mode,
                    "reason": "llm_client_required_for_narrative_modes",
                    "verification": _obj_to_dict(VerificationResult(
                        status="error",
                        confidence=0.0,
                        mode=plan.mode,
                        reason_codes=["llm_client_required_for_narrative_modes"],
                        errors=["llm_client_required_for_narrative_modes"],
                    )),
                }

            model_name = self.cfg.small_model_name if route_decision.model == "small" else self.cfg.large_model_name
            if plan.mode == "mba_framework":
                framework_type = detect_framework_type(plan.raw_question)
                system_prompt, user_prompt = build_framework_prompt(
                    question=plan.raw_question,
                    packed_context=full_context,
                    framework_type=framework_type,
                    ticker=plan.targets[0].ticker if plan.targets else None,
                    fiscal_year=plan.targets[0].fiscal_year if plan.targets else None,
                )
            elif plan.mode in ("comparative_analysis", "multi_period_analysis"):
                unique_tickers = {t.ticker for t in plan.targets if t.ticker}
                unique_years = sorted({int(t.fiscal_year) for t in plan.targets if t.fiscal_year is not None})
                if plan.mode == "multi_period_analysis" or (len(unique_tickers) <= 1 and len(unique_years) >= 2):
                    ticker = next(iter(unique_tickers)) if unique_tickers else (plan.targets[0].ticker if plan.targets else None)
                    packed_contexts: Dict[int, str] = {}
                    for target in plan.targets:
                        if target.fiscal_year is None:
                            continue
                        t_query = f"{query} {ticker or ''} FY{target.fiscal_year}".strip()
                        t_ctx, _, _ = self.retrieval.retrieve(
                            t_query,
                            filters=self._filters_from_target(target),
                        )
                        if t_ctx and t_ctx.strip():
                            packed_contexts[int(target.fiscal_year)] = t_ctx
                    if packed_contexts:
                        system_prompt, user_prompt = build_multi_period_prompt(
                            question=plan.raw_question,
                            packed_contexts=packed_contexts,
                            ticker=ticker or "",
                            periods=sorted(packed_contexts.keys()),
                            metric=plan.targets[0].metric if plan.targets else None,
                        )
                    else:
                        system_prompt, user_prompt = build_json_answer_prompt(plan, full_context)
                else:
                    system_prompt, user_prompt = build_json_answer_prompt(plan, full_context)
            else:
                system_prompt, user_prompt = build_json_answer_prompt(plan, full_context)

            t0 = time.time()
            try:
                model_text, usage = self.llm_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=model_name,
                )
            except Exception as e:
                logger.warning("llm generation failed; using deterministic fallback: %s", e)
                generation_ms = int((time.time() - t0) * 1000)
                usage = {}
                result_obj = _build_llm_fallback_result(
                    plan=plan,
                    full_context=full_context,
                    assumptions=assumptions,
                    verification=verification,
                    error_text=type(e).__name__,
                )
                validation_ok, validation_errors, _ = validate_answer_json(plan, full_context, _safe_json_dumps(result_obj))
                if not validation_ok:
                    # Safety net: keep response available even if strict schema validation flags fallback fields.
                    validation_errors.append(f"fallback_schema_validation:{type(e).__name__}")
                    validation_ok = True
                self.audit.log_generation(
                    run_id=run_id,
                    model_name=f"{model_name}:fallback",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_text=_safe_json_dumps(result_obj),
                    latency_ms=generation_ms,
                    token_usage=usage,
                )
                parsed = None
                model_text = _safe_json_dumps(result_obj)
            else:
                generation_ms = int((time.time() - t0) * 1000)

                self.audit.log_generation(
                    run_id=run_id,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_text=model_text,
                    latency_ms=generation_ms,
                    token_usage=usage,
                )

                validation_ok, validation_errors, parsed = validate_answer_json(plan, full_context, model_text)
                if isinstance(parsed, dict):
                    if plan.mode == "mba_framework":
                        try:
                            framework_type = detect_framework_type(plan.raw_question)
                            fw_ok, fw_errors = validate_framework_output(parsed, framework_type)
                            if not fw_ok:
                                validation_errors.extend(fw_errors)
                        except Exception:
                            logger.exception("framework-specific validation failed")
                    parsed["confidence"] = float(verification.confidence)
                    result_obj = _append_inferences(parsed, assumptions + verification.warnings)
                else:
                    result_obj = {
                        "final_answer": model_text,
                        "claims": [],
                        "tables_used": [],
                        "provenance": {"ticker": plan.targets[0].ticker if plan.targets else None, "fiscal_year": plan.targets[0].fiscal_year if plan.targets else None},
                        "inferences": assumptions + verification.warnings,
                        "confidence": float(verification.confidence),
                    }

        # deterministic output validation too
        if result_obj is not None and plan.mode in (
            "lookup_numeric",
            "compute_metric",
            "valuation",
            "relative_valuation",
            "scenario_analysis",
            "peer_analysis",
        ):
            validation_ok, validation_errors, _ = validate_answer_json(plan, full_context, _safe_json_dumps(result_obj))

        total_ms = int((time.time() - start_t) * 1000)

        self.audit.log_validation(
            run_id=run_id,
            ok=validation_ok,
            errors=validation_errors,
            signals={
                "verification": _obj_to_dict(verification),
                "usage": usage,
                "market_inputs": merged_market_inputs,
            },
            latency_ms=generation_ms,
        )

        if isinstance(result_obj, dict) and "best_evidence" not in result_obj:
            result_obj["best_evidence"] = getattr(verification, "best_evidence", [])

        final_result = {
            "run_id": run_id,
            "ok": validation_ok,
            "action": action if validation_ok else "abstain",
            "mode": plan.mode,
            "decision_time": decision_time,
            "result": result_obj,
            "verification": _obj_to_dict(verification),
            "routing": _obj_to_dict(route_decision),
            "validation_errors": validation_errors,
            "packed_context": full_context,
            "filing_context": filing_context,
            "news_context": news_context,
            "transcript_context": transcript_context,
            "evidence": evidence,
            "target_runs": [{**tr, "target": _obj_to_dict(tr["target"])} for tr in target_runs] if target_runs else None,
            "assumptions": assumptions,
            "market_inputs": merged_market_inputs,
            "timing_ms": {
                "total_ms": total_ms,
                "retrieval_ms": retrieval_ms,
                "verification_ms": gate_ms,
                "generation_ms": generation_ms,
            },
        }

        # 7) Optional signal layer enrichment
        try:
            primary_ticker = plan.targets[0].ticker if plan.targets else None
            primary_fy = plan.targets[0].fiscal_year if plan.targets else None

            current_transcript_text = None
            prior_transcript_text = None

            if primary_ticker and primary_fy:
                try:
                    tc = TranscriptIngestionClient()
                    current_period = f"FY{primary_fy}"
                    prior_period = f"FY{primary_fy - 1}"
                    current_transcript_text, prior_transcript_text = tc.get_current_and_prior_text(
                        ticker=primary_ticker,
                        current_period=current_period,
                        prior_period=prior_period,
                    )
                except Exception:
                    logger.debug("Transcript enrichment unavailable for %s", primary_ticker)

            if primary_ticker:
                final_result = run_hackathon_signal_layer(
                    base_result=final_result,
                    ticker=primary_ticker,
                    fiscal_year=primary_fy,
                    company=None,
                    current_transcript_text=current_transcript_text,
                    prior_transcript_text=prior_transcript_text,
                    recent_news_enabled=True,
                    decision_time=decision_time,
                )
                final_result = _merge_signals_into_answer(final_result)

        except Exception:
            logger.exception("Signal layer failed")

        return final_result


def _build_signal_summary(report: Dict[str, Any], score: Dict[str, Any]) -> str:
    lines: List[str] = []

    ticker = report.get("ticker", "")
    rec = report.get("recommendation", "HOLD")
    strength = report.get("signal_strength", 0.0)
    confidence = report.get("confidence", 0.0)
    fy = report.get("fiscal_year")

    header = f"{ticker}"
    if fy:
        header += f" (FY{fy})"
    lines.append(f"--- Investment Signal: {header} ---")
    lines.append(f"Recommendation: {rec} | Signal: {strength:+.2f} | Confidence: {confidence:.0%}")

    comp = score.get("component_scores", {})
    if comp:
        lines.append("Components: " + ", ".join(f"{k}: {v:+.2f}" for k, v in comp.items()))

    findings = report.get("key_findings", [])
    if findings:
        lines.append("Key Findings:")
        for f in findings[:5]:
            lines.append(f"  - {f}")

    risks = report.get("top_risks", [])
    if risks:
        lines.append("Top Risks:")
        for r in risks[:3]:
            lines.append(f"  - {r.get('category', 'unknown')} (severity={r.get('severity', '?')})")

    return "\n".join(lines)


def _merge_signals_into_answer(final_result: Dict[str, Any]) -> Dict[str, Any]:
    report = final_result.get("hackathon_signal_report")
    score = final_result.get("hackathon_signal_score")
    decision = final_result.get("hackathon_signal_decision") or {}
    if not report or not score:
        return final_result

    signal_text = _build_signal_summary(report, score)
    result_obj = final_result.get("result")

    if isinstance(result_obj, dict):
        base = result_obj.get("final_answer", "")
        result_obj["final_answer"] = f"{base}\n\n{signal_text}" if base else signal_text
        result_obj["signal_recommendation"] = report.get("recommendation", "HOLD")
        result_obj["signal_strength"] = report.get("signal_strength", 0.0)
        result_obj["signal_confidence"] = report.get("confidence", 0.0)
        result_obj["signal_action"] = decision.get("action", "NO_ACT")
        final_result["result"] = result_obj

    return final_result
