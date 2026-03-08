# orchestra.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from audit import AuditLogger
from market_api import (
    MarketDataProvider,
    TTLCache,
    fetch_min_market_inputs,
    merge_market_inputs,
)
from retrieval_tool import FinancialRetrievalTool, RetrievalConfig
from verification import (
    Target,
    TaskPlan,
    VerificationResult,
    build_compute_metric_answer,
    build_json_answer_prompt,
    build_lookup_numeric_answer,
    build_task_plan,
    evidence_requirements,
    gate_evidence,
    validate_answer_json,
    with_strictness,
)
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

    def _fetch_news_context(self, plan: TaskPlan) -> str:
        if not self.cfg.news_client or not plan.targets or not plan.targets[0].ticker:
            return ""

        ticker = plan.targets[0].ticker
        fy = plan.targets[0].fiscal_year
        try:
            rows = self.cfg.news_client.fetch_company_news(ticker=ticker, limit=5)
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
        return "\n\n".join(chunks)

    def _fetch_transcript_context(self, plan: TaskPlan) -> str:
        if not plan.targets or not plan.targets[0].ticker:
            return ""

        ticker = plan.targets[0].ticker
        fy = plan.targets[0].fiscal_year

        try:
            tc = TranscriptIngestionClient()
            if fy is not None:
                current_period = f"FY{fy}"
                prior_period = f"FY{fy - 1}"
                current_text, prior_text = tc.get_current_and_prior_text(
                    ticker=ticker,
                    current_period=current_period,
                    prior_period=prior_period,
                )
                chunks: List[str] = []
                if current_text:
                    chunks.append(f"[TRANSCRIPT {ticker} FY{fy} transcript_{ticker}_{fy}_current]\n{current_text}")
                if prior_text:
                    chunks.append(f"[TRANSCRIPT {ticker} FY{fy-1} transcript_{ticker}_{fy-1}_prior]\n{prior_text}")
                return "\n\n".join(chunks)
            return ""
        except Exception:
            logger.exception("Transcript fetch failed for %s", ticker)
            return ""

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
        # NOTE: routing.decide() in routing.py computes a risk-based RoutingDecision
        # (coverage, reranker margin, retrieval agreement) that can refine this action
        # and drive small vs. large model selection. It is not wired in here yet —
        # the action is taken directly from the verification gate. To enable it, call:
        #   from routing import decide as routing_decide
        #   rd = routing_decide(plan_mode=plan.mode, gate_action=verification.status,
        #                       gate_ok=verification.status in ("answer","answer_with_warning"),
        #                       retrieval_debug=retrieval_debug)
        # then use rd.action and rd.model instead.
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
        news_context = self._fetch_news_context(plan) if plan.retrieval_plan.source_route.news else ""
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

        self.audit.log_gate(
            run_id=run_id,
            plan=_obj_to_dict(plan),
            req=_obj_to_dict(req),
            gate=_obj_to_dict(verification),
            routing={"action": verification.status, "confidence": verification.confidence},
            latency_ms=gate_ms,
        )

        action = self._final_action_from_verification(verification)
        if action not in ("answer", "answer_with_warning"):
            return {
                "run_id": run_id,
                "ok": False,
                "action": action,
                "mode": plan.mode,
                "reason": verification.reason_codes,
                "verification": _obj_to_dict(verification),
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

            model_name = self.cfg.small_model_name if verification.confidence < 0.75 else self.cfg.large_model_name
            system_prompt, user_prompt = build_json_answer_prompt(plan, full_context)

            t0 = time.time()
            try:
                model_text, usage = self.llm_client.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=model_name,
                )
            except Exception as e:
                logger.exception("llm generation failed")
                return {
                    "run_id": run_id,
                    "ok": False,
                    "action": "error",
                    "mode": plan.mode,
                    "reason": f"llm_generation_exception:{type(e).__name__}",
                    "verification": _obj_to_dict(VerificationResult(
                        status="error",
                        confidence=0.0,
                        mode=plan.mode,
                        reason_codes=[f"llm_generation_exception:{type(e).__name__}"],
                        errors=[str(e)],
                    )),
                }
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
        if result_obj is not None and plan.mode in ("lookup_numeric", "compute_metric"):
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

        final_result = {
            "run_id": run_id,
            "ok": validation_ok,
            "action": verification.status if validation_ok else "abstain",
            "mode": plan.mode,
            "result": result_obj,
            "verification": _obj_to_dict(verification),
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
        final_result["result"] = result_obj

    return final_result
