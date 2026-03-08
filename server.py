import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file(): return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
_load_dotenv()

# Internal imports
from local_llm import build_local_llm_client, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL
from market_api import YahooFinanceMarketDataProvider
from news_client_adapter import build_optional_news_client
from orchestrator import FinancialOrchestrator, OrchestratorConfig
try:
    from knowledge_base import TICKERS as _KB_TICKERS
except Exception:
    _KB_TICKERS = []

# --- Setup & Config ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FinSight API", version="1.0.0")

# CORS config to allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Orchestrator instance
_orchestrator: Optional[FinancialOrchestrator] = None

def get_orchestrator() -> FinancialOrchestrator:
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator
        
    base_dir = Path(__file__).resolve().parent
    audit_log = base_dir / "logs" / "audit.jsonl"
    
    known_tickers = {t.upper() for t in _KB_TICKERS if isinstance(t, str) and t.strip()}
    
    llm_client = build_local_llm_client(
        primary_model=os.environ.get("GEMINI_SMALL_MODEL", DEFAULT_PRIMARY_MODEL),
        fallback_model=os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
    )
    
    cfg = OrchestratorConfig(
        base_dir=base_dir,
        audit_log_path=audit_log,
        known_tickers=known_tickers or None,
        market_provider=YahooFinanceMarketDataProvider(),
        news_client=build_optional_news_client(),
    )
    
    _orchestrator = FinancialOrchestrator(cfg=cfg, llm_client=llm_client)
    logger.info("FinancialOrchestrator initialized globally.")
    return _orchestrator

# --- API Models ---
class AnalyzeRequest(BaseModel):
    query: str
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    mode: str = "auto"
    strictness: int = 70

# --- Endpoints ---
@app.post("/api/analyze")
async def analyze_query(req: AnalyzeRequest):
    try:
        orch = get_orchestrator()
        
        # Determine actual mode
        forced_mode = None if req.mode == "auto" else req.mode
        
        result = orch.answer(
            question=req.query,
            market_inputs=None,
            auto_fetch_market=True,
            forced_mode=forced_mode,
            ui_intent=req.mode,
            ui_ticker=req.ticker,
            ui_fiscal_year=req.fiscal_year,
            evidence_strictness=req.strictness,
            decision_time=datetime.now(timezone.utc).isoformat(),
        )
        
        return result
        
    except Exception as e:
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

class DecisionRequest(BaseModel):
    ticker: str = "AAPL"
    fiscal_year: int = 2024
    strictness: int = 70

@app.post("/api/decision")
async def decision_analysis(req: DecisionRequest):
    """
    Full multi-tool signal pipeline.
    Directly calls ALL 5 signal tools to populate every component:
      1) Risk  → extract_risk_signals() on SEC Item 1A text
      2) Tone  → compare_tone() via FinBERT on earnings transcripts
      3) Valuation → run_dcf() using yfinance market data
      4) Growth → yfinance revenue_growth_yoy
      5) News  → classify_news_catalysts() via NewsAPI
    """
    from nlp_signals import extract_risk_signals, compare_tone, classify_news_catalysts
    from news_ingestion import NewsIngestionClient
    from transcript_ingestion import TranscriptIngestionClient
    from signal_scoring import compute_final_signal, signal_action_from_score, to_dict
    from demo_reporting import build_demo_signal_report
    from valuation_engine import run_dcf

    try:
        orch = get_orchestrator()
        dt = datetime.now(timezone.utc).isoformat()

        # ─────────────────────────────────────────────
        # TOOL 1: Retrieve SEC Filing context for risk
        # ─────────────────────────────────────────────
        logger.info("=== TOOL 1: SEC Filing Retrieval ===")
        risk_query = f"{req.ticker} FY{req.fiscal_year} Item 1A risk factors"
        filing_ctx, retrieval_debug, evidence = orch.retrieval.retrieve(risk_query, filters={})

        # Extract Item 1A text and run risk scanner
        from hackathon_pipeline import _extract_item_1a_text_from_context
        item1a_text = _extract_item_1a_text_from_context(filing_ctx)
        risk_signals = extract_risk_signals(item1a_text)
        risk_avg = (
            sum(r.severity for r in risk_signals[:3]) / max(min(len(risk_signals), 3), 1)
            if risk_signals else 0.0
        )
        top_risks = [r.__dict__ for r in risk_signals[:5]]
        logger.info("  Risk severity avg: %s, categories: %s", risk_avg, [r.category for r in risk_signals[:5]])

        # ─────────────────────────────────────────────
        # TOOL 2: Tone analysis via FinBERT on transcripts
        # ─────────────────────────────────────────────
        logger.info("=== TOOL 2: Transcript Tone Analysis ===")
        tone_delta = 0.0
        tone_trend = {}
        try:
            tc = TranscriptIngestionClient()
            current_text, prior_text = tc.get_current_and_prior_text(
                ticker=req.ticker,
                current_period=f"FY{req.fiscal_year}",
                prior_period=f"FY{req.fiscal_year - 1}",
            )
            if current_text and prior_text:
                tone_trend = compare_tone(current_text, prior_text)
                tone_delta = float(tone_trend.get("delta", 0.0))
                logger.info("  Tone delta: %s, direction: %s", tone_delta, tone_trend.get("direction"))
            else:
                logger.info("  No transcript pair available, tone stays at 0.0")
        except Exception as exc:
            logger.warning("  Transcript tone failed: %s", exc)

        # ─────────────────────────────────────────────
        # TOOL 3: DCF Valuation using yfinance market data
        # ─────────────────────────────────────────────
        logger.info("=== TOOL 3: DCF Valuation ===")
        valuation_gap_pct = None
        valuation_summary = {}
        try:
            market_provider = YahooFinanceMarketDataProvider()
            from market_api import fetch_min_market_inputs
            market_data = fetch_min_market_inputs(market_provider, ticker=req.ticker)
            current_price = market_data.get("price")
            logger.info("  Market data: price=%s, market_cap=%s", current_price, market_data.get("market_cap"))

            # Try to get FCF from the orchestrator's retrieval
            fcf_query = f"{req.ticker} FY{req.fiscal_year} Item 8 free cash flow operating cash capital expenditures"
            fcf_ctx, _, _ = orch.retrieval.retrieve(fcf_query, filters={})

            # Use a reasonable FCF estimate via revenue proxy
            rev_query = f"{req.ticker} FY{req.fiscal_year} total revenue net sales"
            rev_ctx, _, _ = orch.retrieval.retrieve(rev_query, filters={})

            # Attempt to extract revenue from the LLM (via orchestrator)
            try:
                rev_result = orch.answer(
                    question=f"What was {req.ticker}'s total revenue in FY{req.fiscal_year}?",
                    auto_fetch_market=False,
                    forced_mode="lookup_numeric",
                    ui_intent="lookup_numeric",
                    ui_ticker=req.ticker,
                    ui_fiscal_year=req.fiscal_year,
                    evidence_strictness=req.strictness,
                )
                rev_val = None
                rev_res = rev_result.get("result", {})
                if isinstance(rev_res, dict):
                    claims = rev_res.get("claims", [])
                    if claims and isinstance(claims[0], dict):
                        raw_val = claims[0].get("value_scaled") or claims[0].get("value_or_summary")
                        if raw_val is not None:
                            try:
                                rev_val = float(raw_val)
                            except (TypeError, ValueError):
                                pass
                if rev_val and rev_val > 0 and current_price:
                    # FCF proxy: ~12% margin for large tech
                    fcf_proxy = rev_val * 0.12
                    shares = market_data.get("market_cap", 0) / current_price if current_price else None

                    assumptions = {
                        "wacc_base": 0.09,
                        "fcf_growth_base": 0.05,
                        "terminal_growth_base": 0.025,
                        "horizon_years": 5,
                        "min_wacc_minus_tg": 0.01,
                    }
                    dcf_result = run_dcf(
                        last_fcf=fcf_proxy,
                        currency="USD",
                        assumptions=assumptions,
                        shares_outstanding=shares,
                    )
                    if dcf_result.intrinsic_value_per_share and current_price:
                        valuation_gap_pct = (dcf_result.intrinsic_value_per_share - current_price) / current_price
                        valuation_summary = {
                            "intrinsic_value": round(dcf_result.intrinsic_value_per_share, 2),
                            "current_price": round(current_price, 2),
                            "valuation_gap_pct": round(valuation_gap_pct, 4),
                            "enterprise_value": round(dcf_result.enterprise_value, 0),
                        }
                        logger.info("  DCF intrinsic: $%.2f vs price $%.2f → gap %.1f%%",
                                    dcf_result.intrinsic_value_per_share, current_price, valuation_gap_pct * 100)
                    else:
                        logger.info("  DCF completed but no per-share value available")
                else:
                    logger.info("  Revenue extraction returned %s — skipping DCF", rev_val)
            except Exception as exc:
                logger.warning("  Revenue extraction failed: %s", exc)
        except Exception as exc:
            logger.warning("  Valuation tool failed: %s", exc)

        # ─────────────────────────────────────────────
        # TOOL 4: Growth (revenue YoY from yfinance)
        # ─────────────────────────────────────────────
        logger.info("=== TOOL 4: Growth Metrics ===")
        revenue_growth_yoy = None
        try:
            import yfinance as yf
            tk = yf.Ticker(req.ticker)
            financials = tk.financials
            if financials is not None and not financials.empty:
                revenue_row = None
                for label in ["Total Revenue", "Revenue", "Net Sales"]:
                    if label in financials.index:
                        revenue_row = financials.loc[label]
                        break
                if revenue_row is not None and len(revenue_row) >= 2:
                    recent = float(revenue_row.iloc[0])
                    prior = float(revenue_row.iloc[1])
                    if prior > 0:
                        revenue_growth_yoy = (recent - prior) / prior
                        logger.info("  Revenue YoY: %.1f%% (%.0f → %.0f)", revenue_growth_yoy * 100, prior, recent)
        except Exception as exc:
            logger.warning("  Growth metrics failed: %s", exc)

        # ─────────────────────────────────────────────
        # TOOL 5: News sentiment via NewsAPI
        # ─────────────────────────────────────────────
        logger.info("=== TOOL 5: News Sentiment ===")
        news_summary = []
        avg_news_score = None
        try:
            news_client = NewsIngestionClient()
            articles = news_client.fetch_recent_news(
                ticker=req.ticker,
                company=None,
                as_of=dt,
                max_results=10,
            )
            catalysts = classify_news_catalysts([a.__dict__ for a in articles])
            news_summary = [c.__dict__ for c in catalysts[:5]]
            if catalysts:
                top = catalysts[:5]
                avg_news_score = sum(c.score for c in top) / len(top)
                logger.info("  News score: %s from %d articles, top catalyst: %s",
                            avg_news_score, len(catalysts), catalysts[0].title[:60] if catalysts else "N/A")
        except Exception as exc:
            logger.warning("  News sentiment failed: %s", exc)

        # ═════════════════════════════════════════════
        # AGGREGATE: Compute final signal from all tools
        # ═════════════════════════════════════════════
        logger.info("=== COMPUTING FINAL SIGNAL ===")
        evidence_chunks = evidence.get("chunks", []) if isinstance(evidence, dict) else []

        score = compute_final_signal(
            risk_severity_avg=risk_avg,
            tone_delta=tone_delta,
            valuation_gap_pct=valuation_gap_pct,
            revenue_growth_yoy=revenue_growth_yoy,
            news_direction_score=avg_news_score,
            evidence_count=len(evidence_chunks),
            contradiction_penalty=0.0,
        )
        score_obj = to_dict(score)
        decision_action = signal_action_from_score(
            signal_score=float(score_obj.get("signal_score", 0.0)),
            confidence=float(score_obj.get("confidence", 0.0)),
        )

        logger.info("  FINAL component_scores: %s", score_obj.get("component_scores"))
        logger.info("  DECISION: %s | strength: %s | confidence: %s",
                     decision_action, score_obj.get("signal_score"), score_obj.get("confidence"))

        report = build_demo_signal_report(
            ticker=req.ticker,
            fiscal_year=req.fiscal_year,
            company=None,
            score_obj=score_obj,
            decision_action=decision_action,
            top_risks=top_risks,
            tone_trend=tone_trend,
            valuation_summary=valuation_summary,
            news_summary=news_summary,
            citations=[],
        )

        return {
            "ok": True,
            "action": "answer",
            "mode": "decision_analysis",
            "hackathon_signal_score": score_obj,
            "hackathon_signal_decision": {
                "action": decision_action,
                "policy": "ACT if confidence>=0.55 and score>=0.35; WATCH if score>=0.10; else NO_ACT",
            },
            "hackathon_signal_report": report.to_dict(),
            "hackathon_signal_markdown": report.to_markdown(),
            "evidence": {"chunks": evidence_chunks},
            "packed_context": filing_ctx,
            "tools_used": {
                "risk": {"tool": "extract_risk_signals", "source": "SEC Filing Item 1A", "risk_count": len(risk_signals)},
                "tone": {"tool": "compare_tone (FinBERT)", "source": "Earnings Transcripts", "delta": tone_delta},
                "valuation": {"tool": "run_dcf", "source": "yfinance + SEC Filings", "gap_pct": valuation_gap_pct},
                "growth": {"tool": "yfinance.financials", "source": "Yahoo Finance", "yoy": revenue_growth_yoy},
                "news": {"tool": "classify_news_catalysts", "source": "NewsAPI", "avg_score": avg_news_score},
            },
        }

    except Exception as e:
        logger.exception("Decision analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}
