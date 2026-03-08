import streamlit as st
import time
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict
from dataclasses import asdict, is_dataclass

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
from orchestrator import FinancialOrchestrator, OrchestratorConfig
from verification import Mode

try:
    from knowledge_base import TICKERS as KB_TICKERS, TARGET_FYS as KB_TARGET_FYS
except Exception:
    KB_TICKERS = []
    KB_TARGET_FYS = []

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
ALL_MODES: list[str] = ["auto"] + list(Mode.__args__)  # type: ignore[attr-defined]
UI_TICKERS: list[str] = sorted({str(t).strip().upper() for t in KB_TICKERS if str(t).strip()})
UI_FISCAL_YEARS: list[int] = sorted({int(y) for y in KB_TARGET_FYS if isinstance(y, int)})


# ---------------------------
# Styling (Bloomberg-ish)
# ---------------------------
TERMINAL_CSS = """
<style>
/* Layout + fonts */
html, body, [class*="css"]  { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: 0.3px; }

/* “Terminal” panels */
.term-panel {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 10px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
}
.term-kpi {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 10px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.03);
}
.term-muted { opacity: 0.75; }
.term-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.04);
  font-size: 12px;
  margin-right: 6px;
}

/* Streamlit tweaks */
[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.10); }
div[data-testid="stMetricValue"] { font-family: inherit; }
</style>
"""


# ---------------------------
# Orchestrator init (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_orchestrator() -> FinancialOrchestrator:
    base_dir = Path(__file__).resolve().parent
    audit_log = base_dir / "logs" / "audit.jsonl"

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    primary_model = os.environ.get("LOCAL_SMALL_MODEL", DEFAULT_PRIMARY_MODEL)
    fallback_model = os.environ.get("LOCAL_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)

    llm_client = build_local_llm_client(
        primary_model=primary_model,
        fallback_model=fallback_model,
        base_url=ollama_url,
    )

    cfg = OrchestratorConfig(
        base_dir=base_dir,
        audit_log_path=audit_log,
        small_model_name=os.environ.get("LOCAL_SMALL_MODEL", "small"),
        large_model_name=os.environ.get("LOCAL_LARGE_MODEL", "large"),
        known_tickers={t.upper() for t in KB_TICKERS if isinstance(t, str) and t.strip()} or None,
        market_provider=YahooFinanceMarketDataProvider(),
    )
    return FinancialOrchestrator(cfg=cfg, llm_client=llm_client)


# ---------------------------
# Helpers
# ---------------------------
def _safe_to_dict(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    return x

def _extract_final_answer(res: Dict[str, Any]) -> str:
    """Extract the core RAG answer (without signal layer — signals rendered separately)."""
    result = res.get("result", {}) or {}
    ans = result.get("final_answer", "")
    assumptions = res.get("assumptions", []) or []
    assumption_text = ""
    if assumptions:
        assumption_text = "\n\nAssumption(s): " + " ".join(str(a) for a in assumptions)

    if isinstance(ans, str) and ans.strip():
        text = ans.strip()
        sep = "--- Investment Signal:"
        if sep in text:
            text = text[:text.index(sep)].rstrip()
        return (text + assumption_text) if text else f"No final answer generated.{assumption_text}"

    reason = res.get("reason")
    if isinstance(reason, str) and reason.strip():
        return f"No final answer generated. Reason: {reason}{assumption_text}"
    return f"No final answer generated.{assumption_text}"

def _health_check(base_dir: Path) -> Dict[str, Any]:
    index_dir = base_dir / "index"
    required = [
        index_dir / "chunks.parquet",
        index_dir / "bm25.pkl",
        index_dir / "faiss.index",
        index_dir / "tables.parquet",
        index_dir / "table_bm25.pkl",
        index_dir / "table_faiss.index",
        base_dir / "data" / "xbrl_companyfacts",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {
        "ok": len(missing) == 0,
        "missing": missing,
    }

def _flatten_evidence(res: Dict[str, Any]) -> Dict[str, Any]:
    ev = res.get("evidence", {}) or {}
    narrative = ev.get("narrative", {}) if isinstance(ev, dict) else {}
    tables = ev.get("tables", {}) if isinstance(ev, dict) else {}
    xbrl = ev.get("xbrl", {}) if isinstance(ev, dict) else {}
    return {"narrative": narrative, "tables": tables, "xbrl": xbrl}

def _read_audit_log(path: Path, tail: int = 300) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        if tail and len(lines) > tail:
            lines = lines[-tail:]
        return "\n".join(lines)
    except Exception:
        return ""

def _available_fiscal_years(base_dir: Path) -> list[int]:
    years: set[int] = set()
    fy_re = re.compile(r"_FY(\d{4})_")

    for rel_dir in ("data/sections", "data/tables", "data/raw_html"):
        d = base_dir / rel_dir
        if not d.exists():
            continue
        for p in d.glob("*"):
            m = fy_re.search(p.name)
            if not m:
                continue
            try:
                years.add(int(m.group(1)))
            except Exception:
                continue

    if years:
        return sorted(years)
    return UI_FISCAL_YEARS


# ---------------------------
# Page
# ---------------------------
def main():
    st.set_page_config(page_title="Finance Analyst Terminal", layout="wide")
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []  # list of normalized runs

    orch = None
    base_dir = Path(__file__).resolve().parent

    # Sidebar (System/Session)
    with st.sidebar:
        st.title("System")

        st.markdown("---")
        st.subheader("System health")
        hc = _health_check(base_dir)
        if hc["ok"]:
            st.success("Index files: READY")
        else:
            st.error("Index files: MISSING")
            st.code("\n".join(hc["missing"]))

        # Ollama status check
        ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        try:
            import requests as _req
            _r = _req.get(f"{ollama_url}/api/tags", timeout=3)
            ollama_ok = _r.status_code == 200
        except Exception:
            ollama_ok = False
        if ollama_ok:
            st.success(f"Ollama: CONNECTED ({ollama_url})")
        else:
            st.error(f"Ollama: NOT REACHABLE ({ollama_url})")

        st.markdown("---")
        st.subheader("Run history")
        st.caption(f"Session runs: {len(st.session_state.history)}")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.rerun()

    # Header
    st.title("Finance Analyst Terminal")
    dynamic_fiscal_years = _available_fiscal_years(base_dir)

    # Top control section (requested flow)
    st.subheader("Controls")
    c_ticker, c_mode, c_fy, c_strict = st.columns([2, 3, 2, 3])
    with c_ticker:
        ticker_options = ["(auto)"] + UI_TICKERS if UI_TICKERS else ["(auto)"]
        ticker_choice = st.selectbox("Ticker", ticker_options, index=0)
        ticker = None if ticker_choice == "(auto)" else ticker_choice
    with c_mode:
        mode = st.selectbox("Mode", ALL_MODES, index=0)
    with c_fy:
        fy_options = ["(auto)"] + [str(y) for y in dynamic_fiscal_years] if dynamic_fiscal_years else ["(auto)"]
        fy_choice = st.selectbox("Fiscal year", fy_options, index=0)
        fiscal_year = None if fy_choice == "(auto)" else int(fy_choice)
    with c_strict:
        strictness = st.slider("Evidence strictness", 0, 100, 70, 1)

    # Query input row
    col_q, col_run = st.columns([7, 1])
    with col_q:
        question = st.text_input("Query", value="", placeholder="e.g., What was AAPL EPS in FY2024?")
    with col_run:
        run = st.button("Run", use_container_width=True)

    # Initialize orchestrator lazily so health can show even if key missing
    try:
        orch = get_orchestrator()
    except Exception as e:
        st.error(f"Orchestrator init failed: {type(e).__name__}: {e}")
        st.stop()

    # Execute
    if run:
        if not isinstance(question, str) or not question.strip():
            st.warning("Enter a query.")
        else:
            # Pass controls as structured hints to avoid polluting planner text.
            forced_mode = None if mode == "auto" else mode
            query = question.strip()

            t0 = time.time()
            try:
                result = orch.answer(
                    query,
                    market_inputs=None,
                    auto_fetch_market=True,
                    forced_mode=forced_mode,
                    ui_intent=mode,
                    ui_ticker=ticker,
                    ui_fiscal_year=fiscal_year,
                    evidence_strictness=strictness,
                )
                latency_s = time.time() - t0
            except TypeError as e:
                # Streamlit can keep a stale cached orchestrator instance across edits.
                # If the old instance doesn't support new kwargs, clear cache and retry once.
                if "unexpected keyword argument" in str(e):
                    get_orchestrator.clear()
                    orch = get_orchestrator()
                    result = orch.answer(
                        query,
                        market_inputs=None,
                        auto_fetch_market=True,
                        forced_mode=forced_mode,
                        ui_intent=mode,
                        ui_ticker=ticker,
                        ui_fiscal_year=fiscal_year,
                        evidence_strictness=strictness,
                    )
                    latency_s = time.time() - t0
                else:
                    st.error(f"Request failed: {type(e).__name__}: {e}")
                    st.stop()
            except Exception as e:
                st.error(f"Request failed: {type(e).__name__}: {e}")
                st.stop()

            # Normalize for UI
            routing = result.get("routing", {}) or {}
            gate = result.get("gate", {}) or {}
            action = result.get("action", "abstain")
            final_answer = _extract_final_answer(result)
            evidence = _flatten_evidence(result)
            packed_context = result.get("packed_context", "") or ""
            if not isinstance(packed_context, str):
                packed_context = str(packed_context)

            record = {
                "ts": int(time.time()),
                "question": question.strip(),
                "ticker": ticker,
                "mode": mode,
                "fiscal_year": fiscal_year,
                "strictness": strictness,
                "run_id": result.get("run_id", "—"),
                "action": action,
                "latency_s": round(latency_s, 2),
                "routing": routing,
                "gate": gate,
                "final_answer": final_answer,
                "evidence": evidence,
                "packed_context": packed_context,
                "raw": result,
            }
            st.session_state.history.insert(0, record)

    # Show latest (if any)
    if st.session_state.history:
        r = st.session_state.history[0]

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Run ID", r["run_id"])
        c2.metric("Action", r["action"])
        c3.metric("Model", (r["routing"] or {}).get("model", "—"))
        c4.metric("Gate score", (r["gate"] or {}).get("score", "—"))
        c5.metric("Latency (s)", r["latency_s"])

        # Answer + Signal panel (unified)
        st.markdown('<div class="term-panel">', unsafe_allow_html=True)
        st.markdown(
            f'<span class="term-badge">ticker={(r.get("ticker") or "—")}</span>'
            f'<span class="term-badge">mode={r["mode"]}</span>'
            f'<span class="term-badge">fy={(r.get("fiscal_year") or "—")}</span>'
            f'<span class="term-badge">strict={r["strictness"]}</span>'
            f'<span class="term-badge">router_mode={(r["routing"] or {}).get("mode","—")}</span>',
            unsafe_allow_html=True
        )
        st.subheader("Answer")
        st.write(r["final_answer"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Signal Dashboard rendered inline after the answer
        raw = r.get("raw", {}) or {}
        if raw.get("hackathon_signal_report"):
            render_signal_inline(raw)

        # Evidence Explorer
        st.subheader("Evidence Explorer")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Top Chunks", "Tables", "XBRL Facts", "Packed Context", "Audit Trace"])

        with tab1:
            narr = (r["evidence"] or {}).get("narrative", {}) or {}
            records = narr.get("records", []) or []
            if not records:
                st.info("No narrative chunks returned.")
            else:
                for i, rec in enumerate(records[:20], 1):
                    title = rec.get("doc_title") or rec.get("source") or f"Chunk {i}"
                    with st.expander(f"{i}. {title}", expanded=(i <= 3)):
                        st.caption(rec.get("source_url") or rec.get("url") or "")
                        st.code((rec.get("text") or "")[:4000])

        with tab2:
            tables = (r["evidence"] or {}).get("tables", {}) or {}
            trecs = tables.get("records", []) or []
            if not trecs:
                st.info("No tables returned.")
            else:
                for i, rec in enumerate(trecs[:20], 1):
                    title = rec.get("title") or rec.get("doc_title") or f"Table {i}"
                    with st.expander(f"{i}. {title}", expanded=(i <= 2)):
                        st.caption(rec.get("source_url") or rec.get("url") or "")
                        st.code((rec.get("text") or "")[:6000])

        with tab3:
            xbrl = (r["evidence"] or {}).get("xbrl", {}) or {}
            facts = xbrl.get("facts", []) or []
            if not facts:
                st.info("No XBRL facts returned.")
            else:
                st.json(facts[:200])

        with tab4:
            pc = r.get("packed_context", "") or ""
            if not pc.strip():
                st.info("No packed context.")
            else:
                st.code(pc[:20000])
                st.download_button(
                    "Download packed_context.txt",
                    data=pc.encode("utf-8"),
                    file_name=f"packed_context_{r['run_id']}.txt",
                    mime="text/plain",
                )

        with tab5:
            st.json({
                "routing": _safe_to_dict(r.get("routing", {})),
                "gate": _safe_to_dict(r.get("gate", {})),
                "validation_errors": r.get("raw", {}).get("validation_errors", []),
            })
            # Tail audit log
            audit_path = base_dir / "logs" / "audit.jsonl"
            audit_tail = _read_audit_log(audit_path, tail=200)
            if audit_tail.strip():
                st.caption("audit.jsonl (tail)")
                st.code(audit_tail)
                st.download_button(
                    "Download full audit.jsonl",
                    data=audit_path.read_bytes() if audit_path.exists() else b"",
                    file_name="audit.jsonl",
                    mime="application/json",
                )

    else:
        st.info("No runs yet. Enter a query and click Run.")

def render_signal_inline(resp: Dict[str, Any]) -> None:
    """Render signal layer output as part of the main answer flow."""
    report = resp.get("hackathon_signal_report") or {}
    score = resp.get("hackathon_signal_score") or {}

    if not report:
        return

    st.markdown("---")
    st.subheader("Investment Signal")

    rec = report.get("recommendation", "N/A")
    rec_colors = {"BUY": "green", "HOLD": "orange", "CAUTIOUS": "orange", "AVOID": "red"}
    rec_color = rec_colors.get(rec, "gray")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="term-kpi"><b>Recommendation</b><br>'
            f'<span style="color:{rec_color};font-size:1.4em;font-weight:bold">{rec}</span></div>',
            unsafe_allow_html=True,
        )
    with c2:
        strength = float(report.get("signal_strength", 0.0))
        st.metric("Signal Strength", f"{strength:+.2f}")
    with c3:
        confidence = float(report.get("confidence", 0.0))
        st.metric("Confidence", f"{confidence:.0%}")
    with c4:
        comp = score.get("component_scores", {})
        dominant = max(comp, key=lambda k: abs(comp[k])) if comp else "—"
        st.metric("Dominant Factor", dominant.title())

    comp = score.get("component_scores", {})
    if comp:
        cols = st.columns(len(comp))
        for col, (k, v) in zip(cols, comp.items()):
            bar_color = "green" if v > 0 else "red" if v < 0 else "gray"
            col.markdown(
                f'<div class="term-kpi" style="text-align:center">'
                f'<span class="term-muted" style="font-size:0.8em">{k.upper()}</span><br>'
                f'<span style="color:{bar_color};font-size:1.2em;font-weight:bold">{v:+.2f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([3, 2])

    with left:
        findings = report.get("key_findings", [])
        if findings:
            st.markdown("**Key Findings**")
            for f in findings:
                st.write(f"- {f}")

        risks = report.get("top_risks", [])
        if risks:
            st.markdown("**Top Risks**")
            for risk in risks[:5]:
                st.write(
                    f"- **{risk.get('category')}** — severity: {risk.get('severity')} | count: {risk.get('count')}"
                )
                for s in (risk.get("snippets") or [])[:2]:
                    st.caption(s)

    with right:
        tone = report.get("tone_trend", {})
        if tone and tone.get("direction"):
            st.markdown("**Tone Trend**")
            direction = tone.get("direction", "flat")
            delta = tone.get("delta", 0.0)
            arrow = {"improving": "arrow_up", "worsening": "arrow_down"}.get(direction, "left_right_arrow")
            st.write(f":{arrow}: {direction.title()} (delta: {delta:+.2f})")

        val = report.get("valuation_summary", {})
        gap = val.get("valuation_gap_pct")
        growth = val.get("revenue_growth_yoy")
        if gap is not None or growth is not None:
            st.markdown("**Valuation**")
            if gap is not None:
                st.write(f"- Valuation gap: {gap:+.1%}")
            if growth is not None:
                st.write(f"- Revenue growth YoY: {growth:+.1%}")

        news = report.get("news_summary", [])
        if news:
            st.markdown("**Recent Catalysts**")
            for n in news[:5]:
                direction = n.get("direction", "neutral")
                icon = {"positive": ":green_circle:", "negative": ":red_circle:"}.get(direction, ":white_circle:")
                st.write(f"{icon} {n.get('title', '')}")

    with st.expander("Full Analyst Report (Markdown)"):
        st.markdown(resp.get("hackathon_signal_markdown", ""))
if __name__ == "__main__":
    main()
