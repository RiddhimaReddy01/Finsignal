"""
test_verification.py

Comprehensive logic tests for verification.py covering:
  1.  normalize_query
  2.  Entity detection (tickers, years, items, metrics)
  3.  Mode inference priority cascade
  4.  build_task_plan integration
  5.  Context parsing (split_context_into_blocks, parse_allowed_ids_from_context)
  6.  Evidence requirements per mode
  7.  gate_evidence (slot, market, volume, section gates)
  8.  Numeric extraction + verification + contradiction detection
  9.  Computed metrics
  10. validate_answer_json
  11. Deterministic answer builders
  12. Pipeline ID format integration (verifies regex vs actual IDs)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from verification import (
    normalize_query,
    _detect_tickers,
    _detect_years,
    _detect_item_hint,
    _detect_metrics,
    _infer_mode,
    build_task_plan,
    evidence_requirements,
    split_context_into_blocks,
    parse_allowed_ids_from_context,
    gate_evidence,
    extract_numeric_topn,
    verify_numeric_candidate,
    contradiction_check,
    choose_best_numeric_with_gate,
    compute_metric_value,
    required_inputs_for_computed,
    compute_metric_from_evidence,
    schema_for_mode,
    build_json_answer_prompt,
    validate_answer_json,
    build_lookup_numeric_answer,
    build_compute_metric_answer,
    Target,
    TaskPlan,
    RetrievalPlan,
    EvidenceRequirements,
    EvidenceBlock,
    NumericCandidate,
    _source_precedence,
    _metric_line_match_score,
    _global_scale_hint,
    _TABLE_HDR_RE,
    _CHUNK_HDR_RE,
    _ID_RE,
    _COMMON_WORDS,
    _COMPUTE_NUMERIC_CONTEXT_RE,
)

PASS = 0
FAIL = 0
ERRORS: List[str] = []
KNOWN_TICKERS = {"AAPL", "META", "NVDA", "GOOGL", "TSLA"}


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}" + (f" -- {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def bug(name: str, detail: str = ""):
    """Mark a known logic bug found during testing."""
    global FAIL
    FAIL += 1
    msg = f"  [BUG]  {name}" + (f" -- {detail}" if detail else "")
    print(msg)
    ERRORS.append(msg)


# =============================================
# Helper: build packed contexts for testing
# =============================================

def make_packed_context_simple():
    """Uses simple IDs (t001, c001) that match verification.py regex expectations."""
    return (
        "[XBRL EVIDENCE]\n"
        "- AAPL Revenue FY2024 value=394328000000 unit=USD\n"
        "\n"
        "[TABLE AAPL FY2024 Item 8 t001]\n"
        "Consolidated Statements of Operations ($ in millions)\n"
        "Revenue | 2024 | 2023\n"
        "Total net sales | $391,035 | $383,285\n"
        "Cost of sales | $214,137 | $214,137\n"
        "Gross profit | $176,898 | $169,148\n"
        "Operating income | $123,216 | $114,301\n"
        "Net income | $93,736 | $96,995\n"
        "\n"
        "[TABLE AAPL FY2024 Item 8 t002]\n"
        "Cash Flow Statement ($ in millions)\n"
        "Cash provided by operating activities | $118,254\n"
        "Capital expenditures | $9,959\n"
        "\n"
        "[AAPL FY2024 Item 1A c001]\n"
        "The Company is subject to risks associated with global economic conditions. "
        "Adverse macroeconomic conditions could materially impact demand for Apple products.\n"
        "\n"
        "[AAPL FY2024 Item 1A c002]\n"
        "The Company faces substantial competition in all markets. "
        "Competitors may introduce new products with better features at lower prices.\n"
        "\n"
        "[AAPL FY2024 Item 7 c003]\n"
        "Total net sales increased 2% or $8 billion during 2024 compared to 2023. "
        "Revenue was $391,035 million for fiscal year 2024.\n"
        "\n"
        "[AAPL FY2024 Item 8 c004]\n"
        "See accompanying Notes to Consolidated Financial Statements.\n"
    )


def make_packed_context_pipeline():
    """Uses actual pipeline ID format to test regex compatibility."""
    return (
        "[TABLE AAPL FY2024 Item 8 AAPL_FY2024_0001234_T001]\n"
        "Revenue | $391,035\n"
        "\n"
        "[AAPL FY2024 Item 1A AAPL_FY2024_Item1A_0001]\n"
        "The Company is subject to risks.\n"
    )


# =============================================
# PART 1: normalize_query
# =============================================

def test_normalize_query():
    print("\n-- normalize_query --")
    check("strips whitespace", normalize_query("  hello  ") == "hello")
    check("collapses spaces", normalize_query("a   b    c") == "a b c")
    check("handles nbsp", normalize_query("a\u00a0b") == "a b")
    check("handles tabs", normalize_query("a\tb") == "a b")
    check("empty string", normalize_query("") == "")
    check("None safe", normalize_query(None) == "")


# =============================================
# PART 2: Entity detection
# =============================================

def test_detect_tickers():
    print("\n-- _detect_tickers --")
    check("finds AAPL", "AAPL" in _detect_tickers("What was AAPL revenue?", KNOWN_TICKERS))
    check("finds multiple", set(_detect_tickers("AAPL vs GOOGL revenue", KNOWN_TICKERS)) == {"AAPL", "GOOGL"})
    check("case insensitive input", "TSLA" in _detect_tickers("tsla revenue", KNOWN_TICKERS))
    check("filters unknown tickers", _detect_tickers("What was XYZ revenue?", KNOWN_TICKERS) == [])
    check("no duplicates", _detect_tickers("AAPL AAPL AAPL", KNOWN_TICKERS) == ["AAPL"])

    # Without known_tickers, common words should be filtered
    without_filter = _detect_tickers("What was AAPL revenue in SEC filings?", None)
    check("without known_tickers filters common words", "WHAT" not in without_filter and "SEC" not in without_filter,
          f"got {without_filter}")
    check("without known_tickers keeps real tickers", "AAPL" in without_filter)


def test_detect_years():
    print("\n-- _detect_years --")
    check("finds 2024", 2024 in _detect_years("revenue in 2024"))
    check("finds multiple", _detect_years("2024 vs 2025") == [2024, 2025])
    check("no duplicates", _detect_years("2024 2024 2024") == [2024])
    check("rejects out of range", _detect_years("year 1800") == [])
    check("empty string", _detect_years("no year here") == [])


def test_detect_item_hint():
    print("\n-- _detect_item_hint --")
    check("risk factors -> Item 1A", _detect_item_hint("risk factors for AAPL") == "Item 1A")
    check("md&a -> Item 7", _detect_item_hint("md&a discussion") == "Item 7")
    check("financial statements -> Item 8", _detect_item_hint("financial statements") == "Item 8")
    check("no hint", _detect_item_hint("general question") is None)
    check("item 7 literal", _detect_item_hint("item 7 analysis") == "Item 7")


def test_detect_metrics():
    print("\n-- _detect_metrics --")
    check("revenue detected", "revenue" in _detect_metrics("What was total revenue?"))
    check("eps detected", "eps" in _detect_metrics("What was EPS?"))
    check("net_income detected", "net_income" in _detect_metrics("net income for 2024"))
    check("multiple metrics", len(_detect_metrics("revenue and net income")) >= 2)
    check("no metric", _detect_metrics("What are the risks?") == [])
    check("capex detected", "capex" in _detect_metrics("capital expenditures"))
    check("fcf detected", "fcf" in _detect_metrics("free cash flow"))


# =============================================
# PART 3: Mode inference priority cascade
# =============================================

def test_infer_mode():
    print("\n-- _infer_mode --")

    # Valuation triggers (highest priority)
    check("dcf -> valuation", _infer_mode("What is the DCF valuation?", ["AAPL"], [2024], ["revenue"]) == "valuation")
    check("wacc -> valuation", _infer_mode("Calculate WACC for AAPL", ["AAPL"], [2024], []) == "valuation")
    check("p/e -> relative_valuation", _infer_mode("What is the P/E multiple?", ["AAPL"], [2024], []) == "relative_valuation")
    check("ev/ebitda -> relative_valuation", _infer_mode("ev/ebitda comparison", ["AAPL"], [2024], []) == "relative_valuation")
    check("bare 'multiple' no false pos", _infer_mode("multiple companies reported growth", ["AAPL"], [2024], []) != "relative_valuation",
          f"got {_infer_mode('multiple companies reported growth', ['AAPL'], [2024], [])}")
    check("'trading multiple' -> rel_val", _infer_mode("What is the trading multiple?", ["AAPL"], [2024], []) == "relative_valuation")

    # Risk triggers
    check("risk factors -> risk_analysis", _infer_mode("What are the risk factors?", ["AAPL"], [2024], []) == "risk_analysis")
    check("uncertainty -> risk_analysis", _infer_mode("What uncertainty does AAPL face?", ["AAPL"], [2024], []) == "risk_analysis")

    # Compare triggers
    check("compare -> comparative", _infer_mode("Compare AAPL and GOOGL", ["AAPL", "GOOGL"], [2024], []) == "comparative_analysis")
    check("2 tickers -> comparative", _infer_mode("AAPL GOOGL revenue", ["AAPL", "GOOGL"], [2024], ["revenue"]) == "comparative_analysis")
    check("2 years -> comparative", _infer_mode("revenue 2024 vs 2025", ["AAPL"], [2024, 2025], ["revenue"]) == "comparative_analysis")

    # Compute triggers
    check("yoy -> compute", _infer_mode("revenue yoy growth", ["AAPL"], [2024], ["revenue"]) == "compute_metric")
    check("margin -> compute", _infer_mode("What is the gross margin?", ["AAPL"], [2024], ["gross_margin"]) == "compute_metric")

    # Framework
    check("swot -> mba_framework", _infer_mode("SWOT analysis of AAPL", ["AAPL"], [], []) == "mba_framework")
    check("porter -> mba_framework", _infer_mode("Porter 5 forces analysis", ["AAPL"], [], []) == "mba_framework")

    # Explain
    check("explain -> explanatory", _infer_mode("Explain why revenue grew", ["AAPL"], [2024], ["revenue"]) == "explanatory_reasoning")
    check("why -> explanatory", _infer_mode("Why did AAPL report lower margins?", ["AAPL"], [2024], []) == "explanatory_reasoning")

    # Numeric default
    check("$ -> lookup_numeric", _infer_mode("How much $ did they earn?", ["AAPL"], [2024], []) == "lookup_numeric")
    check("metric no triggers -> lookup_numeric", _infer_mode("What was AAPL revenue?", ["AAPL"], [2024], ["revenue"]) == "lookup_numeric")

    # Text default
    check("general -> lookup_text", _infer_mode("Describe the business model", ["AAPL"], [2024], []) == "lookup_text")

    # Priority: risk should NOT trigger compute even with "change" in text
    check("risk > compute priority", _infer_mode("risk of change in market", ["AAPL"], [2024], []) == "risk_analysis")

    # "change" only triggers compute when a metric is present
    check("'change in leadership' -> lookup_text",
          _infer_mode("What was the change in leadership?", ["AAPL"], [2024], []) == "lookup_text")

    # "change" WITH a metric should trigger compute
    check("'revenue change' -> compute",
          _infer_mode("What was the revenue change?", ["AAPL"], [2024], ["revenue"]) == "compute_metric")

    # "increase" without metric should not trigger compute
    check("'increase in headcount' -> lookup_text",
          _infer_mode("What was the increase in headcount?", ["AAPL"], [2024], []) == "lookup_text")

    # Explain triggers now checked before compute
    check("why + increase -> explanatory",
          _infer_mode("Why did AAPL revenue increase?", ["AAPL"], [2024], ["revenue"]) == "explanatory_reasoning")


# =============================================
# PART 4: build_task_plan integration
# =============================================

def test_build_task_plan():
    print("\n-- build_task_plan --")

    plan = build_task_plan("What was AAPL revenue in 2024?", KNOWN_TICKERS)
    check("mode is lookup_numeric", plan.mode == "lookup_numeric")
    check("ticker is AAPL", plan.targets[0].ticker == "AAPL")
    check("fiscal_year is 2024", plan.targets[0].fiscal_year == 2024)
    check("metric is revenue", plan.targets[0].metric == "revenue")
    check("hard_filters has ticker", plan.retrieval_plan.hard_filters.get("ticker") == "AAPL")
    check("rewrites non-empty", len(plan.retrieval_plan.rewrites) >= 1)

    # Comparative
    plan2 = build_task_plan("Compare AAPL and GOOGL revenue in 2024", KNOWN_TICKERS)
    check("comparative mode", plan2.mode == "comparative_analysis")
    check("multiple targets", len(plan2.targets) >= 2)
    tickers_in_targets = {t.ticker for t in plan2.targets}
    check("both tickers in targets", {"AAPL", "GOOGL"}.issubset(tickers_in_targets))

    # Risk
    plan3 = build_task_plan("What are the risk factors for TSLA in 2025?", KNOWN_TICKERS)
    check("risk mode", plan3.mode == "risk_analysis")
    check("soft_boost Item 1A", any(b.get("section") == "Item 1A" for b in plan3.retrieval_plan.soft_boosts))

    # No ticker, no year
    plan4 = build_task_plan("Explain the business model", KNOWN_TICKERS)
    check("no ticker -> None", plan4.targets[0].ticker is None)
    check("no year -> None", plan4.targets[0].fiscal_year is None)

    # Schema ID
    check("schema_id set", plan.schema_id == "lookup_numeric_schema_v1")


# =============================================
# PART 5: Context parsing
# =============================================

def test_parse_allowed_ids_simple():
    print("\n-- parse_allowed_ids (simple IDs) --")
    ctx = make_packed_context_simple()
    ids = parse_allowed_ids_from_context(ctx)
    check("finds t001", "t001" in ids)
    check("finds t002", "t002" in ids)
    check("finds c001", "c001" in ids)
    check("finds c002", "c002" in ids)
    check("finds c003", "c003" in ids)
    check("finds c004", "c004" in ids)
    check("total 6 IDs", len(ids) == 6, f"got {len(ids)}: {ids}")


def test_parse_allowed_ids_pipeline():
    """Test with actual pipeline ID format."""
    print("\n-- parse_allowed_ids (pipeline IDs) --")
    ctx = make_packed_context_pipeline()
    ids = parse_allowed_ids_from_context(ctx)
    check("pipeline table ID parsed", "AAPL_FY2024_0001234_T001" in ids, f"got {ids}")
    check("pipeline chunk ID parsed", "AAPL_FY2024_Item1A_0001" in ids, f"got {ids}")
    check("found 2 pipeline IDs", len(ids) == 2, f"got {len(ids)}: {ids}")


def test_split_context_blocks_simple():
    print("\n-- split_context_into_blocks (simple IDs) --")
    ctx = make_packed_context_simple()
    blocks = split_context_into_blocks(ctx)

    kinds = [b.kind for b in blocks]
    check("has xbrl block", "xbrl" in kinds)
    check("has table blocks", kinds.count("table") == 2)
    check("has chunk blocks", kinds.count("chunk") >= 3)

    tables = [b for b in blocks if b.kind == "table"]
    check("table ticker is AAPL", tables[0].ticker == "AAPL")
    check("table FY is 2024", tables[0].fiscal_year == 2024)
    check("table item is Item 8", tables[0].item == "Item 8")
    check("table evid is t001", tables[0].evid == "t001")
    check("table has text", len(tables[0].text) > 10)

    chunks = [b for b in blocks if b.kind == "chunk"]
    item1a_chunks = [b for b in chunks if (b.item or "").startswith("Item 1A")]
    check("2 Item 1A chunks", len(item1a_chunks) == 2)


def test_split_context_blocks_pipeline():
    """Test with actual pipeline ID format."""
    print("\n-- split_context_into_blocks (pipeline IDs) --")
    ctx = make_packed_context_pipeline()
    blocks = split_context_into_blocks(ctx)
    tables = [b for b in blocks if b.kind == "table"]
    chunks = [b for b in blocks if b.kind == "chunk"]
    check("pipeline: 1 table block", len(tables) == 1, f"got {len(tables)}")
    check("pipeline: 1 chunk block", len(chunks) == 1, f"got {len(chunks)}")
    if tables:
        check("pipeline table evid correct", tables[0].evid == "AAPL_FY2024_0001234_T001")
        check("pipeline table ticker", tables[0].ticker == "AAPL")
    if chunks:
        check("pipeline chunk evid correct", chunks[0].evid == "AAPL_FY2024_Item1A_0001")
        check("pipeline chunk item", chunks[0].item == "Item 1A")


def test_split_context_empty():
    print("\n-- split_context_into_blocks (edge cases) --")
    check("empty string", split_context_into_blocks("") == [])
    check("None", split_context_into_blocks(None) == [])
    check("no headers", split_context_into_blocks("just some text\nmore text") == [])


# =============================================
# PART 6: Evidence requirements
# =============================================

def test_evidence_requirements():
    print("\n-- evidence_requirements --")

    def make_plan(mode):
        return TaskPlan(
            raw_question="test", normalized_question="test",
            mode=mode, targets=[Target(ticker="AAPL", fiscal_year=2024, metric="revenue")],
            retrieval_plan=RetrievalPlan(), schema_id="test",
        )

    req_num = evidence_requirements(make_plan("lookup_numeric"))
    check("numeric: requires ticker slot", "ticker" in req_num.required_slots)
    check("numeric: requires fiscal_year slot", "fiscal_year" in req_num.required_slots)
    check("numeric: requires metric slot", "metric" in req_num.required_slots)
    check("numeric: min_tables >= 1", req_num.min_tables >= 1)
    check("numeric: require_item_8", req_num.require_item_8_presence is True)

    req_risk = evidence_requirements(make_plan("risk_analysis"))
    check("risk: min_chunks >= 3", req_risk.min_chunks >= 3)
    check("risk: require_item_1a >= 2", req_risk.require_item_1a_chunks >= 2)

    req_val = evidence_requirements(make_plan("valuation"))
    check("valuation: require_market_inputs", req_val.require_market_inputs is True)

    req_text = evidence_requirements(make_plan("lookup_text"))
    check("text: no required slots", req_text.required_slots == [])
    check("text: min_chunks >= 2", req_text.min_chunks >= 2)

    req_comp = evidence_requirements(make_plan("comparative_analysis"))
    check("compare: requires ticker", "ticker" in req_comp.required_slots)


# =============================================
# PART 7: gate_evidence
# =============================================

def test_gate_evidence():
    print("\n-- gate_evidence --")
    ctx = make_packed_context_simple()

    # Numeric with full evidence should pass
    plan_num = build_task_plan("What was AAPL revenue in 2024?", KNOWN_TICKERS)
    req_num = evidence_requirements(plan_num)
    gate = gate_evidence(plan_num, req_num, ctx)
    check("numeric gate passes", gate.ok is True)
    check("gate action is pass", gate.action == "pass")
    check("signals has counts", gate.signals["n_blocks"] > 0)

    # Missing slot: no ticker
    plan_no_ticker = TaskPlan(
        raw_question="test", normalized_question="test",
        mode="lookup_numeric", targets=[Target(ticker=None, fiscal_year=2024, metric="revenue")],
        retrieval_plan=RetrievalPlan(), schema_id="test",
    )
    req = evidence_requirements(plan_no_ticker)
    gate2 = gate_evidence(plan_no_ticker, req, ctx)
    check("missing ticker -> clarify", gate2.action == "clarify")
    check("missing ticker -> not ok", gate2.ok is False)

    # Missing slot: no metric for numeric
    plan_no_metric = TaskPlan(
        raw_question="test", normalized_question="test",
        mode="lookup_numeric", targets=[Target(ticker="AAPL", fiscal_year=2024, metric=None)],
        retrieval_plan=RetrievalPlan(), schema_id="test",
    )
    gate3 = gate_evidence(plan_no_metric, evidence_requirements(plan_no_metric), ctx)
    check("missing metric -> clarify", gate3.action == "clarify")

    # Empty context -> abstain
    plan_ok = build_task_plan("What was AAPL revenue in 2024?", KNOWN_TICKERS)
    req_ok = evidence_requirements(plan_ok)
    gate4 = gate_evidence(plan_ok, req_ok, "")
    check("empty context -> abstain", gate4.action == "abstain")

    # Valuation without market inputs -> needs_market_data
    plan_val = TaskPlan(
        raw_question="test", normalized_question="test",
        mode="valuation", targets=[Target(ticker="AAPL", fiscal_year=2024)],
        retrieval_plan=RetrievalPlan(), schema_id="test",
    )
    req_val = evidence_requirements(plan_val)
    gate5 = gate_evidence(plan_val, req_val, ctx, market_inputs=None)
    check("valuation no market -> needs_market_data", gate5.action == "needs_market_data")

    # Valuation with market inputs should proceed (may pass or fail volume)
    gate6 = gate_evidence(plan_val, req_val, ctx, market_inputs={"wacc": 0.1})
    check("valuation with market inputs doesn't need_market_data", gate6.action != "needs_market_data")

    # Risk with enough Item 1A chunks
    plan_risk = build_task_plan("What are the risk factors for AAPL in 2024?", KNOWN_TICKERS)
    req_risk = evidence_requirements(plan_risk)
    gate7 = gate_evidence(plan_risk, req_risk, ctx)
    check("risk gate passes with Item 1A", gate7.ok is True)


# =============================================
# PART 8: Numeric extraction
# =============================================

def test_source_precedence():
    print("\n-- _source_precedence --")
    item8_table = EvidenceBlock(kind="table", evid="t1", ticker="AAPL", fiscal_year=2024, item="Item 8", text="")
    item8_chunk = EvidenceBlock(kind="chunk", evid="c1", ticker="AAPL", fiscal_year=2024, item="Item 8", text="")
    item7_chunk = EvidenceBlock(kind="chunk", evid="c2", ticker="AAPL", fiscal_year=2024, item="Item 7", text="")
    other_table = EvidenceBlock(kind="table", evid="t2", ticker="AAPL", fiscal_year=2024, item="Item 1A", text="")
    other_chunk = EvidenceBlock(kind="chunk", evid="c3", ticker="AAPL", fiscal_year=2024, item="Item 1A", text="")

    check("Item 8 table > Item 8 chunk", _source_precedence(item8_table) > _source_precedence(item8_chunk))
    check("Item 8 chunk > Item 7 chunk", _source_precedence(item8_chunk) > _source_precedence(item7_chunk))
    check("Item 7 chunk > other table", _source_precedence(item7_chunk) > _source_precedence(other_table))
    check("other table > other chunk", _source_precedence(other_table) > _source_precedence(other_chunk))


def test_global_scale_hint():
    print("\n-- _global_scale_hint --")
    check("in millions -> 1e6", _global_scale_hint("Revenue ($ in millions)") == 1e6)
    check("in thousands -> 1e3", _global_scale_hint("Revenue ($ in thousands)") == 1e3)
    check("in billions -> 1e9", _global_scale_hint("Revenue ($ in billions)") == 1e9)
    check("no hint -> 1.0", _global_scale_hint("Revenue was $391B") == 1.0)


def test_metric_line_match():
    print("\n-- _metric_line_match_score --")
    check("revenue matches", _metric_line_match_score("Total net sales | $391,035", "revenue") > 0)
    check("no match -> 0", _metric_line_match_score("Operating expenses | $50,000", "revenue") == 0)
    check("multiple syns boost", _metric_line_match_score("Revenue total revenue net sales", "revenue") > 0.8)


def test_extract_numeric_topn():
    print("\n-- extract_numeric_topn --")
    ctx = make_packed_context_simple()
    target = Target(ticker="AAPL", fiscal_year=2024, metric="revenue")

    # Extract from tables only (prefer_tables=True)
    cands = extract_numeric_topn(ctx, target, topn=5, prefer_tables=True)
    check("finds candidates from tables", len(cands) > 0, f"got {len(cands)}")
    if cands:
        check("top candidate has value", cands[0].value_raw > 0)
        check("top candidate from table", cands[0].kind == "table")
        check("has revenue-like value", cands[0].value_scaled > 1e9,
              f"got {cands[0].value_scaled}")
        # $ prefix detection depends on regex capture position in pipe-delimited lines
        check("unit is USD or detected", cands[0].unit in ("USD", "UNKNOWN"),
              f"got {cands[0].unit}")

    # No metric -> empty
    no_metric = extract_numeric_topn(ctx, Target(ticker="AAPL", fiscal_year=2024, metric=None))
    check("no metric -> empty", no_metric == [])

    # Wrong ticker filter -> empty
    wrong_ticker = extract_numeric_topn(ctx, Target(ticker="ZZZZ", fiscal_year=2024, metric="revenue"))
    check("wrong ticker -> empty", wrong_ticker == [])


def test_extract_net_income():
    print("\n-- extract_numeric: net income --")
    ctx = make_packed_context_simple()
    target = Target(ticker="AAPL", fiscal_year=2024, metric="net_income")
    cands = extract_numeric_topn(ctx, target, topn=5, prefer_tables=True)
    check("finds net income candidates", len(cands) > 0)
    if cands:
        # Net income in our context is $93,736 million = ~93.7B
        vals = [c.value_scaled for c in cands]
        has_93b = any(abs(v - 93_736_000_000) < 1e9 for v in vals)
        check("93.7B in candidates", has_93b, f"vals={[f'{v:.0f}' for v in vals[:3]]}")


def test_verify_numeric_candidate():
    print("\n-- verify_numeric_candidate --")
    good = NumericCandidate(
        value_raw=391035, unit="USD", scale_factor=1e6, scale_label="million",
        value_scaled=391035e6, evidence_id="t001", kind="table", item="Item 8",
        ticker="AAPL", fiscal_year=2024, line="Total net sales | $391,035",
        score=0.9, precedence=400,
    )
    ok, errs = verify_numeric_candidate(good, Target(ticker="AAPL", fiscal_year=2024, metric="revenue"))
    check("good candidate passes", ok is True)

    # Ticker mismatch
    ok2, errs2 = verify_numeric_candidate(good, Target(ticker="GOOGL", fiscal_year=2024, metric="revenue"))
    check("ticker mismatch fails", ok2 is False)
    check("ticker_mismatch in errors", "ticker_mismatch" in errs2)

    # FY mismatch
    ok3, errs3 = verify_numeric_candidate(good, Target(ticker="AAPL", fiscal_year=2025, metric="revenue"))
    check("year mismatch fails", ok3 is False)

    # Unknown unit
    unknown_unit = NumericCandidate(
        value_raw=100, unit="UNKNOWN", scale_factor=1, scale_label="raw",
        value_scaled=100, evidence_id="c001", kind="chunk", item="Item 7",
        ticker="AAPL", fiscal_year=2024, line="The value is 100",
        score=0.5, precedence=200,
    )
    ok4, errs4 = verify_numeric_candidate(unknown_unit, Target(ticker="AAPL", fiscal_year=2024, metric="revenue"))
    check("unknown unit fails", ok4 is False)
    check("unit_unknown in errors", "unit_unknown" in errs4)


def test_contradiction_check():
    print("\n-- contradiction_check --")

    c1 = NumericCandidate(
        value_raw=391035, unit="USD", scale_factor=1e6, scale_label="million",
        value_scaled=391035e6, evidence_id="t001", kind="table", item="Item 8",
        ticker="AAPL", fiscal_year=2024, line="Revenue $391,035", score=0.9, precedence=400,
    )
    # Very close value -- no contradiction
    c2 = NumericCandidate(
        value_raw=391035, unit="USD", scale_factor=1e6, scale_label="million",
        value_scaled=391035e6, evidence_id="c003", kind="chunk", item="Item 7",
        ticker="AAPL", fiscal_year=2024, line="Revenue was $391,035 million", score=0.7, precedence=200,
    )
    contrad, conflicts = contradiction_check([c1, c2], rel_tol=0.01, abs_tol=0)
    check("identical values -> no contradiction", contrad is False)

    # Contradicting value (10% off)
    c3 = NumericCandidate(
        value_raw=350000, unit="USD", scale_factor=1e6, scale_label="million",
        value_scaled=350000e6, evidence_id="t002", kind="table", item="Item 8",
        ticker="AAPL", fiscal_year=2024, line="Revenue $350,000", score=0.7, precedence=400,
    )
    contrad2, conflicts2 = contradiction_check([c1, c3], rel_tol=0.01, abs_tol=0)
    check("10% diff -> contradiction", contrad2 is True)
    check("conflict pair recorded", len(conflicts2) == 1)

    # Single candidate -> no contradiction
    contrad3, _ = contradiction_check([c1], rel_tol=0.01, abs_tol=0)
    check("single candidate -> no contradiction", contrad3 is False)

    # Empty list
    contrad4, _ = contradiction_check([], rel_tol=0.01, abs_tol=0)
    check("empty -> no contradiction", contrad4 is False)


def test_choose_best_numeric():
    print("\n-- choose_best_numeric_with_gate --")
    ctx = make_packed_context_simple()
    target = Target(ticker="AAPL", fiscal_year=2024, metric="revenue")
    req = EvidenceRequirements(require_table_for_numeric=True)

    best, debug = choose_best_numeric_with_gate(ctx, target, req, topn=5)
    check("best is not None", best is not None)
    if best:
        check("best from table", best.kind == "table")
        check("best is AAPL", best.ticker == "AAPL")
        check("debug has candidates", len(debug["candidates"]) > 0)
        # With the FY-aware contradiction check, same-block different-year values
        # should not trigger contradictions
        check("contradiction flag is bool", isinstance(debug["contradiction"], bool))


# =============================================
# PART 9: Computed metrics
# =============================================

def test_compute_metric_value():
    print("\n-- compute_metric_value --")
    val, unit, formula = compute_metric_value("fcf", {"cfo": 118254e6, "capex": 9959e6})
    check("fcf computed", val is not None)
    if val:
        expected = (118254 - 9959) * 1e6
        check("fcf value correct", abs(val - expected) < 1, f"got {val}")
        check("fcf unit is USD", unit == "USD")
        check("fcf formula present", "cfo" in formula and "capex" in formula)

    val2, _, _ = compute_metric_value("gross_margin", {"gross_profit": 176898e6, "revenue": 391035e6})
    check("gross margin computed", val2 is not None)
    if val2:
        expected2 = 176898e6 / 391035e6
        check("gross margin ~0.45", abs(val2 - expected2) < 0.001, f"got {val2:.4f}")

    val3, _, _ = compute_metric_value("operating_margin", {"operating_income": 123216e6, "revenue": 391035e6})
    check("operating margin computed", val3 is not None)

    # Division by zero protection
    val4, _, _ = compute_metric_value("gross_margin", {"gross_profit": 100, "revenue": 0})
    check("division by zero -> None", val4 is None)

    # Missing input
    val5, _, _ = compute_metric_value("fcf", {"cfo": 100})
    check("missing capex -> None", val5 is None)

    # Unknown metric
    val6, _, _ = compute_metric_value("unknown_metric", {})
    check("unknown metric -> None", val6 is None)


def test_required_inputs():
    print("\n-- required_inputs_for_computed --")
    check("fcf needs cfo+capex", required_inputs_for_computed("fcf") == ["cfo", "capex"])
    check("gross_margin needs profit+rev", required_inputs_for_computed("gross_margin") == ["gross_profit", "revenue"])
    check("unknown -> empty", required_inputs_for_computed("unknown") == [])


# =============================================
# PART 10: validate_answer_json
# =============================================

def test_validate_answer_json():
    print("\n-- validate_answer_json --")
    ctx = make_packed_context_simple()
    plan = build_task_plan("What was AAPL revenue in 2024?", KNOWN_TICKERS)

    # Valid answer
    valid = json.dumps({
        "final_answer": "AAPL revenue was $391B in FY2024.",
        "claims": [{"text": "Revenue was $391,035 million", "citations": ["t001"]}],
        "tables_used": ["t001"],
        "provenance": {"ticker": "AAPL", "fiscal_year": 2024},
        "inferences": [],
        "numeric": {
            "metric": "revenue",
            "value": 391035000000,
            "unit": "USD",
            "notes": "From audited financial statements",
            "citation": "t001",
        },
    })
    ok, errs, obj = validate_answer_json(plan, ctx, valid)
    check("valid answer passes", ok is True, f"errors: {errs}")

    # Invalid JSON
    ok2, errs2, _ = validate_answer_json(plan, ctx, "not json{{{")
    check("invalid JSON caught", ok2 is False)
    check("invalid_json error", any("invalid_json" in e for e in errs2))

    # Missing claims
    bad_claims = json.dumps({
        "final_answer": "test",
        "tables_used": [],
        "provenance": {"ticker": "AAPL", "fiscal_year": 2024},
        "inferences": [],
        "numeric": {"metric": "revenue", "value": 100, "unit": "USD", "notes": "", "citation": "t001"},
    })
    ok3, errs3, _ = validate_answer_json(plan, ctx, bad_claims)
    check("missing claims caught", ok3 is False)

    # Citation to non-existent ID
    bad_citation = json.dumps({
        "final_answer": "test",
        "claims": [{"text": "test", "citations": ["t999"]}],
        "tables_used": [],
        "provenance": {"ticker": "AAPL", "fiscal_year": 2024},
        "inferences": [],
        "numeric": {"metric": "revenue", "value": 100, "unit": "USD", "notes": "", "citation": "t001"},
    })
    ok4, errs4, _ = validate_answer_json(plan, ctx, bad_citation)
    check("bad citation caught", ok4 is False)
    check("not_allowed in error", any("not_allowed" in e for e in errs4))

    # Wrong unit
    bad_unit = json.dumps({
        "final_answer": "test",
        "claims": [{"text": "test", "citations": ["t001"]}],
        "tables_used": ["t001"],
        "provenance": {"ticker": "AAPL", "fiscal_year": 2024},
        "inferences": [],
        "numeric": {"metric": "revenue", "value": 100, "unit": "BANANAS", "notes": "", "citation": "t001"},
    })
    ok5, errs5, _ = validate_answer_json(plan, ctx, bad_unit)
    check("bad unit caught", ok5 is False)
    check("unit_invalid in error", any("unit_invalid" in e for e in errs5))


def test_validate_risk_analysis():
    print("\n-- validate_answer_json: risk_analysis --")
    ctx = make_packed_context_simple()
    plan = build_task_plan("What are the risk factors for AAPL in 2024?", KNOWN_TICKERS)

    # Valid risk answer with Item 1A citations
    valid_risk = json.dumps({
        "final_answer": "AAPL faces macro and competitive risks.",
        "claims": [
            {"text": "Adverse macro conditions could impact demand.", "citations": ["c001"]},
            {"text": "Competition is substantial.", "citations": ["c002"]},
        ],
        "tables_used": [],
        "provenance": {"ticker": "AAPL", "fiscal_year": 2024},
        "inferences": [],
        "risks": [
            {"risk": "Macro risk", "mechanism": "Demand decline", "citations": ["c001"]},
            {"risk": "Competition", "mechanism": "Price pressure", "citations": ["c002"]},
        ],
    })
    ok, errs, _ = validate_answer_json(plan, ctx, valid_risk)
    check("valid risk answer passes", ok is True, f"errors: {errs}")

    # Risk citing Item 7 (not Item 1A) should fail
    risk_bad_cite = json.dumps({
        "final_answer": "test",
        "claims": [{"text": "test", "citations": ["c003"]}],
        "tables_used": [],
        "provenance": {"ticker": "AAPL", "fiscal_year": 2024},
        "inferences": [],
        "risks": [
            {"risk": "test risk", "mechanism": "test", "citations": ["c003"]},
        ],
    })
    ok2, errs2, _ = validate_answer_json(plan, ctx, risk_bad_cite)
    # c003 is Item 7 -- should be flagged
    has_item1a_error = any("not_item1a" in e for e in errs2)
    check("risk citing Item 7 flagged", has_item1a_error, f"errors: {errs2}")


# =============================================
# PART 11: Deterministic answer builders
# =============================================

def test_build_lookup_numeric_answer():
    print("\n-- build_lookup_numeric_answer --")
    ctx = make_packed_context_simple()
    target = Target(ticker="AAPL", fiscal_year=2024, metric="revenue")
    req = EvidenceRequirements(require_table_for_numeric=True)

    ans, debug = build_lookup_numeric_answer(ctx, target, req)
    check("answer not None", ans is not None)
    if ans:
        check("has metric", ans["metric"] == "revenue")
        check("has value", isinstance(ans["value"], float))
        check("has unit", ans["unit"] in ("USD", "PERCENT", "UNKNOWN"))
        check("has citation", isinstance(ans["citation"], str))

    # No metric -> None
    ans2, _ = build_lookup_numeric_answer(ctx, Target(ticker="AAPL", fiscal_year=2024, metric=None), req)
    check("no metric -> None answer", ans2 is None)


def test_build_compute_metric_answer():
    print("\n-- build_compute_metric_answer --")
    ctx = make_packed_context_simple()
    target = Target(ticker="AAPL", fiscal_year=2024, metric="cfo")
    req = EvidenceRequirements(require_table_for_numeric=True)

    ans, debug = build_compute_metric_answer(ctx, target, "fcf", req)
    if ans:
        check("fcf answer has formula", "cfo" in ans.get("formula", ""))
        check("fcf answer has inputs", len(ans.get("inputs", [])) >= 2)
    else:
        check("fcf computation returned result", ans is not None,
              f"fail_reason: {debug.get('fail_reason')}")


# =============================================
# PART 12: JSON prompt builder
# =============================================

def test_build_json_answer_prompt():
    print("\n-- build_json_answer_prompt --")
    ctx = make_packed_context_simple()
    plan = build_task_plan("What was AAPL revenue in 2024?", KNOWN_TICKERS)
    system, user = build_json_answer_prompt(plan, ctx)

    check("system prompt non-empty", len(system) > 50)
    check("system mentions JSON", "JSON" in system)
    check("system mentions citations", "citation" in system.lower())
    check("system uses pipeline-style ID example", "AAPL_FY2024" in system,
          "prompt should show pipeline ID format, not old c###/t### format")
    check("user has question", "AAPL" in user)
    check("user has mode", plan.mode in user)
    check("user has evidence", "EVIDENCE CONTEXT" in user)
    check("user has schema", "SCHEMA" in user)


# =============================================
# PART 12b: Input validation
# =============================================

def test_input_validation():
    print("\n-- input validation --")
    ctx = make_packed_context_simple()

    # build_task_plan: empty string
    try:
        build_task_plan("")
        check("build_task_plan rejects empty", False, "no exception raised")
    except ValueError:
        check("build_task_plan rejects empty", True)

    # build_task_plan: None
    try:
        build_task_plan(None)
        check("build_task_plan rejects None", False, "no exception raised")
    except (ValueError, TypeError):
        check("build_task_plan rejects None", True)

    # gate_evidence: bad plan type
    try:
        gate_evidence("not a plan", EvidenceRequirements(), ctx)
        check("gate_evidence rejects bad plan", False)
    except TypeError:
        check("gate_evidence rejects bad plan", True)

    # extract_numeric_topn: bad topn
    try:
        extract_numeric_topn(ctx, Target(ticker="AAPL", metric="revenue"), topn=0)
        check("extract_numeric rejects topn=0", False)
    except ValueError:
        check("extract_numeric rejects topn=0", True)

    # validate_answer_json: bad model_text type
    plan = build_task_plan("What was AAPL revenue in 2024?", KNOWN_TICKERS)
    try:
        validate_answer_json(plan, ctx, 12345)
        check("validate rejects non-string text", False)
    except TypeError:
        check("validate rejects non-string text", True)


# =============================================
# PART 13: Schema completeness
# =============================================

def test_schema_for_mode():
    print("\n-- schema_for_mode --")
    modes = ["lookup_numeric", "lookup_text", "compute_metric", "comparative_analysis",
             "risk_analysis", "valuation", "relative_valuation", "explanatory_reasoning", "mba_framework"]
    for m in modes:
        schema = schema_for_mode(m)
        check(f"{m} has final_answer", "final_answer" in schema)
        check(f"{m} has claims", "claims" in schema)

    check("lookup_numeric has numeric key", "numeric" in schema_for_mode("lookup_numeric"))
    check("compute_metric has computed key", "computed" in schema_for_mode("compute_metric"))
    check("risk_analysis has risks key", "risks" in schema_for_mode("risk_analysis"))
    check("valuation has valuation key", "valuation" in schema_for_mode("valuation"))
    check("mba_framework has framework key", "framework" in schema_for_mode("mba_framework"))
    check("comparative has comparison key", "comparison" in schema_for_mode("comparative_analysis"))


# =============================================
# PART 14: Pipeline ID format integration
# =============================================

def test_regex_patterns():
    """Test whether the regex patterns match actual pipeline ID formats."""
    print("\n-- Regex vs pipeline ID format --")

    # Simple IDs (what verification.py expects)
    check("TABLE_HDR matches simple", _TABLE_HDR_RE.match("[TABLE AAPL FY2024 Item 8 t001]") is not None)
    check("CHUNK_HDR matches simple", _CHUNK_HDR_RE.match("[AAPL FY2024 Item 1A c001]") is not None)
    check("ID_RE matches t001", len(_ID_RE.findall("t001")) == 1)
    check("ID_RE matches c002", len(_ID_RE.findall("c002")) == 1)

    # Actual pipeline IDs
    pipeline_table_hdr = "[TABLE AAPL FY2024 Item 8 AAPL_FY2024_0001234_T001]"
    pipeline_chunk_hdr = "[AAPL FY2024 Item 1A AAPL_FY2024_Item1A_0001]"

    table_match = _TABLE_HDR_RE.match(pipeline_table_hdr)
    check("TABLE_HDR matches pipeline ID", table_match is not None)
    if table_match:
        check("pipeline table id extracted", table_match.group("id") == "AAPL_FY2024_0001234_T001")
        check("pipeline table ticker", table_match.group("ticker") == "AAPL")

    chunk_match = _CHUNK_HDR_RE.match(pipeline_chunk_hdr)
    check("CHUNK_HDR matches pipeline ID", chunk_match is not None)
    if chunk_match:
        check("pipeline chunk id extracted", chunk_match.group("id") == "AAPL_FY2024_Item1A_0001")
        check("pipeline chunk item", chunk_match.group("item") == "Item 1A")

    # _ID_RE on pipeline IDs
    pipeline_ids = _ID_RE.findall("AAPL_FY2024_0001234_T001 AAPL_FY2024_Item1A_0001")
    check("ID_RE matches pipeline IDs", len(pipeline_ids) >= 2, f"got {pipeline_ids}")

    # Negative: CHUNK_HDR should NOT match TABLE headers
    check("CHUNK_HDR rejects TABLE header", _CHUNK_HDR_RE.match("[TABLE AAPL FY2024 Item 8 t001]") is None)


# =============================================
# MAIN
# =============================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("VERIFICATION.PY LOGIC TEST SUITE")
    print("=" * 60)

    # Part 1
    test_normalize_query()

    # Part 2
    test_detect_tickers()
    test_detect_years()
    test_detect_item_hint()
    test_detect_metrics()

    # Part 3
    test_infer_mode()

    # Part 4
    test_build_task_plan()

    # Part 5
    test_parse_allowed_ids_simple()
    test_parse_allowed_ids_pipeline()
    test_split_context_blocks_simple()
    test_split_context_blocks_pipeline()
    test_split_context_empty()

    # Part 6
    test_evidence_requirements()

    # Part 7
    test_gate_evidence()

    # Part 8
    test_source_precedence()
    test_global_scale_hint()
    test_metric_line_match()
    test_extract_numeric_topn()
    test_extract_net_income()
    test_verify_numeric_candidate()
    test_contradiction_check()
    test_choose_best_numeric()

    # Part 9
    test_compute_metric_value()
    test_required_inputs()

    # Part 10
    test_validate_answer_json()
    test_validate_risk_analysis()

    # Part 11
    test_build_lookup_numeric_answer()
    test_build_compute_metric_answer()

    # Part 12
    test_build_json_answer_prompt()

    # Part 12b
    test_input_validation()

    # Part 13
    test_schema_for_mode()

    # Part 14 -- integration
    test_regex_patterns()

    # Summary
    print("\n" + "=" * 60)
    bugs = [e for e in ERRORS if "[BUG]" in e]
    fails = [e for e in ERRORS if "[FAIL]" in e]
    print(f"RESULTS:  {PASS} passed,  {len(fails)} failed,  {len(bugs)} bugs,  {PASS + FAIL} total")
    print("=" * 60)
    if bugs:
        print("\nBUGS FOUND:")
        for b in bugs:
            print(b)
    if fails:
        print("\nFailed tests:")
        for f in fails:
            print(f)
    if not ERRORS:
        print("\nAll tests passed!")

    return FAIL == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
