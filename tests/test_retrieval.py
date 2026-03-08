"""
test_retrieval.py

Comprehensive tests for retrieval_tool.py covering:
  1. Unit tests for utility functions
  2. Integration tests across query types (numeric, broad, narrow, cross-company)
  3. Edge-case and logic tests (empty filters, bad filters, override flags)
  4. Context-packing invariants (char caps, dedup, table inclusion)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

BASE = Path(__file__).resolve().parent.parent
IDX = BASE / "index"
sys.path.insert(0, str(BASE))

from retrieval_tool import (
    _is_numeric_query,
    _safe_int,
    _bm25_tokenize,
    rrf_fuse,
    RetrievalConfig,
    FinancialRetrievalTool,
)

# -- Globals --
PASS = 0
FAIL = 0
ERRORS: List[str] = []


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


# =============================================
# PART 1: Unit Tests
# =============================================

def test_is_numeric_query():
    print("\n-- _is_numeric_query --")
    check("revenue is numeric", _is_numeric_query("What was TSLA revenue?"))
    check("net income is numeric", _is_numeric_query("net income for 2025"))
    check("$ is numeric", _is_numeric_query("How much $ did they earn?"))
    check("EPS is numeric", _is_numeric_query("What was EPS last year?"))
    check("margin is numeric", _is_numeric_query("gross margin trend"))
    check("debt is numeric", _is_numeric_query("total debt of Apple"))
    check("risk factors is NOT numeric", not _is_numeric_query("What are the main risk factors?"))
    check("plain narrative is NOT numeric", not _is_numeric_query("Describe Tesla's business model"))
    check("empty string is NOT numeric", not _is_numeric_query(""))
    check("yoy is numeric", _is_numeric_query("revenue yoy growth"))
    check("capex is numeric", _is_numeric_query("capex spending over two years"))
    check("per share is numeric", _is_numeric_query("earnings per share"))
    check("'rationale' is NOT numeric (word boundary)", not _is_numeric_query("What is the rationale for their strategy?"))
    check("'change' removed from markers", not _is_numeric_query("The company saw a change in leadership"))
    check("'marginally' is NOT numeric", not _is_numeric_query("This only marginally affected results"))


def test_safe_int():
    print("\n-- _safe_int --")
    check("int from int", _safe_int(42) == 42)
    check("int from str", _safe_int("2025") == 2025)
    check("int from float", _safe_int(3.9) == 3)
    check("None from garbage", _safe_int("abc") is None)
    check("None from None", _safe_int(None) is None)
    check("None from empty", _safe_int("") is None)


def test_bm25_tokenize():
    print("\n-- _bm25_tokenize --")
    tokens = _bm25_tokenize("Revenue was $3.5B in FY2024!")
    check("lowercased", all(t == t.lower() for t in tokens))
    check("$ preserved", "$3.5b" in tokens)
    check("non-empty", len(tokens) > 0)
    check("empty input", _bm25_tokenize("") == [])
    check("special chars stripped", _bm25_tokenize("hello@world#test") == ["hello", "world", "test"])


def test_rrf_fuse():
    print("\n-- rrf_fuse --")
    list1 = [("a", 10.0), ("b", 8.0), ("c", 6.0)]
    list2 = [("b", 9.0), ("a", 7.0), ("d", 5.0)]
    fused = rrf_fuse([list1, list2], top_k=10, rrf_k=60)
    fused_ids = [x[0] for x in fused]

    check("all unique ids", len(fused_ids) == len(set(fused_ids)))
    check("a and b at top (in both lists)", set(fused_ids[:2]) == {"a", "b"})
    check("d is present", "d" in fused_ids)
    check("scores are descending", all(fused[i][1] >= fused[i + 1][1] for i in range(len(fused) - 1)))

    empty = rrf_fuse([], top_k=10)
    check("empty input -> empty output", empty == [])

    single = rrf_fuse([list1], top_k=2)
    check("single list top_k=2", len(single) == 2)
    check("single list preserves order", single[0][0] == "a")


# =============================================
# PART 2: Integration Tests (live retrieval)
# =============================================

def load_tool() -> FinancialRetrievalTool:
    print("\n-- Loading FinancialRetrievalTool --")
    t0 = time.time()
    tool = FinancialRetrievalTool(
        narrative_chunks_path=IDX / "chunks.parquet",
        narrative_bm25_path=IDX / "bm25.pkl",
        narrative_faiss_path=IDX / "faiss.index",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        table_docs_path=IDX / "tables.parquet",
        table_bm25_path=IDX / "table_bm25.pkl",
        table_faiss_path=IDX / "table_faiss.index",
        companyfacts_dir=BASE / "data" / "xbrl_companyfacts",
        config=RetrievalConfig(use_rerank=False),
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return tool


def run_query(tool, name: str, query: str, filters=None, numeric_query=None) -> Tuple[str, Dict, Dict]:
    print(f"\n-- Query: {name} --")
    print(f"  q = \"{query}\"")
    print(f"  filters = {filters}")
    t0 = time.time()
    ctx, debug, evidence = tool.retrieve(query, filters=filters, numeric_query=numeric_query)
    elapsed = time.time() - t0
    print(f"  latency={debug['latency_ms']['total']}ms  chars={debug['counts']['context_chars']}  "
          f"chunks={debug['counts']['packed_chunks']}  tables={debug['counts']['packed_tables']}")
    return ctx, debug, evidence


def test_numeric_query_with_ticker_filter(tool):
    """Numeric query should activate tables + XBRL, filter to TSLA FY2025."""
    ctx, debug, ev = run_query(
        tool, "Numeric + ticker filter",
        "What was Tesla's total revenue in fiscal year 2025?",
        filters={"ticker": "TSLA", "fiscal_year": 2025},
    )
    check("detected as numeric", ev["numeric_query"] is True)
    check("tables enabled", ev["tables"]["enabled"] is True)
    check("xbrl enabled", ev["xbrl"]["enabled"] is True)
    check("packed tables > 0", debug["counts"]["packed_tables"] > 0)
    check("packed chunks > 0", debug["counts"]["packed_chunks"] > 0)
    check("context non-empty", len(ctx) > 100)
    check("no warnings", debug["warnings"] is None)
    check("TSLA in context", "TSLA" in ctx)
    check("context within char cap", debug["counts"]["context_chars"] <= 18_500)


def test_narrative_broad_query(tool):
    """Broad query should widen window/caps, activate rerank-path (but rerank=off here)."""
    ctx, debug, ev = run_query(
        tool, "Broad narrative query",
        "What are the main risk factors for Apple?",
        filters={"ticker": "AAPL"},
    )
    check("detected as broad", ev["broad_query"] is True)
    check("NOT numeric", ev["numeric_query"] is False)
    check("packed tables == 0 (not numeric)", debug["counts"]["packed_tables"] == 0)
    check("packed chunks > 0", debug["counts"]["packed_chunks"] > 0)
    check("AAPL in context", "AAPL" in ctx)
    check("Item 1A in context", "Item1A" in ctx.replace(" ", ""))


def test_narrow_narrative_query(tool):
    """Specific non-numeric, non-broad query."""
    ctx, debug, ev = run_query(
        tool, "Narrow narrative query",
        "How does NVIDIA describe its competitive position in AI chips?",
        filters={"ticker": "NVDA"},
    )
    check("NOT numeric", ev["numeric_query"] is False)
    check("NOT broad", ev["broad_query"] is False)
    check("packed chunks > 0", debug["counts"]["packed_chunks"] > 0)
    check("NVDA in context", "NVDA" in ctx)


def test_cross_company_no_filter(tool):
    """No ticker filter: should retrieve chunks from multiple tickers."""
    ctx, debug, ev = run_query(
        tool, "Cross-company (no filter)",
        "Compare risk factors related to AI regulation across companies",
        filters=None,
    )
    check("packed chunks > 0", debug["counts"]["packed_chunks"] > 0)
    tickers_in_context = sum(1 for t in ["AAPL", "META", "NVDA", "GOOGL", "TSLA"] if t in ctx)
    check("multiple tickers in context", tickers_in_context >= 2,
          f"found {tickers_in_context} tickers")


def test_numeric_override_false(tool):
    """Force numeric=False on a numeric query: tables/xbrl should be skipped."""
    ctx, debug, ev = run_query(
        tool, "Numeric override=False",
        "What was Google's revenue in 2024?",
        filters={"ticker": "GOOGL", "fiscal_year": 2024},
        numeric_query=False,
    )
    check("forced NOT numeric", ev["numeric_query"] is False)
    check("no tables packed", debug["counts"]["packed_tables"] == 0)
    check("no table candidates", debug["counts"]["table_candidates"] == 0)


def test_numeric_override_true(tool):
    """Force numeric=True on a non-numeric query: should activate tables + XBRL."""
    ctx, debug, ev = run_query(
        tool, "Numeric override=True",
        "Describe Meta's business strategy",
        filters={"ticker": "META", "fiscal_year": 2025},
        numeric_query=True,
    )
    check("forced numeric", ev["numeric_query"] is True)
    check("table candidates > 0", debug["counts"]["table_candidates"] > 0)


def test_filter_nonexistent_ticker(tool):
    """Filter by a ticker not in the corpus: should return empty/minimal context."""
    ctx, debug, ev = run_query(
        tool, "Nonexistent ticker filter",
        "What is the revenue?",
        filters={"ticker": "ZZZZ", "fiscal_year": 2025},
    )
    check("packed chunks == 0", debug["counts"]["packed_chunks"] == 0)
    check("packed tables == 0", debug["counts"]["packed_tables"] == 0)


def test_item_filter(tool):
    """Filter by item name: only chunks from that section."""
    ctx, debug, ev = run_query(
        tool, "Item filter (Item 7 only)",
        "Management discussion of operating results",
        filters={"ticker": "TSLA", "fiscal_year": 2025, "item": "Item 7"},
    )
    check("packed chunks > 0", debug["counts"]["packed_chunks"] > 0)
    check("only Item 7 chunks", all("Item7" in cid or "Item 7" in ctx for cid in ev["narrative"]["selected_chunk_ids"]),
          f"chunk_ids: {ev['narrative']['selected_chunk_ids'][:3]}")


def test_fy_2024_vs_2025(tool):
    """Same query, different FY: results should differ."""
    ctx24, _, ev24 = run_query(
        tool, "TSLA FY2024",
        "Tesla revenue and profitability",
        filters={"ticker": "TSLA", "fiscal_year": 2024},
    )
    ctx25, _, ev25 = run_query(
        tool, "TSLA FY2025",
        "Tesla revenue and profitability",
        filters={"ticker": "TSLA", "fiscal_year": 2025},
    )
    ids_24 = set(ev24["narrative"]["selected_chunk_ids"])
    ids_25 = set(ev25["narrative"]["selected_chunk_ids"])
    check("FY2024 and FY2025 return different chunks", ids_24 != ids_25,
          f"overlap: {len(ids_24 & ids_25)}")


def test_context_dedup(tool):
    """Context should not contain duplicate chunk IDs."""
    ctx, debug, ev = run_query(
        tool, "Dedup check",
        "What are the key risks for NVIDIA?",
        filters={"ticker": "NVDA"},
    )
    chunk_ids = ev["narrative"]["selected_chunk_ids"]
    check("no duplicate chunk IDs", len(chunk_ids) == len(set(chunk_ids)))


def test_latency_reasonable(tool):
    """Retrieval should complete in under 5 seconds (no reranker)."""
    _, debug, _ = run_query(
        tool, "Latency check",
        "Apple's cash flow from operations",
        filters={"ticker": "AAPL", "fiscal_year": 2025},
    )
    total = debug["latency_ms"]["total"]
    check(f"latency < 5000ms (was {total}ms)", total < 5000)


def test_xbrl_evidence_structure(tool):
    """XBRL hits should be well-formed dicts with expected keys."""
    _, _, ev = run_query(
        tool, "XBRL structure check",
        "What was Meta's net income?",
        filters={"ticker": "META", "fiscal_year": 2024},
    )
    check("xbrl hits present", ev["xbrl"]["hits"] is not None and len(ev["xbrl"]["hits"]) > 0)
    if ev["xbrl"]["hits"]:
        hit = ev["xbrl"]["hits"][0]
        check("has ticker", "ticker" in hit)
        check("has concept", "concept" in hit)
        check("has label", "label" in hit)
        check("has value", "value" in hit)
        check("has unit", "unit" in hit)


def test_empty_query(tool):
    """Empty query should not crash."""
    try:
        ctx, debug, ev = run_query(tool, "Empty query", "", filters=None)
        check("did not crash", True)
        check("returns string context", isinstance(ctx, str))
    except Exception as e:
        check("did not crash", False, str(e))


def test_very_long_query(tool):
    """Very long query should not crash."""
    long_q = "revenue " * 200
    try:
        ctx, debug, ev = run_query(tool, "Very long query", long_q, filters={"ticker": "AAPL"})
        check("did not crash", True)
        check("context produced", len(ctx) > 0)
    except Exception as e:
        check("did not crash", False, str(e))


def test_input_validation(tool):
    """Input validation should reject bad types and warn on unknown filter keys."""
    print("\n-- Input validation --")

    try:
        tool.retrieve(123)
        check("rejects non-string query", False, "should have raised TypeError")
    except TypeError:
        check("rejects non-string query", True)

    try:
        tool.retrieve("test", filters="bad")
        check("rejects non-dict filters", False, "should have raised TypeError")
    except TypeError:
        check("rejects non-dict filters", True)

    ctx, debug, ev = tool.retrieve(
        "revenue",
        filters={"ticker": "AAPL", "bogus_key": "ignored"},
    )
    check("unknown filter keys don't crash", True)
    check("still returns results", len(ctx) > 0)


# =============================================
# PART 3: Evidence payload structure tests
# =============================================

def test_evidence_structure(tool):
    """Evidence dict should have all expected top-level keys."""
    print("\n-- Evidence structure --")
    _, _, ev = run_query(
        tool, "Evidence keys",
        "Google revenue growth",
        filters={"ticker": "GOOGL"},
    )
    for key in ["numeric_query", "broad_query", "filters", "narrative", "tables", "xbrl"]:
        check(f"evidence has '{key}'", key in ev)

    narr = ev["narrative"]
    for key in ["bm25", "dense", "fused", "selected_chunk_ids"]:
        check(f"narrative has '{key}'", key in narr)

    check("bm25 returns list of tuples", isinstance(narr["bm25"], list) and len(narr["bm25"]) > 0)
    check("dense returns list of tuples", isinstance(narr["dense"], list) and len(narr["dense"]) > 0)
    check("fused returns list of tuples", isinstance(narr["fused"], list) and len(narr["fused"]) > 0)


# =============================================
# MAIN
# =============================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("RETRIEVAL TOOL TEST SUITE")
    print("=" * 60)

    # Part 1: Unit tests (no model loading)
    test_is_numeric_query()
    test_safe_int()
    test_bm25_tokenize()
    test_rrf_fuse()

    # Part 2 + 3: Integration tests
    tool = load_tool()

    test_numeric_query_with_ticker_filter(tool)
    test_narrative_broad_query(tool)
    test_narrow_narrative_query(tool)
    test_cross_company_no_filter(tool)
    test_numeric_override_false(tool)
    test_numeric_override_true(tool)
    test_filter_nonexistent_ticker(tool)
    test_item_filter(tool)
    test_fy_2024_vs_2025(tool)
    test_context_dedup(tool)
    test_latency_reasonable(tool)
    test_xbrl_evidence_structure(tool)
    test_empty_query(tool)
    test_very_long_query(tool)
    test_input_validation(tool)
    test_evidence_structure(tool)

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS:  {PASS} passed,  {FAIL} failed,  {PASS + FAIL} total")
    print("=" * 60)
    if ERRORS:
        print("\nFailed tests:")
        for e in ERRORS:
            print(e)
    else:
        print("\nAll tests passed!")

    return FAIL == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
