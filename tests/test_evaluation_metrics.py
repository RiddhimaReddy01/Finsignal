"""
test_evaluation_metrics.py

Logic tests for evaluation_metrics.py covering:
  1. read_jsonl parsing
  2. group_by_run grouping
  3. get_event lookup
  4. recall_at_k (standard IR metric, edge cases)
  5. section_hit_rate calculation
  6. unsupported_claim_rate error classification
  7. numeric_exactness tolerance checks
  8. unit_correctness case-insensitive matching
  9. quantiles p50/p95 calculation
  10. fallback_rate counting logic
  11. compute_report end-to-end with synthetic audit log
  12. EvalReport dataclass
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import List

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from evaluation_metrics import (
    read_jsonl,
    group_by_run,
    get_event,
    recall_at_k,
    section_hit_rate,
    unsupported_claim_rate,
    numeric_exactness,
    unit_correctness,
    quantiles,
    fallback_rate,
    compute_report,
    EvalReport,
)

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


def write_jsonl(path: str, records: list):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =============================================
# PART 1: read_jsonl
# =============================================

def test_read_jsonl():
    print("\n-- read_jsonl --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.jsonl")
        records = [{"a": 1}, {"b": 2}, {"c": 3}]
        write_jsonl(path, records)

        rows = read_jsonl(path)
        check("reads 3 records", len(rows) == 3)
        check("first record correct", rows[0] == {"a": 1})
        check("last record correct", rows[2] == {"c": 3})

    # File with blank lines
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "blanks.jsonl")
        with open(path, "w") as f:
            f.write('{"a":1}\n\n\n{"b":2}\n\n')
        rows = read_jsonl(path)
        check("skips blank lines", len(rows) == 2)

    # Corrupt line handling
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "corrupt.jsonl")
        with open(path, "w") as f:
            f.write('{"a":1}\nNOT_JSON_AT_ALL\n{"b":2}\n')
        rows = read_jsonl(path)
        check("skips corrupt lines", len(rows) == 2, f"got {len(rows)}")
        check("preserves valid lines", rows[0] == {"a": 1} and rows[1] == {"b": 2})

    # File not found
    try:
        read_jsonl("/nonexistent/path/audit.jsonl")
        check("raises FileNotFoundError", False)
    except FileNotFoundError:
        check("raises FileNotFoundError", True)


# =============================================
# PART 2: group_by_run
# =============================================

def test_group_by_run():
    print("\n-- group_by_run --")
    rows = [
        {"run_id": "r1", "event": "retrieval"},
        {"run_id": "r1", "event": "generation"},
        {"run_id": "r2", "event": "retrieval"},
        {"event": "orphan"},  # no run_id
    ]
    groups = group_by_run(rows)
    check("2 groups", len(groups) == 2)
    check("r1 has 2 events", len(groups["r1"]) == 2)
    check("r2 has 1 event", len(groups["r2"]) == 1)
    check("orphan excluded", "orphan" not in str(groups))


# =============================================
# PART 3: get_event
# =============================================

def test_get_event():
    print("\n-- get_event --")
    evs = [
        {"event": "retrieval", "data": "a"},
        {"event": "generation", "data": "b"},
        {"event": "validation", "data": "c"},
    ]
    check("finds retrieval", get_event(evs, "retrieval")["data"] == "a")
    check("finds generation", get_event(evs, "generation")["data"] == "b")
    check("finds validation", get_event(evs, "validation")["data"] == "c")
    check("missing -> None", get_event(evs, "gate_routing") is None)
    check("empty list -> None", get_event([], "retrieval") is None)


# =============================================
# PART 4: recall_at_k
# =============================================

def test_recall_at_k():
    print("\n-- recall_at_k --")
    check("perfect recall@5", recall_at_k(["a", "b", "c", "d", "e"], ["a", "b", "c"], 5) == 1.0)
    check("partial recall", recall_at_k(["a", "b", "x", "y", "z"], ["a", "b", "c"], 5) == 2 / 3)
    check("zero recall", recall_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0)
    check("recall@3 truncates", recall_at_k(["a", "x", "y", "b", "c"], ["a", "b"], 3) == 0.5)
    check("empty retrieved", recall_at_k([], ["a", "b"], 5) == 0.0)
    check("empty gold -> 1.0", recall_at_k(["a", "b"], [], 5) == 1.0)
    check("both empty -> 1.0", recall_at_k([], [], 5) == 1.0)
    check("None gold -> 1.0", recall_at_k(["a"], None, 5) == 1.0)
    check("k=1 first only", recall_at_k(["a", "b"], ["a", "b"], 1) == 0.5)
    check("k larger than list", recall_at_k(["a"], ["a"], 100) == 1.0)


# =============================================
# PART 5: section_hit_rate
# =============================================

def test_section_hit_rate():
    print("\n-- section_hit_rate --")
    meta = [
        {"id": "c1", "item": "Item 1A"},
        {"id": "c2", "item": "Item 1A"},
        {"id": "c3", "item": "Item 7"},
        {"id": "c4", "item": "Item 8"},
    ]
    rate = section_hit_rate(meta, "Item 1A")
    check("2/4 = 0.5", rate == 0.5, f"got {rate}")

    all_1a = [{"id": "c1", "item": "Item 1A"}, {"id": "c2", "item": "Item 1A"}]
    check("all match = 1.0", section_hit_rate(all_1a, "Item 1A") == 1.0)

    none_1a = [{"id": "c1", "item": "Item 7"}]
    check("no match = 0.0", section_hit_rate(none_1a, "Item 1A") == 0.0)

    check("empty list = 0.0", section_hit_rate([], "Item 1A") == 0.0)

    # Prefix matching: "Item 1A" should match "Item 1A.1"
    prefix_meta = [{"id": "c1", "item": "Item 1A.1 Risk Factors"}]
    check("prefix match", section_hit_rate(prefix_meta, "Item 1A") == 1.0)

    # Missing item key
    missing = [{"id": "c1"}, {"id": "c2", "item": "Item 1A"}]
    rate2 = section_hit_rate(missing, "Item 1A")
    check("missing item key handled", rate2 == 0.5, f"got {rate2}")


# =============================================
# PART 6: unsupported_claim_rate
# =============================================

def test_unsupported_claim_rate():
    print("\n-- unsupported_claim_rate --")

    # Passing validation
    check("ok=True -> 0.0", unsupported_claim_rate({"ok": True, "errors": []}) == 0.0)

    # All citation errors
    all_bad = {"ok": False, "errors": [
        "claim_0_cit_0_not_allowed:t999",
        "claim_1_missing_citations",
        "computed_input_0_citation_invalid",
    ]}
    rate = unsupported_claim_rate(all_bad)
    check("all citation errors -> 1.0", rate == 1.0, f"got {rate}")

    # Mix of citation and other errors
    mixed = {"ok": False, "errors": [
        "claim_0_cit_0_not_allowed:t999",
        "numeric_unit_invalid",
        "bad_keys: ...",
        "claim_1_missing_citations",
    ]}
    rate2 = unsupported_claim_rate(mixed)
    check("2/4 citation errors -> 0.5", rate2 == 0.5, f"got {rate2}")

    # No errors but not ok (shouldn't happen, but edge case)
    check("not ok, empty errors -> 1.0", unsupported_claim_rate({"ok": False, "errors": []}) == 1.0)

    # None event
    check("None -> 1.0", unsupported_claim_rate(None) == 1.0)
    check("empty dict -> 1.0", unsupported_claim_rate({}) == 1.0)

    # No citation errors at all
    non_cite = {"ok": False, "errors": ["numeric_unit_invalid", "bad_keys"]}
    rate3 = unsupported_claim_rate(non_cite)
    check("no citation errors -> 0.0", rate3 == 0.0, f"got {rate3}")


# =============================================
# PART 7: numeric_exactness
# =============================================

def test_numeric_exactness():
    print("\n-- numeric_exactness --")
    check("exact match -> 1.0", numeric_exactness(100.0, 100.0) == 1.0)
    check("within rel_tol -> 1.0", numeric_exactness(100.05, 100.0, rel_tol=0.001) == 1.0)
    check("outside rel_tol -> 0.0", numeric_exactness(101.0, 100.0, rel_tol=0.001) == 0.0)
    check("within abs_tol -> 1.0", numeric_exactness(101.0, 100.0, rel_tol=0.001, abs_tol=2.0) == 1.0)
    check("gold=0, pred=0 -> 1.0", numeric_exactness(0.0, 0.0) == 1.0)
    check("gold=0, pred=small -> 0.0", numeric_exactness(0.1, 0.0) == 0.0)
    check("gold=0, pred=small with abs_tol -> 1.0", numeric_exactness(0.1, 0.0, abs_tol=0.5) == 1.0)
    check("negative values", numeric_exactness(-100.0, -100.0) == 1.0)
    check("large values", numeric_exactness(391035e6, 391035e6) == 1.0)
    check("large values slight diff", numeric_exactness(391035e6, 391000e6, rel_tol=0.001) == 1.0)


# =============================================
# PART 8: unit_correctness
# =============================================

def test_unit_correctness():
    print("\n-- unit_correctness --")
    check("exact match", unit_correctness("USD", "USD") == 1.0)
    check("case insensitive", unit_correctness("usd", "USD") == 1.0)
    check("mismatch", unit_correctness("USD", "PERCENT") == 0.0)
    check("None vs None", unit_correctness(None, None) == 1.0)
    check("None vs USD", unit_correctness(None, "USD") == 0.0)
    check("empty vs empty", unit_correctness("", "") == 1.0)


# =============================================
# PART 9: quantiles
# =============================================

def test_quantiles():
    print("\n-- quantiles --")
    vals = list(range(1, 101))  # 1..100
    q = quantiles(vals, (0.5, 0.95))
    check("p50 ~ 50", abs(q["p50"] - 50) <= 1, f"got {q['p50']}")
    check("p95 ~ 95", abs(q["p95"] - 95) <= 1, f"got {q['p95']}")

    # Single value
    q2 = quantiles([42.0])
    check("single value p50", q2["p50"] == 42.0)
    check("single value p95", q2["p95"] == 42.0)

    # Empty
    q3 = quantiles([])
    check("empty p50 is nan", math.isnan(q3["p50"]))
    check("empty p95 is nan", math.isnan(q3["p95"]))

    # With Nones (should be filtered)
    q4 = quantiles([10.0, None, 20.0, None, 30.0])
    check("filters Nones", q4["p50"] == 20.0, f"got {q4['p50']}")

    # Two values
    q5 = quantiles([10.0, 90.0])
    check("two values p50", q5["p50"] in (10.0, 90.0))
    check("two values p95", q5["p95"] == 90.0)


# =============================================
# PART 10: fallback_rate
# =============================================

def test_fallback_rate():
    print("\n-- fallback_rate --")

    runs = {
        "r1": [{"event": "gate_routing", "routing": {"action": "pass", "model": "small"}},
               {"event": "validation", "ok": True}],
        "r2": [{"event": "gate_routing", "routing": {"action": "pass", "model": "small"}},
               {"event": "validation", "ok": True}],
    }
    check("no fallbacks -> 0.0", fallback_rate(runs) == 0.0)

    runs2 = {
        "r1": [{"event": "gate_routing", "routing": {"action": "clarify", "model": "none"}}],
        "r2": [{"event": "gate_routing", "routing": {"action": "abstain", "model": "none"}}],
    }
    check("all clarify/abstain -> 1.0", fallback_rate(runs2) == 1.0)

    runs3 = {
        "r1": [{"event": "gate_routing", "routing": {"action": "pass", "model": "large"}},
               {"event": "validation", "ok": True}],
    }
    check("large model -> fallback", fallback_rate(runs3) == 1.0)

    runs4 = {
        "r1": [{"event": "gate_routing", "routing": {"action": "pass", "model": "small"}},
               {"event": "validation", "ok": False}],
    }
    check("validation fail -> fallback", fallback_rate(runs4) == 1.0)

    # Mixed
    runs5 = {
        "r1": [{"event": "gate_routing", "routing": {"action": "pass", "model": "small"}},
               {"event": "validation", "ok": True}],
        "r2": [{"event": "gate_routing", "routing": {"action": "clarify", "model": "none"}}],
        "r3": [{"event": "gate_routing", "routing": {"action": "pass", "model": "large"}},
               {"event": "validation", "ok": True}],
        "r4": [{"event": "gate_routing", "routing": {"action": "pass", "model": "small"}},
               {"event": "validation", "ok": False}],
    }
    rate = fallback_rate(runs5)
    check("3/4 fallback = 0.75", rate == 0.75, f"got {rate}")

    # Empty
    check("empty -> 0.0", fallback_rate({}) == 0.0)


# =============================================
# PART 11: compute_report (end-to-end)
# =============================================

def _build_synthetic_audit():
    """Build a realistic JSONL audit log for two runs."""
    records = []

    # Run 1: successful numeric lookup
    records.append({
        "run_id": "r1", "event": "retrieval", "ts_ms": 1000,
        "question": "What was AAPL revenue?",
        "packed_ids": ["t001", "c003", "c004"],
        "selected": [
            {"id": "t001", "item": "Item 8"},
            {"id": "c003", "item": "Item 7"},
            {"id": "c004", "item": "Item 8"},
        ],
        "latency_ms": 150.0,
    })
    records.append({
        "run_id": "r1", "event": "gate_routing", "ts_ms": 1001,
        "routing": {"action": "pass", "model": "small"},
        "latency_ms": 5.0,
    })
    records.append({
        "run_id": "r1", "event": "generation", "ts_ms": 1002,
        "model": "gpt-4o-mini",
        "output_preview": json.dumps({
            "final_answer": "Revenue was $391B",
            "numeric": {"metric": "revenue", "value": 391035000000, "unit": "USD"},
        }),
        "usage": {"input_tokens": 500, "output_tokens": 100, "cost_usd": 0.002},
        "latency_ms": 800.0,
    })
    records.append({
        "run_id": "r1", "event": "validation", "ts_ms": 1003,
        "ok": True, "errors": [],
        "latency_ms": 2.0,
    })

    # Run 2: failed validation (bad citation)
    records.append({
        "run_id": "r2", "event": "retrieval", "ts_ms": 2000,
        "question": "What are TSLA risks?",
        "packed_ids": ["c010", "c011"],
        "selected": [
            {"id": "c010", "item": "Item 1A"},
            {"id": "c011", "item": "Item 1A"},
        ],
        "latency_ms": 120.0,
    })
    records.append({
        "run_id": "r2", "event": "gate_routing", "ts_ms": 2001,
        "routing": {"action": "pass", "model": "small"},
        "latency_ms": 4.0,
    })
    records.append({
        "run_id": "r2", "event": "generation", "ts_ms": 2002,
        "model": "gpt-4o-mini",
        "usage": {"input_tokens": 400, "output_tokens": 150, "cost_usd": 0.0025},
        "latency_ms": 900.0,
    })
    records.append({
        "run_id": "r2", "event": "validation", "ts_ms": 2003,
        "ok": False,
        "errors": ["claim_0_cit_0_not_allowed:t999", "numeric_unit_invalid"],
        "latency_ms": 1.5,
    })

    return records


def test_compute_report():
    print("\n-- compute_report --")
    records = _build_synthetic_audit()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        write_jsonl(path, records)

        gold = {
            "r1": {
                "gold_ids": ["t001", "c003"],
                "desired_section": "Item 8",
                "gold_numeric": {"value": 391035000000, "unit": "USD"},
            },
            "r2": {
                "gold_ids": ["c010", "c011"],
                "desired_section": "Item 1A",
            },
        }

        report = compute_report(path, labeled_gold=gold)

        check("report is EvalReport", isinstance(report, EvalReport))
        check("recall_at_5 in [0,1]", 0 <= report.recall_at_5 <= 1, f"got {report.recall_at_5}")
        check("recall_at_10 in [0,1]", 0 <= report.recall_at_10 <= 1, f"got {report.recall_at_10}")

        check("recall_at_5 = 1.0", report.recall_at_5 == 1.0, f"got {report.recall_at_5}")

        check("item1a_hit_rate in [0,1]", 0 <= report.item1a_hit_rate <= 1)
        check("unsupported_claim_rate in [0,1]", 0 <= report.unsupported_claim_rate <= 1)

        check("ucr = 0.25", abs(report.unsupported_claim_rate - 0.25) < 0.01,
              f"got {report.unsupported_claim_rate}")

        check("numeric_exactness computed", report.numeric_exactness_mean == 1.0,
              f"got {report.numeric_exactness_mean}")
        check("unit_correctness computed", report.unit_correctness_mean == 1.0,
              f"got {report.unit_correctness_mean}")

        check("latency p50 > 0", report.latency_ms_p50 > 0)
        check("latency p95 >= p50", report.latency_ms_p95 >= report.latency_ms_p50)
        check("cost p50 > 0", report.cost_usd_p50 > 0)

        check("fallback_rate = 0.5", report.fallback_rate == 0.5, f"got {report.fallback_rate}")


def test_compute_report_no_gold():
    print("\n-- compute_report (no gold labels) --")
    records = _build_synthetic_audit()

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        write_jsonl(path, records)

        report = compute_report(path)
        check("works without gold", isinstance(report, EvalReport))
        # No gold -> all recall = 1.0 (trivially satisfied)
        check("recall@5 = 1.0 (no gold)", report.recall_at_5 == 1.0)


# =============================================
# PART 12: EvalReport dataclass
# =============================================

def test_eval_report_dataclass():
    print("\n-- EvalReport dataclass --")
    r = EvalReport(
        recall_at_5=0.85,
        recall_at_10=0.92,
        item1a_hit_rate=0.70,
        unsupported_claim_rate=0.05,
        numeric_exactness_mean=0.90,
        unit_correctness_mean=0.95,
        latency_ms_p50=200.0,
        latency_ms_p95=1500.0,
        cost_usd_p50=0.003,
        cost_usd_p95=0.01,
        fallback_rate=0.15,
    )
    check("recall_at_5 accessible", r.recall_at_5 == 0.85)
    check("recall_at_10 accessible", r.recall_at_10 == 0.92)
    check("item1a_hit_rate accessible", r.item1a_hit_rate == 0.70)
    check("unsupported_claim_rate accessible", r.unsupported_claim_rate == 0.05)
    check("numeric_exactness_mean accessible", r.numeric_exactness_mean == 0.90)
    check("unit_correctness_mean accessible", r.unit_correctness_mean == 0.95)
    check("latency_ms_p50 accessible", r.latency_ms_p50 == 200.0)
    check("latency_ms_p95 accessible", r.latency_ms_p95 == 1500.0)
    check("cost_usd_p50 accessible", r.cost_usd_p50 == 0.003)
    check("cost_usd_p95 accessible", r.cost_usd_p95 == 0.01)
    check("fallback_rate accessible", r.fallback_rate == 0.15)


# =============================================
# MAIN
# =============================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("EVALUATION_METRICS.PY LOGIC TEST SUITE")
    print("=" * 60)

    test_read_jsonl()
    test_group_by_run()
    test_get_event()
    test_recall_at_k()
    test_section_hit_rate()
    test_unsupported_claim_rate()
    test_numeric_exactness()
    test_unit_correctness()
    test_quantiles()
    test_fallback_rate()
    test_compute_report()
    test_compute_report_no_gold()
    test_eval_report_dataclass()

    print("\n" + "=" * 60)
    fails = [e for e in ERRORS if "[FAIL]" in e]
    print(f"RESULTS:  {PASS} passed,  {len(fails)} failed,  {PASS + FAIL} total")
    print("=" * 60)
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
