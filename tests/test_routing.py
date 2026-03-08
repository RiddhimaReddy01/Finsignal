"""
test_routing.py

Logic tests for routing.py covering:
  1. _clip01 clamping
  2. jaccard set similarity
  3. compute_retrieval_signals with full/partial/empty debug
  4. risk_score weighting and hard-mode penalty
  5. choose_model_from_risk threshold behavior
  6. decide() gate-dominated paths (clarify, abstain, needs_market_data)
  7. decide() with passing gate at various risk levels
  8. decide() extreme risk abstain even if gate passes
  9. End-to-end scenario: numeric query with good evidence -> small model
  10. End-to-end scenario: narrative query with poor evidence -> large model
  11. Edge cases (empty inputs, missing keys, boundary values)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from routing import (
    _clip01,
    jaccard,
    compute_retrieval_signals,
    risk_score,
    choose_model_from_risk,
    decide,
    RoutingDecision,
    HARD_MODES,
    COVERAGE_FULL_AT,
    SMALL_MODEL_MAX_RISK,
    ABSTAIN_MIN_RISK,
    W_COVERAGE,
    W_AGREEMENT,
    W_MARGIN,
    W_ANSWER,
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


# =============================================
# PART 1: _clip01
# =============================================

def test_clip01():
    print("\n-- _clip01 --")
    check("0.5 unchanged", _clip01(0.5) == 0.5)
    check("0.0 unchanged", _clip01(0.0) == 0.0)
    check("1.0 unchanged", _clip01(1.0) == 1.0)
    check("negative clipped to 0", _clip01(-0.5) == 0.0)
    check("above 1 clipped to 1", _clip01(1.5) == 1.0)
    check("-100 clipped to 0", _clip01(-100) == 0.0)
    check("100 clipped to 1", _clip01(100) == 1.0)
    check("returns float", isinstance(_clip01(1), float))


# =============================================
# PART 2: jaccard
# =============================================

def test_jaccard():
    print("\n-- jaccard --")
    check("identical sets = 1.0", jaccard(["a", "b", "c"], ["a", "b", "c"]) == 1.0)
    check("disjoint sets = 0.0", jaccard(["a", "b"], ["c", "d"]) == 0.0)
    check("partial overlap", abs(jaccard(["a", "b", "c"], ["b", "c", "d"]) - 0.5) < 0.01)
    check("both empty = 1.0", jaccard([], []) == 1.0)
    check("one empty = 0.0", jaccard(["a"], []) == 0.0)
    check("other empty = 0.0", jaccard([], ["a"]) == 0.0)
    check("None safety a", jaccard(None, ["a"]) == 0.0)
    check("None safety b", jaccard(["a"], None) == 0.0)
    check("None both = 1.0", jaccard(None, None) == 1.0)
    check("duplicates handled", jaccard(["a", "a", "b"], ["a", "b", "b"]) == 1.0)

    j = jaccard(["a", "b"], ["a", "b", "c"])
    check("2/3 overlap", abs(j - 2 / 3) < 0.01, f"got {j}")


# =============================================
# PART 3: compute_retrieval_signals
# =============================================

def test_compute_retrieval_signals_full():
    print("\n-- compute_retrieval_signals (full) --")
    debug = {
        "selected_ids": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"],
        "reranked": [
            {"id": "c1", "score": 9.5},
            {"id": "c2", "score": 7.0},
            {"id": "c3", "score": 6.0},
        ],
        "bm25_ids": ["c1", "c2", "c3", "c4", "c5"],
        "dense_ids": ["c1", "c2", "c3", "c6", "c7"],
    }
    sig = compute_retrieval_signals(debug)

    check("selected_n = 8", sig["selected_n"] == 8)
    check("coverage = 1.0 (8 chunks)", sig["coverage"] == 1.0)
    check("rr_margin > 0", sig["rr_margin"] > 0, f"got {sig['rr_margin']}")
    check("retrieval_agreement in [0,1]", 0 <= sig["retrieval_agreement"] <= 1)
    check("answer_agreement defaults to 0.5", sig["answer_agreement"] == 0.5)

    expected_margin = (9.5 - 7.0) / 9.5
    check("margin ~0.26", abs(sig["rr_margin"] - expected_margin) < 0.01, f"got {sig['rr_margin']}")

    bm25_set = {"c1", "c2", "c3", "c4", "c5"}
    dense_set = {"c1", "c2", "c3", "c6", "c7"}
    expected_jacc = len(bm25_set & dense_set) / len(bm25_set | dense_set)
    check("agreement matches jaccard", abs(sig["retrieval_agreement"] - expected_jacc) < 0.01)


def test_compute_retrieval_signals_sparse():
    print("\n-- compute_retrieval_signals (sparse) --")
    sig = compute_retrieval_signals({"selected_ids": ["c1", "c2"]})
    check("coverage = 2/8 = 0.25", sig["coverage"] == 0.25)
    check("margin = 0 (no reranked)", sig["rr_margin"] == 0.0)
    check("agreement defaults to 0.5", sig["retrieval_agreement"] == 0.5)


def test_compute_retrieval_signals_empty():
    print("\n-- compute_retrieval_signals (empty) --")
    sig = compute_retrieval_signals({})
    check("selected_n = 0", sig["selected_n"] == 0)
    check("coverage = 0", sig["coverage"] == 0.0)
    check("margin = 0", sig["rr_margin"] == 0.0)
    check("agreement = 0.5", sig["retrieval_agreement"] == 0.5)
    check("answer_agreement = 0.5", sig["answer_agreement"] == 0.5)


def test_compute_retrieval_signals_answer_agreement():
    print("\n-- compute_retrieval_signals (answer agreement) --")
    sig_same = compute_retrieval_signals({
        "selected_ids": [],
        "answer_small_norm": "revenue was 391B",
        "answer_large_norm": "revenue was 391B",
    })
    check("same answers = 1.0", sig_same["answer_agreement"] == 1.0)

    sig_diff = compute_retrieval_signals({
        "selected_ids": [],
        "answer_small_norm": "revenue was 391B",
        "answer_large_norm": "revenue was 383B",
    })
    check("different answers = 0.0", sig_diff["answer_agreement"] == 0.0)


# =============================================
# PART 4: risk_score
# =============================================

def test_risk_score():
    print("\n-- risk_score --")

    # Perfect signals -> low risk
    sig_perfect = {"coverage": 1.0, "rr_margin": 1.0, "retrieval_agreement": 1.0, "answer_agreement": 1.0}
    r, reasons = risk_score(sig_perfect, mode="lookup_text")
    check("perfect signals -> risk = 0", r == 0.0, f"got {r}")
    check("no reasons for perfect", len(reasons) == 0, f"got {reasons}")

    # Worst signals -> high risk
    sig_worst = {"coverage": 0.0, "rr_margin": 0.0, "retrieval_agreement": 0.0, "answer_agreement": 0.0}
    r2, reasons2 = risk_score(sig_worst, mode="lookup_text")
    check("worst signals -> risk ~ 1.0", abs(r2 - 1.0) < 1e-9, f"got {r2}")
    check("low_coverage in reasons", "low_coverage" in reasons2)
    check("low_retrieval_agreement in reasons", "low_retrieval_agreement" in reasons2)
    check("low_margin in reasons", "low_margin" in reasons2)

    # Hard mode penalty
    r3, reasons3 = risk_score(sig_perfect, mode="lookup_numeric")
    check("hard mode adds penalty", r3 == 0.10, f"got {r3}")
    check("hard_mode_penalty in reasons", "hard_mode_penalty" in reasons3)

    r4, _ = risk_score(sig_perfect, mode="compute_metric")
    check("compute_metric also penalized", r4 == 0.10)

    r5, _ = risk_score(sig_perfect, mode="valuation")
    check("valuation also penalized", r5 == 0.10)

    # Verify weight distribution: only coverage bad
    sig_bad_cov = {"coverage": 0.0, "rr_margin": 1.0, "retrieval_agreement": 1.0, "answer_agreement": 1.0}
    r6, _ = risk_score(sig_bad_cov, mode="lookup_text")
    check("bad coverage only -> 0.50", abs(r6 - 0.50) < 0.01, f"got {r6}")

    # Only agreement bad
    sig_bad_agr = {"coverage": 1.0, "rr_margin": 1.0, "retrieval_agreement": 0.0, "answer_agreement": 1.0}
    r7, _ = risk_score(sig_bad_agr, mode="lookup_text")
    check("bad agreement only -> 0.20", abs(r7 - 0.20) < 0.01, f"got {r7}")

    # Only margin bad
    sig_bad_mar = {"coverage": 1.0, "rr_margin": 0.0, "retrieval_agreement": 1.0, "answer_agreement": 1.0}
    r8, _ = risk_score(sig_bad_mar, mode="lookup_text")
    check("bad margin only -> 0.20", abs(r8 - 0.20) < 0.01, f"got {r8}")

    # Risk always in [0,1]
    check("risk clipped to [0,1]", 0 <= r2 <= 1)


# =============================================
# PART 5: choose_model_from_risk
# =============================================

def test_choose_model():
    print("\n-- choose_model_from_risk --")
    check("risk 0 -> small", choose_model_from_risk(0.0) == "small")
    check("risk 0.35 -> small", choose_model_from_risk(0.35) == "small")
    check("risk 0.36 -> large", choose_model_from_risk(0.36) == "large")
    check("risk 0.70 -> large", choose_model_from_risk(0.70) == "large")
    check("risk 1.0 -> large", choose_model_from_risk(1.0) == "large")
    check("negative clipped -> small", choose_model_from_risk(-1.0) == "small")
    check("above 1 clipped -> large", choose_model_from_risk(5.0) == "large")


# =============================================
# PART 6: decide() gate-dominated paths
# =============================================

def test_decide_gate_failures():
    print("\n-- decide (gate failures) --")

    # Clarify
    d = decide(plan_mode="lookup_numeric", gate_action="clarify", gate_ok=False, retrieval_debug={})
    check("clarify action", d.action == "clarify")
    check("clarify model = none", d.model == "none")
    check("clarify risk = 1.0", d.risk == 1.0)
    check("clarify reason", "gate_clarify" in d.reasons)

    # Needs market data
    d2 = decide(plan_mode="valuation", gate_action="needs_market_data", gate_ok=False, retrieval_debug={})
    check("needs_market -> clarify", d2.action == "clarify")
    check("needs_market reason", "gate_needs_market_data" in d2.reasons)

    # Abstain
    d3 = decide(plan_mode="lookup_numeric", gate_action="abstain", gate_ok=False, retrieval_debug={})
    check("abstain action", d3.action == "abstain")
    check("abstain model = none", d3.model == "none")
    check("abstain risk = 1.0", d3.risk == 1.0)

    # Unknown gate action when not ok
    d4 = decide(plan_mode="lookup_text", gate_action="unknown_action", gate_ok=False, retrieval_debug={})
    check("unknown gate action -> abstain", d4.action == "abstain")


# =============================================
# PART 7: decide() with passing gate
# =============================================

def test_decide_gate_pass():
    print("\n-- decide (gate passes) --")

    # Good evidence -> small model
    good_debug = {
        "selected_ids": ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"],
        "reranked": [{"id": "c1", "score": 9.0}, {"id": "c2", "score": 5.0}],
        "bm25_ids": ["c1", "c2", "c3", "c4", "c5"],
        "dense_ids": ["c1", "c2", "c3", "c4", "c5"],
    }
    d = decide(plan_mode="lookup_text", gate_action="pass", gate_ok=True, retrieval_debug=good_debug)
    check("good evidence -> answer", d.action == "answer")
    check("good evidence -> small", d.model == "small", f"got {d.model}, risk={d.risk}")
    check("risk < 0.35", d.risk <= 0.35, f"got {d.risk}")

    # Poor evidence -> large model
    poor_debug = {
        "selected_ids": ["c1"],
        "reranked": [{"id": "c1", "score": 3.0}, {"id": "c2", "score": 2.9}],
        "bm25_ids": ["c1", "c2"],
        "dense_ids": ["c5", "c6"],
    }
    d2 = decide(plan_mode="lookup_text", gate_action="pass", gate_ok=True, retrieval_debug=poor_debug)
    check("poor evidence -> answer", d2.action == "answer")
    check("poor evidence -> large", d2.model == "large", f"got {d2.model}, risk={d2.risk}")
    check("risk > 0.35", d2.risk > 0.35, f"got {d2.risk}")


# =============================================
# PART 8: decide() extreme risk abstain
# =============================================

def test_decide_extreme_risk():
    print("\n-- decide (extreme risk) --")
    empty_debug = {
        "selected_ids": [],
        "reranked": [],
        "bm25_ids": [],
        "dense_ids": [],
    }
    d = decide(plan_mode="lookup_numeric", gate_action="pass", gate_ok=True, retrieval_debug=empty_debug)
    check("extreme risk -> abstain", d.action == "abstain", f"got action={d.action}, risk={d.risk}")
    check("extreme risk > 0.90", d.risk > 0.90, f"got {d.risk}")
    check("risk_too_high reason", "risk_too_high" in d.reasons)
    check("model is none", d.model == "none")


# =============================================
# PART 9: End-to-end scenarios
# =============================================

def test_e2e_numeric_good():
    print("\n-- E2E: numeric query, good evidence --")
    debug = {
        "selected_ids": [f"c{i}" for i in range(12)],
        "reranked": [{"id": "c0", "score": 8.5}, {"id": "c1", "score": 4.0}],
        "bm25_ids": [f"c{i}" for i in range(10)],
        "dense_ids": [f"c{i}" for i in range(10)],
    }
    d = decide(plan_mode="lookup_numeric", gate_action="pass", gate_ok=True, retrieval_debug=debug)
    check("answers", d.action == "answer")
    check("mode preserved", d.mode == "lookup_numeric")
    check("has hard_mode_penalty", "hard_mode_penalty" in d.reasons)
    check("risk reasonable", 0 < d.risk < 0.5, f"got {d.risk}")


def test_e2e_narrative_poor():
    print("\n-- E2E: narrative query, poor evidence --")
    debug = {
        "selected_ids": ["c1", "c2"],
        "reranked": [{"id": "c1", "score": 3.0}, {"id": "c2", "score": 2.8}],
        "bm25_ids": ["c1"],
        "dense_ids": ["c5"],
    }
    d = decide(plan_mode="explanatory_reasoning", gate_action="pass", gate_ok=True, retrieval_debug=debug)
    check("narrative poor -> answer or abstain", d.action in ("answer", "abstain"))
    check("model is large or none", d.model in ("large", "none"), f"got {d.model}")


# =============================================
# PART 10: RoutingDecision dataclass
# =============================================

def test_routing_decision_dataclass():
    print("\n-- RoutingDecision dataclass --")
    d = RoutingDecision(action="answer", mode="lookup_text", model="small", risk=0.2, reasons=[], signals={})
    check("action accessible", d.action == "answer")
    check("mode accessible", d.mode == "lookup_text")
    check("model accessible", d.model == "small")
    check("risk accessible", d.risk == 0.2)
    check("reasons accessible", d.reasons == [])
    check("signals accessible", d.signals == {})


# =============================================
# PART 11: Edge cases
# =============================================

def test_edge_cases():
    print("\n-- edge cases --")

    # None retrieval_debug
    d = decide(plan_mode="lookup_text", gate_action="pass", gate_ok=True, retrieval_debug=None)
    check("None debug doesn't crash", d.action in ("answer", "abstain"))

    # Single reranked item (no margin computable)
    sig = compute_retrieval_signals({"reranked": [{"id": "c1", "score": 9.0}]})
    check("single reranked -> margin 0", sig["rr_margin"] == 0.0)

    # Reranked with bad score types
    sig2 = compute_retrieval_signals({"reranked": [{"id": "c1"}, {"id": "c2"}]})
    check("missing scores -> margin 0", sig2["rr_margin"] == 0.0)

    # Coverage boundaries
    sig3 = compute_retrieval_signals({"selected_ids": ["c1"] * 4})
    check("4 chunks -> coverage 0.5", sig3["coverage"] == 0.5, f"got {sig3['coverage']}")

    sig4 = compute_retrieval_signals({"selected_ids": ["c1"] * 20})
    check("20 chunks -> coverage capped at 1.0", sig4["coverage"] == 1.0)


# =============================================
# PART 12: Module-level constants
# =============================================

def test_module_constants():
    print("\n-- module constants --")
    check("HARD_MODES is frozenset", isinstance(HARD_MODES, frozenset))
    check("lookup_numeric in HARD_MODES", "lookup_numeric" in HARD_MODES)
    check("compute_metric in HARD_MODES", "compute_metric" in HARD_MODES)
    check("valuation in HARD_MODES", "valuation" in HARD_MODES)
    check("COVERAGE_FULL_AT > 0", COVERAGE_FULL_AT > 0)
    check("SMALL_MODEL_MAX_RISK in (0,1)", 0 < SMALL_MODEL_MAX_RISK < 1)
    check("ABSTAIN_MIN_RISK > SMALL_MODEL_MAX_RISK", ABSTAIN_MIN_RISK > SMALL_MODEL_MAX_RISK)
    check("weights sum to 1.0", abs((W_COVERAGE + W_AGREEMENT + W_MARGIN + W_ANSWER) - 1.0) < 1e-9)


# =============================================
# PART 13: Input validation
# =============================================

def test_input_validation():
    print("\n-- input validation --")
    try:
        decide(plan_mode="", gate_action="pass", gate_ok=True, retrieval_debug={})
        check("rejects empty plan_mode", False)
    except ValueError:
        check("rejects empty plan_mode", True)

    try:
        decide(plan_mode="lookup_text", gate_action=123, gate_ok=True, retrieval_debug={})
        check("rejects non-string gate_action", False)
    except TypeError:
        check("rejects non-string gate_action", True)


# =============================================
# MAIN
# =============================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("ROUTING.PY LOGIC TEST SUITE")
    print("=" * 60)

    test_clip01()
    test_jaccard()
    test_compute_retrieval_signals_full()
    test_compute_retrieval_signals_sparse()
    test_compute_retrieval_signals_empty()
    test_compute_retrieval_signals_answer_agreement()
    test_risk_score()
    test_choose_model()
    test_decide_gate_failures()
    test_decide_gate_pass()
    test_decide_extreme_risk()
    test_e2e_numeric_good()
    test_e2e_narrative_poor()
    test_routing_decision_dataclass()
    test_edge_cases()
    test_module_constants()
    test_input_validation()

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
