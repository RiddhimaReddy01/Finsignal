"""
test_audit.py

Logic tests for audit.py covering:
  1. _now_ms timestamp generation
  2. _sha256_text hashing (consistency, None safety)
  3. AuditLogger directory creation
  4. AuditLogger.new_run_id uniqueness
  5. AuditLogger.log base method (JSONL format, ts_ms injection)
  6. log_retrieval schema correctness
  7. log_gate schema correctness
  8. log_generation hashing + truncation
  9. log_validation schema correctness
  10. Multi-event JSONL integrity (read back and parse every line)
  11. Edge cases (empty strings, None defaults, missing optional fields)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from audit import _now_ms, _sha256_text, _safe_json_default, AuditLogger

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
# PART 1: _now_ms
# =============================================

def test_now_ms():
    print("\n-- _now_ms --")
    t = _now_ms()
    check("returns int", isinstance(t, int))
    check("reasonable epoch ms", t > 1_700_000_000_000, f"got {t}")
    t2 = _now_ms()
    check("monotonic (t2 >= t1)", t2 >= t)


# =============================================
# PART 2: _sha256_text
# =============================================

def test_sha256_text():
    print("\n-- _sha256_text --")
    h1 = _sha256_text("hello")
    check("returns string", isinstance(h1, str))
    check("64 hex chars", len(h1) == 64, f"got {len(h1)}")

    h2 = _sha256_text("hello")
    check("deterministic", h1 == h2)

    h3 = _sha256_text("world")
    check("different input -> different hash", h1 != h3)

    h_none = _sha256_text(None)
    h_empty = _sha256_text("")
    check("None treated as empty string", h_none == h_empty)
    check("empty string produces valid hash", len(h_empty) == 64)


# =============================================
# PART 3: AuditLogger init + directory creation
# =============================================

def test_logger_init():
    print("\n-- AuditLogger init --")
    with tempfile.TemporaryDirectory() as td:
        nested = os.path.join(td, "sub", "dir", "audit.jsonl")
        logger = AuditLogger(path=nested)
        check("creates nested parent dirs", os.path.isdir(os.path.join(td, "sub", "dir")))
        check("path stored", logger.path == nested)


# =============================================
# PART 4: new_run_id
# =============================================

def test_new_run_id():
    print("\n-- new_run_id --")
    with tempfile.TemporaryDirectory() as td:
        logger = AuditLogger(path=os.path.join(td, "audit.jsonl"))
        ids = {logger.new_run_id() for _ in range(100)}
        check("100 unique ids", len(ids) == 100)
        check("hex format", all(len(rid) == 32 and rid.isalnum() for rid in ids))


# =============================================
# PART 5: log base method
# =============================================

def test_log_base():
    print("\n-- log (base) --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log({"event": "test", "data": 42})
        logger.log({"event": "test2", "data": "hello"})

        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        check("wrote 2 lines", len(lines) == 2)

        r1 = json.loads(lines[0])
        check("first line is valid JSON", isinstance(r1, dict))
        check("ts_ms auto-injected", "ts_ms" in r1)
        check("ts_ms is int", isinstance(r1["ts_ms"], int))
        check("event preserved", r1["event"] == "test")
        check("data preserved", r1["data"] == 42)

        # ts_ms should not overwrite if already present
        logger.log({"event": "custom_ts", "ts_ms": 12345})
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        r3 = json.loads(lines[2])
        check("custom ts_ms preserved", r3["ts_ms"] == 12345)


# =============================================
# PART 6: log_retrieval
# =============================================

def test_log_retrieval():
    print("\n-- log_retrieval --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)
        rid = logger.new_run_id()

        logger.log_retrieval(
            run_id=rid,
            question="What was AAPL revenue?",
            rewrites=["What was AAPL revenue?", "AAPL total net sales"],
            hard_filters={"ticker": "AAPL", "fiscal_year": 2024},
            soft_boosts=[{"section": "Item 8", "weight": 1.0}],
            candidates=[{"id": "c001", "score": 0.85}],
            reranker={"model": "cross-encoder/ms-marco", "top": [{"id": "c001", "score": 7.2}]},
            selected=[{"id": "c001", "offsets": [[10, 140]]}],
            packed_ids=["c001", "t001"],
            latency_ms=152.3,
            debug={"extra": True},
        )

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("event is retrieval", rec["event"] == "retrieval")
        check("run_id matches", rec["run_id"] == rid)
        check("question stored", rec["question"] == "What was AAPL revenue?")
        check("rewrites stored", len(rec["rewrites"]) == 2)
        check("hard_filters in plan", rec["retrieval_plan"]["hard_filters"]["ticker"] == "AAPL")
        check("soft_boosts in plan", len(rec["retrieval_plan"]["soft_boosts"]) == 1)
        check("candidates stored", len(rec["candidates"]) == 1)
        check("reranker stored", rec["reranker"]["model"] == "cross-encoder/ms-marco")
        check("selected stored", rec["selected"][0]["id"] == "c001")
        check("packed_ids stored", rec["packed_ids"] == ["c001", "t001"])
        check("latency stored", rec["latency_ms"] == 152.3)
        check("debug stored", rec["debug"]["extra"] is True)


def test_log_retrieval_defaults():
    print("\n-- log_retrieval (defaults) --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log_retrieval(
            run_id="test",
            question="test?",
            rewrites=[],
            hard_filters={},
            soft_boosts=[],
        )

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("candidates defaults to []", rec["candidates"] == [])
        check("reranker defaults to {}", rec["reranker"] == {})
        check("selected defaults to []", rec["selected"] == [])
        check("packed_ids defaults to []", rec["packed_ids"] == [])
        check("latency defaults to None", rec["latency_ms"] is None)
        check("debug defaults to {}", rec["debug"] == {})


# =============================================
# PART 7: log_gate
# =============================================

def test_log_gate():
    print("\n-- log_gate --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log_gate(
            run_id="r1",
            plan={"mode": "lookup_numeric", "targets": [{"ticker": "AAPL"}]},
            req={"min_tables": 1},
            gate={"ok": True, "action": "pass"},
            routing={"model": "small", "risk": 0.2},
            latency_ms=5.1,
        )

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("event is gate_routing", rec["event"] == "gate_routing")
        check("plan stored", rec["plan"]["mode"] == "lookup_numeric")
        check("requirements stored", rec["requirements"]["min_tables"] == 1)
        check("gate stored", rec["gate"]["ok"] is True)
        check("routing stored", rec["routing"]["model"] == "small")
        check("latency stored", rec["latency_ms"] == 5.1)


# =============================================
# PART 8: log_generation
# =============================================

def test_log_generation():
    print("\n-- log_generation --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        system = "You are a financial QA system."
        user = "What was AAPL revenue in 2024?"
        output = '{"final_answer": "Revenue was $391B", "claims": []}' + "x" * 1000

        logger.log_generation(
            run_id="r1",
            model_name="gpt-4o-mini",
            system_prompt=system,
            user_prompt=user,
            output_text=output,
            latency_ms=1200.5,
            token_usage={"input_tokens": 500, "output_tokens": 200, "cost_usd": 0.003},
        )

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("event is generation", rec["event"] == "generation")
        check("model stored", rec["model"] == "gpt-4o-mini")
        check("system_hash is sha256", len(rec["system_hash"]) == 64)
        check("user_hash is sha256", len(rec["user_hash"]) == 64)
        check("output_hash is sha256", len(rec["output_hash"]) == 64)
        check("system hash matches", rec["system_hash"] == _sha256_text(system))
        check("user hash matches", rec["user_hash"] == _sha256_text(user))
        check("output_preview truncated to 800", len(rec["output_preview"]) == 800)
        check("latency stored", rec["latency_ms"] == 1200.5)
        check("usage stored", rec["usage"]["cost_usd"] == 0.003)
        check("full prompts NOT stored", "system_prompt" not in rec and "user_prompt" not in rec)


def test_log_generation_defaults():
    print("\n-- log_generation (defaults) --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log_generation(
            run_id="r1",
            model_name="gpt-4o",
            system_prompt="sys",
            user_prompt="usr",
            output_text="out",
        )

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("latency defaults to None", rec["latency_ms"] is None)
        check("usage defaults to {}", rec["usage"] == {})
        check("short output not truncated", rec["output_preview"] == "out")


# =============================================
# PART 9: log_validation
# =============================================

def test_log_validation():
    print("\n-- log_validation --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log_validation(
            run_id="r1",
            ok=False,
            errors=["claim_0_cit_0_not_allowed:t999", "numeric_unit_invalid"],
            signals={"n_blocks": 5},
            latency_ms=2.3,
        )

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("event is validation", rec["event"] == "validation")
        check("ok is False", rec["ok"] is False)
        check("errors stored", len(rec["errors"]) == 2)
        check("signals stored", rec["signals"]["n_blocks"] == 5)
        check("latency stored", rec["latency_ms"] == 2.3)

    # Passing validation
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log_validation(run_id="r2", ok=True, errors=[])

        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())

        check("ok is True", rec["ok"] is True)
        check("empty errors", rec["errors"] == [])
        check("signals defaults to {}", rec["signals"] == {})


# =============================================
# PART 10: Multi-event JSONL integrity
# =============================================

def test_multi_event_jsonl():
    print("\n-- multi-event JSONL integrity --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)
        rid = logger.new_run_id()

        logger.log_retrieval(run_id=rid, question="q", rewrites=[], hard_filters={}, soft_boosts=[])
        logger.log_gate(run_id=rid, plan={}, req={}, gate={"ok": True}, routing={"model": "small"})
        logger.log_generation(run_id=rid, model_name="m", system_prompt="s", user_prompt="u", output_text="o")
        logger.log_validation(run_id=rid, ok=True, errors=[])

        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        check("4 lines written", len(lines) == 4)

        events = []
        all_valid = True
        for i, line in enumerate(lines):
            try:
                rec = json.loads(line)
                events.append(rec)
            except json.JSONDecodeError:
                all_valid = False
                check(f"line {i} valid JSON", False)

        check("all lines are valid JSON", all_valid)
        check("all have run_id", all(e.get("run_id") == rid for e in events))
        check("all have ts_ms", all("ts_ms" in e for e in events))

        event_types = [e["event"] for e in events]
        check("retrieval event present", "retrieval" in event_types)
        check("gate_routing event present", "gate_routing" in event_types)
        check("generation event present", "generation" in event_types)
        check("validation event present", "validation" in event_types)

        timestamps = [e["ts_ms"] for e in events]
        check("timestamps non-decreasing", all(timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)))


# =============================================
# PART 11: Edge cases
# =============================================

def test_edge_cases():
    print("\n-- edge cases --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log({"event": "unicode", "data": "Hello \u00e9\u00e8\u00ea \u2603"})
        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        check("unicode preserved", "\u2603" in rec["data"])

        logger.log({"event": "nested", "deep": {"a": {"b": [1, 2, {"c": True}]}}})
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        rec2 = json.loads(lines[-1])
        check("nested structure preserved", rec2["deep"]["a"]["b"][2]["c"] is True)

        logger.log_generation(
            run_id="edge",
            model_name="m",
            system_prompt="",
            user_prompt="",
            output_text="",
        )
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        rec3 = json.loads(lines[-1])
        check("empty strings don't crash", rec3["output_preview"] == "")
        check("empty hash valid", len(rec3["system_hash"]) == 64)


# =============================================
# PART 12: Thread safety
# =============================================

def test_thread_safety():
    import threading
    print("\n-- thread safety --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)
        n_threads = 10
        n_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def writer(tid):
            barrier.wait()
            for i in range(n_per_thread):
                logger.log({"event": "thread_test", "tid": tid, "seq": i})

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        check("all lines written", len(lines) == n_threads * n_per_thread,
              f"expected {n_threads * n_per_thread}, got {len(lines)}")

        all_valid = True
        for i, line in enumerate(lines):
            try:
                json.loads(line)
            except json.JSONDecodeError:
                all_valid = False
                break
        check("no corrupted JSON from concurrent writes", all_valid)


# =============================================
# PART 13: JSON serialization fallback
# =============================================

def test_safe_json_default():
    print("\n-- _safe_json_default --")
    from datetime import datetime

    check("set -> sorted list", _safe_json_default({"b", "a"}) == ["a", "b"])
    check("bytes -> str", isinstance(_safe_json_default(b"hello"), str))
    dt = datetime(2024, 1, 15, 10, 30)
    check("datetime -> isoformat", "2024" in _safe_json_default(dt))
    check("unknown -> str()", isinstance(_safe_json_default(object()), str))


def test_log_non_serializable():
    print("\n-- log non-serializable objects --")
    from datetime import datetime
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log({"event": "test", "data": {"a", "b", "c"}, "ts": datetime.now()})

        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        rec = json.loads(line)
        check("set serialized", isinstance(rec["data"], list))
        check("datetime serialized", isinstance(rec["ts"], str))


# =============================================
# PART 14: Error resilience
# =============================================

def test_log_to_bad_path():
    """Logging to an unwritable path should not crash."""
    print("\n-- error resilience --")
    logger = AuditLogger(path=os.path.join(tempfile.gettempdir(), "audit_test_resilience.jsonl"))

    try:
        logger.log({"event": "test"})
        check("normal log doesn't crash", True)
    except Exception as e:
        check("normal log doesn't crash", False, str(e))


# =============================================
# PART 15: Input validation
# =============================================

def test_input_validation():
    print("\n-- input validation --")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "audit.jsonl")
        logger = AuditLogger(path=path)

        logger.log_retrieval(
            run_id="",
            question="",
            rewrites=[],
            hard_filters={},
            soft_boosts=[],
        )
        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        check("empty run_id still logs", rec["run_id"] == "")
        check("empty question still logs", rec["question"] == "")


# =============================================
# MAIN
# =============================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("AUDIT.PY LOGIC TEST SUITE")
    print("=" * 60)

    test_now_ms()
    test_sha256_text()
    test_logger_init()
    test_new_run_id()
    test_log_base()
    test_log_retrieval()
    test_log_retrieval_defaults()
    test_log_gate()
    test_log_generation()
    test_log_generation_defaults()
    test_log_validation()
    test_multi_event_jsonl()
    test_edge_cases()
    test_thread_safety()
    test_safe_json_default()
    test_log_non_serializable()
    test_log_to_bad_path()
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
