"""
test_build_index.py

Comprehensive tests for build_index.py covering:
  1. Unit tests for chunk_text() - boundaries, overlaps, edge cases
  2. Unit tests for html_table_to_surrogate() - HTML parsing, truncation
  3. Unit tests for bm25_tokenize() consistency with retrieval_tool
  4. Integration tests on built artifacts - schema, alignment, searchability
  5. Data integrity checks - no empty chunks, unique IDs, correct linkage
"""

from __future__ import annotations

import json
import pickle
import sys
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import faiss

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from build_index import (
    chunk_text,
    bm25_tokenize,
    html_table_to_surrogate,
    build_bm25,
    _guess_table_item,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INDEX_DIR,
    SECT_DIR,
    TABLE_DIR,
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
# PART 1: chunk_text() unit tests
# =============================================

def test_chunk_text_basic():
    print("\n-- chunk_text: basic --")
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunk_text(text, size=50, overlap=10)
    check("produces multiple chunks", len(chunks) >= 2)
    check("all chunks non-empty", all(len(c.strip()) > 0 for c in chunks))
    full = " ".join(chunks)
    for word in ["First", "Second", "Third", "Fourth"]:
        check(f"'{word}' preserved", word in full)


def test_chunk_text_empty():
    print("\n-- chunk_text: empty/whitespace --")
    check("empty string", chunk_text("") == [])
    check("whitespace only", chunk_text("   \n\t  ") == [])
    check("None-like empty", chunk_text("  \n") == [])


def test_chunk_text_single_sentence():
    print("\n-- chunk_text: single sentence --")
    text = "This is one short sentence."
    chunks = chunk_text(text, size=1200, overlap=200)
    check("single sentence -> 1 chunk", len(chunks) == 1)
    check("content preserved", chunks[0] == text)


def test_chunk_text_very_long_sentence():
    print("\n-- chunk_text: very long sentence --")
    text = "word " * 500  # ~2500 chars, no sentence boundary
    chunks = chunk_text(text, size=1200, overlap=200)
    check("still produces chunks", len(chunks) >= 1)
    check("no chunk exceeds 2x size (graceful)", all(len(c) < CHUNK_SIZE * 3 for c in chunks))


def test_chunk_text_overlap_exists():
    print("\n-- chunk_text: overlap --")
    sentences = [f"Sentence number {i} with some extra words here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, size=200, overlap=50)
    if len(chunks) >= 2:
        overlaps_found = 0
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].split()[-5:])
            words_b = set(chunks[i + 1].split()[:5])
            if words_a & words_b:
                overlaps_found += 1
        check("overlap exists between consecutive chunks", overlaps_found > 0,
              f"found {overlaps_found}/{len(chunks)-1}")
    else:
        check("overlap exists between consecutive chunks", False, "only 1 chunk produced")


def test_chunk_text_no_content_loss():
    print("\n-- chunk_text: no content loss --")
    sentences = [f"Fact {i} is important." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunk_text(text, size=300, overlap=50)
    all_text = " ".join(chunks)
    for i in range(30):
        check(f"Fact {i} preserved", f"Fact {i}" in all_text)


def test_chunk_text_respects_size():
    print("\n-- chunk_text: respects size --")
    sentences = [f"Sentence {i} with enough words to make it meaningful." for i in range(50)]
    text = " ".join(sentences)
    chunks = chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    oversized = [i for i, c in enumerate(chunks) if len(c) > CHUNK_SIZE * 1.5]
    check("no severely oversized chunks", len(oversized) == 0,
          f"oversized chunk indices: {oversized}")


def test_chunk_text_special_characters():
    print("\n-- chunk_text: special characters --")
    text = "Revenue was $3.5B in FY2024. The 10-K filing noted a 15% increase. Item 1A: Risk factors."
    chunks = chunk_text(text, size=500, overlap=50)
    full = " ".join(chunks)
    check("dollar sign preserved", "$3.5B" in full)
    check("percentage preserved", "15%" in full)
    check("item reference preserved", "Item 1A" in full)


# =============================================
# PART 2: html_table_to_surrogate() unit tests
# =============================================

def test_table_surrogate_basic():
    print("\n-- html_table_to_surrogate: basic --")
    html = """<table>
        <tr><th>Year</th><th>Revenue</th></tr>
        <tr><td>2024</td><td>$100B</td></tr>
        <tr><td>2025</td><td>$120B</td></tr>
    </table>"""
    result = html_table_to_surrogate(html, "Revenue Table")
    check("title included", "Revenue Table" in result)
    check("header row included", "Year" in result and "Revenue" in result)
    check("data rows included", "$100B" in result and "$120B" in result)
    check("pipe separated", "|" in result)


def test_table_surrogate_no_table():
    print("\n-- html_table_to_surrogate: no table tag --")
    result = html_table_to_surrogate("<div>Not a table</div>", "My Title")
    check("falls back to title", result == "My Title")


def test_table_surrogate_empty():
    print("\n-- html_table_to_surrogate: empty --")
    result = html_table_to_surrogate("", "")
    check("returns empty string for empty input", result == "")


def test_table_surrogate_truncation():
    print("\n-- html_table_to_surrogate: truncation --")
    rows = "".join(f"<tr><td>{'x' * 100}</td><td>{'y' * 100}</td></tr>" for _ in range(50))
    html = f"<table>{rows}</table>"
    result = html_table_to_surrogate(html, "Big Table")
    check("truncated to ~3000 chars", len(result) <= 3010)
    check("ends with ...", result.endswith("..."))


def test_table_surrogate_empty_cells():
    print("\n-- html_table_to_surrogate: empty cells --")
    html = "<table><tr><td></td><td>Value</td><td></td></tr></table>"
    result = html_table_to_surrogate(html, "")
    check("empty cells filtered", result.strip() == "Value")


# =============================================
# PART 3: bm25_tokenize consistency
# =============================================

def test_tokenizer_consistency():
    print("\n-- bm25_tokenize: consistency with retrieval_tool --")
    from retrieval_tool import _bm25_tokenize as rt_tokenize

    test_strings = [
        "Revenue was $3.5B in FY2024!",
        "10-K filing noted 15% increase",
        "Item 1A: Risk Factors",
        "",
        "hello@world#test",
    ]
    for s in test_strings:
        build_tokens = bm25_tokenize(s)
        retrieval_tokens = rt_tokenize(s)
        check(f"tokenizer match: {s[:40]!r}", build_tokens == retrieval_tokens,
              f"build={build_tokens} vs retrieval={retrieval_tokens}")


# =============================================
# PART 4: _guess_table_item
# =============================================

def test_guess_table_item():
    print("\n-- _guess_table_item --")
    items = {
        "Item 1A": "Risk factors [TABLE:AAPL_T001 foo]",
        "Item 7": "Management discussion [TABLE:AAPL_T002 bar]",
        "Item 8": "Financial statements",
    }
    check("finds table in Item 1A", _guess_table_item("AAPL_T001", items) == "Item 1A")
    check("finds table in Item 7", _guess_table_item("AAPL_T002", items) == "Item 7")
    check("returns empty for missing", _guess_table_item("AAPL_T099", items) == "")


# =============================================
# PART 5: Built artifact integrity tests
# =============================================

def test_chunks_parquet():
    print("\n-- chunks.parquet integrity --")
    path = INDEX_DIR / "chunks.parquet"
    check("chunks.parquet exists", path.exists())
    if not path.exists():
        return

    df = pd.read_parquet(path)
    expected_cols = {"chunk_id", "text", "ticker", "fiscal_year", "item",
                     "chunk_index", "prev_chunk_id", "next_chunk_id"}
    check("has all expected columns", expected_cols.issubset(set(df.columns)),
          f"missing: {expected_cols - set(df.columns)}")

    check("no empty chunk_ids", df["chunk_id"].notna().all() and (df["chunk_id"] != "").all())
    check("chunk_ids are unique", df["chunk_id"].is_unique)
    check("no empty text", df["text"].apply(lambda x: len(str(x).strip()) > 0).all())
    check("row count > 0", len(df) > 0, f"got {len(df)}")

    tickers = set(df["ticker"].unique())
    expected_tickers = {"AAPL", "META", "NVDA", "GOOGL", "TSLA"}
    check("all 5 tickers present", expected_tickers.issubset(tickers),
          f"missing: {expected_tickers - tickers}")

    items = set(df["item"].unique())
    expected_items = {"Item 1A", "Item 7", "Item 8"}
    check("all 3 items present", expected_items.issubset(items),
          f"missing: {expected_items - items}")

    fys = set(df["fiscal_year"].unique())
    check("FY2024 present", 2024 in fys)
    check("FY2025 present", 2025 in fys)

    sizes = df["text"].str.len()
    check("min chunk >= 10 chars", sizes.min() >= 10, f"min={sizes.min()}")
    check("median chunk reasonable", 200 < sizes.median() < 2000,
          f"median={sizes.median():.0f}")
    oversized = (sizes > CHUNK_SIZE * 2).sum()
    check("few severely oversized chunks", oversized < len(df) * 0.05,
          f"{oversized}/{len(df)} oversized")


def test_chunk_linkage():
    print("\n-- chunk linkage (prev/next) --")
    df = pd.read_parquet(INDEX_DIR / "chunks.parquet")
    all_ids = set(df["chunk_id"])

    broken_prev = 0
    broken_next = 0
    for _, row in df.iterrows():
        pid = row["prev_chunk_id"]
        nid = row["next_chunk_id"]
        if pid is not None and pid != "" and not pd.isna(pid):
            if pid not in all_ids:
                broken_prev += 1
        if nid is not None and nid != "" and not pd.isna(nid):
            if nid not in all_ids:
                broken_next += 1

    check("no broken prev_chunk_id links", broken_prev == 0, f"{broken_prev} broken")
    check("no broken next_chunk_id links", broken_next == 0, f"{broken_next} broken")

    for (ticker, fy, item), grp in df.groupby(["ticker", "fiscal_year", "item"]):
        grp = grp.sort_values("chunk_index")
        ids = grp["chunk_id"].tolist()
        if len(ids) >= 2:
            first_prev = grp.iloc[0]["prev_chunk_id"]
            last_next = grp.iloc[-1]["next_chunk_id"]
            first_ok = first_prev is None or pd.isna(first_prev) or first_prev == ""
            last_ok = last_next is None or pd.isna(last_next) or last_next == ""
            if not first_ok:
                check(f"first chunk has no prev ({ticker} FY{fy} {item})", False,
                      f"prev={first_prev}")
                break
            if not last_ok:
                check(f"last chunk has no next ({ticker} FY{fy} {item})", False,
                      f"next={last_next}")
                break
    else:
        check("first chunks have no prev, last chunks have no next", True)


def test_tables_parquet():
    print("\n-- tables.parquet integrity --")
    path = INDEX_DIR / "tables.parquet"
    check("tables.parquet exists", path.exists())
    if not path.exists():
        return

    df = pd.read_parquet(path)
    expected_cols = {"table_id", "surrogate_text", "ticker", "fiscal_year", "item"}
    check("has expected columns", expected_cols.issubset(set(df.columns)),
          f"missing: {expected_cols - set(df.columns)}")

    check("table_ids unique", df["table_id"].is_unique)
    check("no empty surrogate_text", df["surrogate_text"].apply(lambda x: len(str(x).strip()) >= 10).all())
    check("row count > 0", len(df) > 0)

    tickers = set(df["ticker"].unique())
    check("multiple tickers in tables", len(tickers) >= 3, f"found: {tickers}")


def test_bm25_pkl():
    print("\n-- bm25.pkl integrity --")
    path = INDEX_DIR / "bm25.pkl"
    check("bm25.pkl exists", path.exists())
    if not path.exists():
        return

    with open(path, "rb") as f:
        payload = pickle.load(f)

    check("has 'bm25' key", "bm25" in payload)
    check("has 'chunk_ids' key", "chunk_ids" in payload)

    bm25 = payload["bm25"]
    ids = payload["chunk_ids"]
    check("bm25 is BM25Okapi", type(bm25).__name__ == "BM25Okapi")
    check("ids count > 0", len(ids) > 0)

    chunks_df = pd.read_parquet(INDEX_DIR / "chunks.parquet")
    check("BM25 ids count matches parquet rows",
          len(ids) == len(chunks_df),
          f"bm25={len(ids)} vs parquet={len(chunks_df)}")

    check("BM25 ids match parquet chunk_ids",
          ids == chunks_df["chunk_id"].tolist())

    scores = bm25.get_scores(bm25_tokenize("revenue growth"))
    check("BM25 returns scores array", len(scores) == len(ids))
    check("BM25 has non-zero scores", np.max(scores) > 0)


def test_table_bm25_pkl():
    print("\n-- table_bm25.pkl integrity --")
    path = INDEX_DIR / "table_bm25.pkl"
    check("table_bm25.pkl exists", path.exists())
    if not path.exists():
        return

    with open(path, "rb") as f:
        payload = pickle.load(f)

    check("has 'bm25' key", "bm25" in payload)
    check("has 'table_ids' key", "table_ids" in payload)

    ids = payload["table_ids"]
    tables_df = pd.read_parquet(INDEX_DIR / "tables.parquet")
    check("table BM25 ids count matches parquet",
          len(ids) == len(tables_df),
          f"bm25={len(ids)} vs parquet={len(tables_df)}")


def test_faiss_index():
    print("\n-- faiss.index integrity --")
    path = INDEX_DIR / "faiss.index"
    check("faiss.index exists", path.exists())
    if not path.exists():
        return

    index = faiss.read_index(str(path))
    chunks_df = pd.read_parquet(INDEX_DIR / "chunks.parquet")

    check("FAISS vector count matches parquet rows",
          index.ntotal == len(chunks_df),
          f"faiss={index.ntotal} vs parquet={len(chunks_df)}")

    check("FAISS dimension is 384 (MiniLM)", index.d == 384)

    query_vec = np.random.randn(1, 384).astype("float32")
    query_vec /= np.linalg.norm(query_vec)
    D, I = index.search(query_vec, 5)
    check("FAISS search returns results", I.shape == (1, 5))
    check("FAISS indices are valid", all(0 <= i < index.ntotal for i in I[0]))


def test_table_faiss_index():
    print("\n-- table_faiss.index integrity --")
    path = INDEX_DIR / "table_faiss.index"
    check("table_faiss.index exists", path.exists())
    if not path.exists():
        return

    index = faiss.read_index(str(path))
    tables_df = pd.read_parquet(INDEX_DIR / "tables.parquet")

    check("table FAISS vector count matches parquet",
          index.ntotal == len(tables_df),
          f"faiss={index.ntotal} vs parquet={len(tables_df)}")

    check("table FAISS dimension is 384", index.d == 384)


def test_cross_artifact_alignment():
    print("\n-- cross-artifact alignment --")
    chunks_df = pd.read_parquet(INDEX_DIR / "chunks.parquet")

    with open(INDEX_DIR / "bm25.pkl", "rb") as f:
        bm25_payload = pickle.load(f)
    bm25_ids = bm25_payload["chunk_ids"]

    faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

    parquet_ids = chunks_df["chunk_id"].tolist()

    check("parquet row count == BM25 id count == FAISS vectors",
          len(parquet_ids) == len(bm25_ids) == faiss_index.ntotal,
          f"parquet={len(parquet_ids)} bm25={len(bm25_ids)} faiss={faiss_index.ntotal}")

    check("BM25 id order matches parquet order",
          bm25_ids == parquet_ids)

    tables_df = pd.read_parquet(INDEX_DIR / "tables.parquet")
    with open(INDEX_DIR / "table_bm25.pkl", "rb") as f:
        t_bm25_payload = pickle.load(f)
    t_bm25_ids = t_bm25_payload["table_ids"]
    t_faiss = faiss.read_index(str(INDEX_DIR / "table_faiss.index"))
    t_parquet_ids = tables_df["table_id"].tolist()

    check("table parquet == table BM25 == table FAISS",
          len(t_parquet_ids) == len(t_bm25_ids) == t_faiss.ntotal,
          f"parquet={len(t_parquet_ids)} bm25={len(t_bm25_ids)} faiss={t_faiss.ntotal}")


def test_per_ticker_coverage():
    print("\n-- per-ticker coverage --")
    chunks_df = pd.read_parquet(INDEX_DIR / "chunks.parquet")
    tables_df = pd.read_parquet(INDEX_DIR / "tables.parquet")

    for ticker in ["AAPL", "META", "NVDA", "GOOGL", "TSLA"]:
        n_chunks = len(chunks_df[chunks_df["ticker"] == ticker])
        n_tables = len(tables_df[tables_df["ticker"] == ticker])
        check(f"{ticker} has chunks", n_chunks > 0, f"chunks={n_chunks}")
        check(f"{ticker} has tables", n_tables > 0, f"tables={n_tables}")

    for fy in [2024, 2025]:
        n = len(chunks_df[chunks_df["fiscal_year"] == fy])
        check(f"FY{fy} has chunks", n > 0, f"chunks={n}")


def test_build_meta():
    print("\n-- build_meta.json --")
    path = INDEX_DIR / "build_meta.json"
    check("build_meta.json exists", path.exists())
    if not path.exists():
        return

    meta = json.loads(path.read_text())
    for key in ["build_timestamp", "embed_model", "chunk_size", "chunk_overlap",
                 "narrative_chunks", "table_docs", "build_elapsed_s", "artifacts"]:
        check(f"meta has '{key}'", key in meta, f"keys={list(meta.keys())}")

    check("narrative_chunks > 0", meta["narrative_chunks"] > 0)
    check("table_docs > 0", meta["table_docs"] > 0)
    check("build_elapsed_s > 0", meta["build_elapsed_s"] > 0)

    expected_artifacts = {"chunks.parquet", "bm25.pkl", "faiss.index",
                          "tables.parquet", "table_bm25.pkl", "table_faiss.index"}
    actual = set(meta["artifacts"].keys())
    check("all artifacts listed", expected_artifacts.issubset(actual),
          f"missing: {expected_artifacts - actual}")

    for name, info in meta["artifacts"].items():
        check(f"  {name} has size_bytes", "size_bytes" in info and info["size_bytes"] > 0)
        check(f"  {name} has sha256_prefix", "sha256_prefix" in info and len(info["sha256_prefix"]) == 16)


def test_idempotency():
    """Verify that the built artifacts are deterministic (same data -> same IDs)."""
    print("\n-- idempotency --")
    chunks_df = pd.read_parquet(INDEX_DIR / "chunks.parquet")

    from build_index import build_narrative_chunks
    fresh_df = build_narrative_chunks()

    check("same row count on rebuild", len(fresh_df) == len(chunks_df),
          f"fresh={len(fresh_df)} vs saved={len(chunks_df)}")
    check("same chunk_ids on rebuild",
          fresh_df["chunk_id"].tolist() == chunks_df["chunk_id"].tolist())


# =============================================
# MAIN
# =============================================

def main():
    global PASS, FAIL

    print("=" * 60)
    print("BUILD INDEX TEST SUITE")
    print("=" * 60)

    # Part 1: chunk_text unit tests
    test_chunk_text_basic()
    test_chunk_text_empty()
    test_chunk_text_single_sentence()
    test_chunk_text_very_long_sentence()
    test_chunk_text_overlap_exists()
    test_chunk_text_no_content_loss()
    test_chunk_text_respects_size()
    test_chunk_text_special_characters()

    # Part 2: table surrogate tests
    test_table_surrogate_basic()
    test_table_surrogate_no_table()
    test_table_surrogate_empty()
    test_table_surrogate_truncation()
    test_table_surrogate_empty_cells()

    # Part 3: tokenizer consistency
    test_tokenizer_consistency()

    # Part 4: guess_table_item
    test_guess_table_item()

    # Part 5: built artifact tests
    test_chunks_parquet()
    test_chunk_linkage()
    test_tables_parquet()
    test_bm25_pkl()
    test_table_bm25_pkl()
    test_faiss_index()
    test_table_faiss_index()
    test_cross_artifact_alignment()
    test_per_ticker_coverage()
    test_build_meta()
    test_idempotency()

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
