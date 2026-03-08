"""
build_index.py

Reads the output of knowledge_base.py (sections + tables JSON) and builds
all retrieval indexes needed by retrieval_tool.py:

  index/
    chunks.parquet        narrative chunks with linked-list pointers
    bm25.pkl              BM25Okapi over narrative chunks
    faiss.index           FAISS IndexFlatIP (normalized embeddings)
    tables.parquet        table surrogate docs
    table_bm25.pkl        BM25Okapi over table surrogates
    table_faiss.index     FAISS IndexFlatIP for tables
    build_meta.json       build timestamp, source hashes, row counts
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import faiss
from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# =========================
# CONFIG (env-overridable)
# =========================

BASE = Path(os.environ.get("FIN_TOOL_BASE", r"C:\Users\riddh\OneDrive\Desktop\financial_analysis_tool"))
SECT_DIR = BASE / "data" / "sections"
TABLE_DIR = BASE / "data" / "tables"
INDEX_DIR = BASE / "index"

EMBED_MODEL = os.environ.get("FIN_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.environ.get("FIN_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.environ.get("FIN_CHUNK_OVERLAP", "200"))
EMBED_BATCH = int(os.environ.get("FIN_EMBED_BATCH", "64"))


# =========================
# Tokenizer (must match retrieval_tool.py)
# =========================

def bm25_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9$\.\-% ]+", " ", text)
    return [t for t in text.split() if t]


# =========================
# 1. NARRATIVE CHUNKING
# =========================

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping character-level chunks at sentence boundaries."""
    if not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) + 1 > size and current:
            chunks.append(current.strip())
            words = current.split()
            keep_chars = 0
            keep_words = []
            for w in reversed(words):
                if keep_chars + len(w) + 1 > overlap:
                    break
                keep_words.insert(0, w)
                keep_chars += len(w) + 1
            current = " ".join(keep_words) + " " + sent if keep_words else sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())
    return chunks


def _load_sections_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Skipping corrupted/unreadable file %s: %s", path.name, exc)
        return None


def build_narrative_chunks() -> pd.DataFrame:
    """Load all sections JSONs, chunk the text, return a DataFrame."""
    rows: List[Dict] = []
    skipped = 0

    for path in sorted(SECT_DIR.glob("*_sections.json")):
        obj = _load_sections_json(path)
        if obj is None:
            skipped += 1
            continue

        ticker = obj.get("ticker")
        fy = obj.get("fiscal_year")
        if not ticker or fy is None:
            logger.warning("Skipping %s: missing ticker or fiscal_year", path.name)
            skipped += 1
            continue

        accession = obj.get("accession", "")
        source_url = obj.get("source_url", "")

        for item_name, item_text in obj.get("items", {}).items():
            if not item_text or not item_text.strip():
                continue

            pieces = chunk_text(item_text)
            for ci, piece in enumerate(pieces):
                chunk_id = f"{ticker}_FY{fy}_{item_name.replace(' ', '')}_{ci:04d}"
                rows.append({
                    "chunk_id": chunk_id,
                    "text": piece,
                    "ticker": ticker,
                    "fiscal_year": fy,
                    "item": item_name,
                    "chunk_index": ci,
                    "accession": accession,
                    "source_url": source_url,
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                })

    if skipped:
        logger.warning("Skipped %d section files", skipped)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.error("No chunks produced — check data/sections/ directory")
        return df

    for (ticker, fy, item), grp in df.groupby(["ticker", "fiscal_year", "item"]):
        idxs = grp.sort_values("chunk_index").index.tolist()
        for pos, idx in enumerate(idxs):
            if pos > 0:
                df.at[idx, "prev_chunk_id"] = df.at[idxs[pos - 1], "chunk_id"]
            if pos < len(idxs) - 1:
                df.at[idx, "next_chunk_id"] = df.at[idxs[pos + 1], "chunk_id"]

    logger.info("Narrative chunks: %d", len(df))
    return df


# =========================
# 2. TABLE SURROGATE DOCS
# =========================

def html_table_to_surrogate(html: str, title: str) -> str:
    """Convert an HTML <table> to a compact text surrogate for retrieval."""
    soup = BeautifulSoup(html, "html.parser")
    tbl = soup.find("table")
    if tbl is None:
        return title

    lines: List[str] = []
    if title:
        lines.append(title)

    for row in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))

    text = "\n".join(lines)
    if len(text) > 3000:
        text = text[:3000] + "..."
    return text


def _guess_table_item(table_id: str, filing_text_items: Dict[str, str]) -> str:
    """Best-effort: check which section's placeholder mentions this table_id."""
    for item_name, text in filing_text_items.items():
        if table_id in text:
            return item_name
    return ""


def build_table_docs() -> pd.DataFrame:
    """Load all tables JSONs, build surrogate text, return DataFrame."""
    sect_cache: Dict[Tuple[str, int], Dict[str, str]] = {}
    for p in sorted(SECT_DIR.glob("*_sections.json")):
        obj = _load_sections_json(p)
        if obj and obj.get("ticker") and obj.get("fiscal_year") is not None:
            sect_cache[(obj["ticker"], obj["fiscal_year"])] = obj.get("items", {})

    rows: List[Dict] = []
    skipped = 0

    for path in sorted(TABLE_DIR.glob("*_tables.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping corrupted table file %s: %s", path.name, exc)
            skipped += 1
            continue

        ticker = obj.get("ticker")
        fy = obj.get("fiscal_year")
        if not ticker or fy is None:
            logger.warning("Skipping %s: missing ticker or fiscal_year", path.name)
            skipped += 1
            continue

        accession = obj.get("accession", "")
        source_url = obj.get("source_url", "")
        items_text = sect_cache.get((ticker, fy), {})

        for tbl in obj.get("tables", []):
            table_id = tbl.get("table_id", "")
            title = tbl.get("title", "")
            html = tbl.get("html", "")
            surrogate = html_table_to_surrogate(html, title)
            if len(surrogate.strip()) < 10:
                continue

            item = _guess_table_item(table_id, items_text)

            rows.append({
                "table_id": table_id,
                "surrogate_text": surrogate,
                "ticker": ticker,
                "fiscal_year": fy,
                "item": item,
                "accession": accession,
                "source_url": source_url,
                "title": title,
            })

    if skipped:
        logger.warning("Skipped %d table files", skipped)

    df = pd.DataFrame(rows)
    logger.info("Table docs: %d", len(df))
    return df


# =========================
# 3. BM25 INDEX BUILDER
# =========================

def build_bm25(ids: List[str], texts: List[str]) -> Tuple[BM25Okapi, List[str]]:
    corpus = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus)
    return bm25, ids


# =========================
# 4. FAISS INDEX BUILDER
# =========================

def build_faiss(embedder: SentenceTransformer, texts: List[str], batch_size: int = EMBED_BATCH) -> faiss.Index:
    logger.info("Encoding %d texts ...", len(texts))
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=batch_size,
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS index: %d vectors, dim=%d", index.ntotal, dim)
    return index


# =========================
# 5. BUILD METADATA
# =========================

def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def write_build_meta(
    index_dir: Path,
    *,
    n_chunks: int,
    n_tables: int,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
    source_sections: List[str],
    source_tables: List[str],
    elapsed_s: float,
) -> Path:
    meta = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "narrative_chunks": n_chunks,
        "table_docs": n_tables,
        "build_elapsed_s": round(elapsed_s, 1),
        "source_sections": source_sections,
        "source_tables": source_tables,
        "artifacts": {},
    }

    for p in sorted(index_dir.iterdir()):
        if p.name == "build_meta.json":
            continue
        meta["artifacts"][p.name] = {
            "size_bytes": p.stat().st_size,
            "sha256_prefix": _file_sha256(p),
        }

    out = index_dir / "build_meta.json"
    out.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote build_meta.json")
    return out


# =========================
# MAIN
# =========================

def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logger.info("BASE: %s", BASE)
    logger.info("EMBED_MODEL: %s  CHUNK_SIZE: %d  OVERLAP: %d", EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP)

    logger.info("Loading embedding model: %s", EMBED_MODEL)
    embedder = SentenceTransformer(EMBED_MODEL)

    # -- Narrative --
    logger.info("=== NARRATIVE CHUNKS ===")
    chunks_df = build_narrative_chunks()
    if chunks_df.empty:
        logger.error("Aborting: no narrative chunks produced")
        return

    chunks_df.to_parquet(INDEX_DIR / "chunks.parquet", index=False)
    logger.info("Saved chunks.parquet (%d rows)", len(chunks_df))

    chunk_ids = chunks_df["chunk_id"].tolist()
    chunk_texts = chunks_df["text"].tolist()

    logger.info("Building narrative BM25 ...")
    bm25, bm25_ids = build_bm25(chunk_ids, chunk_texts)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": bm25_ids}, f)
    logger.info("Saved bm25.pkl")

    logger.info("Building narrative FAISS ...")
    n_index = build_faiss(embedder, chunk_texts)
    faiss.write_index(n_index, str(INDEX_DIR / "faiss.index"))
    logger.info("Saved faiss.index")

    # -- Tables --
    logger.info("=== TABLE DOCS ===")
    tables_df = build_table_docs()
    tables_df.to_parquet(INDEX_DIR / "tables.parquet", index=False)
    logger.info("Saved tables.parquet (%d rows)", len(tables_df))

    table_ids = tables_df["table_id"].tolist()
    table_texts = tables_df["surrogate_text"].tolist()

    if len(table_ids) > 0:
        logger.info("Building table BM25 ...")
        t_bm25, t_bm25_ids = build_bm25(table_ids, table_texts)
        with open(INDEX_DIR / "table_bm25.pkl", "wb") as f:
            pickle.dump({"bm25": t_bm25, "table_ids": t_bm25_ids}, f)
        logger.info("Saved table_bm25.pkl")

        logger.info("Building table FAISS ...")
        t_index = build_faiss(embedder, table_texts)
        faiss.write_index(t_index, str(INDEX_DIR / "table_faiss.index"))
        logger.info("Saved table_faiss.index")
    else:
        logger.warning("No table docs produced — skipping table indexes")

    elapsed = time.time() - t_start

    # -- Build metadata --
    source_sections = sorted(p.name for p in SECT_DIR.glob("*_sections.json"))
    source_tables = sorted(p.name for p in TABLE_DIR.glob("*_tables.json"))

    write_build_meta(
        INDEX_DIR,
        n_chunks=len(chunks_df),
        n_tables=len(tables_df),
        embed_model=EMBED_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        source_sections=source_sections,
        source_tables=source_tables,
        elapsed_s=elapsed,
    )

    # -- Summary --
    logger.info("=== DONE (%.1fs) ===", elapsed)
    logger.info("Narrative: %d chunks", len(chunks_df))
    logger.info("Tables:    %d docs", len(tables_df))
    logger.info("Output:    %s", INDEX_DIR)
    for p in sorted(INDEX_DIR.iterdir()):
        sz = p.stat().st_size
        unit = "KB" if sz < 1_000_000 else "MB"
        val = sz / 1024 if unit == "KB" else sz / (1024 * 1024)
        logger.info("  %-30s %8.1f %s", p.name, val, unit)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    main()
