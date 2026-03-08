"""
retrieval_tool.py

Finance-grade retrieval tool for SEC 10-K RAG with:
- Narrative retrieval: BM25 + FAISS + RRF + optional cross-encoder rerank
- Table retrieval: BM25 + FAISS + RRF (+ optional rerank)
- XBRL retrieval: lightweight “numeric evidence” lookup from companyfacts JSON (retrieval only)
- Small retrieve, big read: window expansion using prev/next links (narrative chunks)
- Context packing: dedupe + caps + section diversity + optional “must include table evidence” for numeric queries

This file is retrieval-only (no generation/verification). It returns:
- packed_context: string with evidence blocks
- debug: dict with details + selected ids
- evidence: structured payload of chunk hits, table hits, and xbrl hits (if any)

Expected artifacts (paths are configurable):
NARRATIVE:
- chunks.parquet               (chunk_id, text, ticker, fiscal_year, item, chunk_index, prev_chunk_id, next_chunk_id, ...)
- bm25.pkl                     {"chunk_ids": [...], "bm25": BM25Okapi}
- faiss.index                  FAISS IndexFlatIP (embeddings normalized)
TABLES (optional but supported):
- tables.parquet               table docs (table_id, surrogate_text, ticker, fiscal_year, item, accession, source_url, title, ...)
- table_bm25.pkl               {"table_ids": [...], "bm25": BM25Okapi}
- table_faiss.index            FAISS IndexFlatIP for table surrogate embeddings
XBRL (optional but supported):
- data/xbrl_companyfacts/{TICKER}_companyfacts.json

Dependencies:
pip install pandas numpy faiss-cpu rank-bm25 sentence-transformers
(optional) pip install transformers torch  # if you use CrossEncoder

Notes:
- This module assumes FAISS vector order matches the corresponding ids list.
- For “metadata filters”, we filter post-retrieval (fast enough for small corpora; can be optimized later).
"""
# retrieval_tool.py
from __future__ import annotations

import json
import logging
import os
import pickle
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

logger = logging.getLogger(__name__)

# ============================================================
# Utils
# ============================================================

def _now_ms() -> int:
    return int(time.time() * 1000)

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def _canon_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s.lower() if s else None

def _canon_ticker(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    return s if s else None

_ITEM_CANON_RE = re.compile(r"(?i)\bitem\s*([0-9]{1,2}[a-z]?)\b")

def _canon_item(x: Any) -> Optional[str]:
    """
    Canonicalize to 'Item 8', 'Item 1A', etc. so verification regex matches.
    """
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = _ITEM_CANON_RE.search(s)
    if m:
        return f"Item {m.group(1).upper()}"
    # allow bare "8" or "1A"
    if re.fullmatch(r"(?i)[0-9]{1,2}[a-z]?", s):
        return f"Item {s.upper()}"
    # some upstreams store "ITEM_8" / "ITEM-8"
    s2 = re.sub(r"[_\-]+", " ", s, flags=re.I).strip()
    m2 = _ITEM_CANON_RE.search(s2)
    if m2:
        return f"Item {m2.group(1).upper()}"
    return s

def _bm25_tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9$\.\-% ]+", " ", text)
    return [t for t in text.split() if t]

def rrf_fuse(
    ranked_lists: List[List[Tuple[str, float]]],
    top_k: int = 50,
    rrf_k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion.
    Each list is [(id, score)] but scores are ignored; rank is used.
    """
    score: Dict[str, float] = {}
    for lst in ranked_lists:
        for r, (doc_id, _) in enumerate(lst):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (rrf_k + r + 1)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)[:top_k]

# ============================================================
# Numeric query detection (routing heuristic)
# ============================================================

_NUMERIC_SYMBOL_MARKERS = {"$", "%"}
_NUMERIC_WORD_PATTERN = re.compile(
    r"\b(?:"  # metric-ish
    r"usd|million|billion|percent|eps|per share"
    r"|revenue|net income|gross profit|gross margin|operating income|cash flow"
    r"|free cash|fcf|capex|capital expenditure|assets|liabilities"
    r"|debt|share repurchase|buyback|dividend|guidance"
    r"|yoy|year over year|increase|decrease|margin|ratio"
    r")\b",
    re.IGNORECASE,
)
_NUMERIC_INTENT_PATTERN = re.compile(
    r"\b(?:how much|amount|total|value|calculate|compute|estimate|what is|give me)\b",
    re.IGNORECASE,
)

def _is_numeric_query(q: str) -> bool:
    """
    Retrieval router heuristic: numeric intent => include tables + XBRL evidence.

    Guard against over-triggering by requiring either:
      - explicit numeric symbol ($, %), OR
      - (numeric-word hit AND numeric-intent phrase)
    """
    q = q or ""
    if any(s in q for s in _NUMERIC_SYMBOL_MARKERS):
        return True
    return bool(_NUMERIC_WORD_PATTERN.search(q) and _NUMERIC_INTENT_PATTERN.search(q))

# ============================================================
# Narrative Retriever (chunks)
# ============================================================

class NarrativeRetriever:
    """
    BM25 + FAISS dense retrieval over narrative chunks with post-filtering by metadata.
    """

    def __init__(
        self,
        chunks_df: pd.DataFrame,
        bm25: BM25Okapi,
        bm25_chunk_ids: List[str],
        faiss_index: faiss.Index,
        embedder: SentenceTransformer,
        faiss_chunk_ids: List[str],
    ):
        self.chunks_df = chunks_df
        self.chunk_row: Dict[str, Dict[str, Any]] = {
            row["chunk_id"]: row for row in chunks_df.to_dict(orient="records")
        }
        self.bm25 = bm25
        self.bm25_chunk_ids = bm25_chunk_ids

        self.faiss_index = faiss_index
        self.embedder = embedder
        self.faiss_chunk_ids = faiss_chunk_ids

    def _apply_filters(self, cid: str, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        row = self.chunk_row.get(cid)
        if row is None:
            return False

        for k, v in filters.items():
            if v is None:
                continue
            if k == "ticker":
                if _canon_ticker(row.get(k)) != _canon_ticker(v):
                    return False
            elif k == "fiscal_year":
                if _safe_int(row.get(k)) != _safe_int(v):
                    return False
            elif k == "item":
                if _canon_item(row.get(k)) != _canon_item(v):
                    return False
            else:
                if row.get(k) != v:
                    return False
        return True

    def bm25_search(self, query: str, k: int = 50, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(_bm25_tokenize(query))
        idxs = np.argsort(scores)[::-1]
        out: List[Tuple[str, float]] = []
        for i in idxs[: k * 10]:
            cid = self.bm25_chunk_ids[i]
            if self._apply_filters(cid, filters):
                out.append((cid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def dense_search(self, query: str, k: int = 50, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        qemb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.faiss_index.search(qemb, k * 10)
        out: List[Tuple[str, float]] = []
        for j in range(I.shape[1]):
            idx = int(I[0][j])
            if idx < 0:
                continue
            if idx >= len(self.faiss_chunk_ids):
                continue
            cid = self.faiss_chunk_ids[idx]
            if self._apply_filters(cid, filters):
                out.append((cid, float(D[0][j])))
            if len(out) >= k:
                break
        return out

# ============================================================
# Table Retriever (table docs)
# ============================================================

class TableRetriever:
    """
    BM25 + FAISS dense retrieval over table surrogate docs.

    Expected columns in tables_df:
      - table_id (str)
      - surrogate_text (str)
      - ticker, fiscal_year, item, accession, source_url, title (best-effort)
    """

    def __init__(
        self,
        tables_df: pd.DataFrame,
        bm25: BM25Okapi,
        bm25_table_ids: List[str],
        faiss_index: faiss.Index,
        embedder: SentenceTransformer,
        faiss_table_ids: List[str],
    ):
        self.tables_df = tables_df
        self.table_row: Dict[str, Dict[str, Any]] = {
            row["table_id"]: row for row in tables_df.to_dict(orient="records")
        }
        self.bm25 = bm25
        self.bm25_table_ids = bm25_table_ids
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.faiss_table_ids = faiss_table_ids

    def _apply_filters(self, tid: str, filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        row = self.table_row.get(tid)
        if row is None:
            return False

        for k, v in filters.items():
            if v is None:
                continue
            if k == "ticker":
                if _canon_ticker(row.get(k)) != _canon_ticker(v):
                    return False
            elif k == "fiscal_year":
                if _safe_int(row.get(k)) != _safe_int(v):
                    return False
            elif k == "item":
                if _canon_item(row.get(k)) != _canon_item(v):
                    return False
            else:
                if row.get(k) != v:
                    return False
        return True

    def bm25_search(self, query: str, k: int = 30, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        scores = self.bm25.get_scores(_bm25_tokenize(query))
        idxs = np.argsort(scores)[::-1]
        out: List[Tuple[str, float]] = []
        for i in idxs[: k * 10]:
            tid = self.bm25_table_ids[i]
            if self._apply_filters(tid, filters):
                out.append((tid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    def dense_search(self, query: str, k: int = 30, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        qemb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.faiss_index.search(qemb, k * 10)
        out: List[Tuple[str, float]] = []
        for j in range(I.shape[1]):
            idx = int(I[0][j])
            if idx < 0:
                continue
            if idx >= len(self.faiss_table_ids):
                continue
            tid = self.faiss_table_ids[idx]
            if self._apply_filters(tid, filters):
                out.append((tid, float(D[0][j])))
            if len(out) >= k:
                break
        return out

# ============================================================
# Optional Cross-Encoder reranker
# ============================================================

class CrossEncoderReranker:
    def __init__(self, model_name: str):
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder unavailable. Install sentence-transformers with cross-encoder deps.")
        self.ce = CrossEncoder(model_name)

    def rerank_texts(
        self,
        query: str,
        candidates: List[Tuple[str, str]],  # (id, text)
        top_n: int = 10,
        batch_size: int = 32,
    ) -> List[Tuple[str, float]]:
        pairs = [(query, txt) for _, txt in candidates]
        scores = self.ce.predict(pairs, batch_size=batch_size)
        ranked = sorted(zip([cid for cid, _ in candidates], scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(s)) for cid, s in ranked[:top_n]]

# ============================================================
# Context Packing
# ============================================================

class ContextPacker:
    def __init__(self, chunk_row: Dict[str, Dict[str, Any]], table_row: Optional[Dict[str, Dict[str, Any]]] = None):
        self.chunk_row = chunk_row
        self.table_row = table_row or {}

    def expand_window(self, chunk_ids: List[str], window: int = 1) -> List[str]:
        out: List[str] = []
        seen = set()
        for cid in chunk_ids:
            if cid not in self.chunk_row:
                continue
            to_add = [cid]

            cur = cid
            for _ in range(window):
                prev_id = self.chunk_row[cur].get("prev_chunk_id")
                if prev_id and prev_id in self.chunk_row:
                    to_add.insert(0, prev_id)
                    cur = prev_id
                else:
                    break

            cur = cid
            for _ in range(window):
                next_id = self.chunk_row[cur].get("next_chunk_id")
                if next_id and next_id in self.chunk_row:
                    to_add.append(next_id)
                    cur = next_id
                else:
                    break

            for x in to_add:
                if x not in seen:
                    out.append(x)
                    seen.add(x)
        return out

    def pack(
        self,
        chunk_ids_ranked: List[str],
        table_ids_ranked: List[str],
        *,
        numeric_query: bool,
        window: int = 1,
        max_chars: int = 18_000,
        max_chunks_per_item: int = 8,
        min_gap: int = 2,
        max_tables: int = 4,
        require_table_if_numeric: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        debug: Dict[str, Any] = {"packed": {"tables": [], "chunks": []}}

        # ---- pick tables first (if numeric) ----
        picked_tables: List[str] = []
        if numeric_query:
            for tid in table_ids_ranked:
                if tid in self.table_row:
                    picked_tables.append(tid)
                if len(picked_tables) >= max_tables:
                    break

        # ---- pick narrative chunks with caps ----
        picked_chunks: List[str] = []
        picked_positions: Dict[Tuple[Any, Any, Any], List[int]] = {}
        counts: Dict[Tuple[Any, Any, Any], int] = {}

        def try_add_chunk(cid: str) -> bool:
            if cid not in self.chunk_row:
                return False
            row = self.chunk_row[cid]
            key = (_canon_ticker(row.get("ticker")), _safe_int(row.get("fiscal_year")), _canon_item(row.get("item")))
            counts.setdefault(key, 0)
            picked_positions.setdefault(key, [])
            if counts[key] >= max_chunks_per_item:
                return False

            pos = _safe_int(row.get("chunk_index")) or 0
            if any(abs(pos - p) < min_gap for p in picked_positions[key]):
                return False

            picked_chunks.append(cid)
            counts[key] += 1
            picked_positions[key].append(pos)
            return True

        for cid in chunk_ids_ranked:
            try_add_chunk(cid)

        expanded_chunks = self.expand_window(picked_chunks, window=window)

        # ---- compose final context ----
        blocks: List[str] = []
        total = 0

        # tables first
        for tid in picked_tables:
            row = self.table_row[tid]
            ticker = _canon_ticker(row.get("ticker")) or str(row.get("ticker") or "").upper()
            fy = _safe_int(row.get("fiscal_year"))
            item = _canon_item(row.get("item")) or "Item NA"

            title = (row.get("title") or row.get("caption") or "").strip()
            header = f"[TABLE {ticker} FY{fy if fy is not None else 'NA'} {item} {tid}]"
            body = (row.get("surrogate_text") or "").strip()
            block = header + ("\n" + title if title else "") + "\n" + body + "\n\n"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
            debug["packed"]["tables"].append(tid)

        if numeric_query and require_table_if_numeric and len(debug["packed"]["tables"]) == 0:
            debug["warning"] = "numeric_query_no_table_evidence"

        # narrative chunks
        seen_chunks = set()
        for cid in expanded_chunks:
            if cid in seen_chunks:
                continue
            if cid not in self.chunk_row:
                continue
            seen_chunks.add(cid)
            row = self.chunk_row[cid]

            ticker = _canon_ticker(row.get("ticker")) or str(row.get("ticker") or "").upper()
            fy = _safe_int(row.get("fiscal_year"))
            item = _canon_item(row.get("item")) or "Item NA"

            header = f"[{ticker} FY{fy if fy is not None else 'NA'} {item} {cid}]\n"
            body = (row.get("text") or "").strip()
            block = header + body + "\n\n"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
            debug["packed"]["chunks"].append(cid)

        debug["packed"]["chars"] = total
        return "".join(blocks), debug

# ============================================================
# XBRL store (retrieval-only evidence)
# ============================================================

@dataclass
class XBRLEvidence:
    ticker: str
    concept: str
    label: str
    fy: Optional[int]
    end: Optional[str]
    value: Optional[float]
    unit: Optional[str]
    source: str  # file path or accession if known

class XBRLStore:
    def __init__(self, companyfacts_dir: Path):
        self.dir = companyfacts_dir
        self.cache: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def load(self, ticker: str) -> Optional[dict]:
        t = (ticker or "").upper().strip()
        if not t:
            return None
        with self._lock:
            if t in self.cache:
                return self.cache[t]
        path = self.dir / f"{t}_companyfacts.json"
        if not path.exists():
            logger.debug("XBRL companyfacts not found for %s", t)
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load XBRL for %s: %s", t, exc)
            return None
        with self._lock:
            self.cache[t] = obj
        return obj

    @staticmethod
    def _iter_facts(obj: dict) -> Iterable[Tuple[str, dict]]:
        facts = obj.get("facts", {})
        for tax in ("us-gaap", "ifrs-full"):
            for concept, payload in facts.get(tax, {}).items():
                yield concept, payload

    @staticmethod
    def _best_label(payload: dict) -> str:
        return payload.get("label") or payload.get("description") or ""

    @staticmethod
    def _pick_value_for_fy(payload: dict, fy: Optional[int]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        units = payload.get("units", {})
        best = None  # (is_10k, end, val, unit)
        for unit, arr in units.items():
            for rec in arr:
                form = (rec.get("form") or "").upper()
                is_10k = 1 if form == "10-K" else 0
                rec_fy = rec.get("fy")
                if fy is not None and rec_fy != fy:
                    continue
                end = rec.get("end")
                val = rec.get("val")
                if val is None:
                    continue
                cand = (is_10k, end or "", float(val), unit)
                if best is None or cand > best:
                    best = cand
        if best is None:
            return None, None, None
        _, end, val, unit = best
        return val, unit, end

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        top_k: int = 5,
    ) -> List[XBRLEvidence]:
        q = (query or "").lower()
        if not q:
            return []
        tickers = [ticker.upper()] if ticker else list(self.cache.keys())
        out: List[Tuple[float, XBRLEvidence]] = []

        for t in tickers:
            obj = self.load(t)
            if obj is None:
                continue
            for concept, payload in self._iter_facts(obj):
                label = self._best_label(payload)
                hay = (concept + " " + label).lower()

                score = 0.0
                for tok in set(_bm25_tokenize(query)):
                    if tok and tok in hay:
                        score += 1.0
                if score <= 0:
                    continue

                val, unit, end = self._pick_value_for_fy(payload, fiscal_year)
                ev = XBRLEvidence(
                    ticker=t,
                    concept=concept,
                    label=label,
                    fy=fiscal_year,
                    end=end,
                    value=val,
                    unit=unit,
                    source=str(self.dir / f"{t}_companyfacts.json"),
                )
                out.append((score, ev))

        out.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in out[:top_k]]

# ============================================================
# Main Retrieval Tool
# ============================================================

@dataclass
class RetrievalConfig:
    # narrative
    bm25_k: int = 80
    dense_k: int = 80
    fused_k: int = 30
    rrf_k: int = 60

    # table
    table_bm25_k: int = 30
    table_dense_k: int = 30
    table_fused_k: int = 15

    # reranking
    use_rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 10
    rerank_batch_size: int = 32
    rerank_only_if_broad: bool = True

    # packing
    window: int = 1
    max_chars: int = 18_000
    max_chunks_per_item: int = 8
    min_gap: int = 2
    max_tables: int = 4
    require_table_if_numeric: bool = True

    # broad query detection
    broad_markers: Tuple[str, ...] = (
        "main", "key", "major", "overall", "summarize", "summary", "top", "biggest",
        "risk factors", "what are the risks", "overview"
    )

class FinancialRetrievalTool:
    """
    Retrieval-only tool that returns packed context + evidence payload.
    """

    VALID_FILTER_KEYS = {"ticker", "fiscal_year", "item", "accession"}

    def __init__(
        self,
        *,
        narrative_chunks_path: Path,
        narrative_bm25_path: Path,
        narrative_faiss_path: Path,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        # optional table assets
        table_docs_path: Optional[Path] = None,
        table_bm25_path: Optional[Path] = None,
        table_faiss_path: Optional[Path] = None,
        # optional xbrl
        companyfacts_dir: Optional[Path] = None,
        config: Optional[RetrievalConfig] = None,
        # IMPORTANT: id mapping artifacts (recommended)
        narrative_faiss_ids_path: Optional[Path] = None,
        table_faiss_ids_path: Optional[Path] = None,
    ):
        self.cfg = config or RetrievalConfig()

        # ---- narrative load ----
        logger.info("Loading narrative chunks from %s", narrative_chunks_path)
        self.chunks_df = pd.read_parquet(narrative_chunks_path)

        with open(narrative_bm25_path, "rb") as f:
            payload = pickle.load(f)
        self.n_bm25 = payload["bm25"]
        self.n_bm25_ids = payload["chunk_ids"]

        self.n_faiss = faiss.read_index(str(narrative_faiss_path))
        logger.info("Loading embedding model: %s", embed_model)
        self.embedder = SentenceTransformer(embed_model)

        # --- FAISS id mapping ---
        # Correct mapping requires that faiss_ids list matches the order used to build the FAISS index.
        if narrative_faiss_ids_path is None:
            # convention: <faiss.index>.ids.pkl
            candidate = narrative_faiss_path.with_suffix(narrative_faiss_path.suffix + ".ids.pkl")
            narrative_faiss_ids_path = candidate if candidate.exists() else None

        if narrative_faiss_ids_path and narrative_faiss_ids_path.exists():
            with open(narrative_faiss_ids_path, "rb") as f:
                self.n_faiss_ids = pickle.load(f)
            logger.info("Loaded narrative FAISS id map: %s (%d ids)", narrative_faiss_ids_path, len(self.n_faiss_ids))
        else:
            # fallback (works only if dataframe order == faiss build order)
            self.n_faiss_ids = self.chunks_df["chunk_id"].tolist()
            logger.warning(
                "Narrative FAISS id map not found; using chunks_df order (may be incorrect). "
                "Provide narrative_faiss_ids_path or create '<faiss.index>.ids.pkl'."
            )

        if len(self.n_bm25_ids) != len(self.n_faiss_ids):
            logger.warning(
                "BM25 ids (%d) and FAISS ids (%d) count mismatch — index may be stale",
                len(self.n_bm25_ids), len(self.n_faiss_ids),
            )

        self.narrative = NarrativeRetriever(
            chunks_df=self.chunks_df,
            bm25=self.n_bm25,
            bm25_chunk_ids=self.n_bm25_ids,
            faiss_index=self.n_faiss,
            embedder=self.embedder,
            faiss_chunk_ids=self.n_faiss_ids,
        )
        logger.info("Narrative retriever ready: %d chunks", len(self.n_faiss_ids))

        # ---- optional table load ----
        self.tables_enabled = False
        self.table_retriever: Optional[TableRetriever] = None
        self.tables_df: Optional[pd.DataFrame] = None

        if table_docs_path and table_bm25_path and table_faiss_path:
            logger.info("Loading table indexes")
            self.tables_df = pd.read_parquet(table_docs_path)

            with open(table_bm25_path, "rb") as f:
                tp = pickle.load(f)
            t_bm25 = tp["bm25"]
            t_bm25_ids = tp.get("table_ids") or tp.get("doc_ids") or tp.get("ids")
            if t_bm25_ids is None:
                raise ValueError("table_bm25.pkl must contain 'table_ids' (or compatible)")

            t_faiss = faiss.read_index(str(table_faiss_path))

            # --- FAISS id mapping for tables ---
            if table_faiss_ids_path is None:
                candidate = table_faiss_path.with_suffix(table_faiss_path.suffix + ".ids.pkl")
                table_faiss_ids_path = candidate if candidate.exists() else None

            if table_faiss_ids_path and table_faiss_ids_path.exists():
                with open(table_faiss_ids_path, "rb") as f:
                    t_faiss_ids = pickle.load(f)
                logger.info("Loaded table FAISS id map: %s (%d ids)", table_faiss_ids_path, len(t_faiss_ids))
            else:
                t_faiss_ids = self.tables_df["table_id"].tolist()
                logger.warning(
                    "Table FAISS id map not found; using tables_df order (may be incorrect). "
                    "Provide table_faiss_ids_path or create '<table_faiss.index>.ids.pkl'."
                )

            self.table_retriever = TableRetriever(
                tables_df=self.tables_df,
                bm25=t_bm25,
                bm25_table_ids=t_bm25_ids,
                faiss_index=t_faiss,
                embedder=self.embedder,
                faiss_table_ids=t_faiss_ids,
            )
            self.tables_enabled = True
            logger.info("Table retriever ready: %d docs", len(t_faiss_ids))
        else:
            logger.info("Table retrieval disabled (paths not provided)")

        # ---- optional xbrl ----
        self.xbrl_enabled = False
        self.xbrl: Optional[XBRLStore] = None
        if companyfacts_dir:
            self.xbrl = XBRLStore(companyfacts_dir)
            self.xbrl_enabled = True
            logger.info("XBRL store enabled: %s", companyfacts_dir)

        # ---- packer ----
        table_row = self.table_retriever.table_row if self.table_retriever else {}
        self.packer = ContextPacker(self.narrative.chunk_row, table_row=table_row)

        # ---- optional reranker ----
        self.reranker: Optional[CrossEncoderReranker] = None
        if self.cfg.use_rerank:
            if CrossEncoder is None:
                raise RuntimeError("CrossEncoder not available. Install sentence-transformers cross-encoder deps.")
            self.reranker = CrossEncoderReranker(self.cfg.rerank_model)
            logger.info("Cross-encoder reranker loaded: %s", self.cfg.rerank_model)

        logger.info("FinancialRetrievalTool ready")

    def _is_broad(self, q: str) -> bool:
        ql = (q or "").lower()
        return any(m in ql for m in self.cfg.broad_markers)

    def _validate_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if filters is None:
            return None
        if not isinstance(filters, dict):
            raise TypeError(f"filters must be a dict, got {type(filters).__name__}")
        unknown = set(filters.keys()) - self.VALID_FILTER_KEYS
        if unknown:
            logger.warning("Unknown filter keys ignored: %s (valid: %s)", unknown, self.VALID_FILTER_KEYS)

        cleaned: Dict[str, Any] = {k: v for k, v in filters.items() if k in self.VALID_FILTER_KEYS}

        # canonicalize known keys
        if "ticker" in cleaned:
            cleaned["ticker"] = _canon_ticker(cleaned["ticker"])
        if "fiscal_year" in cleaned:
            cleaned["fiscal_year"] = _safe_int(cleaned["fiscal_year"])
        if "item" in cleaned:
            cleaned["item"] = _canon_item(cleaned["item"])

        return cleaned

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        *,
        numeric_query: Optional[bool] = None,
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
          packed_context: str
          debug: dict
          evidence: dict with ids + metadata
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        filters = self._validate_filters(filters)

        t0 = _now_ms()
        numeric = _is_numeric_query(query) if numeric_query is None else bool(numeric_query)
        broad = self._is_broad(query)
        logger.debug("Query: %r | numeric=%s broad=%s filters=%s", query[:120], numeric, broad, filters)

        warnings: List[str] = []

        # ---------- Narrative hybrid ----------
        bm: List[Tuple[str, float]] = []
        try:
            bm = self.narrative.bm25_search(query, k=self.cfg.bm25_k, filters=filters)
        except Exception:
            logger.exception("BM25 narrative search failed")
            warnings.append("bm25_narrative_failed")
        t1 = _now_ms()

        dn: List[Tuple[str, float]] = []
        try:
            dn = self.narrative.dense_search(query, k=self.cfg.dense_k, filters=filters)
        except Exception:
            logger.exception("Dense narrative search failed")
            warnings.append("dense_narrative_failed")
        t2 = _now_ms()

        fused = rrf_fuse([bm, dn], top_k=self.cfg.fused_k, rrf_k=self.cfg.rrf_k)
        fused_ids = [cid for cid, _ in fused]
        t3 = _now_ms()
        logger.debug("Narrative: bm25=%d dense=%d fused=%d", len(bm), len(dn), len(fused_ids))

        # ---------- Table hybrid (only if enabled and numeric) ----------
        table_fused_ids: List[str] = []
        table_hits: List[Tuple[str, float]] = []

        if numeric and self.tables_enabled and self.table_retriever is not None:
            try:
                tbm = self.table_retriever.bm25_search(query, k=self.cfg.table_bm25_k, filters=filters)
                td = self.table_retriever.dense_search(query, k=self.cfg.table_dense_k, filters=filters)
                table_hits = rrf_fuse([tbm, td], top_k=self.cfg.table_fused_k, rrf_k=self.cfg.rrf_k)
                table_fused_ids = [tid for tid, _ in table_hits]
                logger.debug("Tables: bm25=%d dense=%d fused=%d", len(tbm), len(td), len(table_fused_ids))
            except Exception:
                logger.exception("Table retrieval failed")
                warnings.append("table_retrieval_failed")
        t4 = _now_ms()

        # ---------- Optional rerank (narrative only by default) ----------
        reranked_chunk_ids = fused_ids
        reranked_scores: List[Tuple[str, float]] = []
        do_rerank = bool(self.cfg.use_rerank) and (not self.cfg.rerank_only_if_broad or broad)

        if do_rerank and self.reranker is not None:
            try:
                candidates = [
                    (cid, self.narrative.chunk_row[cid]["text"])
                    for cid in fused_ids if cid in self.narrative.chunk_row
                ]
                reranked_scores = self.reranker.rerank_texts(
                    query,
                    candidates,
                    top_n=self.cfg.rerank_top_n,
                    batch_size=self.cfg.rerank_batch_size,
                )
                reranked_chunk_ids = [cid for cid, _ in reranked_scores]
                logger.debug("Reranked: %d -> %d", len(candidates), len(reranked_chunk_ids))
            except Exception:
                logger.exception("Reranking failed, falling back to fused order")
                warnings.append("rerank_failed")
        t5 = _now_ms()

        # ---------- XBRL evidence (retrieval only) ----------
        xbrl_evs: List[XBRLEvidence] = []
        if numeric and self.xbrl_enabled and self.xbrl is not None:
            try:
                tkr = (filters or {}).get("ticker")
                fy = (filters or {}).get("fiscal_year")
                fy_int = _safe_int(fy) if fy is not None else None
                xbrl_evs = self.xbrl.search(query, ticker=tkr, fiscal_year=fy_int, top_k=5)
                logger.debug("XBRL hits: %d", len(xbrl_evs))
            except Exception:
                logger.exception("XBRL search failed")
                warnings.append("xbrl_search_failed")
        t6 = _now_ms()

        # ---------- Pack context ----------
        packed_context, pack_debug = self.packer.pack(
            chunk_ids_ranked=reranked_chunk_ids,
            table_ids_ranked=table_fused_ids,
            numeric_query=numeric,
            window=self.cfg.window if not broad else max(2, self.cfg.window),
            max_chars=self.cfg.max_chars if not broad else int(self.cfg.max_chars * 1.4),
            max_chunks_per_item=self.cfg.max_chunks_per_item if not broad else max(self.cfg.max_chunks_per_item * 2, 12),
            min_gap=self.cfg.min_gap if not broad else 1,
            max_tables=self.cfg.max_tables,
            require_table_if_numeric=self.cfg.require_table_if_numeric,
        )

        # Prepend XBRL evidence within the char budget
        xbrl_chars = 0
        if numeric and xbrl_evs:
            x_lines = ["[XBRL EVIDENCE]"]
            for ev in xbrl_evs:
                x_lines.append(
                    f"- {ev.ticker} {ev.concept} ({ev.label}) "
                    f"FY{ev.fy or 'NA'} end={ev.end or 'NA'} value={ev.value} unit={ev.unit} "
                    f"[source:{Path(ev.source).name}]"
                )
            x_block = "\n".join(x_lines) + "\n\n"
            xbrl_chars = len(x_block)
            if xbrl_chars + len(packed_context) <= self.cfg.max_chars * 1.05:
                packed_context = x_block + packed_context
            else:
                trimmed = "\n".join(x_lines[:3]) + "\n\n"
                xbrl_chars = len(trimmed)
                packed_context = trimmed + packed_context

        t7 = _now_ms()

        if pack_debug.get("warning"):
            warnings.append(pack_debug["warning"])

        total_context_chars = int(pack_debug.get("packed", {}).get("chars", 0)) + xbrl_chars

        # ---------- Evidence payload ----------
        evidence: Dict[str, Any] = {
            "numeric_query": numeric,
            "broad_query": broad,
            "filters": filters,
            "narrative": {
                "bm25": bm[:10],
                "dense": dn[:10],
                "fused": fused,
                "reranked": reranked_scores if reranked_scores else None,
                "selected_chunk_ids": pack_debug["packed"]["chunks"],
            },
            "tables": {
                "enabled": self.tables_enabled,
                "fused": table_hits if table_hits else None,
                "selected_table_ids": pack_debug["packed"]["tables"],
            },
            "xbrl": {
                "enabled": self.xbrl_enabled,
                "hits": [ev.__dict__ for ev in xbrl_evs] if xbrl_evs else None,
            },
        }

        debug: Dict[str, Any] = {
            "latency_ms": {
                "bm25": t1 - t0,
                "dense": t2 - t1,
                "rrf": t3 - t2,
                "tables": t4 - t3,
                "rerank": t5 - t4,
                "xbrl": t6 - t5,
                "pack": t7 - t6,
                "total": t7 - t0,
            },
            "counts": {
                "bm25_candidates": len(bm),
                "dense_candidates": len(dn),
                "fused_candidates": len(fused_ids),
                "reranked_top": len(reranked_chunk_ids),
                "table_candidates": len(table_fused_ids),
                "packed_chunks": len(pack_debug["packed"]["chunks"]),
                "packed_tables": len(pack_debug["packed"]["tables"]),
                "context_chars": total_context_chars,
            },
            "warnings": warnings if warnings else None,
        }

        logger.info(
            "retrieve done: total=%dms chunks=%d tables=%d chars=%d warnings=%s",
            debug["latency_ms"]["total"],
            debug["counts"]["packed_chunks"],
            debug["counts"]["packed_tables"],
            total_context_chars,
            warnings or "none",
        )

        return packed_context, debug, evidence


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    BASE = Path(os.environ.get("FIN_TOOL_BASE", Path(__file__).resolve().parent))
    IDX = BASE / "index"

    tool = FinancialRetrievalTool(
        narrative_chunks_path=IDX / "chunks.parquet",
        narrative_bm25_path=IDX / "bm25.pkl",
        narrative_faiss_path=IDX / "faiss.index",
        # OPTIONAL BUT RECOMMENDED:
        narrative_faiss_ids_path=IDX / "faiss.index.ids.pkl",
        embed_model="sentence-transformers/all-MiniLM-L6-v2",
        table_docs_path=IDX / "tables.parquet",
        table_bm25_path=IDX / "table_bm25.pkl",
        table_faiss_path=IDX / "table_faiss.index",
        table_faiss_ids_path=IDX / "table_faiss.index.ids.pkl",
        companyfacts_dir=BASE / "data" / "xbrl_companyfacts",
        config=RetrievalConfig(
            use_rerank=True,
            rerank_only_if_broad=True,
        ),
    )
