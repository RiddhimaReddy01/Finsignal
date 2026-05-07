from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
IDX = BASE / "index"
sys.path.insert(0, str(BASE))

from retrieval_tool import FinancialRetrievalTool, RetrievalConfig


@pytest.fixture(scope="session")
def tool() -> FinancialRetrievalTool:
    return FinancialRetrievalTool(
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
