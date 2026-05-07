from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from news_ingestion import NewsIngestionClient


class NewsClientAdapter:
    """
    Adapter from NewsIngestionClient -> orchestrator.NewsClient protocol.
    Uses NewsIngestionClient caching by default.
    """

    def __init__(self, client: Optional[NewsIngestionClient] = None):
        self.client = client or NewsIngestionClient()

    def fetch_company_news(
        self,
        *,
        ticker: str,
        limit: int = 5,
        as_of: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        rows = self.client.fetch_recent_news(
            ticker=ticker,
            as_of=as_of,
            max_results=max(1, int(limit)),
            use_cache=True,
            force_refresh=False,
        )
        out: List[Dict[str, Any]] = []
        for r in rows[: max(1, int(limit))]:
            out.append({
                "id": r.article_id,
                "title": r.title,
                "summary": r.description or r.content,
                "text": r.content,
                "published_at": r.published_at,
                "url": r.url,
                "source": r.source_name,
            })
        return out


def build_optional_news_client() -> Optional[NewsClientAdapter]:
    if not (os.environ.get("NEWSAPI_KEY") or "").strip():
        return None
    try:
        return NewsClientAdapter()
    except Exception:
        return None
