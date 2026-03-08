from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


@dataclass(frozen=True)
class NewsArticle:
    article_id: str
    ticker: str
    company: Optional[str]
    title: str
    description: str
    content: str
    source_name: str
    author: Optional[str]
    published_at: str
    url: str
    language: Optional[str] = None


@dataclass(frozen=True)
class NewsFetchConfig:
    api_key: str
    page_size: int = 25
    lookback_days: int = 14
    language: str = "en"
    timeout_s: int = 30
    cache_dir: str = "data/cache/news"
    cache_ttl_s: int = 1800


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _norm_text(x: Any) -> str:
    s = str(x or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _article_id(ticker: str, title: str, published_at: str, url: str) -> str:
    raw = f"{ticker}|{title}|{published_at}|{url}".encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:16]


def _parse_newsapi_article(raw: Dict[str, Any], ticker: str, company: Optional[str]) -> Optional[NewsArticle]:
    title = _norm_text(raw.get("title"))
    desc = _norm_text(raw.get("description"))
    content = _norm_text(raw.get("content"))
    url = _norm_text(raw.get("url"))
    source_name = _norm_text((raw.get("source") or {}).get("name"))
    published_at = _norm_text(raw.get("publishedAt"))

    if not title or not url or not published_at:
        return None

    return NewsArticle(
        article_id=_article_id(ticker=ticker, title=title, published_at=published_at, url=url),
        ticker=ticker.upper(),
        company=company,
        title=title,
        description=desc,
        content=content,
        source_name=source_name or "unknown",
        author=_norm_text(raw.get("author")) or None,
        published_at=published_at,
        url=url,
        language=raw.get("language"),
    )


def _dedupe_articles(items: List[NewsArticle]) -> List[NewsArticle]:
    seen: set[str] = set()
    out: List[NewsArticle] = []
    for a in items:
        key = f"{a.title.lower()}|{a.source_name.lower()}|{a.published_at[:10]}"
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


class NewsIngestionClient:
    def __init__(self, cfg: Optional[NewsFetchConfig] = None):
        api_key = (cfg.api_key if cfg else "") or os.environ.get("NEWSAPI_KEY", "")
        if not api_key:
            raise ValueError("Missing NEWSAPI_KEY.")
        self.cfg = cfg or NewsFetchConfig(api_key=api_key)
        self._cache_dir = Path(self.cfg.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(
        self,
        *,
        ticker: str,
        company: Optional[str],
        lookback_days: int,
        max_results: int,
    ) -> str:
        sig = {
            "ticker": str(ticker or "").upper(),
            "company": str(company or "").strip().lower(),
            "lookback_days": int(lookback_days),
            "max_results": int(max_results),
            "language": str(self.cfg.language or "").lower(),
        }
        raw = json.dumps(sig, sort_keys=True, separators=(",", ":")).encode("utf-8", errors="ignore")
        return hashlib.sha1(raw).hexdigest()[:24]

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[List[NewsArticle]]:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            fetched_ts = int(obj.get("fetched_ts", 0))
            if fetched_ts <= 0:
                return None
            age_s = int(_utc_now().timestamp()) - fetched_ts
            if age_s > int(self.cfg.cache_ttl_s):
                return None
            rows = obj.get("articles") or []
            out: List[NewsArticle] = []
            for row in rows:
                if isinstance(row, dict):
                    out.append(NewsArticle(**row))
            if out:
                logger.info("News cache hit for key=%s (%d articles)", key, len(out))
            return out
        except Exception:
            logger.exception("Failed reading news cache: %s", path)
            return None

    def _write_cache(self, key: str, articles: List[NewsArticle]) -> None:
        path = self._cache_path(key)
        payload = {
            "fetched_ts": int(_utc_now().timestamp()),
            "articles": [asdict(a) for a in articles],
        }
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            logger.exception("Failed writing news cache: %s", path)

    def fetch_recent_news(
        self,
        *,
        ticker: str,
        company: Optional[str] = None,
        lookback_days: Optional[int] = None,
        max_results: Optional[int] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> List[NewsArticle]:
        ticker = str(ticker).strip().upper()
        if not ticker:
            raise ValueError("ticker is required")

        days = int(lookback_days or self.cfg.lookback_days)
        page_size = int(max_results or self.cfg.page_size)
        cache_key = self._cache_key(
            ticker=ticker,
            company=company,
            lookback_days=days,
            max_results=page_size,
        )
        if use_cache and not force_refresh:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        date_from = (_utc_now() - timedelta(days=days)).date().isoformat()

        query_parts = [ticker]
        if company:
            query_parts.append(f'"{company}"')
        q = " OR ".join(query_parts)

        params = {
            "q": q,
            "from": date_from,
            "language": self.cfg.language,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "apiKey": self.cfg.api_key,
        }

        resp = requests.get(NEWSAPI_URL, params=params, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        payload = resp.json()

        raw_articles = payload.get("articles", []) or []
        articles: List[NewsArticle] = []
        for raw in raw_articles:
            parsed = _parse_newsapi_article(raw, ticker=ticker, company=company)
            if parsed is not None:
                articles.append(parsed)

        articles = _dedupe_articles(articles)
        if use_cache:
            self._write_cache(cache_key, articles)
        logger.info("Fetched %d news articles for %s", len(articles), ticker)
        return articles

    def persist_jsonl(self, articles: List[NewsArticle], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")

    def load_jsonl(self, path: Path) -> List[NewsArticle]:
        if not path.exists():
            return []
        out: List[NewsArticle] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(NewsArticle(**json.loads(line)))
        return out


def build_news_context(articles: List[NewsArticle], max_chars: int = 6000) -> str:
    parts: List[str] = []
    used = 0
    for i, a in enumerate(articles, start=1):
        block = (
            f"[news_{i}]\n"
            f"ticker={a.ticker} source={a.source_name} published_at={a.published_at}\n"
            f"title: {a.title}\n"
            f"description: {a.description}\n"
            f"content: {a.content}\n"
            f"url: {a.url}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)
