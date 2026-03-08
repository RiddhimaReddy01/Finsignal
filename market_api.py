# market_api.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple


# ---------------------------
# Provider interface
# ---------------------------

class MarketDataProvider(Protocol):
    """
    Fetches market data on-demand. Implementations can use:
    - yfinance
    - polygon.io
    - alphavantage
    - IEX Cloud
    - your internal market data service

    Return values should be best-effort and may be None if unavailable.
    """

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Expected keys (best-effort):
          - price: float
          - market_cap: float
          - currency: str
          - asof: str (ISO date/time)
        """
        ...

    def get_risk_free_rate(self, *, tenor: str = "10y") -> Optional[float]:
        """Return decimal rate, e.g., 0.042 for 4.2%."""
        ...

    def get_beta(self, ticker: str) -> Optional[float]:
        """Return equity beta."""
        ...


class YahooFinanceMarketDataProvider:
    """
    Best-effort market data provider backed by yfinance.
    """

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        try:
            import yfinance as yf
        except Exception:
            return {}
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            return {}
        out: Dict[str, Any] = {
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency"),
            "asof": datetime.now(timezone.utc).isoformat(),
        }
        return {k: v for k, v in out.items() if v is not None}

    def get_risk_free_rate(self, *, tenor: str = "10y") -> Optional[float]:
        if tenor != "10y":
            return None
        try:
            import yfinance as yf
            hist = yf.Ticker("^TNX").history(period="5d")
            if hist.empty:
                return None
            close = float(hist["Close"].dropna().iloc[-1])
            return close / 100.0
        except Exception:
            return None

    def get_beta(self, ticker: str) -> Optional[float]:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info or {}
            beta = info.get("beta")
            return float(beta) if beta is not None else None
        except Exception:
            return None


# ---------------------------
# Cache
# ---------------------------

@dataclass
class TTLCache:
    ttl_s: int = 300
    _store: Dict[str, Tuple[float, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        pass

    def get(self, key: str) -> Any:
        hit = self._store.get(key)
        if not hit:
            return None
        ts, val = hit
        if (time.time() - ts) > self.ttl_s:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: Any) -> None:
        self._store[key] = (time.time(), val)


# ---------------------------
# Normalization / merge logic
# ---------------------------

RECOMMENDED_KEYS = {
    # hard market
    "price",              # current price
    "market_cap",         # optional
    "risk_free_rate",     # decimal (10y)
    "beta",               # optional

    # valuation assumptions
    "wacc",               # required for DCF unless you compute it
    "terminal_growth",    # or "exit_multiple"
    "exit_multiple",      # alternative terminal value approach

    # equity bridge
    "shares_outstanding", # needed if you want per-share intrinsic value from equity value
    "net_debt",           # optional
    "cash",               # optional
    "debt",               # optional

    # meta
    "currency",
    "asof",
}

def merge_market_inputs(user_inputs: Optional[Dict[str, Any]], fetched: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Precedence: user overrides fetched. Always return a dict (possibly empty).
    """
    out: Dict[str, Any] = {}
    for src in (fetched or {}, user_inputs or {}):
        if not isinstance(src, dict):
            continue
        for k, v in src.items():
            if v is not None:
                out[k] = v
    return out


def fetch_min_market_inputs(
    provider: MarketDataProvider,
    *,
    ticker: str,
    tenor: str = "10y",
    cache: Optional[TTLCache] = None,
) -> Dict[str, Any]:
    """
    Fetches only the minimal set that is reasonably obtainable from a market API.
    Does NOT invent WACC/terminal growth — those are assumptions.
    """
    t = (ticker or "").upper().strip()
    if not t:
        return {}

    cache = cache or TTLCache(ttl_s=300)

    # quote
    q_key = f"quote:{t}"
    quote = cache.get(q_key)
    if quote is None:
        quote = provider.get_quote(t) or {}
        cache.set(q_key, quote)

    # risk-free
    rf_key = f"rf:{tenor}"
    rf = cache.get(rf_key)
    if rf is None:
        rf = provider.get_risk_free_rate(tenor=tenor)
        cache.set(rf_key, rf)

    # beta
    b_key = f"beta:{t}"
    beta = cache.get(b_key)
    if beta is None:
        beta = provider.get_beta(t)
        cache.set(b_key, beta)

    out: Dict[str, Any] = {}
    if isinstance(quote, dict):
        if quote.get("price") is not None:
            out["price"] = float(quote["price"])
        if quote.get("market_cap") is not None:
            out["market_cap"] = float(quote["market_cap"])
        if quote.get("currency"):
            out["currency"] = str(quote["currency"])
        if quote.get("asof"):
            out["asof"] = str(quote["asof"])

    if rf is not None:
        out["risk_free_rate"] = float(rf)
    if beta is not None:
        out["beta"] = float(beta)

    return out