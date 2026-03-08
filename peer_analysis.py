# peer_analysis.py
# ============================================================
# Peer Group Construction & Comparative Valuation
#
# Addresses the gap where relative_valuation compares against
# a single manually-provided peer median. This module:
#
#   1. Automatically selects a peer group from a sector map
#   2. Fetches multiples for each peer via market_api
#   3. Computes peer median, mean, range for each multiple
#   4. Ranks the target vs peers
#   5. Produces a structured peer comparison output
#
# Plugs into relative_valuation_engine.py and orchestrator.py.
# ============================================================

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)

MultipleType = Literal["P_E", "P_S", "EV_SALES", "EV_EBITDA"]


# ============================================================
# 1. Sector/Industry Peer Map
# ============================================================

# Maps tickers to their sector and a default peer group.
# In production this would come from a database or API;
# for the hackathon, we hardcode the universe we index.

SECTOR_MAP: Dict[str, Dict[str, Any]] = {
    "AAPL": {"sector": "Technology", "industry": "Consumer Electronics", "peers": ["MSFT", "GOOGL", "META", "AMZN", "SAMSUNG"]},
    "MSFT": {"sector": "Technology", "industry": "Software", "peers": ["AAPL", "GOOGL", "META", "ORCL", "CRM"]},
    "NVDA": {"sector": "Technology", "industry": "Semiconductors", "peers": ["AMD", "INTC", "AVGO", "QCOM", "TSM"]},
    "GOOGL": {"sector": "Technology", "industry": "Internet Services", "peers": ["META", "MSFT", "AMZN", "AAPL", "NFLX"]},
    "META": {"sector": "Technology", "industry": "Social Media", "peers": ["GOOGL", "SNAP", "PINS", "TWTR", "MSFT"]},
    "TSLA": {"sector": "Consumer Discretionary", "industry": "Electric Vehicles", "peers": ["F", "GM", "RIVN", "LCID", "TM"]},
    "AMZN": {"sector": "Consumer Discretionary", "industry": "E-Commerce", "peers": ["GOOGL", "MSFT", "AAPL", "WMT", "BABA"]},
    # Semiconductor peers
    "AMD": {"sector": "Technology", "industry": "Semiconductors", "peers": ["NVDA", "INTC", "AVGO", "QCOM", "TSM"]},
    "INTC": {"sector": "Technology", "industry": "Semiconductors", "peers": ["NVDA", "AMD", "AVGO", "QCOM", "TSM"]},
    # Add more as the indexed universe grows
}

# Fallback: broad tech peers
DEFAULT_TECH_PEERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA"]


def get_peer_group(
    ticker: str,
    *,
    max_peers: int = 5,
    custom_peers: Optional[List[str]] = None,
    exclude_self: bool = True,
) -> List[str]:
    """
    Get the peer group for a ticker.

    Priority:
    1. Custom peers if provided
    2. Sector map lookup
    3. Default tech peers (fallback)
    """
    t = (ticker or "").upper().strip()

    if custom_peers:
        peers = [p.upper().strip() for p in custom_peers if p.strip()]
    elif t in SECTOR_MAP:
        peers = list(SECTOR_MAP[t]["peers"])
    else:
        peers = list(DEFAULT_TECH_PEERS)

    if exclude_self:
        peers = [p for p in peers if p != t]

    return peers[:max_peers]


def get_sector_info(ticker: str) -> Dict[str, str]:
    """Get sector and industry for a ticker."""
    t = (ticker or "").upper().strip()
    info = SECTOR_MAP.get(t, {})
    return {
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
    }


# ============================================================
# 2. Peer Data Fetching
# ============================================================

@dataclass
class PeerMultiples:
    """Market multiples for a single company."""
    ticker: str
    price: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_sales: Optional[float] = None
    ev_ebitda: Optional[float] = None
    beta: Optional[float] = None
    fetch_success: bool = False
    error: Optional[str] = None


def fetch_peer_multiples(
    ticker: str,
    market_provider: Any,
) -> PeerMultiples:
    """
    Fetch market multiples for a single ticker.

    Uses the existing MarketDataProvider interface from market_api.py.
    Falls back gracefully on any failure.
    """
    t = (ticker or "").upper().strip()
    result = PeerMultiples(ticker=t)

    try:
        quote = market_provider.get_quote(t) if market_provider else {}
        if not isinstance(quote, dict):
            quote = {}

        result.price = _safe_float(quote.get("currentPrice") or quote.get("price"))
        result.market_cap = _safe_float(quote.get("marketCap") or quote.get("market_cap"))

        # Try to get multiples directly from the quote (yfinance provides these)
        try:
            import yfinance as yf
            info = yf.Ticker(t).info or {}
            result.pe_ratio = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
            result.ps_ratio = _safe_float(info.get("priceToSalesTrailing12Months"))
            result.ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
            result.ev_sales = _safe_float(info.get("enterpriseToRevenue"))
            result.beta = _safe_float(info.get("beta"))
            result.fetch_success = True
        except Exception as e:
            # Fall back to just what market_provider gave us
            result.beta = _safe_float(market_provider.get_beta(t)) if market_provider else None
            result.fetch_success = result.price is not None
            if not result.fetch_success:
                result.error = str(e)

    except Exception as e:
        result.error = str(e)
        logger.warning("Failed to fetch multiples for %s: %s", t, e)

    return result


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


import math


# ============================================================
# 3. Peer Group Statistics
# ============================================================

@dataclass
class MultipleStats:
    """Descriptive statistics for a single multiple across the peer group."""
    multiple_type: str
    values: Dict[str, float]  # ticker -> value
    count: int
    median: Optional[float]
    mean: Optional[float]
    min_val: Optional[float]
    max_val: Optional[float]
    std: Optional[float]


def compute_multiple_stats(
    peer_data: List[PeerMultiples],
    multiple_type: MultipleType,
) -> MultipleStats:
    """Compute descriptive stats for one multiple across peers."""
    attr_map = {
        "P_E": "pe_ratio",
        "P_S": "ps_ratio",
        "EV_SALES": "ev_sales",
        "EV_EBITDA": "ev_ebitda",
    }
    attr = attr_map.get(multiple_type, "pe_ratio")

    values = {}
    for pm in peer_data:
        v = getattr(pm, attr, None)
        if v is not None and v > 0:
            values[pm.ticker] = v

    vals = list(values.values())
    n = len(vals)

    return MultipleStats(
        multiple_type=multiple_type,
        values=values,
        count=n,
        median=statistics.median(vals) if n >= 1 else None,
        mean=statistics.mean(vals) if n >= 1 else None,
        min_val=min(vals) if vals else None,
        max_val=max(vals) if vals else None,
        std=statistics.stdev(vals) if n >= 2 else None,
    )


# ============================================================
# 4. Peer Comparison Result
# ============================================================

@dataclass
class PeerRanking:
    """Where the target ranks vs peers on a given multiple."""
    multiple_type: str
    target_ticker: str
    target_value: Optional[float]
    peer_median: Optional[float]
    premium_to_median_pct: Optional[float]
    rank: Optional[int]        # 1 = cheapest
    total_ranked: int
    assessment: str            # "discount", "in_line", "premium"


@dataclass
class PeerAnalysisResult:
    """Complete peer group analysis output."""
    target_ticker: str
    sector: str
    industry: str
    peer_tickers: List[str]
    target_multiples: PeerMultiples
    peer_multiples: List[PeerMultiples]
    stats: Dict[str, MultipleStats]       # multiple_type -> stats
    rankings: List[PeerRanking]
    summary: str
    data_quality: Dict[str, Any]


def rank_target_vs_peers(
    target: PeerMultiples,
    peer_data: List[PeerMultiples],
    multiple_type: MultipleType,
) -> PeerRanking:
    """Rank the target against its peers on one multiple."""
    attr_map = {
        "P_E": "pe_ratio",
        "P_S": "ps_ratio",
        "EV_SALES": "ev_sales",
        "EV_EBITDA": "ev_ebitda",
    }
    attr = attr_map.get(multiple_type, "pe_ratio")

    target_val = getattr(target, attr, None)
    all_vals = []

    for pm in peer_data:
        v = getattr(pm, attr, None)
        if v is not None and v > 0:
            all_vals.append((pm.ticker, v))

    if target_val is not None and target_val > 0:
        all_vals.append((target.ticker, target_val))

    # Sort ascending (lower multiple = cheaper)
    all_vals.sort(key=lambda x: x[1])
    total = len(all_vals)

    rank = None
    for i, (t, v) in enumerate(all_vals, 1):
        if t == target.ticker:
            rank = i
            break

    # Compute peer-only median (excluding target)
    peer_only_vals = [v for t, v in all_vals if t != target.ticker]
    peer_median = statistics.median(peer_only_vals) if peer_only_vals else None

    premium = None
    assessment = "insufficient"
    if target_val is not None and peer_median is not None and peer_median > 0:
        premium = (target_val - peer_median) / peer_median
        if premium > 0.15:
            assessment = "premium"
        elif premium < -0.15:
            assessment = "discount"
        else:
            assessment = "in_line"

    return PeerRanking(
        multiple_type=multiple_type,
        target_ticker=target.ticker,
        target_value=target_val,
        peer_median=peer_median,
        premium_to_median_pct=premium,
        rank=rank,
        total_ranked=total,
        assessment=assessment,
    )


# ============================================================
# 5. Main Analysis Runner
# ============================================================

def run_peer_analysis(
    *,
    ticker: str,
    market_provider: Any,
    custom_peers: Optional[List[str]] = None,
    max_peers: int = 5,
    multiples_to_compute: Optional[List[MultipleType]] = None,
) -> PeerAnalysisResult:
    """
    Full peer group analysis.

    Usage:
        from market_api import YahooFinanceMarketDataProvider
        result = run_peer_analysis(
            ticker="AAPL",
            market_provider=YahooFinanceMarketDataProvider(),
        )
        print(result.summary)
        for r in result.rankings:
            print(f"  {r.multiple_type}: rank {r.rank}/{r.total_ranked} ({r.assessment})")
    """
    t = (ticker or "").upper().strip()
    sector_info = get_sector_info(t)
    peers = get_peer_group(t, max_peers=max_peers, custom_peers=custom_peers)

    if multiples_to_compute is None:
        multiples_to_compute = ["P_E", "EV_EBITDA", "P_S", "EV_SALES"]

    # Fetch target
    target_data = fetch_peer_multiples(t, market_provider)

    # Fetch peers
    peer_data = []
    for p in peers:
        pd = fetch_peer_multiples(p, market_provider)
        peer_data.append(pd)

    # Compute stats per multiple
    stats = {}
    for mt in multiples_to_compute:
        stats[mt] = compute_multiple_stats(peer_data, mt)

    # Rank target vs peers
    rankings = []
    for mt in multiples_to_compute:
        rankings.append(rank_target_vs_peers(target_data, peer_data, mt))

    # Data quality assessment
    successful_fetches = sum(1 for p in peer_data if p.fetch_success)
    data_quality = {
        "peers_requested": len(peers),
        "peers_fetched": successful_fetches,
        "fetch_rate": successful_fetches / max(len(peers), 1),
        "target_fetched": target_data.fetch_success,
        "multiples_with_data": sum(1 for s in stats.values() if s.count >= 2),
        "total_multiples": len(multiples_to_compute),
    }

    # Summary
    summary_parts = [f"{t} peer analysis ({sector_info['industry']})."]
    for r in rankings:
        if r.target_value is not None and r.peer_median is not None:
            premium_str = f"{r.premium_to_median_pct:+.1%}" if r.premium_to_median_pct is not None else "N/A"
            summary_parts.append(
                f"{r.multiple_type}: {r.target_value:.1f}x vs peer median {r.peer_median:.1f}x "
                f"({premium_str}, rank {r.rank}/{r.total_ranked}, {r.assessment})."
            )

    return PeerAnalysisResult(
        target_ticker=t,
        sector=sector_info["sector"],
        industry=sector_info["industry"],
        peer_tickers=peers,
        target_multiples=target_data,
        peer_multiples=peer_data,
        stats=stats,
        rankings=rankings,
        summary=" ".join(summary_parts),
        data_quality=data_quality,
    )


# ============================================================
# 6. Integration helpers for orchestrator
# ============================================================

def get_peer_median_for_multiple(
    ticker: str,
    multiple_type: MultipleType,
    market_provider: Any,
    *,
    max_peers: int = 5,
) -> Optional[float]:
    """
    Quick helper: get just the peer median for one multiple.

    This is the simplest integration point — call this from
    orchestrator.py when running relative_valuation mode to
    automatically provide the peer_median parameter.
    """
    try:
        result = run_peer_analysis(
            ticker=ticker,
            market_provider=market_provider,
            max_peers=max_peers,
            multiples_to_compute=[multiple_type],
        )
        for r in result.rankings:
            if r.multiple_type == multiple_type:
                return r.peer_median
    except Exception:
        logger.exception("Peer median fetch failed for %s %s", ticker, multiple_type)
    return None


def peer_analysis_to_signal(result: PeerAnalysisResult) -> Dict[str, Any]:
    """
    Convert peer analysis into a format consumable by quant_decision_engine.

    Returns data that can feed into build_relative_valuation_signal().
    """
    # Use the first available ranking as the primary signal
    primary = None
    for r in result.rankings:
        if r.target_value is not None and r.peer_median is not None:
            primary = r
            break

    if primary is None:
        return {"peer_median": None, "assessment": "insufficient", "rankings": []}

    return {
        "peer_median": primary.peer_median,
        "target_value": primary.target_value,
        "premium_pct": primary.premium_to_median_pct,
        "assessment": primary.assessment,
        "multiple_type": primary.multiple_type,
        "rank": primary.rank,
        "total_ranked": primary.total_ranked,
        "peer_tickers": result.peer_tickers,
        "data_quality": result.data_quality,
        "all_rankings": [
            {
                "multiple": r.multiple_type,
                "target": r.target_value,
                "peer_median": r.peer_median,
                "premium": r.premium_to_median_pct,
                "rank": r.rank,
                "total": r.total_ranked,
                "assessment": r.assessment,
            }
            for r in result.rankings
            if r.target_value is not None
        ],
    }
