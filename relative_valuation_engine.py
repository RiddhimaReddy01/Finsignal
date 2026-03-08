# relative_valuation_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

MultipleType = Literal["P_E", "P_S", "EV_SALES", "EV_EBITDA"]


@dataclass
class RelativeValuationResult:
    multiple_type: MultipleType
    numerator: float
    denominator: float
    multiple: float
    currency: str
    notes: str
    sources: Dict[str, Any]


def compute_multiple(
    *,
    multiple_type: MultipleType,
    currency: str = "USD",
    # market
    price: Optional[float] = None,
    market_cap: Optional[float] = None,
    enterprise_value: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
    # filing denominators
    eps: Optional[float] = None,       # per-share
    revenue: Optional[float] = None,   # total
    ebitda: Optional[float] = None,    # total
    sources: Optional[Dict[str, Any]] = None,
) -> RelativeValuationResult:
    sources = sources or {}
    ccy = currency or "USD"

    if multiple_type == "P_E":
        if price is None:
            raise ValueError("P/E requires price")
        if eps is None or float(eps) == 0.0:
            raise ValueError("P/E requires non-zero EPS")
        mult = float(price) / float(eps)
        return RelativeValuationResult(
            multiple_type=multiple_type,
            numerator=float(price),
            denominator=float(eps),
            multiple=mult,
            currency=ccy,
            notes="P/E = Price / EPS (EPS is per-share).",
            sources=sources,
        )

    if multiple_type == "P_S":
        if price is None:
            raise ValueError("P/S requires price")
        if shares_outstanding is None or float(shares_outstanding) == 0.0:
            raise ValueError("P/S requires shares_outstanding")
        if revenue is None or float(revenue) == 0.0:
            raise ValueError("P/S requires non-zero revenue")
        rev_per_share = float(revenue) / float(shares_outstanding)
        mult = float(price) / rev_per_share
        return RelativeValuationResult(
            multiple_type=multiple_type,
            numerator=float(price),
            denominator=rev_per_share,
            multiple=mult,
            currency=ccy,
            notes="P/S = Price / (Revenue per share).",
            sources=sources,
        )

    if multiple_type == "EV_SALES":
        ev = enterprise_value
        if ev is None:
            if market_cap is None:
                raise ValueError("EV/Sales requires enterprise_value or market_cap")
            ev = float(market_cap)
            sources["ev_proxy"] = "EV approximated as market cap (no net debt provided)."
        if revenue is None or float(revenue) == 0.0:
            raise ValueError("EV/Sales requires non-zero revenue")
        mult = float(ev) / float(revenue)
        return RelativeValuationResult(
            multiple_type=multiple_type,
            numerator=float(ev),
            denominator=float(revenue),
            multiple=mult,
            currency=ccy,
            notes="EV/Sales = Enterprise Value / Revenue.",
            sources=sources,
        )

    if multiple_type == "EV_EBITDA":
        ev = enterprise_value
        if ev is None:
            if market_cap is None:
                raise ValueError("EV/EBITDA requires enterprise_value or market_cap")
            ev = float(market_cap)
            sources["ev_proxy"] = "EV approximated as market cap (no net debt provided)."
        if ebitda is None or float(ebitda) == 0.0:
            raise ValueError("EV/EBITDA requires non-zero EBITDA")
        mult = float(ev) / float(ebitda)
        return RelativeValuationResult(
            multiple_type=multiple_type,
            numerator=float(ev),
            denominator=float(ebitda),
            multiple=mult,
            currency=ccy,
            notes="EV/EBITDA = Enterprise Value / EBITDA.",
            sources=sources,
        )

    raise ValueError(f"Unsupported multiple_type: {multiple_type}")