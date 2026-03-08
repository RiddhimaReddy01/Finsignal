# valuation_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DCFResult:
    currency: str
    enterprise_value: float
    equity_value: Optional[float]
    intrinsic_value_per_share: Optional[float]
    fcf_forecast: List[float]
    pv_fcf: List[float]
    terminal_value: float
    pv_terminal_value: float
    assumptions_used: Dict[str, Any]
    sensitivity: Dict[str, Any]


def _pv(x: float, r: float, t: int) -> float:
    return float(x) / ((1.0 + float(r)) ** int(t))


def forecast_fcf(last_fcf: float, growth: float, years: int) -> List[float]:
    out = []
    base = float(last_fcf)
    g = float(growth)
    for _ in range(years):
        base = base * (1.0 + g)
        out.append(base)
    return out


def terminal_value_gordon(last_fcf_year: float, wacc: float, tg: float, *, min_spread: float) -> float:
    w = float(wacc)
    g = float(tg)
    if w <= g + float(min_spread):
        raise ValueError("Invalid terminal assumptions: require WACC > terminal growth + buffer")
    return float(last_fcf_year) * (1.0 + g) / (w - g)


def run_dcf(
    *,
    last_fcf: float,
    currency: str,
    assumptions: Dict[str, Any],
    net_debt: Optional[float] = None,            # debt - cash
    shares_outstanding: Optional[float] = None,  # shares count
) -> DCFResult:
    ccy = currency or "USD"
    years = int(assumptions.get("horizon_years", 5))

    wacc_base = float(assumptions.get("wacc_base", 0.085))
    fcf_g = float(assumptions.get("fcf_growth_base", 0.05))
    tg_base = float(assumptions.get("terminal_growth_base", 0.025))
    min_spread = float(assumptions.get("min_wacc_minus_tg", 0.01))

    fcf_forecast = forecast_fcf(last_fcf, fcf_g, years)
    pv_fcf = [_pv(cf, wacc_base, t=i) for i, cf in enumerate(fcf_forecast, start=1)]

    tv = terminal_value_gordon(fcf_forecast[-1], wacc_base, tg_base, min_spread=min_spread)
    pv_tv = _pv(tv, wacc_base, t=years)

    ev = float(sum(pv_fcf) + pv_tv)

    eq = None
    ivps = None
    if net_debt is not None:
        eq = ev - float(net_debt)
        if shares_outstanding is not None and float(shares_outstanding) > 0:
            ivps = eq / float(shares_outstanding)

    # Sensitivity grid
    wacc_grid = assumptions.get("wacc_grid") or [wacc_base]
    tg_grid = assumptions.get("terminal_growth_grid") or [tg_base]
    grid = []
    for w in wacc_grid:
        row = []
        for tg in tg_grid:
            try:
                tv_ = terminal_value_gordon(fcf_forecast[-1], float(w), float(tg), min_spread=min_spread)
                pv_fcf_ = [_pv(cf, float(w), t=i) for i, cf in enumerate(fcf_forecast, start=1)]
                ev_ = float(sum(pv_fcf_) + _pv(tv_, float(w), t=years))
                eq_ = None
                ivps_ = None
                if net_debt is not None:
                    eq_ = ev_ - float(net_debt)
                    if shares_outstanding is not None and float(shares_outstanding) > 0:
                        ivps_ = eq_ / float(shares_outstanding)
                row.append({"wacc": float(w), "tg": float(tg), "ev": ev_, "eq": eq_, "ivps": ivps_})
            except Exception as e:
                row.append({"wacc": float(w), "tg": float(tg), "error": str(e)})
        grid.append(row)

    return DCFResult(
        currency=ccy,
        enterprise_value=ev,
        equity_value=eq,
        intrinsic_value_per_share=ivps,
        fcf_forecast=fcf_forecast,
        pv_fcf=pv_fcf,
        terminal_value=tv,
        pv_terminal_value=pv_tv,
        assumptions_used=assumptions,
        sensitivity={"wacc_grid": wacc_grid, "terminal_growth_grid": tg_grid, "grid": grid},
    )