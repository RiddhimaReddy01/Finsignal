# assumptions_policy.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional

TerminalMethod = Literal["gordon", "exit_multiple"]
DiscountMethod = Literal["fixed_wacc", "capm"]


@dataclass(frozen=True)
class AssumptionsPolicy:
    """
    System-defined assumptions for valuation. Users do NOT set these in the UI.
    You version + hash this so the output is reproducible and auditable.
    """
    version: str = "assumptions_v1"

    # Forecast
    horizon_years: int = 5
    fcf_growth_base: float = 0.05
    fcf_growth_grid: tuple[float, ...] = (0.03, 0.05, 0.07)

    # Discounting
    discount_method: DiscountMethod = "fixed_wacc"
    wacc_base: float = 0.085
    wacc_grid: tuple[float, ...] = (0.075, 0.085, 0.095)

    # CAPM knobs (used only if discount_method="capm")
    equity_risk_premium: float = 0.05
    beta_fallback: float = 1.0
    risk_free_fallback: float = 0.04

    # Terminal value
    terminal_method: TerminalMethod = "gordon"
    terminal_growth_base: float = 0.025
    terminal_growth_grid: tuple[float, ...] = (0.015, 0.020, 0.025, 0.030)

    # Safety constraints
    min_wacc_minus_tg: float = 0.01  # enforce WACC >= tg + buffer


def policy_to_jsonable(policy: AssumptionsPolicy) -> Dict[str, Any]:
    d = asdict(policy)
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


def policy_hash(policy: AssumptionsPolicy) -> str:
    s = json.dumps(policy_to_jsonable(policy), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_assumptions(
    *,
    strictness: int,
    policy: Optional[AssumptionsPolicy] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns the assumptions object the LLM must echo for valuation modes.
    Strictness can be used later to widen/narrow grids. MVP keeps policy fixed.
    """
    policy = policy or AssumptionsPolicy()
    a = policy_to_jsonable(policy)

    if overrides and isinstance(overrides, dict):
        for k, v in overrides.items():
            if k in a and v is not None:
                a[k] = v

    a["policy_version"] = policy.version
    a["policy_hash"] = policy_hash(policy)
    a["strictness"] = int(strictness)
    return a