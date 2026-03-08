# scenario_analysis.py
# ============================================================
# Interactive Scenario Analysis Engine
#
# Goes beyond the static sensitivity grid in valuation_engine.py.
# Supports:
#   - Named scenarios (base/bull/bear/stress)
#   - What-if parameter overrides ("what if WACC increases by 200bps")
#   - Multi-parameter sweeps
#   - Stress testing with predefined stress profiles
#   - Scenario comparison with delta attribution
#   - Monte Carlo simulation (simplified)
#
# Plugs into the existing run_dcf() and assumptions_policy.py.
# ============================================================

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from assumptions_policy import AssumptionsPolicy, build_assumptions, policy_to_jsonable
from valuation_engine import DCFResult, run_dcf

ScenarioType = Literal["base", "bull", "bear", "stress", "custom"]


# ============================================================
# 1. Scenario Definitions
# ============================================================

@dataclass
class ScenarioSpec:
    """A named scenario with parameter overrides."""
    name: str
    scenario_type: ScenarioType
    description: str
    overrides: Dict[str, Any]  # keys match assumptions_policy fields


# Predefined scenarios — these are common stress/bull/bear cases
# that a quant would use for sensitivity testing
PREDEFINED_SCENARIOS: Dict[str, ScenarioSpec] = {
    "base": ScenarioSpec(
        name="Base Case",
        scenario_type="base",
        description="Current assumptions from policy defaults.",
        overrides={},
    ),
    "bull": ScenarioSpec(
        name="Bull Case",
        scenario_type="bull",
        description="Higher growth, lower discount rate. Market tailwinds scenario.",
        overrides={
            "fcf_growth_base": 0.08,
            "wacc_base": 0.075,
            "terminal_growth_base": 0.030,
        },
    ),
    "bear": ScenarioSpec(
        name="Bear Case",
        scenario_type="bear",
        description="Lower growth, higher discount rate. Recession/headwinds scenario.",
        overrides={
            "fcf_growth_base": 0.02,
            "wacc_base": 0.10,
            "terminal_growth_base": 0.015,
        },
    ),
    "stress_recession": ScenarioSpec(
        name="Recession Stress",
        scenario_type="stress",
        description="Deep recession: negative FCF growth, elevated WACC, compressed terminal.",
        overrides={
            "fcf_growth_base": -0.05,
            "wacc_base": 0.12,
            "terminal_growth_base": 0.010,
        },
    ),
    "stress_rate_shock": ScenarioSpec(
        name="Rate Shock (+200bps)",
        scenario_type="stress",
        description="Interest rate shock: WACC increases by 200 basis points from base.",
        overrides={
            "wacc_base": 0.105,  # 8.5% + 2% = 10.5%
        },
    ),
    "stress_growth_collapse": ScenarioSpec(
        name="Growth Collapse",
        scenario_type="stress",
        description="FCF growth drops to zero, terminal growth at minimum.",
        overrides={
            "fcf_growth_base": 0.00,
            "terminal_growth_base": 0.010,
        },
    ),
    "bull_ai_tailwind": ScenarioSpec(
        name="AI Tailwind",
        scenario_type="bull",
        description="AI-driven revenue acceleration with margin expansion.",
        overrides={
            "fcf_growth_base": 0.12,
            "wacc_base": 0.08,
            "terminal_growth_base": 0.030,
        },
    ),
}


# ============================================================
# 2. What-If Parser
# ============================================================

_WHATIF_PATTERNS = [
    # "what if WACC increases by 200bps"
    (re.compile(r"wacc\s+(?:increases?|goes?\s+up|rises?)\s+(?:by\s+)?(\d+)\s*(?:bps|basis\s*points?)", re.I),
     lambda m: {"wacc_delta_bps": int(m.group(1))}),

    # "what if WACC decreases by 100bps"
    (re.compile(r"wacc\s+(?:decreases?|goes?\s+down|drops?|falls?)\s+(?:by\s+)?(\d+)\s*(?:bps|basis\s*points?)", re.I),
     lambda m: {"wacc_delta_bps": -int(m.group(1))}),

    # "what if WACC is 10%"
    (re.compile(r"wacc\s+(?:is|=|equals?|were?|at)\s+(\d+(?:\.\d+)?)\s*%", re.I),
     lambda m: {"wacc_base": float(m.group(1)) / 100}),

    # "what if terminal growth is 1.5%"
    (re.compile(r"terminal\s+(?:growth|g)\s+(?:is|=|at|were?)\s+(\d+(?:\.\d+)?)\s*%", re.I),
     lambda m: {"terminal_growth_base": float(m.group(1)) / 100}),

    # "what if growth drops to 2%" (avoid matching terminal growth)
    (re.compile(r"(?<!terminal\s)(?:fcf\s+)?growth\s+(?:drops?|falls?|decreases?|is|=)\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%", re.I),
     lambda m: {"fcf_growth_base": float(m.group(1)) / 100}),

    # "what if growth increases to 10%" (avoid matching terminal growth)
    (re.compile(r"(?<!terminal\s)(?:fcf\s+)?growth\s+(?:increases?|rises?|goes?\s+up|is|=)\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%", re.I),
     lambda m: {"fcf_growth_base": float(m.group(1)) / 100}),

    # "what if revenue growth drops to 2%"
    (re.compile(r"revenue\s+growth\s+(?:drops?|falls?|is)\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%", re.I),
     lambda m: {"fcf_growth_base": float(m.group(1)) / 100}),
]


def parse_whatif(question: str, base_assumptions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a what-if question into assumption overrides.

    Returns a dict of parameter overrides that can be passed to build_assumptions().
    """
    overrides = {}
    q = question or ""

    for pattern, extractor in _WHATIF_PATTERNS:
        m = pattern.search(q)
        if m:
            extracted = extractor(m)
            # Handle relative deltas
            if "wacc_delta_bps" in extracted:
                current_wacc = float(base_assumptions.get("wacc_base", 0.085))
                overrides["wacc_base"] = current_wacc + extracted["wacc_delta_bps"] / 10000
            else:
                overrides.update(extracted)

    return overrides


def is_whatif_query(question: str) -> bool:
    """Detect if this is a what-if / scenario question."""
    q = (question or "").lower()
    triggers = [
        "what if", "what would", "what happens if", "scenario",
        "stress test", "sensitivity", "how would", "impact of",
        "assuming", "if wacc", "if growth", "bull case", "bear case",
    ]
    return any(t in q for t in triggers)


# ============================================================
# 3. Scenario Runner
# ============================================================

@dataclass
class ScenarioResult:
    """Result of running a single scenario."""
    scenario: ScenarioSpec
    assumptions_used: Dict[str, Any]
    dcf: DCFResult
    enterprise_value: float
    equity_value: Optional[float]
    intrinsic_value_per_share: Optional[float]


@dataclass
class ScenarioComparison:
    """Comparison between base case and scenario."""
    scenario_name: str
    base_ev: float
    scenario_ev: float
    ev_delta: float
    ev_delta_pct: float
    base_ivps: Optional[float]
    scenario_ivps: Optional[float]
    ivps_delta: Optional[float]
    ivps_delta_pct: Optional[float]
    parameter_changes: Dict[str, Dict[str, float]]  # param -> {base, scenario, delta}


@dataclass
class ScenarioAnalysisResult:
    """Full scenario analysis output."""
    ticker: str
    last_fcf: float
    currency: str
    base_result: ScenarioResult
    scenario_results: List[ScenarioResult]
    comparisons: List[ScenarioComparison]
    monte_carlo: Optional[Dict[str, Any]] = None


def run_scenario(
    *,
    scenario: ScenarioSpec,
    last_fcf: float,
    currency: str,
    base_policy: Optional[AssumptionsPolicy] = None,
    net_debt: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
    strictness: int = 55,
) -> ScenarioResult:
    """Run a single scenario through DCF."""
    assumptions = build_assumptions(
        strictness=strictness,
        policy=base_policy,
        overrides=scenario.overrides or None,
    )

    dcf = run_dcf(
        last_fcf=last_fcf,
        currency=currency,
        assumptions=assumptions,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
    )

    return ScenarioResult(
        scenario=scenario,
        assumptions_used=assumptions,
        dcf=dcf,
        enterprise_value=dcf.enterprise_value,
        equity_value=dcf.equity_value,
        intrinsic_value_per_share=dcf.intrinsic_value_per_share,
    )


def compare_scenarios(
    base: ScenarioResult,
    other: ScenarioResult,
) -> ScenarioComparison:
    """Compute deltas between base and another scenario."""
    ev_delta = other.enterprise_value - base.enterprise_value
    ev_delta_pct = ev_delta / base.enterprise_value if base.enterprise_value != 0 else 0

    ivps_delta = None
    ivps_delta_pct = None
    if base.intrinsic_value_per_share is not None and other.intrinsic_value_per_share is not None:
        ivps_delta = other.intrinsic_value_per_share - base.intrinsic_value_per_share
        if base.intrinsic_value_per_share != 0:
            ivps_delta_pct = ivps_delta / base.intrinsic_value_per_share

    # Track which parameters changed
    param_changes = {}
    tracked_params = ["wacc_base", "fcf_growth_base", "terminal_growth_base", "horizon_years"]
    for p in tracked_params:
        base_val = base.assumptions_used.get(p)
        other_val = other.assumptions_used.get(p)
        if base_val is not None and other_val is not None and base_val != other_val:
            param_changes[p] = {
                "base": float(base_val),
                "scenario": float(other_val),
                "delta": float(other_val) - float(base_val),
            }

    return ScenarioComparison(
        scenario_name=other.scenario.name,
        base_ev=base.enterprise_value,
        scenario_ev=other.enterprise_value,
        ev_delta=ev_delta,
        ev_delta_pct=ev_delta_pct,
        base_ivps=base.intrinsic_value_per_share,
        scenario_ivps=other.intrinsic_value_per_share,
        ivps_delta=ivps_delta,
        ivps_delta_pct=ivps_delta_pct,
        parameter_changes=param_changes,
    )


def run_scenario_analysis(
    *,
    ticker: str,
    last_fcf: float,
    currency: str = "USD",
    scenario_names: Optional[List[str]] = None,
    custom_scenarios: Optional[List[ScenarioSpec]] = None,
    net_debt: Optional[float] = None,
    shares_outstanding: Optional[float] = None,
    strictness: int = 55,
    run_monte_carlo: bool = False,
    mc_iterations: int = 1000,
) -> ScenarioAnalysisResult:
    """
    Run full scenario analysis: base + named scenarios + custom scenarios.

    Usage:
        result = run_scenario_analysis(
            ticker="AAPL",
            last_fcf=110_000_000_000,
            scenario_names=["bull", "bear", "stress_recession"],
            net_debt=50_000_000_000,
            shares_outstanding=15_400_000_000,
        )
    """
    # Always run base
    base_scenario = PREDEFINED_SCENARIOS["base"]
    base_result = run_scenario(
        scenario=base_scenario,
        last_fcf=last_fcf,
        currency=currency,
        net_debt=net_debt,
        shares_outstanding=shares_outstanding,
        strictness=strictness,
    )

    # Collect scenarios to run
    scenarios_to_run: List[ScenarioSpec] = []

    for name in (scenario_names or []):
        if name in PREDEFINED_SCENARIOS and name != "base":
            scenarios_to_run.append(PREDEFINED_SCENARIOS[name])

    if custom_scenarios:
        scenarios_to_run.extend(custom_scenarios)

    # Run all scenarios
    scenario_results = []
    comparisons = []

    for spec in scenarios_to_run:
        try:
            result = run_scenario(
                scenario=spec,
                last_fcf=last_fcf,
                currency=currency,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
                strictness=strictness,
            )
            scenario_results.append(result)
            comparisons.append(compare_scenarios(base_result, result))
        except Exception as e:
            # Log but don't fail the whole analysis
            scenario_results.append(ScenarioResult(
                scenario=spec,
                assumptions_used={},
                dcf=None,
                enterprise_value=0,
                equity_value=None,
                intrinsic_value_per_share=None,
            ))

    # Optional Monte Carlo
    mc_result = None
    if run_monte_carlo:
        mc_result = _run_monte_carlo(
            last_fcf=last_fcf,
            currency=currency,
            net_debt=net_debt,
            shares_outstanding=shares_outstanding,
            iterations=mc_iterations,
            strictness=strictness,
        )

    return ScenarioAnalysisResult(
        ticker=ticker,
        last_fcf=last_fcf,
        currency=currency,
        base_result=base_result,
        scenario_results=scenario_results,
        comparisons=comparisons,
        monte_carlo=mc_result,
    )


# ============================================================
# 4. Simplified Monte Carlo
# ============================================================

def _run_monte_carlo(
    *,
    last_fcf: float,
    currency: str,
    net_debt: Optional[float],
    shares_outstanding: Optional[float],
    iterations: int = 1000,
    strictness: int = 55,
) -> Dict[str, Any]:
    """
    Simplified Monte Carlo over WACC and growth assumptions.

    Samples from triangular distributions around base assumptions.
    """
    policy = AssumptionsPolicy()
    results = []

    for _ in range(iterations):
        # Sample WACC: triangular(0.06, base, 0.14)
        wacc = random.triangular(0.06, 0.14, policy.wacc_base)
        # Sample FCF growth: triangular(-0.05, base, 0.15)
        fcf_g = random.triangular(-0.05, 0.15, policy.fcf_growth_base)
        # Sample terminal growth: triangular(0.005, base, 0.04)
        tg = random.triangular(0.005, 0.04, policy.terminal_growth_base)

        # Enforce WACC > tg + buffer
        if wacc <= tg + policy.min_wacc_minus_tg:
            wacc = tg + policy.min_wacc_minus_tg + 0.01

        overrides = {
            "wacc_base": wacc,
            "fcf_growth_base": fcf_g,
            "terminal_growth_base": tg,
        }

        try:
            assumptions = build_assumptions(strictness=strictness, overrides=overrides)
            dcf = run_dcf(
                last_fcf=last_fcf,
                currency=currency,
                assumptions=assumptions,
                net_debt=net_debt,
                shares_outstanding=shares_outstanding,
            )
            if dcf.intrinsic_value_per_share is not None:
                results.append(dcf.intrinsic_value_per_share)
            else:
                results.append(dcf.enterprise_value)
        except Exception:
            continue

    if not results:
        return {"error": "All iterations failed"}

    results.sort()
    n = len(results)

    return {
        "iterations": iterations,
        "successful": n,
        "mean": sum(results) / n,
        "median": results[n // 2],
        "p10": results[int(n * 0.10)],
        "p25": results[int(n * 0.25)],
        "p75": results[int(n * 0.75)],
        "p90": results[int(n * 0.90)],
        "min": results[0],
        "max": results[-1],
        "std": (sum((x - sum(results) / n) ** 2 for x in results) / n) ** 0.5,
    }
