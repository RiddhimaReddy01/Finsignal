# multi_period_analysis.py
# ============================================================
# Multi-Period Comparative Analysis
#
# Handles time-series questions like:
#   "How did Apple's margins change from FY2023 to FY2025?"
#   "Compare AAPL revenue growth across 2022-2025"
#   "What's the trend in NVDA's risk language over 3 years?"
#
# What this adds beyond the existing comparative_analysis mode:
#   - Detects multi-period intent for the SAME ticker
#   - Retrieves evidence per period with period-specific filters
#   - Computes deltas, CAGR, trends across periods
#   - Classifies trend direction (improving/stable/deteriorating)
#   - Builds structured time-series output
#
# Plugs into orchestrator.py as an enhancement to
# comparative_analysis when targets share the same ticker
# but differ in fiscal_year.
# ============================================================

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

TrendDirection = Literal["improving", "stable", "deteriorating", "volatile", "insufficient"]


@dataclass
class PeriodDataPoint:
    """A single metric value for a specific period."""
    fiscal_year: int
    value: float
    unit: str
    evidence_id: str
    confidence: float  # how reliable this extraction was


@dataclass
class TimeSeries:
    """A metric tracked across multiple periods."""
    metric: str
    ticker: str
    points: List[PeriodDataPoint]
    unit: str

    # Computed fields
    deltas: List[Dict[str, Any]] = field(default_factory=list)
    cagr: Optional[float] = None
    trend: TrendDirection = "insufficient"
    trend_reasoning: str = ""


@dataclass
class MultiPeriodResult:
    """Complete multi-period analysis output."""
    ticker: str
    periods: List[int]
    series: List[TimeSeries]
    summary: str
    risk_trend: Optional[Dict[str, Any]] = None
    tone_trend: Optional[Dict[str, Any]] = None


# ============================================================
# 1. Detection — is this a multi-period question?
# ============================================================

_MULTI_YEAR_PATTERNS = [
    re.compile(r"(\d{4})\s*(?:to|-|through|–|—)\s*(\d{4})", re.I),
    re.compile(r"(?:from|between)\s+(?:fy\s*)?(\d{4})\s+(?:to|and|through)\s+(?:fy\s*)?(\d{4})", re.I),
    re.compile(r"(?:over|across|last)\s+(\d+)\s+years?", re.I),
    re.compile(r"(?:trend|evolution|trajectory|change|changed|changing)", re.I),
    re.compile(r"(?:year\s*over\s*year|yoy|y/y|annual\s+change)", re.I),
]

_TREND_WORDS = {"trend", "evolution", "trajectory", "over time", "historical", "changed", "improved", "worsened", "grown", "declined"}


def is_multi_period_query(question: str, tickers: List[str], years: List[int]) -> bool:
    """
    Detect if this is a multi-period question for the same entity.

    Returns True if:
    - Same ticker mentioned with 2+ years
    - Trend/evolution language is present
    - Year range pattern detected (e.g., "2022 to 2025")
    """
    q = (question or "").lower()

    # Explicit year range
    for pat in _MULTI_YEAR_PATTERNS[:2]:
        if pat.search(question):
            return True

    # "last N years" pattern
    m = _MULTI_YEAR_PATTERNS[2].search(question)
    if m:
        return True

    # Trend language with a single ticker
    if len(tickers) == 1 and any(w in q for w in _TREND_WORDS):
        return True

    # Same ticker with 2+ years explicitly mentioned
    if len(tickers) <= 1 and len(years) >= 2:
        return True

    return False


def extract_period_range(question: str, detected_years: List[int]) -> List[int]:
    """
    Extract the full list of fiscal years to analyze.

    Handles:
    - "2022 to 2025" → [2022, 2023, 2024, 2025]
    - "last 3 years" → [current-2, current-1, current] (filled from detected_years)
    - explicit years → sorted list
    """
    q = question or ""

    # Explicit range
    for pat in _MULTI_YEAR_PATTERNS[:2]:
        m = pat.search(q)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if 1990 <= y1 <= 2100 and 1990 <= y2 <= 2100:
                lo, hi = min(y1, y2), max(y1, y2)
                return list(range(lo, hi + 1))

    # "last N years"
    m = _MULTI_YEAR_PATTERNS[2].search(q)
    if m and detected_years:
        n = int(m.group(1))
        latest = max(detected_years)
        return list(range(latest - n + 1, latest + 1))

    # Fall back to detected years
    if detected_years:
        return sorted(set(detected_years))

    return []


# ============================================================
# 2. Per-Period Evidence Retrieval
# ============================================================

def build_period_retrieval_targets(
    ticker: str,
    periods: List[int],
    metric: Optional[str],
    item_hint: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Build retrieval filter sets for each period.

    Returns a list of dicts, each suitable for passing as `filters`
    to FinancialRetrievalTool.retrieve().
    """
    targets = []
    for fy in sorted(periods):
        filters = {"ticker": ticker, "fiscal_year": fy}
        if item_hint:
            filters["item"] = item_hint
        targets.append({
            "fiscal_year": fy,
            "filters": filters,
            "query_suffix": f"{ticker} FY{fy}" + (f" {metric}" if metric else ""),
        })
    return targets


# ============================================================
# 3. Time-Series Computation
# ============================================================

def compute_deltas(points: List[PeriodDataPoint]) -> List[Dict[str, Any]]:
    """Compute year-over-year deltas between consecutive data points."""
    if len(points) < 2:
        return []

    sorted_pts = sorted(points, key=lambda p: p.fiscal_year)
    deltas = []

    for i in range(1, len(sorted_pts)):
        prev = sorted_pts[i - 1]
        curr = sorted_pts[i]
        abs_delta = curr.value - prev.value

        if prev.value != 0:
            pct_delta = abs_delta / abs(prev.value)
        else:
            pct_delta = None

        deltas.append({
            "from_fy": prev.fiscal_year,
            "to_fy": curr.fiscal_year,
            "from_value": prev.value,
            "to_value": curr.value,
            "absolute_delta": abs_delta,
            "pct_delta": pct_delta,
            "from_evidence": prev.evidence_id,
            "to_evidence": curr.evidence_id,
        })

    return deltas


def compute_cagr(points: List[PeriodDataPoint]) -> Optional[float]:
    """Compute Compound Annual Growth Rate across the full period."""
    if len(points) < 2:
        return None

    sorted_pts = sorted(points, key=lambda p: p.fiscal_year)
    first = sorted_pts[0]
    last = sorted_pts[-1]
    years = last.fiscal_year - first.fiscal_year

    if years <= 0 or first.value <= 0 or last.value <= 0:
        return None

    try:
        cagr = (last.value / first.value) ** (1.0 / years) - 1.0
        return cagr
    except (ZeroDivisionError, ValueError, OverflowError):
        return None


def classify_trend(deltas: List[Dict[str, Any]], metric: str) -> Tuple[TrendDirection, str]:
    """
    Classify the overall trend direction from period deltas.

    Logic:
    - All deltas positive → improving
    - All deltas negative → deteriorating
    - Mixed with small magnitude → stable
    - Mixed with large swings → volatile
    """
    if not deltas:
        return "insufficient", "Not enough data points for trend classification."

    pct_deltas = [d["pct_delta"] for d in deltas if d["pct_delta"] is not None]
    if not pct_deltas:
        return "insufficient", "Could not compute percentage changes."

    positive = sum(1 for d in pct_deltas if d > 0.02)
    negative = sum(1 for d in pct_deltas if d < -0.02)
    total = len(pct_deltas)
    avg_magnitude = sum(abs(d) for d in pct_deltas) / total

    # For metrics where lower is better (like debt_to_equity), invert the interpretation
    inverted_metrics = {"debt_to_equity", "current_ratio_inverse"}
    is_inverted = metric.lower() in inverted_metrics

    if positive == total:
        direction = "deteriorating" if is_inverted else "improving"
        reasoning = f"Consistent positive trajectory: all {total} periods showed increases."
    elif negative == total:
        direction = "improving" if is_inverted else "deteriorating"
        reasoning = f"Consistent negative trajectory: all {total} periods showed decreases."
    elif avg_magnitude < 0.03:
        direction = "stable"
        reasoning = f"Average absolute change of {avg_magnitude:.1%} is below the 3% materiality threshold."
    elif positive >= total * 0.7:
        direction = "deteriorating" if is_inverted else "improving"
        reasoning = f"Predominantly positive: {positive}/{total} periods increased (avg magnitude {avg_magnitude:.1%})."
    elif negative >= total * 0.7:
        direction = "improving" if is_inverted else "deteriorating"
        reasoning = f"Predominantly negative: {negative}/{total} periods decreased (avg magnitude {avg_magnitude:.1%})."
    else:
        direction = "volatile"
        reasoning = f"Mixed signals: {positive} up, {negative} down, {total - positive - negative} flat. Average swing {avg_magnitude:.1%}."

    return direction, reasoning


def build_time_series(
    metric: str,
    ticker: str,
    points: List[PeriodDataPoint],
) -> TimeSeries:
    """Assemble a complete TimeSeries with computed analytics."""
    if not points:
        return TimeSeries(
            metric=metric, ticker=ticker, points=[], unit="UNKNOWN",
            trend="insufficient", trend_reasoning="No data points extracted.",
        )

    sorted_pts = sorted(points, key=lambda p: p.fiscal_year)
    unit = sorted_pts[0].unit

    deltas = compute_deltas(sorted_pts)
    cagr = compute_cagr(sorted_pts)
    trend, reasoning = classify_trend(deltas, metric)

    if cagr is not None:
        reasoning += f" CAGR: {cagr:+.2%} over {sorted_pts[-1].fiscal_year - sorted_pts[0].fiscal_year} years."

    return TimeSeries(
        metric=metric,
        ticker=ticker,
        points=sorted_pts,
        unit=unit,
        deltas=deltas,
        cagr=cagr,
        trend=trend,
        trend_reasoning=reasoning,
    )


# ============================================================
# 4. Risk Language Trend (multi-period risk analysis)
# ============================================================

def compute_risk_trend(
    risk_signals_by_year: Dict[int, List[Any]],
) -> Dict[str, Any]:
    """
    Compare risk signal intensity across fiscal years.

    Input: {2023: [RiskSignal, ...], 2024: [...], 2025: [...]}
    Output: trend summary with per-category evolution.
    """
    if len(risk_signals_by_year) < 2:
        return {"trend": "insufficient", "detail": "Need 2+ years of risk data."}

    # Aggregate severity per category per year
    categories: Dict[str, Dict[int, float]] = {}
    for fy, signals in sorted(risk_signals_by_year.items()):
        for sig in signals:
            cat = getattr(sig, "category", None) or (sig.get("category") if isinstance(sig, dict) else None) or "unknown"
            sev = float(getattr(sig, "severity", 0) or (sig.get("severity", 0) if isinstance(sig, dict) else 0))
            categories.setdefault(cat, {})[fy] = max(categories.get(cat, {}).get(fy, 0), sev)

    # Compute trend per category
    category_trends = []
    years = sorted(risk_signals_by_year.keys())
    for cat, year_sevs in categories.items():
        if len(year_sevs) < 2:
            continue
        first_sev = year_sevs.get(years[0], 0)
        last_sev = year_sevs.get(years[-1], 0)
        delta = last_sev - first_sev

        if delta > 0.1:
            direction = "escalating"
        elif delta < -0.1:
            direction = "diminishing"
        else:
            direction = "stable"

        category_trends.append({
            "category": cat,
            "direction": direction,
            "first_year": years[0],
            "first_severity": round(first_sev, 3),
            "last_year": years[-1],
            "last_severity": round(last_sev, 3),
            "delta": round(delta, 3),
        })

    # Overall
    escalating = sum(1 for t in category_trends if t["direction"] == "escalating")
    diminishing = sum(1 for t in category_trends if t["direction"] == "diminishing")

    if escalating > diminishing:
        overall = "risk_increasing"
    elif diminishing > escalating:
        overall = "risk_decreasing"
    else:
        overall = "risk_stable"

    return {
        "trend": overall,
        "periods": years,
        "category_trends": sorted(category_trends, key=lambda t: abs(t["delta"]), reverse=True),
        "escalating_count": escalating,
        "diminishing_count": diminishing,
    }


# ============================================================
# 5. Prompt Builder for Multi-Period
# ============================================================

def build_multi_period_prompt(
    question: str,
    packed_contexts: Dict[int, str],
    ticker: str,
    periods: List[int],
    metric: Optional[str] = None,
    example_citation: str = "EVIDENCE_ID",
) -> Tuple[str, str]:
    """
    Build a prompt specifically for multi-period comparative analysis.

    packed_contexts: {fiscal_year: packed_context_string}
    """
    period_labels = ", ".join(f"FY{y}" for y in sorted(periods))

    # Combine contexts with clear period markers
    combined_context_parts = []
    for fy in sorted(periods):
        ctx = packed_contexts.get(fy, "")
        if ctx.strip():
            combined_context_parts.append(f"=== PERIOD: FY{fy} ===\n{ctx}")

    combined_context = "\n\n".join(combined_context_parts)

    metric_instruction = ""
    if metric:
        metric_instruction = f"\nPRIMARY METRIC: {metric}\nExtract this metric for each period and compute year-over-year changes.\n"

    system_prompt = f"""You are an evidence-grounded financial analyst performing multi-period comparative analysis.
You are analyzing {ticker} across periods: {period_labels}.

CRITICAL RULES:
- Use ONLY the supplied evidence context. Do not invent data.
- Every factual claim must cite an evidence ID from the context.
- When comparing periods, cite evidence from BOTH periods being compared.
- Clearly distinguish between extracted facts and analytical inferences.
- Return ONLY valid JSON.
{metric_instruction}
ANALYSIS APPROACH:
1. Extract key metrics for each period from the evidence.
2. Compute absolute and percentage changes between consecutive periods.
3. Identify the overall trend direction (improving/stable/deteriorating/volatile).
4. Highlight any inflection points or structural changes.
5. Note any data gaps where a metric couldn't be extracted for a period.

OUTPUT SCHEMA:
{{
  "final_answer": "2-3 sentence summary of the multi-period trend for {ticker}",
  "comparison": {{
    "targets": [{', '.join(f'{{"ticker": "{ticker}", "fiscal_year": {fy}}}' for fy in sorted(periods))}],
    "metrics": [
      {{
        "metric": "string",
        "values": [
          {{"fiscal_year": {sorted(periods)[0]}, "value": "number", "unit": "string", "citation": "{example_citation}"}},
        ],
        "trend": "improving|stable|deteriorating|volatile",
        "cagr": "number|null",
        "summary": "string"
      }}
    ],
    "key_changes": [
      {{
        "period": "FY{sorted(periods)[0]} → FY{sorted(periods)[-1]}",
        "description": "string",
        "citations": ["{example_citation}"]
      }}
    ],
    "summary": "string"
  }},
  "claims": [],
  "tables_used": [],
  "provenance": {{"ticker": "{ticker}", "fiscal_year": null}},
  "inferences": [],
  "confidence": 0.0
}}"""

    user_prompt = f"""QUESTION:
{question}

ENTITY: {ticker}
PERIODS: {period_labels}

EVIDENCE CONTEXT (organized by period):
{combined_context}

Analyze the multi-period trend. Return ONLY valid JSON."""

    return system_prompt, user_prompt
