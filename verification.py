# verification.py
from __future__ import annotations

import json
import logging
import re
from difflib import get_close_matches
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Literal

from financial_metrics import FinancialMetricsEngine

logger = logging.getLogger(__name__)

# ============================================================
# 0) Types
# ============================================================

Mode = Literal[
    "lookup_numeric",
    "lookup_text",
    "lookup_text_filing",
    "lookup_text_management",
    "lookup_text_news",
    "compute_metric",
    "comparative_analysis",
    "risk_analysis",
    "valuation",
    "relative_valuation",
    "explanatory_reasoning",
    "mba_framework",
]

VerificationStatus = Literal[
    "answer",
    "answer_with_warning",
    "clarify",
    "abstain",
    "error",
]

SourceType = Literal["filing", "news", "transcript", "market_data"]


@dataclass
class Target:
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    metric: Optional[str] = None
    item_hint: Optional[str] = None


@dataclass
class SourceRoutePlan:
    filing: bool = True
    news: bool = False
    transcript: bool = False
    market_data: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class RetrievalPlan:
    hard_filters: Dict[str, Any] = field(default_factory=dict)
    soft_boosts: List[Dict[str, Any]] = field(default_factory=list)
    rewrites: List[str] = field(default_factory=list)
    source_route: SourceRoutePlan = field(default_factory=SourceRoutePlan)


@dataclass
class TaskPlan:
    raw_question: str
    normalized_question: str
    mode: Mode
    targets: List[Target]
    retrieval_plan: RetrievalPlan
    schema_id: str


@dataclass
class EvidenceRequirements:
    required_slots: List[str] = field(default_factory=list)
    required_core_sources: List[SourceType] = field(default_factory=list)  # backward-compatible
    required_all_sources: List[SourceType] = field(default_factory=list)
    required_any_sources: List[SourceType] = field(default_factory=list)

    min_total_blocks: int = 1
    min_chunks: int = 0
    min_tables: int = 0
    min_item1a_like: int = 0
    min_sources: int = 1

    require_item8_or_xbrl_for_numeric: bool = False
    require_market_inputs: bool = False
    require_computed_inputs: bool = False
    multi_entity: bool = False
    preferred_sections: List[str] = field(default_factory=list)

    numeric_contradiction_rel_tol: float = 0.05
    numeric_contradiction_abs_tol: float = 0.0

    evidence_strictness: int = 55


@dataclass
class VerificationResult:
    status: VerificationStatus
    confidence: float
    mode: Mode
    reason_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    best_evidence: List[Dict[str, Any]] = field(default_factory=list)
    source_coverage: Dict[str, bool] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("answer", "answer_with_warning")

    @property
    def action(self) -> str:
        return self.status


@dataclass
class EvidenceBlock:
    kind: Literal["table", "chunk", "xbrl", "news", "transcript"]
    evid: str
    source_type: SourceType
    ticker: Optional[str]
    fiscal_year: Optional[int]
    item: Optional[str]
    text: str
    period_label: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NumericCandidate:
    value_raw: float
    unit: str
    scale_factor: float
    scale_label: str
    value_scaled: float
    evidence_id: str
    kind: str
    source_type: str
    item: Optional[str]
    ticker: Optional[str]
    fiscal_year: Optional[int]
    line: str
    score: float
    precedence: int


@dataclass
class ComputedMetric:
    name: str
    value: float
    unit: str
    formula: str
    inputs: Dict[str, Any]


# ============================================================
# 1) Lexicons / planning helpers
# ============================================================

def normalize_query(q: str) -> str:
    q = (q or "").strip().replace("\u00a0", " ")
    q = re.sub(r"[ \t]+", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


_TICKER_RE = re.compile(r"\b[A-Z]{1,6}\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_FY_PATTERNS = (
    re.compile(r"\bfy\s*[-]?\s*(19\d{2}|20\d{2})\b", re.I),
    re.compile(r"\bfiscal\s+(?:year\s+)?(19\d{2}|20\d{2})\b", re.I),
    re.compile(r"\bannual\s+(19\d{2}|20\d{2})\b", re.I),
    re.compile(r"\b(19\d{2}|20\d{2})\s+10-k\b", re.I),
)
_COMPARE_TERMS = re.compile(r"\b(vs|versus|compare|compared|relative to|against)\b", re.I)

ITEM_MAP = {
    "risk factors": "Item 1A",
    "item 1a": "Item 1A",
    "md&a": "Item 7",
    "management discussion": "Item 7",
    "item 7": "Item 7",
    "financial statements": "Item 8",
    "item 8": "Item 8",
}

METRIC_SYNS: Dict[str, List[str]] = {
    "revenue": ["revenue", "net sales", "total net sales", "total revenue", "sales"],
    "net_income": ["net income", "net earnings", "net income (loss)", "earnings"],
    "operating_income": ["operating income", "income from operations", "operating profit"],
    "gross_profit": [
        "gross profit",
        "gross margin",            # Apple 10-K label: "Gross margin"
        "gross profit (loss)",
    ],
    "capex": [
        "capex",
        "capital expenditures",
        "capital expenditure",
        "purchase of property and equipment",
        "purchases of property and equipment",
        "purchase of property, plant and equipment",
        "purchases of property, plant and equipment",
        "payments to acquire property, plant and equipment",
        "payments for acquisition of property, plant and equipment",
        "additions to property, plant and equipment",
    ],
    "eps": ["earnings per share", "eps", "diluted eps", "basic eps"],
    "cfo": [
        "cash provided by operating activities",
        "operating cash flow",
        "net cash from operating activities",
        "net cash generated by operating activities",   # Apple 10-K label
        "net cash provided by operating activities",
        "cash flows from operations",
        "cash flow from operating activities",
    ],
    "ebitda": ["ebitda", "earnings before interest taxes depreciation and amortization"],
    "current_assets": ["current assets", "total current assets"],
    "current_liabilities": ["current liabilities", "total current liabilities"],
    "total_debt": ["total debt", "long-term debt", "long term debt", "debt"],
    "total_equity": ["total equity", "stockholders' equity", "shareholders' equity", "equity"],
    "total_assets": ["total assets"],
    "fcf": ["free cash flow", "fcf"],
    "gross_margin": ["gross margin", "gross profit margin"],
    "operating_margin": ["operating margin"],
    "net_margin": ["net margin", "net income margin"],
    "ebitda_margin": ["ebitda margin"],
    "current_ratio": ["current ratio"],
    "debt_to_equity": ["debt to equity", "debt/equity", "leverage ratio"],
    "roe": ["return on equity", "roe"],
    "revenue_yoy": ["revenue yoy", "revenue growth", "year over year revenue growth", "yoy revenue"],
}

MONETARY_METRICS = {
    "revenue", "net_income", "operating_income", "gross_profit",
    "capex", "cfo", "ebitda", "fcf",
    "current_assets", "current_liabilities",
    "total_debt", "total_equity", "total_assets"
}

RATIO_METRICS = {
    "gross_margin", "operating_margin", "net_margin",
    "ebitda_margin", "current_ratio", "debt_to_equity", "roe"
}

COMPUTE_TRIGGERS = [
    "yoy", "year over year", "growth", "cagr", "margin", "ratio",
    "difference", "delta", "free cash flow", "fcf",
]

VALUATION_TRIGGERS = [
    "valuation", "intrinsic value", "discounted cash flow", "dcf",
    "ev/ebitda", "p/e", "price to earnings", "trading multiple",
    "valuation multiple", "earnings multiple", "ebitda multiple",
    "relative valuation", "wacc", "beta", "market cap", "share price",
]

RISK_TRIGGERS = ["risk", "risks", "risk factors", "item 1a", "uncertainty", "exposure"]
FRAMEWORK_TRIGGERS = ["framework", "mba", "swot", "porter", "5 forces", "pestel", "4ps"]
EXPLAIN_TRIGGERS = ["explain", "why", "rationale", "driver", "drivers", "reason"]
LATEST_TRIGGERS = ["latest", "recent", "today", "this week", "news", "headline", "headlines"]
TRANSCRIPT_TRIGGERS = ["said", "mentioned", "guidance", "call", "management", "transcript", "earnings call"]

_COMMON_WORDS = frozenset({
    "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "DO", "FOR",
    "FROM", "HAD", "HAS", "HAVE", "HE", "HER", "HIS", "HOW", "I", "IF",
    "IN", "IS", "IT", "ITS", "MY", "NO", "NOT", "OF", "ON", "OR", "OUR",
    "OUT", "OWN", "SAY", "SHE", "SO", "THAN", "THAT", "THE", "THEIR",
    "THEM", "THEN", "THERE", "THESE", "THEY", "THIS", "TO", "UP", "US",
    "WAS", "WE", "WERE", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY",
    "WILL", "WITH", "YOU", "YOUR", "SEC", "FY", "USD", "VS", "NA", "ALL",
    "ANY", "CAN", "DID", "GET", "GOT", "HIM", "LET", "MAY", "NEW", "NOW",
    "OLD", "ONE", "PUT", "RAN", "SET", "TOP", "TWO", "USE", "WAY",
})

_COMPANY_ALIAS_BY_TICKER: Dict[str, Tuple[str, ...]] = {
    "AAPL": ("apple", "apple inc"),
    "MSFT": ("microsoft", "microsoft corp", "microsoft corporation"),
    "NVDA": ("nvidia", "nvidia corp", "nvidia corporation"),
    "TSLA": ("tesla", "tesla inc"),
    "GOOGL": ("google", "alphabet", "alphabet inc", "google llc"),
    "META": ("meta", "meta platforms", "facebook", "facebook inc"),
    "AMZN": ("amazon", "amazon.com", "amazon inc"),
}
_FUZZY_ALIAS_MIN_RATIO = 0.88


def _detect_tickers_from_company_names(q: str, known_tickers: Optional[Set[str]]) -> List[str]:
    ql = (q or "").lower()
    if not ql:
        return []

    out: List[str] = []
    seen: Set[str] = set()

    for ticker, aliases in _COMPANY_ALIAS_BY_TICKER.items():
        if known_tickers and ticker not in known_tickers:
            continue
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", ql):
                if ticker not in seen:
                    out.append(ticker)
                    seen.add(ticker)
                break

    # Fuzzy fallback for light typos like "Nvdia" / "Tesal"
    alias_to_ticker: Dict[str, str] = {}
    for ticker, aliases in _COMPANY_ALIAS_BY_TICKER.items():
        if known_tickers and ticker not in known_tickers:
            continue
        for alias in aliases:
            if " " not in alias:
                alias_to_ticker[alias] = ticker

    for w in re.findall(r"[a-z]{3,}", ql):
        close = get_close_matches(w, alias_to_ticker.keys(), n=1, cutoff=_FUZZY_ALIAS_MIN_RATIO)
        if not close:
            continue
        ticker = alias_to_ticker[close[0]]
        if ticker not in seen:
            out.append(ticker)
            seen.add(ticker)

    return out


def _detect_tickers(q: str, known_tickers: Optional[Set[str]]) -> List[str]:
    cands = [t for t in _TICKER_RE.findall(q.upper()) if not t.isdigit()]
    cands.extend(_detect_tickers_from_company_names(q, known_tickers))
    if known_tickers:
        cands = [t for t in cands if t in known_tickers]
    else:
        cands = [t for t in cands if t not in _COMMON_WORDS]

    out: List[str] = []
    seen: Set[str] = set()
    for t in cands:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _detect_years(q: str) -> List[int]:
    ys: List[int] = []

    for rx in _FY_PATTERNS:
        for m in rx.finditer(q):
            y = int(m.group(1))
            if 1990 <= y <= 2100:
                ys.append(y)

    for m in _YEAR_RE.finditer(q):
        y = int(m.group(1))
        if 1990 <= y <= 2100:
            ys.append(y)

    out: List[int] = []
    seen: Set[int] = set()
    for y in ys:
        if y not in seen:
            out.append(y)
            seen.add(y)
    return out


def _detect_item_hint(q: str) -> Optional[str]:
    ql = q.lower()
    for k, v in ITEM_MAP.items():
        if k in ql:
            return v
    return None


def _detect_metrics(q: str) -> List[str]:
    ql = q.lower()
    hits: List[str] = []
    for canon, syns in METRIC_SYNS.items():
        if any(s in ql for s in syns):
            hits.append(canon)

    out: List[str] = []
    seen: Set[str] = set()
    for m in hits:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out


def choose_mode_mvp(
    question: str,
    *,
    tickers: List[str],
    years: List[int],
    metrics: List[str],
    forced_mode: Optional[str] = None,
    ui_intent: Optional[str] = None,
) -> Mode:
    if forced_mode and forced_mode != "auto":
        return forced_mode  # type: ignore[return-value]

    ql = (question or "").lower()

    if ui_intent:
        ui = ui_intent.lower()
        if ui in ("numeric", "facts"):
            return "lookup_numeric"
        if ui in ("text", "narrative"):
            return "lookup_text_filing"

    if bool(_COMPARE_TERMS.search(question)) or len(tickers) >= 2 or len(years) >= 2:
        return "comparative_analysis"

    if any(t in ql for t in VALUATION_TRIGGERS):
        if any(x in ql for x in ["dcf", "discounted cash flow", "wacc", "terminal", "intrinsic value"]):
            return "valuation"
        return "relative_valuation"

    if any(t in ql for t in RISK_TRIGGERS):
        return "risk_analysis"
    if any(t in ql for t in FRAMEWORK_TRIGGERS):
        return "mba_framework"
    if any(t in ql for t in EXPLAIN_TRIGGERS):
        return "explanatory_reasoning"
    if any(t in ql for t in LATEST_TRIGGERS):
        return "lookup_text_news"
    if any(t in ql for t in TRANSCRIPT_TRIGGERS):
        return "lookup_text_management"

    if any(t in ql for t in COMPUTE_TRIGGERS):
        return "compute_metric"

    if metrics or re.search(r"[\$%]", ql) or any(w in ql for w in ["how much", "amount", "total", "percent"]):
        return "lookup_numeric"

    return "lookup_text_filing"


def _resolve_primary_metric(question: str, metrics: List[str], mode: Mode) -> Optional[str]:
    if not metrics:
        return None
    if mode != "compute_metric":
        return metrics[0]

    ql = (question or "").lower()
    if ("yoy" in ql or "year over year" in ql) and "revenue" in ql:
        return "revenue_yoy"
    if "gross margin" in ql:
        return "gross_margin"
    if "operating margin" in ql:
        return "operating_margin"
    if "net margin" in ql:
        return "net_margin"
    if "ebitda margin" in ql:
        return "ebitda_margin"
    if "debt to equity" in ql or "debt/equity" in ql:
        return "debt_to_equity"
    if "return on equity" in ql or re.search(r"\broe\b", ql):
        return "roe"
    if "current ratio" in ql:
        return "current_ratio"
    if "free cash flow" in ql or re.search(r"\bfcf\b", ql):
        return "fcf"
    return metrics[0]


def _build_source_route(question: str, mode: Mode) -> SourceRoutePlan:
    ql = (question or "").lower()
    route = SourceRoutePlan(filing=True)

    if mode == "lookup_numeric":
        route.reasons.append("audited_facts_prefer_filing")
    elif mode == "compute_metric":
        route.reasons.append("derived_metric_requires_filing_inputs")
    elif mode == "risk_analysis":
        route.transcript = True
        route.reasons.extend(["risk_prefers_item1a", "transcript_as_supplement"])
    elif mode == "relative_valuation":
        route.market_data = True
        route.reasons.extend(["relative_valuation_requires_market_data", "filing_denominator_required"])
    elif mode == "valuation":
        route.market_data = True
        route.transcript = True
        route.reasons.extend(["valuation_uses_filing_inputs", "market_data_optional_or_enhancing", "transcript_as_context"])
    elif mode in ("explanatory_reasoning", "lookup_text", "lookup_text_filing"):
        route.reasons.append("filing_narrative_lookup")
    elif mode == "lookup_text_management":
        route.filing = False
        route.transcript = True
        route.reasons.append("management_commentary_requires_transcript")
    elif mode == "lookup_text_news":
        route.filing = False
        route.news = True
        route.reasons.append("latest_query_requires_news")
    elif mode == "mba_framework":
        route.news = True
        route.transcript = True
        route.reasons.extend(["framework_can_use_multi_source_evidence"])

    return route


def build_task_plan(
    raw_question: str,
    known_tickers: Optional[Set[str]] = None,
    *,
    forced_mode: Optional[str] = None,
    ui_intent: Optional[str] = None,
    forced_ticker: Optional[str] = None,
    forced_fiscal_year: Optional[int] = None,
) -> TaskPlan:
    if not isinstance(raw_question, str) or not raw_question.strip():
        raise ValueError("raw_question must be a non-empty string")

    q = normalize_query(raw_question)
    tickers = _detect_tickers(q, known_tickers)
    years = _detect_years(q)
    item_hint = _detect_item_hint(q)
    metrics = _detect_metrics(q)

    if forced_ticker:
        ft = forced_ticker.strip().upper()
        if known_tickers is None or ft in known_tickers:
            tickers = [ft] + [t for t in tickers if t != ft]

    if forced_fiscal_year is not None:
        fy = int(forced_fiscal_year)
        years = [fy] + [y for y in years if y != fy]

    mode = choose_mode_mvp(
        q,
        tickers=tickers,
        years=years,
        metrics=metrics,
        forced_mode=forced_mode,
        ui_intent=ui_intent,
    )

    primary_metric = _resolve_primary_metric(q, metrics, mode)

    if mode == "comparative_analysis":
        targets: List[Target] = []
        if tickers and years:
            for t in tickers:
                targets.append(Target(ticker=t, fiscal_year=years[0], metric=primary_metric, item_hint=item_hint))
        elif tickers:
            for t in tickers:
                targets.append(Target(ticker=t, fiscal_year=years[0] if years else None, metric=primary_metric, item_hint=item_hint))
        else:
            targets = [Target(metric=primary_metric, item_hint=item_hint)]
    else:
        targets = [Target(
            ticker=tickers[0] if tickers else None,
            fiscal_year=years[0] if years else None,
            metric=primary_metric,
            item_hint=item_hint,
        )]

    hard_filters: Dict[str, Any] = {
        "ticker": targets[0].ticker,
        "fiscal_year": targets[0].fiscal_year,
    }
    if item_hint:
        hard_filters["item"] = item_hint

    soft_boosts: List[Dict[str, Any]] = []
    if mode == "risk_analysis":
        soft_boosts.append({"section": "Item 1A", "weight": 1.0})
    if mode in ("lookup_numeric", "compute_metric", "valuation", "relative_valuation"):
        soft_boosts.append({"section": "Item 8", "weight": 1.0})
    if item_hint:
        soft_boosts.append({"section": item_hint, "weight": 1.0})

    rewrites = [q]
    if primary_metric:
        for s in METRIC_SYNS.get(primary_metric, [])[:3]:
            if s.lower() not in q.lower():
                rewrites.append(f"{q} {s}")
    if item_hint and item_hint.lower() not in q.lower():
        rewrites.append(f"{q} {item_hint}")

    deduped: List[str] = []
    seen: Set[str] = set()
    for r in rewrites:
        rr = normalize_query(r)
        if rr and rr not in seen:
            deduped.append(rr)
            seen.add(rr)

    source_route = _build_source_route(q, mode)

    return TaskPlan(
        raw_question=raw_question,
        normalized_question=q,
        mode=mode,
        targets=targets,
        retrieval_plan=RetrievalPlan(
            hard_filters=hard_filters,
            soft_boosts=soft_boosts,
            rewrites=deduped[:6],
            source_route=source_route,
        ),
        schema_id=f"{mode}_schema_v1",
    )


# ============================================================
# 2) Evidence requirements by mode
# ============================================================

EVIDENCE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "lookup_numeric": {
        "required_slots": ["ticker", "fiscal_year", "metric"],
        "required_any": ["filing"],
        "preferred": ["xbrl_table"],
        "require_item8_or_xbrl_for_numeric": True,
    },
    "compute_metric": {
        "required_slots": ["ticker", "fiscal_year", "metric"],
        "required_any": ["filing"],
        "required_inputs": True,
        "require_item8_or_xbrl_for_numeric": True,
    },
    "comparative_analysis": {
        "required_slots": ["ticker"],
        "required_any": ["filing"],
        "multi_entity": True,
    },
    "risk_analysis": {
        "required_slots": ["ticker", "fiscal_year"],
        "required_any": ["filing"],
        "preferred_sections": ["Item 1A"],
        "min_item1a_like": 1,
        "min_chunks": 1,
        "min_total_blocks": 2,
    },
    "lookup_text_filing": {
        "required_any": ["filing"],
        "min_chunks": 1,
    },
    "lookup_text_management": {
        "required_all": ["transcript"],
    },
    "lookup_text_news": {
        "required_all": ["news"],
    },
    "relative_valuation": {
        "required_slots": ["ticker", "fiscal_year"],
        "required_all": ["market_data", "filing"],
        "require_market_inputs": True,
        "min_tables": 1,
        "min_total_blocks": 2,
    },
    "valuation": {
        "required_slots": ["ticker", "fiscal_year"],
        "required_any": ["filing"],
        "min_tables": 1,
        "min_total_blocks": 2,
    },
    "mba_framework": {
        "required_any": ["filing", "news", "transcript"],
        "min_sources": 2,
    },
    "explanatory_reasoning": {
        "required_any": ["filing"],
        "min_total_blocks": 1,
    },
    # backward-compat
    "lookup_text": {
        "required_any": ["filing"],
        "min_chunks": 1,
    },
}


def evidence_requirements(plan: TaskPlan) -> EvidenceRequirements:
    rule = EVIDENCE_REQUIREMENTS.get(plan.mode, {})
    req = EvidenceRequirements()

    req.required_slots = list(rule.get("required_slots", []))
    req.required_any_sources = list(rule.get("required_any", []))
    req.required_all_sources = list(rule.get("required_all", []))
    req.required_core_sources = list(req.required_all_sources or req.required_any_sources)

    req.min_total_blocks = int(rule.get("min_total_blocks", req.min_total_blocks))
    req.min_chunks = int(rule.get("min_chunks", req.min_chunks))
    req.min_tables = int(rule.get("min_tables", req.min_tables))
    req.min_item1a_like = int(rule.get("min_item1a_like", req.min_item1a_like))
    req.min_sources = int(rule.get("min_sources", req.min_sources))

    req.require_item8_or_xbrl_for_numeric = bool(
        rule.get("require_item8_or_xbrl_for_numeric", req.require_item8_or_xbrl_for_numeric)
    )
    req.require_market_inputs = bool(rule.get("require_market_inputs", req.require_market_inputs))
    req.require_computed_inputs = bool(rule.get("required_inputs", req.require_computed_inputs))
    req.multi_entity = bool(rule.get("multi_entity", req.multi_entity))
    req.preferred_sections = list(rule.get("preferred_sections", []))
    return req


def with_strictness(req: EvidenceRequirements, strictness: int) -> EvidenceRequirements:
    req.evidence_strictness = int(strictness)
    return req


# ============================================================
# 3) Context parsing
# ============================================================

_TABLE_HDR_RE = re.compile(
    r"^\[\s*TABLE\s+(?P<ticker>[A-Z]{1,6})\s+FY\s*(?P<fy>\d{4})\s+(?P<item>Item\s+[0-9A-Z\.]+)\s+(?P<id>[^\]]+?)\s*\]\s*$",
    re.I,
)
_CHUNK_HDR_RE = re.compile(
    r"^\[\s*(?!TABLE\b)(?P<ticker>[A-Z]{1,6})\s+FY\s*(?P<fy>\d{4})\s+(?P<item>Item\s+[0-9A-Z\.]+)\s+(?P<id>[^\]]+?)\s*\]\s*$",
    re.I,
)
_NEWS_HDR_RE = re.compile(
    r"^\[\s*NEWS\s+(?P<ticker>[A-Z]{1,6})\s+(?:FY\s*(?P<fy>\d{4})\s+)?(?P<id>[^\]]+?)\s*\]\s*$",
    re.I,
)
_TRANSCRIPT_HDR_RE = re.compile(
    r"^\[\s*TRANSCRIPT\s+(?P<ticker>[A-Z]{1,6})\s+(?:FY\s*(?P<fy>\d{4})\s+)?(?P<id>[^\]]+?)\s*\]\s*$",
    re.I,
)
_ID_RE = re.compile(r"(?:(?<!\w)[tc]\d+(?!\w)|[A-Z]{1,6}_FY\d{4}_\S+)", re.I)


def _canon_item(item: Optional[str]) -> Optional[str]:
    if not item:
        return None
    s = re.sub(r"\s+", " ", item.strip())
    s = re.sub(r"^item\s+", "Item ", s, flags=re.I)
    return s


def parse_allowed_ids_from_context(packed_context: str) -> Set[str]:
    allowed: Set[str] = set()
    if "[XBRL EVIDENCE]" in (packed_context or ""):
        allowed.add("xbrl")
    for line in (packed_context or "").splitlines():
        line = line.strip()
        if not (line.startswith("[") and line.endswith("]")):
            continue
        for rx in (_TABLE_HDR_RE, _CHUNK_HDR_RE, _NEWS_HDR_RE, _TRANSCRIPT_HDR_RE):
            m = rx.match(line)
            if m:
                allowed.add(m.group("id").strip())
                break
        else:
            allowed.update(x.strip() for x in _ID_RE.findall(line))
    return allowed


def split_context_into_blocks(packed_context: str) -> List[EvidenceBlock]:
    if not packed_context:
        return []

    lines = packed_context.splitlines()
    blocks: List[EvidenceBlock] = []
    i = 0

    def _consume_text(start_idx: int) -> Tuple[str, int]:
        j = start_idx
        while j < len(lines):
            nxt = lines[j].strip()
            if (
                nxt == "[XBRL EVIDENCE]"
                or _TABLE_HDR_RE.match(nxt)
                or _CHUNK_HDR_RE.match(nxt)
                or _NEWS_HDR_RE.match(nxt)
                or _TRANSCRIPT_HDR_RE.match(nxt)
            ):
                break
            j += 1
        return "\n".join(lines[start_idx:j]).strip(), j

    while i < len(lines):
        line = lines[i].strip()

        if line == "[XBRL EVIDENCE]":
            txt, new_i = _consume_text(i + 1)
            if txt:
                blocks.append(EvidenceBlock(
                    kind="xbrl",
                    evid="xbrl",
                    source_type="filing",
                    ticker=None,
                    fiscal_year=None,
                    item=None,
                    text=txt,
                ))
            i = new_i
            continue

        m = _TABLE_HDR_RE.match(line)
        if m:
            txt, new_i = _consume_text(i + 1)
            blocks.append(EvidenceBlock(
                kind="table",
                evid=m.group("id").strip(),
                source_type="filing",
                ticker=m.group("ticker").upper(),
                fiscal_year=int(m.group("fy")),
                item=_canon_item(m.group("item")),
                text=txt,
            ))
            i = new_i
            continue

        m = _CHUNK_HDR_RE.match(line)
        if m:
            txt, new_i = _consume_text(i + 1)
            blocks.append(EvidenceBlock(
                kind="chunk",
                evid=m.group("id").strip(),
                source_type="filing",
                ticker=m.group("ticker").upper(),
                fiscal_year=int(m.group("fy")),
                item=_canon_item(m.group("item")),
                text=txt,
            ))
            i = new_i
            continue

        m = _NEWS_HDR_RE.match(line)
        if m:
            txt, new_i = _consume_text(i + 1)
            fy = int(m.group("fy")) if m.group("fy") else None
            blocks.append(EvidenceBlock(
                kind="news",
                evid=m.group("id").strip(),
                source_type="news",
                ticker=m.group("ticker").upper(),
                fiscal_year=fy,
                item=None,
                text=txt,
            ))
            i = new_i
            continue

        m = _TRANSCRIPT_HDR_RE.match(line)
        if m:
            txt, new_i = _consume_text(i + 1)
            fy = int(m.group("fy")) if m.group("fy") else None
            blocks.append(EvidenceBlock(
                kind="transcript",
                evid=m.group("id").strip(),
                source_type="transcript",
                ticker=m.group("ticker").upper(),
                fiscal_year=fy,
                item=None,
                text=txt,
            ))
            i = new_i
            continue

        i += 1

    return blocks


# ============================================================
# 4) Verification layer
# ============================================================

def _slot_value(target: Target, slot: str) -> Any:
    if slot == "ticker":
        return target.ticker
    if slot == "fiscal_year":
        return target.fiscal_year
    if slot == "metric":
        return target.metric
    return None


def _score_to_band(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _source_precedence(block: EvidenceBlock, mode: Mode) -> int:
    item = (block.item or "").lower()

    if mode in ("lookup_numeric", "compute_metric", "valuation", "relative_valuation"):
        if block.kind == "xbrl":
            return 500
        if block.kind == "table" and item.startswith("item 8"):
            return 450
        if block.kind == "chunk" and item.startswith("item 8"):
            return 350
        if block.kind == "transcript":
            return 200
        if block.kind == "news":
            return 150

    if mode == "risk_analysis":
        if block.kind == "chunk" and item.startswith("item 1a"):
            return 450
        if block.kind == "transcript":
            return 260
        if block.kind == "news":
            return 180
        if block.kind == "chunk":
            return 140

    if mode in (
        "explanatory_reasoning",
        "lookup_text",
        "lookup_text_filing",
        "lookup_text_management",
        "lookup_text_news",
        "mba_framework",
    ):
        if block.kind == "news":
            return 320
        if block.kind == "transcript":
            return 300
        if block.kind == "chunk" and item.startswith("item 7"):
            return 260
        if block.kind == "chunk" and item.startswith("item 1a"):
            return 240
        if block.kind == "table":
            return 200
        if block.kind == "xbrl":
            return 180

    if block.kind == "xbrl":
        return 300
    if block.kind == "table":
        return 250
    if block.kind == "chunk":
        return 200
    if block.kind == "transcript":
        return 150
    if block.kind == "news":
        return 120
    return 0


def _topic_relevance_score(text: str, metric: Optional[str], question: str) -> float:
    ql = (question or "").lower()
    tl = (text or "").lower()
    score = 0.0

    if metric:
        syns = METRIC_SYNS.get(metric, [metric])
        if any(s.lower() in tl for s in syns):
            score += 0.35

    question_terms = [w for w in re.findall(r"[a-zA-Z]{4,}", ql) if w not in {"what", "when", "where", "which", "their", "about"}]
    overlap = sum(1 for w in question_terms if w in tl)
    if question_terms:
        score += min(0.35, 0.08 * overlap)

    if "risk" in ql and ("risk" in tl or "uncertaint" in tl or "exposure" in tl):
        score += 0.20
    if "guidance" in ql and "guidance" in tl:
        score += 0.20
    if "supply chain" in ql and "supply chain" in tl:
        score += 0.25

    return min(1.0, score)


def _entity_year_match(block: EvidenceBlock, target: Target) -> Tuple[bool, bool]:
    ticker_ok = True
    year_ok = True
    if target.ticker and block.ticker:
        ticker_ok = (block.ticker == target.ticker)
    if target.fiscal_year is not None and block.fiscal_year is not None:
        year_ok = (int(block.fiscal_year) == int(target.fiscal_year))
    return ticker_ok, year_ok


def _detect_source_coverage(blocks: List[EvidenceBlock], market_inputs: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    return {
        "filing": any(b.source_type == "filing" for b in blocks),
        "news": any(b.source_type == "news" for b in blocks),
        "transcript": any(b.source_type == "transcript" for b in blocks),
        "market_data": bool(isinstance(market_inputs, dict) and (market_inputs.get("price") is not None or market_inputs.get("market_cap") is not None)),
    }


def _best_blocks_for_mode(plan: TaskPlan, blocks: List[EvidenceBlock], limit: int = 5) -> List[EvidenceBlock]:
    target = plan.targets[0] if plan.targets else Target()
    scored: List[Tuple[float, EvidenceBlock]] = []

    for b in blocks:
        ticker_ok, year_ok = _entity_year_match(b, target)
        rel = _topic_relevance_score(b.text, target.metric, plan.raw_question)
        base = _source_precedence(b, plan.mode) / 500.0
        s = base + rel
        if ticker_ok:
            s += 0.20
        if year_ok:
            s += 0.15
        scored.append((s, b))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored[:limit]]


def verify_evidence(
    plan: TaskPlan,
    req: EvidenceRequirements,
    packed_context: str,
    *,
    market_inputs: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    strictness = max(0, min(100, int(getattr(req, "evidence_strictness", 55))))
    # 55 is the neutral/default point.
    strict_delta = (strictness - 55) / 45.0

    def _scaled_min(base: int, floor: int = 0) -> int:
        if base <= 0:
            return 0
        # strictness 0 -> ~0.5x, strictness 100 -> ~1.5x
        factor = 1.0 + (0.5 * strict_delta)
        return max(floor, int(round(base * factor)))

    eff_min_total_blocks = _scaled_min(req.min_total_blocks, floor=1)
    eff_min_chunks = _scaled_min(req.min_chunks, floor=0)
    eff_min_tables = _scaled_min(req.min_tables, floor=0)
    eff_min_item1a_like = _scaled_min(req.min_item1a_like, floor=0)
    eff_min_sources = _scaled_min(req.min_sources, floor=1)

    # Confidence policy becomes more lenient/strict with evidence_strictness.
    # strictness 0  -> answer >= 0.70, warn >= 0.40
    # strictness 55 -> answer >= 0.80, warn >= 0.50
    # strictness 100-> answer >= 0.90, warn >= 0.60
    answer_threshold = max(0.0, min(0.99, 0.80 + (0.10 * strict_delta)))
    warn_threshold = max(0.0, min(answer_threshold, 0.50 + (0.10 * strict_delta)))

    blocks = split_context_into_blocks(packed_context)
    source_coverage = _detect_source_coverage(blocks, market_inputs)
    route = plan.retrieval_plan.source_route

    signals: Dict[str, Any] = {
        "strictness": strictness,
        "thresholds": {
            "answer": answer_threshold,
            "warning": warn_threshold,
            "min_total_blocks": eff_min_total_blocks,
            "min_chunks": eff_min_chunks,
            "min_tables": eff_min_tables,
            "min_item1a_like": eff_min_item1a_like,
            "min_sources": eff_min_sources,
        },
        "n_blocks": len(blocks),
        "n_tables": sum(1 for b in blocks if b.kind == "table"),
        "n_chunks": sum(1 for b in blocks if b.kind == "chunk"),
        "n_xbrl": sum(1 for b in blocks if b.kind == "xbrl"),
        "n_news": sum(1 for b in blocks if b.kind == "news"),
        "n_transcript": sum(1 for b in blocks if b.kind == "transcript"),
        "n_item1a_like": sum(
            1 for b in blocks
            if (b.kind == "chunk" and (b.item or "").lower().startswith("item 1a")) or (b.kind == "transcript" and "risk" in b.text.lower())
        ),
    }
    planned_required_missing: List[str] = []
    for src, required in (
        ("filing", bool(route.filing)),
        ("news", bool(route.news)),
        ("transcript", bool(route.transcript)),
        ("market_data", bool(route.market_data)),
    ):
        if required and not source_coverage.get(src, False):
            planned_required_missing.append(src)
    if planned_required_missing:
        signals["planned_required_missing"] = planned_required_missing

    target = plan.targets[0] if plan.targets else Target()

    # 1) Slot sufficiency
    missing_slots = [s for s in req.required_slots if not _slot_value(target, s)]
    if missing_slots:
        return VerificationResult(
            status="clarify",
            confidence=0.10,
            mode=plan.mode,
            reason_codes=[f"missing_slot:{s}" for s in missing_slots],
            errors=[f"missing_slot:{s}" for s in missing_slots],
            source_coverage=source_coverage,
            signals=signals,
        )

    # 2) Source sufficiency
    missing_required_all = [s for s in req.required_all_sources if not source_coverage.get(s, False)]
    if missing_required_all:
        reason_codes = [f"required_source_missing:{s}" for s in missing_required_all]
        return VerificationResult(
            status="abstain",
            confidence=0.15,
            mode=plan.mode,
            reason_codes=reason_codes,
            errors=reason_codes[:],
            source_coverage=source_coverage,
            signals=signals,
        )

    if req.required_any_sources:
        if not any(source_coverage.get(s, False) for s in req.required_any_sources):
            reason_codes = [f"required_any_source_missing:{s}" for s in req.required_any_sources]
            return VerificationResult(
                status="abstain",
                confidence=0.15,
                mode=plan.mode,
                reason_codes=reason_codes,
                errors=reason_codes[:],
                source_coverage=source_coverage,
                signals=signals,
            )

    if req.multi_entity:
        entities = {t.ticker for t in plan.targets if t.ticker}
        if len(entities) < 2:
            return VerificationResult(
                status="clarify",
                confidence=0.10,
                mode=plan.mode,
                reason_codes=["missing_slot:comparative_entities"],
                errors=["missing_slot:comparative_entities"],
                source_coverage=source_coverage,
                signals=signals,
            )

    if signals["n_blocks"] < eff_min_total_blocks:
        return VerificationResult(
            status="abstain",
            confidence=0.20,
            mode=plan.mode,
            reason_codes=["insufficient_blocks"],
            errors=["insufficient_blocks"],
            source_coverage=source_coverage,
            signals=signals,
        )

    n_sources_present = sum(1 for k in ("filing", "news", "transcript", "market_data") if source_coverage.get(k, False))
    if n_sources_present < eff_min_sources:
        return VerificationResult(
            status="abstain",
            confidence=0.20,
            mode=plan.mode,
            reason_codes=["insufficient_source_diversity"],
            errors=["insufficient_source_diversity"],
            source_coverage=source_coverage,
            signals=signals,
        )

    if signals["n_chunks"] < eff_min_chunks:
        return VerificationResult(
            status="answer_with_warning",
            confidence=0.55,
            mode=plan.mode,
            reason_codes=["weak_chunk_coverage"],
            warnings=["weak_chunk_coverage"],
            source_coverage=source_coverage,
            signals=signals,
        )

    if signals["n_tables"] < eff_min_tables:
        return VerificationResult(
            status="answer_with_warning",
            confidence=0.58,
            mode=plan.mode,
            reason_codes=["weak_table_coverage"],
            warnings=["weak_table_coverage"],
            source_coverage=source_coverage,
            signals=signals,
        )

    if req.require_item8_or_xbrl_for_numeric:
        has_strong_numeric = (signals["n_xbrl"] > 0) or any(
            b.kind == "table" and (b.item or "").lower().startswith("item 8")
            for b in blocks
        )
        if not has_strong_numeric:
            return VerificationResult(
                status="answer_with_warning",
                confidence=0.60,
                mode=plan.mode,
                reason_codes=["numeric_without_item8_or_xbrl"],
                warnings=["numeric_without_item8_or_xbrl"],
                source_coverage=source_coverage,
                signals=signals,
            )

    if eff_min_item1a_like and signals["n_item1a_like"] < eff_min_item1a_like:
        return VerificationResult(
            status="answer_with_warning",
            confidence=0.57,
            mode=plan.mode,
            reason_codes=["weak_risk_evidence"],
            warnings=["weak_risk_evidence"],
            source_coverage=source_coverage,
            signals=signals,
        )

    if req.require_market_inputs and not source_coverage["market_data"]:
        return VerificationResult(
            status="abstain",
            confidence=0.20,
            mode=plan.mode,
            reason_codes=["required_source_missing:market_data"],
            errors=["required_source_missing:market_data"],
            source_coverage=source_coverage,
            signals=signals,
        )

    if req.require_computed_inputs and plan.mode == "compute_metric":
        metric_name = target.metric or ""
        needed_inputs = required_inputs_for_computed(metric_name)
        if needed_inputs:
            for inp in needed_inputs:
                inp_metric = "revenue" if inp in ("revenue_t", "revenue_t1") else inp
                fy = target.fiscal_year
                if inp == "revenue_t1" and isinstance(target.fiscal_year, int):
                    fy = int(target.fiscal_year) - 1
                test_target = Target(
                    ticker=target.ticker,
                    fiscal_year=fy,
                    metric=inp_metric,
                    item_hint="Item 8",
                )
                best, _ = choose_best_numeric_with_gate(packed_context, test_target, req, topn=4)
                if best is None:
                    rc = f"required_input_missing:{inp}"
                    return VerificationResult(
                        status="abstain",
                        confidence=0.22,
                        mode=plan.mode,
                        reason_codes=[rc],
                        errors=[rc],
                        source_coverage=source_coverage,
                        signals=signals,
                    )

    # 3) Entity and period consistency
    entity_good = 0
    year_good = 0
    for b in blocks:
        ticker_ok, year_ok = _entity_year_match(b, target)
        if ticker_ok:
            entity_good += 1
        if year_ok:
            year_good += 1

    if blocks:
        entity_ratio = entity_good / len(blocks)
        year_ratio = year_good / len(blocks)
    else:
        entity_ratio = 0.0
        year_ratio = 0.0

    # 4) Quality / confidence
    confidence = 0.0
    reason_codes: List[str] = []
    warnings: List[str] = []
    if planned_required_missing:
        reason_codes.extend([f"required_source_missing:{s}" for s in planned_required_missing])

    confidence += 0.20
    reason_codes.append("source_sufficiency_met")

    if entity_ratio >= 0.80:
        confidence += 0.22
        reason_codes.append("ticker_matched")
    elif entity_ratio >= 0.50:
        confidence += 0.10
        warnings.append("partial_ticker_match")
    else:
        warnings.append("weak_ticker_match")

    if target.fiscal_year is None:
        reason_codes.append("year_not_required_or_unspecified")
        confidence += 0.05
    else:
        if year_ratio >= 0.80:
            confidence += 0.18
            reason_codes.append("fiscal_year_matched")
        elif year_ratio >= 0.50:
            confidence += 0.08
            warnings.append("partial_year_match")
        else:
            warnings.append("weak_year_match")

    if plan.mode in ("lookup_numeric", "compute_metric", "valuation", "relative_valuation"):
        if signals["n_xbrl"] > 0:
            confidence += 0.22
            reason_codes.append("xbrl_present")
        elif any(b.kind == "table" and (b.item or "").lower().startswith("item 8") for b in blocks):
            confidence += 0.18
            reason_codes.append("audited_item8_table_present")
        elif any(b.kind == "chunk" and (b.item or "").lower().startswith("item 8") for b in blocks):
            confidence += 0.10
            warnings.append("item8_text_without_table")
    elif plan.mode == "risk_analysis":
        if any(b.kind == "chunk" and (b.item or "").lower().startswith("item 1a") for b in blocks):
            confidence += 0.18
            reason_codes.append("item1a_present")
        elif any(b.kind == "transcript" for b in blocks):
            confidence += 0.10
            warnings.append("risk_from_transcript_only")
    else:
        if any(b.kind == "news" for b in blocks):
            confidence += 0.08
            reason_codes.append("news_present")
        if any(b.kind == "transcript" for b in blocks):
            confidence += 0.08
            reason_codes.append("transcript_present")
        if any(b.source_type == "filing" for b in blocks):
            confidence += 0.08
            reason_codes.append("filing_present")

    # 5) Best evidence
    best_blocks = _best_blocks_for_mode(plan, blocks, limit=5)
    best_evidence = [
        {
            "evidence_id": b.evid,
            "kind": b.kind,
            "source_type": b.source_type,
            "ticker": b.ticker,
            "fiscal_year": b.fiscal_year,
            "item": b.item,
            "preview": (b.text[:180] + "...") if len(b.text) > 180 else b.text,
            "precedence": _source_precedence(b, plan.mode),
        }
        for b in best_blocks
    ]

    confidence = min(0.99, max(0.05, confidence))

    # 6) Policy
    if confidence >= answer_threshold:
        status: VerificationStatus = "answer"
    elif confidence >= warn_threshold:
        status = "answer_with_warning"
    else:
        status = "abstain"

    return VerificationResult(
        status=status,
        confidence=confidence,
        mode=plan.mode,
        reason_codes=reason_codes,
        warnings=warnings,
        errors=[],
        best_evidence=best_evidence,
        source_coverage=source_coverage,
        signals={**signals, "confidence_band": _score_to_band(confidence)},
    )


# Backward-compatible wrapper
def gate_evidence(
    plan: TaskPlan,
    req: EvidenceRequirements,
    packed_context: str,
    *,
    market_inputs: Optional[Dict[str, Any]] = None,
) -> VerificationResult:
    try:
        return verify_evidence(plan, req, packed_context, market_inputs=market_inputs)
    except Exception as e:
        logger.exception("gate_evidence failed")
        return VerificationResult(
            status="error",
            confidence=0.0,
            mode=plan.mode,
            reason_codes=[f"verification_exception:{type(e).__name__}"],
            errors=[str(e)],
            source_coverage={},
            signals={},
        )


# ============================================================
# 5) Numeric extraction
# ============================================================

_NUM_RE = re.compile(
    r"(?P<prefix>\$)?\s*(?P<num>\(?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\)?)\s*(?P<suffix>%|bps|million|billion|thousand|bn|(?:m|k)(?=[^a-zA-Z]))?",
    re.I,
)
_SCALE = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9,
}


def _to_float(num_str: str) -> float:
    s = (num_str or "").strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()
    val = float(s.replace(",", ""))
    return -val if neg else val


def _global_scale_hint(text: str) -> float:
    t = (text or "").lower()
    if "in billions" in t or "($ in billions" in t:
        return 1e9
    if "in millions" in t or "($ in millions" in t:
        return 1e6
    if "in thousands" in t or "($ in thousands" in t:
        return 1e3
    return 1.0


def _looks_like_year_token(val: float) -> bool:
    iv = int(val)
    return abs(val - iv) < 1e-9 and 1900 <= iv <= 2100


def _metric_line_match_score(line: str, metric: str) -> float:
    ll = (line or "").lower()
    syns = [s.lower() for s in METRIC_SYNS.get(metric, [metric])]
    hits = sum(1 for s in syns if s in ll)
    if hits == 0:
        return 0.0
    return min(1.0, 0.60 + 0.15 * hits)


def _metric_supported_in_line(line: str, metric: str) -> bool:
    ll = (line or "").lower()
    syns = [s.lower() for s in METRIC_SYNS.get(metric, [metric])]
    return any(s in ll for s in syns)


def _infer_unit(prefix: Optional[str], suffix: str, metric: str, val: float, scale_hint: float, kind: str) -> str:
    suffix = (suffix or "").lower()
    if suffix == "%":
        return "PERCENT"
    if suffix == "bps":
        return "BPS"
    if prefix == "$":
        return "USD"
    if metric in MONETARY_METRICS:
        return "USD"
    if metric == "eps":
        return "RATIO"
    if metric in RATIO_METRICS:
        if 0 <= val <= 10:
            return "RATIO"
        if 0 <= val <= 100:
            return "PERCENT"
    if kind in ("table", "xbrl") and scale_hint in (1e3, 1e6, 1e9) and metric not in RATIO_METRICS and metric != "eps":
        return "USD"
    return "UNKNOWN"


def extract_numeric_topn(
    packed_context: str,
    target: Target,
    *,
    topn: int = 5,
    prefer_strong_filing: bool = True,
) -> List[NumericCandidate]:
    blocks = split_context_into_blocks(packed_context)
    metric = target.metric
    fy = target.fiscal_year
    ticker = target.ticker

    if not metric:
        return []

    cands: List[NumericCandidate] = []

    for b in blocks:
        if b.source_type != "filing":
            continue
        if prefer_strong_filing and b.kind not in ("xbrl", "table", "chunk"):
            continue
        if ticker and b.ticker and b.ticker != ticker:
            continue
        if fy is not None and b.fiscal_year is not None and int(b.fiscal_year) != int(fy):
            continue

        scale_hint = _global_scale_hint(b.text)
        prec = _source_precedence(b, "lookup_numeric")

        lines = (b.text or "").splitlines()
        for i, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue

            mscore = _metric_line_match_score(line, metric)
            if mscore <= 0:
                continue

            number_lines = [line]
            if b.kind in ("table", "xbrl"):
                for j in (i + 1, i + 2):
                    if j < len(lines):
                        nxt = lines[j].strip()
                        if nxt:
                            number_lines.append(nxt)

            year_bonus = 0.2 if fy is not None and str(fy) in line else 0.0
            item8_bonus = 0.15 if (b.item or "").lower().startswith("item 8") or b.kind == "xbrl" else 0.0
            table_bonus = 0.12 if b.kind in ("table", "xbrl") else 0.0

            for cand_line in number_lines:
                for m in _NUM_RE.finditer(cand_line):
                    val = _to_float(m.group("num"))
                    if _looks_like_year_token(val):
                        continue

                    suffix = (m.group("suffix") or "").lower()
                    prefix = m.group("prefix")
                    unit = _infer_unit(prefix, suffix, metric, val, scale_hint, b.kind)

                    if suffix in _SCALE:
                        scale_factor = _SCALE[suffix]
                        scale_label = suffix
                    else:
                        scale_factor = scale_hint
                        scale_label = (
                            "billion" if scale_hint == 1e9 else
                            "million" if scale_hint == 1e6 else
                            "thousand" if scale_hint == 1e3 else
                            "raw"
                        )

                    value_scaled = float(val) * float(scale_factor)
                    score = 0.55 * mscore + year_bonus + item8_bonus + table_bonus
                    if cand_line != line and b.kind in ("table", "xbrl"):
                        score += 0.08

                    cands.append(NumericCandidate(
                        value_raw=float(val),
                        unit=unit,
                        scale_factor=float(scale_factor),
                        scale_label=scale_label,
                        value_scaled=float(value_scaled),
                        evidence_id=b.evid,
                        kind=b.kind,
                        source_type=b.source_type,
                        item=b.item,
                        ticker=b.ticker,
                        fiscal_year=b.fiscal_year,
                        line=f"{line} || {cand_line}" if cand_line != line else line,
                        score=float(score),
                        precedence=prec,
                    ))

    cands.sort(key=lambda c: (c.score, c.precedence, abs(c.value_scaled)), reverse=True)
    return cands[:topn]


def verify_numeric_candidate(c: NumericCandidate, target: Target) -> Tuple[bool, List[str]]:
    errs: List[str] = []

    if target.ticker and c.ticker and c.ticker != target.ticker:
        errs.append("ticker_mismatch")
    if target.fiscal_year is not None and c.fiscal_year is not None and int(c.fiscal_year) != int(target.fiscal_year):
        errs.append("fiscal_year_mismatch")

    metric = (target.metric or "").lower()

    if c.unit == "UNKNOWN":
        errs.append("unit_unknown")
    if metric in MONETARY_METRICS and c.unit != "USD":
        errs.append("unit_mismatch_expected_usd")
    if metric in RATIO_METRICS and c.unit not in {"PERCENT", "RATIO"}:
        errs.append("unit_mismatch_expected_ratio")
    if metric and not _metric_supported_in_line(c.line, metric):
        errs.append("metric_not_supported_in_line")

    ok = not any(
        e in errs for e in [
            "ticker_mismatch",
            "fiscal_year_mismatch",
            "unit_unknown",
            "unit_mismatch_expected_usd",
            "unit_mismatch_expected_ratio",
        ]
    )
    return ok, errs


def contradiction_check(
    cands: List[NumericCandidate],
    *,
    rel_tol: float,
    abs_tol: float,
) -> Tuple[bool, List[Tuple[NumericCandidate, NumericCandidate, float]]]:
    conflicts: List[Tuple[NumericCandidate, NumericCandidate, float]] = []
    if len(cands) < 2:
        return False, conflicts

    base = cands[0]
    for other in cands[1:]:
        if base.unit != other.unit:
            continue
        if base.fiscal_year is not None and other.fiscal_year is not None and int(base.fiscal_year) != int(other.fiscal_year):
            continue

        a = float(base.value_scaled)
        b = float(other.value_scaled)
        if a == 0 and b == 0:
            continue

        rel = abs(a - b) / max(1e-12, abs(a))
        absd = abs(a - b)

        if ((abs_tol and absd > abs_tol) or (rel_tol and rel > rel_tol)) and other.precedence + 100 >= base.precedence:
            conflicts.append((base, other, rel))

    return len(conflicts) > 0, conflicts


def choose_best_numeric_with_gate(
    packed_context: str,
    target: Target,
    req: EvidenceRequirements,
    *,
    topn: int = 5,
) -> Tuple[Optional[NumericCandidate], Dict[str, Any]]:
    debug: Dict[str, Any] = {"candidates": [], "candidate_errors": [], "contradiction": False, "conflicts": []}

    cands = extract_numeric_topn(packed_context, target, topn=max(topn, 8), prefer_strong_filing=True)
    debug["candidates"] = [
        {
            "evidence_id": c.evidence_id,
            "kind": c.kind,
            "unit": c.unit,
            "value_raw": c.value_raw,
            "value_scaled": c.value_scaled,
            "score": c.score,
            "precedence": c.precedence,
            "line": c.line[:220],
        }
        for c in cands
    ]

    if not cands:
        return None, {**debug, "fail_reason": "no_candidates"}

    verified: List[NumericCandidate] = []
    for c in cands:
        ok, errs = verify_numeric_candidate(c, target)
        debug["candidate_errors"].append({"evidence_id": c.evidence_id, "errors": errs})
        if ok:
            verified.append(c)

    if not verified:
        return None, {**debug, "fail_reason": "no_verified_candidates"}

    contrad, conflicts = contradiction_check(
        verified,
        rel_tol=req.numeric_contradiction_rel_tol,
        abs_tol=req.numeric_contradiction_abs_tol,
    )
    debug["contradiction"] = contrad
    debug["conflicts"] = [
        {
            "base_id": a.evidence_id,
            "other_id": b.evidence_id,
            "rel_diff": rel,
            "base_scaled": a.value_scaled,
            "other_scaled": b.value_scaled,
        }
        for a, b, rel in conflicts
    ]

    verified.sort(key=lambda c: (c.score, c.precedence, abs(c.value_scaled)), reverse=True)
    return verified[0], debug


# ============================================================
# 6) Deterministic computed metrics
# ============================================================

def required_inputs_for_computed(metric_name: str) -> List[str]:
    m = (metric_name or "").lower().strip()
    if m in ("fcf", "free_cash_flow"):
        return ["cfo", "capex"]
    if m == "gross_margin":
        return ["gross_profit", "revenue"]
    if m in ("operating_margin", "op_margin"):
        return ["operating_income", "revenue"]
    if m == "net_margin":
        return ["net_income", "revenue"]
    if m == "ebitda_margin":
        return ["ebitda", "revenue"]
    if m == "current_ratio":
        return ["current_assets", "current_liabilities"]
    if m == "debt_to_equity":
        return ["total_debt", "total_equity"]
    if m in ("roe", "return_on_equity"):
        return ["net_income", "total_equity"]
    if m in ("revenue_yoy", "yoy_revenue_growth"):
        return ["revenue_t", "revenue_t1"]
    return []


def compute_metric_value(metric_name: str, inputs: Dict[str, float]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    m = (metric_name or "").lower().strip()
    try:
        if m in ("fcf", "free_cash_flow") and {"cfo", "capex"} <= inputs.keys():
            r = FinancialMetricsEngine.free_cash_flow(inputs["cfo"], inputs["capex"])
            return r.value, r.unit, r.formula
        if m == "gross_margin" and {"gross_profit", "revenue"} <= inputs.keys():
            r = FinancialMetricsEngine.gross_margin(inputs["gross_profit"], inputs["revenue"])
            return r.value, r.unit, r.formula
        if m in ("operating_margin", "op_margin") and {"operating_income", "revenue"} <= inputs.keys():
            r = FinancialMetricsEngine.operating_margin(inputs["operating_income"], inputs["revenue"])
            return r.value, r.unit, r.formula
        if m == "net_margin" and {"net_income", "revenue"} <= inputs.keys():
            r = FinancialMetricsEngine.net_margin(inputs["net_income"], inputs["revenue"])
            return r.value, r.unit, r.formula
        if m == "ebitda_margin" and {"ebitda", "revenue"} <= inputs.keys():
            r = FinancialMetricsEngine.ebitda_margin(inputs["ebitda"], inputs["revenue"])
            return r.value, r.unit, r.formula
        if m == "current_ratio" and {"current_assets", "current_liabilities"} <= inputs.keys():
            r = FinancialMetricsEngine.current_ratio(inputs["current_assets"], inputs["current_liabilities"])
            return r.value, r.unit, r.formula
        if m == "debt_to_equity" and {"total_debt", "total_equity"} <= inputs.keys():
            r = FinancialMetricsEngine.debt_to_equity(inputs["total_debt"], inputs["total_equity"])
            return r.value, r.unit, r.formula
        if m in ("roe", "return_on_equity") and {"net_income", "total_equity"} <= inputs.keys():
            r = FinancialMetricsEngine.roe(inputs["net_income"], inputs["total_equity"])
            return r.value, r.unit, r.formula
        if m in ("revenue_yoy", "yoy_revenue_growth") and {"revenue_t", "revenue_t1"} <= inputs.keys():
            r = FinancialMetricsEngine.revenue_yoy(inputs["revenue_t"], inputs["revenue_t1"])
            return r.value, r.unit, r.formula
    except ValueError:
        return None, None, None

    return None, None, None


def compute_metric_from_evidence(
    packed_context: str,
    target: Target,
    computed_metric_name: str,
    req: EvidenceRequirements,
) -> Tuple[Optional[ComputedMetric], Dict[str, Any]]:
    need = required_inputs_for_computed(computed_metric_name)
    debug: Dict[str, Any] = {"required_inputs": need, "extractions": {}, "fail_reason": None}

    if not need:
        debug["fail_reason"] = "unknown_computed_metric"
        return None, debug

    extracted_vals: Dict[str, float] = {}
    extracted_meta: Dict[str, Any] = {}

    for inp in need:
        metric_to_extract = "revenue" if inp in ("revenue_t", "revenue_t1") else inp
        fy = target.fiscal_year
        if inp == "revenue_t1" and isinstance(target.fiscal_year, int):
            fy = int(target.fiscal_year) - 1

        temp_target = Target(ticker=target.ticker, fiscal_year=fy, metric=metric_to_extract, item_hint="Item 8")
        best, ex_debug = choose_best_numeric_with_gate(packed_context, temp_target, req, topn=6)
        debug["extractions"][inp] = {"best": None, "debug": ex_debug}

        if not best:
            debug["fail_reason"] = f"missing_input:{inp}"
            return None, debug

        extracted_vals[inp] = float(best.value_scaled if best.unit == "USD" else best.value_raw)
        extracted_meta[inp] = {
            "value_scaled": best.value_scaled,
            "value_raw": best.value_raw,
            "unit": best.unit,
            "evidence_id": best.evidence_id,
            "line": best.line,
        }
        debug["extractions"][inp]["best"] = extracted_meta[inp]

    value, unit, formula = compute_metric_value(computed_metric_name, extracted_vals)
    if value is None:
        debug["fail_reason"] = "compute_failed"
        return None, debug

    return ComputedMetric(
        name=computed_metric_name,
        value=float(value),
        unit=unit or "UNKNOWN",
        formula=formula or "",
        inputs=extracted_meta,
    ), debug


# ============================================================
# 7) Prompt schema / validation
# ============================================================

def schema_for_mode(mode: Mode) -> Dict[str, Any]:
    base = {
        "final_answer": "string",
        "claims": [{
            "claim_type": "fact|ratio|risk|summary|comparison|valuation|framework",
            "entity": "string|null",
            "metric_or_topic": "string|null",
            "period": "string|null",
            "unit": "string|null",
            "value_or_summary": "number|string|null",
            "citations": ["EVIDENCE_ID"],
            "formula": "string|null",
            "inputs": [{"name": "string", "value": "number|string", "unit": "string|null", "citation": "EVIDENCE_ID|null"}],
        }],
        "tables_used": ["EVIDENCE_ID"],
        "provenance": {"ticker": "string|null", "fiscal_year": "int|null"},
        "inferences": ["string"],
        "confidence": "number",
    }

    if mode == "lookup_numeric":
        return {
            **base,
            "numeric": {
                "metric": "string",
                "value": "number",
                "unit": "USD|PERCENT|RATIO|UNKNOWN",
                "notes": "string",
                "citation": "EVIDENCE_ID",
            },
        }

    if mode == "compute_metric":
        return {
            **base,
            "computed": {
                "metric": "string",
                "value": "number",
                "unit": "USD|PERCENT|RATIO|UNKNOWN",
                "formula": "string",
                "inputs": [{"name": "string", "value": "number", "unit": "string", "citation": "EVIDENCE_ID"}],
            },
        }

    if mode == "comparative_analysis":
        return {
            **base,
            "comparison": {
                "targets": [{"ticker": "string|null", "fiscal_year": "int|null"}],
                "facts": [{"target": "int", "text": "string", "citations": ["EVIDENCE_ID"]}],
                "summary": "string",
            },
        }

    if mode == "risk_analysis":
        return {
            **base,
            "risks": [{"risk": "string", "mechanism": "string", "citations": ["EVIDENCE_ID"]}],
        }

    if mode == "mba_framework":
        return {
            **base,
            "framework": {
                "type": "SWOT|PORTER|PESTEL|OTHER",
                "bullets": [{"bucket": "string", "text": "string", "citations": ["EVIDENCE_ID"]}],
            },
        }

    if mode == "valuation":
        return {
            **base,
            "valuation": {
                "type": "DCF",
                "verified_inputs": [{"name": "string", "value": "number", "unit": "string", "citation": "EVIDENCE_ID"}],
                "assumptions": [{"name": "string", "value": "number|string", "source": "policy|external"}],
                "outputs": [{"name": "string", "value": "number", "unit": "string"}],
                "sensitivity": "object",
            },
        }

    if mode == "relative_valuation":
        return {
            **base,
            "relative_valuation": {
                "multiple": "string",
                "numerator": {"name": "string", "value": "number", "citation": "external|market"},
                "denominator": {"name": "string", "value": "number", "citation": "EVIDENCE_ID"},
                "value": "number",
            },
        }

    return base


def _example_ids_from_context(packed_context: str) -> Tuple[str, str]:
    allowed = parse_allowed_ids_from_context(packed_context or "")
    cands = [x for x in allowed if x != "xbrl"]
    example_citation = cands[0] if cands else "EVIDENCE_ID"
    example_table = next((x for x in cands if x.lower().startswith("t") or "_t" in x.lower()), example_citation)
    return example_citation, example_table


def build_json_answer_prompt(plan: TaskPlan, packed_context: str) -> Tuple[str, str]:
    schema = schema_for_mode(plan.mode)
    example_citation, example_table = _example_ids_from_context(packed_context)
    schema_str = json.dumps(schema, indent=2).replace("EVIDENCE_ID", example_citation)

    system = (
        "You are an evidence-grounded financial QA system.\n"
        "Use ONLY the supplied evidence context.\n"
        "Return ONLY valid JSON.\n"
        "Every factual statement must be cited with evidence IDs exactly as they appear in the context headers.\n"
        "Separate evidence-backed facts from inferences.\n"
        "Do not invent citations, numbers, years, companies, or tables.\n"
    )

    user = (
        f"QUESTION:\n{plan.raw_question}\n\n"
        f"MODE:\n{plan.mode}\n\n"
        f"EVIDENCE CONTEXT:\n{packed_context}\n\n"
        f"OUTPUT MUST MATCH THIS SCHEMA EXACTLY:\n{schema_str}\n"
    )

    return system, user


_CLAIM_KEYS = {
    "claim_type",
    "entity",
    "metric_or_topic",
    "period",
    "unit",
    "value_or_summary",
    "citations",
    "formula",
    "inputs",
}


def _normalize_claims_shape(plan: TaskPlan, obj: Dict[str, Any]) -> None:
    claims = obj.get("claims")
    if not isinstance(claims, list):
        return

    tgt = plan.targets[0] if plan.targets else Target()
    period = f"FY{tgt.fiscal_year}" if isinstance(tgt.fiscal_year, int) else None
    default_topic = tgt.metric
    if default_topic is None:
        if plan.mode == "risk_analysis":
            default_topic = "risk"
        elif "news" in plan.mode:
            default_topic = "news"
        elif "management" in plan.mode:
            default_topic = "management commentary"

    out: List[Dict[str, Any]] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        if _CLAIM_KEYS.issubset(set(c.keys())):
            out.append({k: c.get(k) for k in _CLAIM_KEYS})
            continue

        # backward-compatible upcast from {"text","citations"}.
        text = c.get("text")
        cits = c.get("citations") if isinstance(c.get("citations"), list) else []
        if isinstance(text, str):
            out.append({
                "claim_type": "summary",
                "entity": tgt.ticker,
                "metric_or_topic": default_topic,
                "period": period,
                "unit": None,
                "value_or_summary": text,
                "citations": cits,
                "formula": None,
                "inputs": [],
            })
    obj["claims"] = out


def validate_answer_json(
    plan: TaskPlan,
    packed_context: str,
    model_text: str,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    errors: List[str] = []
    try:
        obj = json.loads(model_text)
    except Exception as e:
        return False, [f"invalid_json:{e}"], None

    if not isinstance(obj, dict):
        return False, ["top_level_not_object"], obj

    _normalize_claims_shape(plan, obj)

    if plan.mode == "comparative_analysis" and not isinstance(obj.get("comparison"), dict):
        obj["comparison"] = {
            "targets": [
                {"ticker": t.ticker, "fiscal_year": t.fiscal_year}
                for t in (plan.targets or [])
            ],
            "facts": [
                {"target": 0, "text": str(c.get("value_or_summary", "")), "citations": c.get("citations", [])}
                for c in (obj.get("claims") or [])
                if isinstance(c, dict)
            ],
            "summary": obj.get("final_answer", "") if isinstance(obj.get("final_answer"), str) else "",
        }
    if plan.mode == "mba_framework" and not isinstance(obj.get("framework"), dict):
        obj["framework"] = {
            "type": "SWOT",
            "bullets": [
                {"bucket": "Summary", "text": str(c.get("value_or_summary", "")), "citations": c.get("citations", [])}
                for c in (obj.get("claims") or [])
                if isinstance(c, dict)
            ],
        }

    expected_keys = set(schema_for_mode(plan.mode).keys())
    if set(obj.keys()) != expected_keys:
        errors.append(f"bad_keys: expected={sorted(expected_keys)} got={sorted(obj.keys())}")

    if not isinstance(obj.get("final_answer"), str):
        errors.append("final_answer_not_string")
    if not isinstance(obj.get("claims"), list):
        errors.append("claims_not_list")
    if not isinstance(obj.get("confidence"), (float, int)):
        errors.append("confidence_not_number")

    allowed_ids = parse_allowed_ids_from_context(packed_context)
    blocks = split_context_into_blocks(packed_context)
    block_map: Dict[str, EvidenceBlock] = {b.evid: b for b in blocks}

    claims = obj.get("claims", [])
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            errors.append(f"claim_{i}_not_object")
            continue
        if set(c.keys()) != _CLAIM_KEYS:
            errors.append(f"claim_{i}_bad_keys")
        cits = c.get("citations")
        if not isinstance(cits, list) or not cits:
            errors.append(f"claim_{i}_missing_citations")
            continue
        for j, cid in enumerate(cits):
            if not isinstance(cid, str):
                errors.append(f"claim_{i}_cit_{j}_not_string")
                continue
            if cid not in allowed_ids:
                errors.append(f"claim_{i}_cit_{j}_not_allowed:{cid}")
        if c.get("inputs") is not None and not isinstance(c.get("inputs"), list):
            errors.append(f"claim_{i}_inputs_not_list")

    if isinstance(obj.get("tables_used"), list):
        obj["tables_used"] = [t for t in obj["tables_used"] if isinstance(t, str) and t in allowed_ids]
    else:
        errors.append("tables_used_not_list")

    prov = obj.get("provenance")
    if not isinstance(prov, dict):
        errors.append("provenance_not_object")
    else:
        if set(prov.keys()) != {"ticker", "fiscal_year"}:
            errors.append("provenance_bad_keys")

    return len(errors) == 0, errors, obj


# ============================================================
# 8) Deterministic output builders
# ============================================================

def build_lookup_numeric_answer(
    packed_context: str,
    target: Target,
    req: EvidenceRequirements,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    best, debug = choose_best_numeric_with_gate(packed_context, target, req, topn=6)
    if not best:
        return None, debug

    unit = best.unit if best.unit in ("USD", "PERCENT", "RATIO") else "UNKNOWN"
    value = float(best.value_scaled) if unit == "USD" else float(best.value_raw)

    return {
        "metric": target.metric or "",
        "value": value,
        "unit": unit,
        "notes": f"Extracted from {best.kind}; scale={best.scale_label}",
        "citation": best.evidence_id,
    }, debug


def build_compute_metric_answer(
    packed_context: str,
    target: Target,
    computed_metric_name: str,
    req: EvidenceRequirements,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    cm, debug = compute_metric_from_evidence(packed_context, target, computed_metric_name, req)
    if not cm:
        return None, debug

    inputs_list = []
    for name, meta in cm.inputs.items():
        inputs_list.append({
            "name": name,
            "value": float(meta["value_scaled"]) if meta["unit"] == "USD" else float(meta["value_raw"]),
            "unit": meta["unit"],
            "citation": meta["evidence_id"],
        })

    return {
        "metric": cm.name,
        "value": float(cm.value),
        "unit": cm.unit if cm.unit in ("USD", "PERCENT", "RATIO") else "UNKNOWN",
        "formula": cm.formula,
        "inputs": inputs_list,
    }, debug
