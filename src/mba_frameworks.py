# mba_frameworks.py
# ============================================================
# MBA Framework Prompt Engineering
#
# Provides framework-specific:
#   - Bucket definitions with descriptions
#   - Tailored system prompts
#   - Scoring rubrics per bucket
#   - Output schemas with pre-specified bucket names
#   - Detection logic to classify which framework the user wants
#
# Supported frameworks:
#   SWOT, PORTER, PESTEL, VRIO, BCG, ANSOFF, VALUE_CHAIN
#
# Plugs into verification.py's mba_framework mode by replacing
# the generic build_json_answer_prompt with framework-aware prompts.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

FrameworkType = Literal["SWOT", "PORTER", "PESTEL", "VRIO", "BCG", "ANSOFF", "VALUE_CHAIN", "OTHER"]


# ============================================================
# 1. Framework Definitions
# ============================================================

@dataclass(frozen=True)
class BucketDef:
    """A single bucket/category within a framework."""
    name: str
    description: str
    evidence_hints: List[str]  # what to look for in evidence
    scoring_rubric: str        # how to assess strength


@dataclass(frozen=True)
class FrameworkDef:
    """Complete definition of an MBA framework."""
    framework_type: FrameworkType
    display_name: str
    description: str
    buckets: List[BucketDef]
    analysis_guidance: str     # how the LLM should approach the analysis
    output_instruction: str    # specific formatting instructions


SWOT = FrameworkDef(
    framework_type="SWOT",
    display_name="SWOT Analysis",
    description="Evaluates internal strengths/weaknesses and external opportunities/threats.",
    buckets=[
        BucketDef(
            name="Strengths",
            description="Internal capabilities and resources that give the company a competitive advantage.",
            evidence_hints=["market share", "brand", "patents", "margins", "cash position", "talent", "technology moat"],
            scoring_rubric="Strong = sustainable, hard to replicate, directly tied to financial outperformance. Weak = generic, easily matched by competitors.",
        ),
        BucketDef(
            name="Weaknesses",
            description="Internal limitations, resource gaps, or vulnerabilities that could hinder performance.",
            evidence_hints=["concentration risk", "debt levels", "margin compression", "regulatory exposure", "talent retention", "geographic dependence"],
            scoring_rubric="Critical = directly threatens revenue or profitability within 1-2 years. Minor = manageable with existing resources.",
        ),
        BucketDef(
            name="Opportunities",
            description="External factors or trends the company could exploit for growth or value creation.",
            evidence_hints=["market expansion", "new products", "M&A", "regulatory tailwinds", "AI adoption", "emerging markets", "secular trends"],
            scoring_rubric="High = large addressable market, company has right to win. Low = speculative, requires significant execution.",
        ),
        BucketDef(
            name="Threats",
            description="External factors that could negatively impact the company's position or performance.",
            evidence_hints=["competition", "regulation", "macro headwinds", "disruption", "geopolitical", "supply chain", "litigation"],
            scoring_rubric="Severe = high probability AND high impact. Moderate = manageable but requires monitoring.",
        ),
    ],
    analysis_guidance=(
        "For each bucket, provide 2-5 specific, evidence-backed points. "
        "Prioritize factors that are MATERIAL to financial performance, not generic observations. "
        "Cross-reference filing evidence (10-K) with news and transcripts where available. "
        "Strengths and Weaknesses must come from internal company data (filings, transcripts). "
        "Opportunities and Threats should incorporate external sources (news, market data) alongside filing risk factors."
    ),
    output_instruction=(
        "Each bullet must cite specific evidence IDs. "
        "Rate each point as HIGH/MEDIUM/LOW impact. "
        "End with a 2-sentence synthesis of the overall SWOT position."
    ),
)

PORTER = FrameworkDef(
    framework_type="PORTER",
    display_name="Porter's Five Forces",
    description="Analyzes industry competitive intensity and attractiveness through five structural forces.",
    buckets=[
        BucketDef(
            name="Threat of New Entrants",
            description="How easy is it for new competitors to enter this market?",
            evidence_hints=["barriers to entry", "capital requirements", "patents", "regulatory approval", "brand loyalty", "economies of scale", "switching costs"],
            scoring_rubric="HIGH threat = low barriers, many potential entrants, commoditized market. LOW = high barriers, strong moats.",
        ),
        BucketDef(
            name="Bargaining Power of Suppliers",
            description="How much leverage do suppliers have over the company?",
            evidence_hints=["supplier concentration", "component scarcity", "TSMC", "single source", "raw materials", "switching costs", "vertical integration"],
            scoring_rubric="HIGH power = few suppliers, critical inputs, no substitutes. LOW = many alternatives, company has scale leverage.",
        ),
        BucketDef(
            name="Bargaining Power of Buyers",
            description="How much leverage do customers have?",
            evidence_hints=["customer concentration", "price sensitivity", "switching costs", "enterprise vs consumer", "contract terms", "channel power"],
            scoring_rubric="HIGH power = few large buyers, low switching costs, price-sensitive market. LOW = fragmented buyers, high switching costs.",
        ),
        BucketDef(
            name="Threat of Substitutes",
            description="How easily can customers switch to alternative products or services?",
            evidence_hints=["alternative technologies", "open source", "cross-platform", "commoditization", "disruption risk", "behavioral switching costs"],
            scoring_rubric="HIGH threat = many viable alternatives, low switching costs. LOW = unique product, strong ecosystem lock-in.",
        ),
        BucketDef(
            name="Competitive Rivalry",
            description="How intense is competition among existing players?",
            evidence_hints=["market share", "pricing pressure", "number of competitors", "growth rate", "differentiation", "exit barriers", "R&D spending"],
            scoring_rubric="HIGH rivalry = many equal competitors, slow growth, high fixed costs. LOW = clear leader, growing market, differentiated products.",
        ),
    ],
    analysis_guidance=(
        "Assess each force on a scale of LOW / MODERATE / HIGH based on concrete evidence. "
        "Focus on the CURRENT state of each force, not historical. "
        "Use filing evidence for company-specific factors (supplier relationships, customer concentration). "
        "Use news for industry-level dynamics (new entrants, regulatory changes). "
        "The goal is to determine overall industry attractiveness for investment."
    ),
    output_instruction=(
        "For each force, state the assessment (LOW/MODERATE/HIGH), then provide 2-4 evidence-backed reasons. "
        "Conclude with an overall industry attractiveness rating and 2-sentence investment implication."
    ),
)

PESTEL = FrameworkDef(
    framework_type="PESTEL",
    display_name="PESTEL Analysis",
    description="Scans the macro-environment across six dimensions to identify external factors affecting the company.",
    buckets=[
        BucketDef(
            name="Political",
            description="Government policies, trade relations, political stability, and regulatory environment.",
            evidence_hints=["tariffs", "export controls", "trade war", "sanctions", "government contracts", "lobbying", "political risk"],
            scoring_rubric="Material = directly impacts revenue, supply chain, or market access. Immaterial = distant or speculative political risk.",
        ),
        BucketDef(
            name="Economic",
            description="Macroeconomic conditions affecting demand, costs, and capital availability.",
            evidence_hints=["interest rates", "inflation", "GDP growth", "consumer spending", "FX exposure", "recession risk", "credit conditions"],
            scoring_rubric="Material = company revenue is cyclically exposed OR balance sheet is rate-sensitive. Immaterial = company is defensive/counter-cyclical.",
        ),
        BucketDef(
            name="Social",
            description="Demographic trends, consumer behavior shifts, workforce dynamics, and cultural factors.",
            evidence_hints=["demographics", "remote work", "ESG", "consumer preferences", "talent market", "diversity", "health trends", "urbanization"],
            scoring_rubric="Material = trend directly affects demand for company's products or talent pipeline. Immaterial = generic societal trend.",
        ),
        BucketDef(
            name="Technological",
            description="Technology trends, innovation cycles, R&D dynamics, and disruption potential.",
            evidence_hints=["AI", "automation", "cloud", "cybersecurity", "R&D spending", "patents", "technology adoption curves", "Moore's law"],
            scoring_rubric="Material = company is either beneficiary or vulnerable to the technology shift. Immaterial = tangential to core business.",
        ),
        BucketDef(
            name="Environmental",
            description="Climate-related risks, sustainability requirements, and resource constraints.",
            evidence_hints=["carbon emissions", "sustainability", "climate risk", "renewable energy", "supply chain resilience", "water scarcity", "ESG reporting"],
            scoring_rubric="Material = regulatory compliance cost OR physical risk to operations. Immaterial = minimal exposure to environmental factors.",
        ),
        BucketDef(
            name="Legal",
            description="Regulatory compliance, litigation exposure, intellectual property, and legal proceedings.",
            evidence_hints=["antitrust", "litigation", "patent disputes", "data privacy", "GDPR", "securities law", "settlement", "compliance costs"],
            scoring_rubric="Material = active litigation with quantifiable exposure OR regulatory action pending. Immaterial = routine compliance.",
        ),
    ],
    analysis_guidance=(
        "For each dimension, identify 1-3 MATERIAL factors specific to this company, not generic macro commentary. "
        "Every point must be tied to evidence from the filing, news, or transcripts. "
        "Political and Legal factors should reference specific regulations or jurisdictions. "
        "Economic factors should reference the company's actual exposure (revenue mix, debt structure). "
        "Rate each factor's time horizon: NEAR-TERM (0-1yr), MEDIUM-TERM (1-3yr), or LONG-TERM (3+yr)."
    ),
    output_instruction=(
        "For each dimension, provide the assessment with time horizon, then 1-3 cited evidence points. "
        "Conclude with the top 3 macro factors most likely to impact the investment thesis."
    ),
)

VRIO = FrameworkDef(
    framework_type="VRIO",
    display_name="VRIO Framework",
    description="Evaluates whether a company's resources and capabilities create sustainable competitive advantage.",
    buckets=[
        BucketDef(
            name="Valuable",
            description="Does the resource enable the firm to exploit opportunities or neutralize threats?",
            evidence_hints=["revenue contribution", "market position", "cost advantage", "customer value", "pricing power"],
            scoring_rubric="YES = resource directly drives financial outperformance. NO = resource exists but doesn't differentiate.",
        ),
        BucketDef(
            name="Rare",
            description="Is the resource controlled by only a small number of competing firms?",
            evidence_hints=["market share", "patents", "unique technology", "exclusive partnerships", "talent concentration"],
            scoring_rubric="YES = few or no competitors possess this. NO = widely available in the industry.",
        ),
        BucketDef(
            name="Inimitable",
            description="Is it costly or difficult for competitors to obtain or develop this resource?",
            evidence_hints=["R&D investment", "time to build", "network effects", "ecosystem lock-in", "regulatory moats", "brand history"],
            scoring_rubric="YES = would take years and billions to replicate. NO = competitors could build or buy this within 1-2 years.",
        ),
        BucketDef(
            name="Organized",
            description="Is the firm organized to capture the value of this resource?",
            evidence_hints=["management structure", "compensation alignment", "capital allocation", "strategic focus", "operational efficiency"],
            scoring_rubric="YES = company actively leverages this resource in strategy and operations. NO = resource exists but is underutilized.",
        ),
    ],
    analysis_guidance=(
        "Identify 3-5 key resources or capabilities from the filing evidence. "
        "For each resource, evaluate all four VRIO dimensions sequentially. "
        "A resource must pass all four tests (V+R+I+O) to be a source of SUSTAINED competitive advantage. "
        "V alone = competitive parity. V+R = temporary advantage. V+R+I = unused competitive advantage. V+R+I+O = sustained advantage."
    ),
    output_instruction=(
        "Present as a matrix: rows = resources, columns = V/R/I/O with YES/NO and brief evidence. "
        "Classify each resource's competitive implication. "
        "Conclude with whether the company has a sustainable moat."
    ),
)


# ============================================================
# 2. Framework Registry
# ============================================================

FRAMEWORK_REGISTRY: Dict[FrameworkType, FrameworkDef] = {
    "SWOT": SWOT,
    "PORTER": PORTER,
    "PESTEL": PESTEL,
    "VRIO": VRIO,
}

# Detection patterns — order matters (most specific first)
_FRAMEWORK_PATTERNS: List[Tuple[re.Pattern, FrameworkType]] = [
    (re.compile(r"\bporter'?s?\s*(five|5)\s*forces?\b", re.I), "PORTER"),
    (re.compile(r"\bporter\b", re.I), "PORTER"),
    (re.compile(r"\b5\s*forces?\b", re.I), "PORTER"),
    (re.compile(r"\bpestel\b", re.I), "PESTEL"),
    (re.compile(r"\bpest\b", re.I), "PESTEL"),
    (re.compile(r"\bvrio\b", re.I), "VRIO"),
    (re.compile(r"\bswot\b", re.I), "SWOT"),
    # Fallback heuristics
    (re.compile(r"\b(competitive\s+landscape|industry\s+analysis|competitive\s+forces)\b", re.I), "PORTER"),
    (re.compile(r"\b(macro|macroeconomic|external\s+environment)\b", re.I), "PESTEL"),
    (re.compile(r"\b(competitive\s+advantage|moat|sustainable\s+advantage|resources?\s+and\s+capabilities)\b", re.I), "VRIO"),
    (re.compile(r"\b(strengths?|weaknesses?|opportunities?|threats?)\b", re.I), "SWOT"),
]


def detect_framework_type(question: str) -> FrameworkType:
    """Classify which MBA framework the user is requesting."""
    q = (question or "").strip()
    for pattern, ftype in _FRAMEWORK_PATTERNS:
        if pattern.search(q):
            return ftype
    return "SWOT"  # default


def get_framework_def(framework_type: FrameworkType) -> FrameworkDef:
    return FRAMEWORK_REGISTRY.get(framework_type, SWOT)


# ============================================================
# 3. Framework-Specific Prompt Builder
# ============================================================

def build_framework_prompt(
    question: str,
    packed_context: str,
    framework_type: Optional[FrameworkType] = None,
    *,
    ticker: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    example_citation: str = "EVIDENCE_ID",
) -> Tuple[str, str]:
    """
    Build a framework-specific system + user prompt.

    Replaces the generic build_json_answer_prompt for mba_framework mode.
    Returns (system_prompt, user_prompt).
    """
    if framework_type is None:
        framework_type = detect_framework_type(question)

    fdef = get_framework_def(framework_type)

    # Build bucket specification
    bucket_spec_lines = []
    for i, bucket in enumerate(fdef.buckets, 1):
        hints = ", ".join(bucket.evidence_hints[:5])
        bucket_spec_lines.append(
            f"  {i}. **{bucket.name}**: {bucket.description}\n"
            f"     Look for: {hints}\n"
            f"     Scoring: {bucket.scoring_rubric}"
        )
    bucket_spec = "\n".join(bucket_spec_lines)

    # Build output schema
    bucket_names = [b.name for b in fdef.buckets]
    schema_buckets = ", ".join(f'"{b}"' for b in bucket_names)

    entity_label = ticker or "the company"
    fy_label = f" for FY{fiscal_year}" if fiscal_year else ""

    system_prompt = f"""You are an evidence-grounded financial analyst performing a {fdef.display_name}.
{fdef.description}

CRITICAL RULES:
- Use ONLY the supplied evidence context. Do not invent information.
- Every factual claim must cite an evidence ID exactly as it appears in the context headers.
- Separate evidence-backed facts from analytical inferences.
- Do not fabricate citations, numbers, years, or company names.
- Return ONLY valid JSON matching the schema below.

FRAMEWORK SPECIFICATION — {fdef.display_name}:
{bucket_spec}

ANALYSIS APPROACH:
{fdef.analysis_guidance}

OUTPUT REQUIREMENTS:
{fdef.output_instruction}

OUTPUT SCHEMA (return ONLY this JSON):
{{
  "final_answer": "2-3 sentence executive summary of the {fdef.display_name} for {entity_label}{fy_label}",
  "framework": {{
    "type": "{framework_type}",
    "buckets": [
      {{
        "bucket": "{bucket_names[0]}",
        "assessment": "HIGH|MODERATE|LOW or summary phrase",
        "points": [
          {{
            "text": "specific evidence-backed point",
            "impact": "HIGH|MEDIUM|LOW",
            "time_horizon": "NEAR_TERM|MEDIUM_TERM|LONG_TERM",
            "citations": ["{example_citation}"]
          }}
        ]
      }}
    ],
    "synthesis": "2-sentence overall assessment connecting the buckets to investment implications"
  }},
  "claims": [
    {{
      "claim_type": "framework",
      "entity": "{ticker or 'null'}",
      "metric_or_topic": "{framework_type.lower()}",
      "period": "{'FY' + str(fiscal_year) if fiscal_year else 'null'}",
      "unit": null,
      "value_or_summary": "string",
      "citations": ["{example_citation}"],
      "formula": null,
      "inputs": []
    }}
  ],
  "tables_used": [],
  "provenance": {{"ticker": "{ticker or 'null'}", "fiscal_year": {fiscal_year or 'null'}}},
  "inferences": ["list any assumptions made"],
  "confidence": 0.0
}}

VALID BUCKET NAMES: [{schema_buckets}]
You MUST include ALL {len(fdef.buckets)} buckets. Each bucket MUST have at least 1 point with citations."""

    user_prompt = f"""QUESTION:
{question}

ENTITY: {entity_label}{fy_label}
FRAMEWORK: {fdef.display_name} ({framework_type})

EVIDENCE CONTEXT:
{packed_context}

Analyze using the {fdef.display_name} framework. Return ONLY valid JSON matching the schema."""

    return system_prompt, user_prompt


# ============================================================
# 4. Schema for validation
# ============================================================

def framework_schema(framework_type: FrameworkType) -> Dict[str, Any]:
    """Return the expected output schema for validation."""
    fdef = get_framework_def(framework_type)
    return {
        "final_answer": "string",
        "framework": {
            "type": framework_type,
            "buckets": [{
                "bucket": f"{fdef.buckets[0].name}|...|{fdef.buckets[-1].name}",
                "assessment": "string",
                "points": [{
                    "text": "string",
                    "impact": "HIGH|MEDIUM|LOW",
                    "time_horizon": "NEAR_TERM|MEDIUM_TERM|LONG_TERM",
                    "citations": ["EVIDENCE_ID"],
                }],
            }],
            "synthesis": "string",
        },
        "claims": [{"claim_type": "framework", "entity": "string|null", "metric_or_topic": "string", "period": "string|null", "unit": "null", "value_or_summary": "string", "citations": ["EVIDENCE_ID"], "formula": "null", "inputs": []}],
        "tables_used": ["EVIDENCE_ID"],
        "provenance": {"ticker": "string|null", "fiscal_year": "int|null"},
        "inferences": ["string"],
        "confidence": "number",
    }


def validate_framework_output(
    output: Dict[str, Any],
    framework_type: FrameworkType,
) -> Tuple[bool, List[str]]:
    """Validate that framework output has all required buckets."""
    errors = []
    fdef = get_framework_def(framework_type)
    expected_buckets = {b.name for b in fdef.buckets}

    fw = output.get("framework")
    if not isinstance(fw, dict):
        return False, ["framework_not_dict"]

    actual_buckets = set()
    for b in (fw.get("buckets") or []):
        if isinstance(b, dict):
            actual_buckets.add(b.get("bucket", ""))

    missing = expected_buckets - actual_buckets
    if missing:
        errors.append(f"missing_buckets:{','.join(sorted(missing))}")

    for b in (fw.get("buckets") or []):
        if not isinstance(b, dict):
            continue
        points = b.get("points") or []
        if not points:
            errors.append(f"empty_bucket:{b.get('bucket', '?')}")
        for p in points:
            if isinstance(p, dict) and not (p.get("citations") or []):
                errors.append(f"uncited_point_in:{b.get('bucket', '?')}")

    return len(errors) == 0, errors
