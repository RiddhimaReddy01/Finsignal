#!/usr/bin/env python3
"""
Evidence Quality Analysis - Compute tool confidence from actual evidence blocks.

Measures:
1. Source quality (SEC > transcript > news > social)
2. Evidence coherence (do blocks agree with each other?)
3. Recency (how recent is the evidence?)
4. Internal contradictions (semantic analysis)
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvidenceBlock:
    """Structured evidence block with metadata."""
    text: str
    source_type: str       # "filing", "transcript", "news", "social", "table"
    date: Optional[str]    # ISO format: "2024-01-15"
    relevance_score: float # 0.0-1.0 how relevant to the tool's question
    sentiment: Optional[float]  # -1.0 (negative) to +1.0 (positive)


class SourceQualityScorer:
    """Score evidence source reliability."""

    QUALITY_MAP = {
        "filing": 0.95,           # SEC 10-K, highly regulated
        "sec": 0.95,
        "10k": 0.95,
        "10q": 0.94,
        "transcript": 0.90,       # Earnings call, official
        "earnings": 0.90,
        "call": 0.88,
        "table": 0.92,            # Financial tables from filing
        "xbrl": 0.93,
        "news": 0.65,             # News can be biased/incomplete
        "article": 0.65,
        "social": 0.40,           # Social media, high noise
        "tweet": 0.35,
        "blog": 0.50,
        "unknown": 0.60,
    }

    @classmethod
    def score(cls, source_type: str) -> float:
        """Get quality score for source type (0.0-1.0)."""
        source_lower = source_type.lower().strip()
        return cls.QUALITY_MAP.get(source_lower, 0.60)


class RecencyScorer:
    """Score evidence recency."""

    @classmethod
    def score(cls, date_str: Optional[str], reference_date: str = "2024-12-31") -> float:
        """
        Score evidence recency (0.0-1.0).

        Scoring:
        - <3 months old: 1.0 (fresh)
        - 3-6 months: 0.95
        - 6-12 months: 0.85
        - 1-2 years: 0.65
        - 2+ years: 0.40 (stale)
        """
        if not date_str:
            return 0.50  # Unknown date = medium

        try:
            # Simple year-based scoring (assumes YYYY-MM-DD format)
            date_year = int(date_str.split("-")[0])
            ref_year = int(reference_date.split("-")[0])
            years_old = ref_year - date_year

            if years_old <= 0:
                return 1.0
            elif years_old == 1:
                return 0.85
            elif years_old == 2:
                return 0.65
            else:
                return max(0.40, 0.40 + (3 - years_old) * 0.05)
        except:
            return 0.50


class CoherenceAnalyzer:
    """Analyze coherence (agreement) between evidence blocks."""

    SENTIMENT_KEYWORDS = {
        "positive": [
            "growth", "strong", "improving", "positive", "robust", "solid",
            "increasing", "accelerating", "expansion", "upgrade", "beat",
            "success", "gains", "rally", "outperform", "efficient"
        ],
        "negative": [
            "risk", "weak", "decline", "negative", "weakness", "deteriorate",
            "decreasing", "slowing", "contraction", "downgrade", "miss",
            "challenge", "loss", "pressure", "underperform", "inefficient"
        ]
    }

    @classmethod
    def extract_sentiment(cls, text: str) -> float:
        """
        Extract sentiment from text (-1.0 to +1.0).

        Simple keyword counting:
        - Count positive words
        - Count negative words
        - Balance: (pos - neg) / (pos + neg)
        """
        text_lower = text.lower()

        pos_count = sum(1 for word in cls.SENTIMENT_KEYWORDS["positive"]
                       if word in text_lower)
        neg_count = sum(1 for word in cls.SENTIMENT_KEYWORDS["negative"]
                       if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    @classmethod
    def measure_coherence(
        cls,
        evidence_blocks: List[EvidenceBlock],
        expected_direction: float,  # The tool's raw score direction
    ) -> float:
        """
        Measure evidence coherence (0.0-1.0).

        High coherence: All blocks point same direction as tool score
        Low coherence: Blocks contradict tool score or each other

        Returns: coherence_score (0.3-0.95)
        """
        if len(evidence_blocks) <= 1:
            return 0.85  # Single block is coherent by definition

        # Extract sentiment from each block
        sentiments = []
        for block in evidence_blocks:
            # Use provided sentiment or extract from text
            if block.sentiment is not None:
                sentiments.append(block.sentiment)
            else:
                sentiments.append(cls.extract_sentiment(block.text))

        # Direction: positive (>0) or negative (<0)
        expected_dir = 1.0 if expected_direction > 0 else -1.0

        # Count how many align with expected direction
        aligned = sum(1 for s in sentiments if (s * expected_dir) > 0)
        alignment_ratio = aligned / len(sentiments)

        # Check if sentiments are consistent with each other
        # High variance = low coherence
        avg_sentiment = sum(sentiments) / len(sentiments)
        variance = sum((s - avg_sentiment) ** 2 for s in sentiments) / len(sentiments)
        variance_score = 1.0 / (1.0 + variance)  # Invert: low variance = high score

        # Combine: alignment + internal consistency
        coherence = 0.5 * alignment_ratio + 0.5 * variance_score

        # Map to 0.30-0.95 scale
        return 0.30 + (0.65 * coherence)


class ContradictionDetector:
    """Detect internal contradictions in evidence blocks."""

    @classmethod
    def detect_contradictions(
        cls,
        evidence_blocks: List[EvidenceBlock],
        tool_name: str,
    ) -> List[str]:
        """
        Detect contradictions within evidence blocks.

        Examples:
        - Risk tool: "supply chain risk HIGH" vs "supply chain STABLE"
        - Tone tool: "management positive outlook" vs "guidance lowered"
        - Valuation: "undervalued by DCF" vs "overvalued by comparables"
        """
        contradictions = []

        if len(evidence_blocks) <= 1:
            return contradictions

        # Extract key phrases from each block
        for i in range(len(evidence_blocks)):
            for j in range(i + 1, len(evidence_blocks)):
                block_i = evidence_blocks[i]
                block_j = evidence_blocks[j]

                # Check for opposite sentiments
                if block_i.sentiment is not None and block_j.sentiment is not None:
                    sentiment_diff = abs(block_i.sentiment - block_j.sentiment)
                    if sentiment_diff > 1.5:  # Very different (e.g., -0.8 vs +0.7)
                        contradictions.append(
                            f"blocks_{i}_{j}_sentiment_mismatch"
                        )
                        continue

                # Check for semantic contradictions
                text_i = block_i.text.lower()
                text_j = block_j.text.lower()

                if cls._is_semantic_contradiction(text_i, text_j, tool_name):
                    contradictions.append(
                        f"blocks_{i}_{j}_semantic_contradiction"
                    )

        return contradictions

    @classmethod
    def _is_semantic_contradiction(cls, text_i: str, text_j: str, tool_name: str) -> bool:
        """Detect semantic contradictions specific to each tool."""

        if tool_name == "risk":
            # Risk: look for "HIGH/LOW risk" pairs
            high_risk_terms = ["high risk", "significant risk", "material risk", "severe"]
            low_risk_terms = ["low risk", "minimal risk", "stable", "well-managed", "mitigated"]

            has_high = any(term in text_i for term in high_risk_terms)
            has_low = any(term in text_i for term in low_risk_terms)

            has_high_j = any(term in text_j for term in high_risk_terms)
            has_low_j = any(term in text_j for term in low_risk_terms)

            # Contradiction: one says HIGH, other says LOW (for same topic)
            same_topic = cls._extract_topic(text_i) == cls._extract_topic(text_j)
            if same_topic and has_high and has_low_j or (has_low and has_high_j):
                return True

        elif tool_name == "tone":
            # Tone: look for "positive/negative" sentiment shifts
            positive_terms = ["improving", "positive", "strong", "encouraged", "optimistic"]
            negative_terms = ["declining", "negative", "weak", "concerned", "pessimistic"]

            has_pos_i = any(term in text_i for term in positive_terms)
            has_neg_i = any(term in text_i for term in negative_terms)

            has_pos_j = any(term in text_j for term in positive_terms)
            has_neg_j = any(term in text_j for term in negative_terms)

            if (has_pos_i and has_neg_j) or (has_neg_i and has_pos_j):
                return True

        elif tool_name == "valuation":
            # Valuation: look for "overvalued/undervalued" from different methods
            undervalued_terms = ["undervalued", "attractive", "cheap", "discount", "intrinsic value >"]
            overvalued_terms = ["overvalued", "expensive", "premium", "stretched", "intrinsic value <"]

            has_under_i = any(term in text_i for term in undervalued_terms)
            has_over_i = any(term in text_i for term in overvalued_terms)

            has_under_j = any(term in text_j for term in undervalued_terms)
            has_over_j = any(term in text_j for term in overvalued_terms)

            if (has_under_i and has_over_j) or (has_over_i and has_under_j):
                return True

        return False

    @classmethod
    def _extract_topic(cls, text: str) -> str:
        """Extract main topic (e.g., 'supply chain', 'regulatory')."""
        # Simple: first noun phrase
        match = re.search(r"(\w+(?:\s+\w+)?)\s+(?:risk|issue|concern)", text)
        if match:
            return match.group(1).lower()
        return ""


class BaseConfidenceCalculator:
    """Calculate base confidence from evidence analysis."""

    @classmethod
    def calculate(
        cls,
        evidence_blocks: List[EvidenceBlock],
        tool_score: float,
    ) -> float:
        """
        Calculate base confidence (0.0-0.95).

        Combines:
        1. Source quality (0.4-0.95)
        2. Evidence coherence (0.3-0.95)
        3. Quantity bonus (1.0-1.3x)
        4. Recency (0.4-1.0)
        """

        if not evidence_blocks:
            return 0.0

        # 1. Source quality
        total_quality = sum(
            SourceQualityScorer.score(block.source_type)
            for block in evidence_blocks
        )
        avg_source_quality = total_quality / len(evidence_blocks)

        # 2. Coherence
        coherence = CoherenceAnalyzer.measure_coherence(
            evidence_blocks,
            tool_score
        )

        # 3. Quantity bonus (diminishing returns)
        quantity_bonus = 1.0 + min(0.3, len(evidence_blocks) * 0.03)

        # 4. Recency
        recency_scores = [
            RecencyScorer.score(block.date) for block in evidence_blocks
        ]
        avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5

        # Combine
        base_confidence = (
            avg_source_quality *      # 0.4 - 0.95
            coherence *               # 0.3 - 0.95
            avg_recency *             # 0.4 - 1.0
            quantity_bonus            # 1.0 - 1.3
        )

        # Cap at 0.95
        return min(0.95, max(0.0, base_confidence))


if __name__ == "__main__":
    # Example usage
    blocks = [
        EvidenceBlock(
            text="Supply chain disruption risk remains HIGH due to geopolitical tensions",
            source_type="filing",
            date="2024-01-15",
            relevance_score=0.95,
            sentiment=None,
        ),
        EvidenceBlock(
            text="Supply chain is STABLE with diversified suppliers across regions",
            source_type="filing",
            date="2024-01-15",
            relevance_score=0.95,
            sentiment=None,
        ),
        EvidenceBlock(
            text="Recent news shows concerning supply delays affecting production",
            source_type="news",
            date="2024-01-10",
            relevance_score=0.80,
            sentiment=None,
        ),
    ]

    # Analyze
    tool_score = -0.65  # High risk

    source_quality = sum(
        SourceQualityScorer.score(b.source_type) for b in blocks
    ) / len(blocks)
    print(f"Average Source Quality: {source_quality:.2f}")

    coherence = CoherenceAnalyzer.measure_coherence(blocks, tool_score)
    print(f"Evidence Coherence: {coherence:.2f}")

    contradictions = ContradictionDetector.detect_contradictions(blocks, "risk")
    print(f"Contradictions Found: {len(contradictions)}")
    for c in contradictions:
        print(f"  - {c}")

    base_conf = BaseConfidenceCalculator.calculate(blocks, tool_score)
    print(f"Base Confidence: {base_conf:.2f}")
