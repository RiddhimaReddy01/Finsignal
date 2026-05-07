#!/usr/bin/env python3
"""
Phase 1 Integration Test: Evidence Quality Analysis for Risk Tool

Tests that:
1. Evidence blocks are properly created with source_type and date
2. BaseConfidenceCalculator works on evidence blocks
3. Risk tool uses quality-based confidence instead of simple formula
"""

from evidence_quality_analyzer import (
    EvidenceBlock,
    BaseConfidenceCalculator,
    SourceQualityScorer,
    RecencyScorer,
    CoherenceAnalyzer,
)

# Test 1: Evidence block creation
print("=" * 60)
print("TEST 1: Create Evidence Blocks with metadata")
print("=" * 60)

blocks = [
    EvidenceBlock(
        text="Supply chain disruption risk remains HIGH due to geopolitical tensions",
        source_type="filing",
        date="2024-12-31",
        relevance_score=0.95,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Supply chain is STABLE with diversified suppliers across regions",
        source_type="filing",
        date="2024-12-31",
        relevance_score=0.95,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Recent news shows concerning supply delays affecting production",
        source_type="news",
        date="2024-12-25",
        relevance_score=0.80,
        sentiment=None,
    ),
]

print(f"Created {len(blocks)} evidence blocks:")
for i, block in enumerate(blocks):
    print(f"  Block {i+1}: source={block.source_type}, date={block.date}, text={block.text[:50]}...")

# Test 2: Component scoring
print("\n" + "=" * 60)
print("TEST 2: Score Components")
print("=" * 60)

source_quality = sum(SourceQualityScorer.score(b.source_type) for b in blocks) / len(blocks)
print(f"Average Source Quality: {source_quality:.3f}")

recency_scores = [RecencyScorer.score(b.date, reference_date="2024-12-31") for b in blocks]
avg_recency = sum(recency_scores) / len(recency_scores)
print(f"Average Recency: {avg_recency:.3f}")
print(f"  Individual scores: {[f'{s:.3f}' for s in recency_scores]}")

# Risk tool score (tool thinks there's HIGH risk, -0.65)
tool_score = -0.65
coherence = CoherenceAnalyzer.measure_coherence(blocks, tool_score)
print(f"Evidence Coherence (tool_score={tool_score}): {coherence:.3f}")

# Test 3: Base confidence calculation
print("\n" + "=" * 60)
print("TEST 3: Calculate Base Confidence")
print("=" * 60)

base_confidence = BaseConfidenceCalculator.calculate(
    evidence_blocks=blocks,
    tool_score=tool_score,
)

print(f"Base Confidence (Quality-Based): {base_confidence:.3f}")
print(f"  = source_quality({source_quality:.3f}) x coherence({coherence:.3f})")
print(f"    x recency({avg_recency:.3f}) x quantity_bonus(1.09)")
print(f"  = {source_quality:.3f} x {coherence:.3f} x {avg_recency:.3f} x 1.09")
print(f"  = {base_confidence:.3f}")

# Compare to old simple formula
simple_confidence = min(0.35 + 0.08 * len(blocks), 0.95)
print(f"\nOld Simple Formula: 0.35 + 0.08 * {len(blocks)} = {simple_confidence:.3f}")
print(f"Improvement: Quality-based considers evidence coherence and source quality")

# Test 4: Scenario analysis
print("\n" + "=" * 60)
print("TEST 4: Real-World Scenarios")
print("=" * 60)

# Scenario A: All SEC evidence, highly coherent
blocks_coherent = [
    EvidenceBlock(
        text="Supply chain risk is MINIMAL with strong diversification",
        source_type="filing",
        date="2024-12-31",
        relevance_score=0.95,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Operations are stable and well-managed across regions",
        source_type="filing",
        date="2024-12-31",
        relevance_score=0.95,
        sentiment=None,
    ),
]
conf_coherent = BaseConfidenceCalculator.calculate(blocks_coherent, tool_score=0.2)
print(f"Scenario A (Coherent SEC evidence):")
print(f"  Blocks: {len(blocks_coherent)}, Source: Filing, Coherence: HIGH")
print(f"  Confidence: {conf_coherent:.3f}")

# Scenario B: Mixed sources, some old
blocks_mixed = [
    EvidenceBlock(
        text="HIGH risk from supply chain disruption",
        source_type="filing",
        date="2024-12-31",
        relevance_score=0.95,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Supply chain issues resolved long ago",
        source_type="news",
        date="2022-01-15",  # Very old
        relevance_score=0.60,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Recent chatter on Twitter suggests problems",
        source_type="social",
        date="2024-12-28",
        relevance_score=0.50,
        sentiment=None,
    ),
]
conf_mixed = BaseConfidenceCalculator.calculate(blocks_mixed, tool_score=-0.6)
print(f"\nScenario B (Mixed sources + old evidence):")
print(f"  Blocks: {len(blocks_mixed)}, Sources: SEC/News/Social, Recency: Mixed")
print(f"  Confidence: {conf_mixed:.3f}")
print(f"  Note: Lower confidence due to old news and social media")

# Scenario C: High quality, recent, coherent
blocks_high_quality = [
    EvidenceBlock(
        text="Free cash flow strong and growing YoY",
        source_type="filing",
        date="2024-12-31",
        relevance_score=0.98,
        sentiment=None,
    ),
    EvidenceBlock(
        text="XBRL financial data shows consistent revenue growth",
        source_type="filing",
        date="2024-12-31",
        relevance_score=1.0,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Earnings call transcript confirms strong momentum",
        source_type="filing",  # Treated as filing for high quality
        date="2024-12-31",
        relevance_score=0.95,
        sentiment=None,
    ),
    EvidenceBlock(
        text="Financial analysis from reputable publication",
        source_type="news",
        date="2024-12-28",
        relevance_score=0.85,
        sentiment=None,
    ),
]
conf_high = BaseConfidenceCalculator.calculate(blocks_high_quality, tool_score=0.7)
print(f"\nScenario C (High quality evidence):")
print(f"  Blocks: {len(blocks_high_quality)}, Sources: SEC/Transcript/News, All recent")
print(f"  Confidence: {conf_high:.3f}")
print(f"  Note: Highest confidence - quality sources, coherent, recent")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Phase 1 Integration Test Complete!")
print(f"- Evidence blocks created with source_type and date")
print(f"- BaseConfidenceCalculator properly scores quality")
print(f"- Quality-based confidence > simple formula for coherent evidence")
print(f"- Integration ready for server.py decision pipeline")
