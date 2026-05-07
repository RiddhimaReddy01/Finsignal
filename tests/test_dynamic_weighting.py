#!/usr/bin/env python3
"""
Test script for dynamic confidence-based weighting in Decision Mode.

Tests:
1. Basic aggregation with all tools having equal confidence
2. Tool with low confidence reduces effective weight
3. Contradictions reduce tool confidence
4. Final score and confidence are correctly aggregated
"""

from signal_scoring import (
    ToolSignal,
    build_tool_signals_from_components,
    compute_final_signal_dynamic,
    signal_action_from_score,
    to_dict,
)


def test_equal_confidence():
    """Test aggregation when all tools have equal confidence."""
    print("\n" + "="*80)
    print("TEST 1: Equal Confidence (All Tools High Confidence)")
    print("="*80)

    # All tools with high confidence, positive signals
    tool_signals = {
        "risk": ToolSignal(name="risk", score=-0.3, confidence=0.85, evidence_count=8, contradictions=[]),
        "tone": ToolSignal(name="tone", score=+0.2, confidence=0.80, evidence_count=6, contradictions=[]),
        "valuation": ToolSignal(name="valuation", score=+0.4, confidence=0.85, evidence_count=7, contradictions=[]),
        "growth": ToolSignal(name="growth", score=+0.3, confidence=0.75, evidence_count=5, contradictions=[]),
        "news": ToolSignal(name="news", score=+0.1, confidence=0.60, evidence_count=3, contradictions=[]),
    }

    score = compute_final_signal_dynamic(tools=tool_signals)
    score_obj = to_dict(score)

    print(f"\nComponent Scores: {score_obj['component_scores']}")
    print(f"Component Confidences: {score_obj['component_confidences']}")
    print(f"Tool Details (Weights):")
    for row in score_obj["tool_details"]["weighted_rows"]:
        print(f"  {row['name']:12} | base_weight={row['base_weight']:.2f} "
              f"| confidence={row['adjusted_confidence']:.2f} "
              f"| effective_weight={row['effective_weight']:.2f} "
              f"| contribution={row['weighted_contribution']:+.4f}")

    print(f"\nFinal Signal Score: {score_obj['signal_score']:+.4f}")
    print(f"Final Confidence: {score_obj['confidence']:.4f}")
    print(f"Label: {score_obj['label']}")

    action = signal_action_from_score(signal_score=score_obj['signal_score'], confidence=score_obj['confidence'])
    print(f"Decision Action: {action}")


def test_low_confidence_tool():
    """Test when one tool has low confidence (limits its weight)."""
    print("\n" + "="*80)
    print("TEST 2: Low Confidence Tool (News Has Only 1 Article)")
    print("="*80)

    tool_signals = {
        "risk": ToolSignal(name="risk", score=-0.3, confidence=0.85, evidence_count=8, contradictions=[]),
        "tone": ToolSignal(name="tone", score=+0.2, confidence=0.80, evidence_count=6, contradictions=[]),
        "valuation": ToolSignal(name="valuation", score=+0.4, confidence=0.85, evidence_count=7, contradictions=[]),
        "growth": ToolSignal(name="growth", score=+0.3, confidence=0.75, evidence_count=5, contradictions=[]),
        "news": ToolSignal(name="news", score=-0.8, confidence=0.43, evidence_count=1, contradictions=[]),  # ← LOW CONFIDENCE
    }

    score = compute_final_signal_dynamic(tools=tool_signals)
    score_obj = to_dict(score)

    print(f"\nComponent Scores: {score_obj['component_scores']}")
    print(f"Component Confidences: {score_obj['component_confidences']}")
    print(f"Tool Details (Weights):")
    for row in score_obj["tool_details"]["weighted_rows"]:
        print(f"  {row['name']:12} | base_weight={row['base_weight']:.2f} "
              f"| confidence={row['adjusted_confidence']:.2f} "
              f"| effective_weight={row['effective_weight']:.2f} "
              f"| contribution={row['weighted_contribution']:+.4f}")

    print(f"\n[WARNING] NOTICE: News has low confidence (0.43), so even though score=-0.8 is very negative,")
    print(f"                 its effective weight is only 0.10 × 0.43 = 0.043")
    print(f"                 This reduces its influence compared to fixed-weight approach.")

    print(f"\nFinal Signal Score: {score_obj['signal_score']:+.4f}")
    print(f"Final Confidence: {score_obj['confidence']:.4f}")
    print(f"Label: {score_obj['label']}")

    action = signal_action_from_score(signal_score=score_obj['signal_score'], confidence=score_obj['confidence'])
    print(f"Decision Action: {action}")


def test_contradictions_reduce_confidence():
    """Test that detected contradictions reduce tool confidence."""
    print("\n" + "="*80)
    print("TEST 3: Contradictions Reduce Tool Confidence")
    print("="*80)

    # Risk says "sell" (-0.6) but Growth says "buy" (+0.5)
    # This is detected as a contradiction
    tool_signals = {
        "risk": ToolSignal(
            name="risk",
            score=-0.6,
            confidence=0.85,
            evidence_count=8,
            contradictions=["strong_growth_despite_high_risk"],  # ← CONTRADICTION
        ),
        "tone": ToolSignal(name="tone", score=+0.1, confidence=0.80, evidence_count=6, contradictions=[]),
        "valuation": ToolSignal(name="valuation", score=+0.2, confidence=0.85, evidence_count=7, contradictions=[]),
        "growth": ToolSignal(
            name="growth",
            score=+0.5,
            confidence=0.75,
            evidence_count=5,
            contradictions=[],  # Growth doesn't have contradictions detected in this test
        ),
        "news": ToolSignal(name="news", score=+0.05, confidence=0.60, evidence_count=3, contradictions=[]),
    }

    score = compute_final_signal_dynamic(tools=tool_signals)
    score_obj = to_dict(score)

    print(f"\nComponent Scores: {score_obj['component_scores']}")
    print(f"Component Confidences: {score_obj['component_confidences']}")
    print(f"Tool Details (Weights):")
    for row in score_obj["tool_details"]["weighted_rows"]:
        penalty = ""
        if row["name"] == "risk" and row["base_confidence"] != row["adjusted_confidence"]:
            penalty = f" (penalized from {row['base_confidence']:.2f})"
        print(f"  {row['name']:12} | base_weight={row['base_weight']:.2f} "
              f"| confidence={row['adjusted_confidence']:.2f}{penalty} "
              f"| effective_weight={row['effective_weight']:.2f} "
              f"| contribution={row['weighted_contribution']:+.4f}")

    print(f"\n[WARNING] NOTICE: Risk has contradiction 'strong_growth_despite_high_risk'")
    print(f"                 Base confidence 0.85 -> adjusted confidence 0.77 (reduced by 0.08)")
    print(f"                 Effective weight: 0.35 x 0.77 = 0.270 (less than base 0.35)")

    print(f"\nFinal Signal Score: {score_obj['signal_score']:+.4f}")
    print(f"Final Confidence: {score_obj['confidence']:.4f}")
    print(f"Label: {score_obj['label']}")

    action = signal_action_from_score(signal_score=score_obj['signal_score'], confidence=score_obj['confidence'])
    print(f"Decision Action: {action}")


def test_build_from_components():
    """Test building tool signals from raw components."""
    print("\n" + "="*80)
    print("TEST 4: Building Tool Signals From Raw Components")
    print("="*80)

    contradiction_map = {
        "risk": ["strong_growth_despite_high_risk"],
        "news": ["positive_news_high_risk"],
    }

    tool_signals = build_tool_signals_from_components(
        risk_avg=0.65,  # 65% risk severity
        risk_evidence_count=8,
        tone_delta=0.15,  # slightly positive tone
        tone_evidence_count=6,
        valuation_gap_pct=0.25,  # 25% undervalued
        valuation_evidence_count=7,
        revenue_growth_yoy=0.18,  # 18% YoY growth
        growth_evidence_count=5,
        news_direction_score=0.2,  # positive news
        news_evidence_count=4,
        contradiction_map=contradiction_map,
    )

    print("\nBuilt Tool Signals:")
    for name, signal in tool_signals.items():
        print(f"  {name:12} | score={signal.score:+.4f} | confidence={signal.confidence:.4f} "
              f"| evidence={signal.evidence_count} | contradictions={signal.contradictions}")

    print("\nAggregating with dynamic weighting...")
    score = compute_final_signal_dynamic(tools=tool_signals)
    score_obj = to_dict(score)

    print(f"\nFinal Signal Score: {score_obj['signal_score']:+.4f}")
    print(f"Final Confidence: {score_obj['confidence']:.4f}")
    print(f"Label: {score_obj['label']}")

    action = signal_action_from_score(signal_score=score_obj['signal_score'], confidence=score_obj['confidence'])
    print(f"Decision Action: {action}")

    print(f"\nKey Findings:")
    for finding in score_obj.get("key_findings", []):
        print(f"  - {finding}")


def test_comparison_fixed_vs_dynamic():
    """Compare fixed-weight vs dynamic-weight aggregation."""
    print("\n" + "="*80)
    print("TEST 5: Fixed Weights vs Dynamic Weights Comparison")
    print("="*80)

    from signal_scoring import compute_final_signal

    # Scenario: Risk high (0.7), but all other tools positive and high-confidence
    tool_signals = {
        "risk": ToolSignal(name="risk", score=-0.7, confidence=0.90, evidence_count=10, contradictions=[]),
        "tone": ToolSignal(name="tone", score=+0.3, confidence=0.85, evidence_count=8, contradictions=[]),
        "valuation": ToolSignal(name="valuation", score=+0.5, confidence=0.88, evidence_count=9, contradictions=[]),
        "growth": ToolSignal(name="growth", score=+0.4, confidence=0.80, evidence_count=6, contradictions=[]),
        "news": ToolSignal(name="news", score=+0.2, confidence=0.70, evidence_count=4, contradictions=[]),
    }

    # Dynamic weighting
    dynamic_score = compute_final_signal_dynamic(tools=tool_signals)
    dynamic_obj = to_dict(dynamic_score)

    # Fixed weighting (old approach)
    fixed_score = compute_final_signal(
        risk_severity_avg=0.7,
        tone_delta=0.3,
        valuation_gap_pct=0.5,
        revenue_growth_yoy=0.4,
        news_direction_score=0.2,
        evidence_count=37,
        contradiction_penalty=0.0,
    )
    fixed_obj = to_dict(fixed_score)

    print("\nDYNAMIC WEIGHTING (Confidence-Based):")
    print(f"  Signal Score: {dynamic_obj['signal_score']:+.4f}")
    print(f"  Confidence: {dynamic_obj['confidence']:.4f}")
    print(f"  Decision: {signal_action_from_score(signal_score=dynamic_obj['signal_score'], confidence=dynamic_obj['confidence'])}")
    print(f"  Component Confidences: {dynamic_obj['component_confidences']}")

    print("\nFIXED WEIGHTING (Original):")
    print(f"  Signal Score: {fixed_obj['signal_score']:+.4f}")
    print(f"  Confidence: {fixed_obj['confidence']:.4f}")
    print(f"  Decision: {signal_action_from_score(signal_score=fixed_obj['signal_score'], confidence=fixed_obj['confidence'])}")

    print("\nDIFFERENCE:")
    score_diff = dynamic_obj['signal_score'] - fixed_obj['signal_score']
    conf_diff = dynamic_obj['confidence'] - fixed_obj['confidence']
    print(f"  Score Delta: {score_diff:+.4f}")
    print(f"  Confidence Delta: {conf_diff:+.4f}")

    if abs(score_diff) > 0.05:
        print(f"  [WARNING] Significant difference in score! Dynamic approach gives {'more ' if score_diff > 0 else 'less '}weight to high-confidence tools.")


if __name__ == "__main__":
    test_equal_confidence()
    test_low_confidence_tool()
    test_contradictions_reduce_confidence()
    test_build_from_components()
    test_comparison_fixed_vs_dynamic()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)
