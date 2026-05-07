# Dynamic Confidence-Based Weighting - Implementation Guide

## Overview

Replaced fixed-weight signal aggregation with **dynamic confidence-based weighting** in Decision Mode. This is a significant architectural improvement that:

1. **Weights scale with tool confidence**: `effective_weight = base_weight × tool_confidence`
2. **Contradictions reduce tool confidence**: Each detected contradiction reduces a tool's confidence by 0.08
3. **Final aggregations are confidence-weighted**: Both score and confidence use dynamic weights
4. **Provides transparency**: UI can show per-tool confidences and contradictions

---

## What Changed

### 1. New Data Structures (signal_scoring.py)

```python
@dataclass
class ToolSignal:
    name: str                    # risk, tone, valuation, growth, news
    score: float                 # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0
    evidence_count: int          # number of evidence blocks
    contradictions: List[str]    # detected contradictions
```

```python
@dataclass
class SignalScore:
    signal_score: float
    confidence: float
    label: str
    component_scores: Dict[str, float]           # Per-tool scores
    component_confidences: Dict[str, float]      # NEW: Per-tool confidences
    key_findings: List[str]
    risk_flags: List[str]
    tool_details: Dict[str, Any]                 # NEW: Weighted rows for transparency
```

### 2. New Functions (signal_scoring.py)

| Function | Purpose |
|----------|---------|
| `_detect_tool_contradiction()` | Detect weak signals |
| `_adjust_confidence_for_contradictions()` | Apply penalty per contradiction |
| `_compute_base_tool_confidence()` | Base confidence = 0.35 + 0.08 × evidence_count |
| `build_tool_signals_from_components()` | Create ToolSignal objects from raw tool metrics |
| `compute_final_signal_dynamic()` | NEW: Aggregates using dynamic confidence weights |

### 3. Updated Server Logic (server.py)

**Before**:
```python
score = compute_final_signal(
    risk_severity_avg=risk_avg,
    tone_delta=tone_delta,
    ...
    evidence_count=len(all_chunks),
    contradiction_penalty=0.0,  # Fixed contradiction handling
)
```

**After**:
```python
# Count evidence per tool
risk_evidence = tool_evidence.get("risk", [])
tone_evidence = tool_evidence.get("tone", [])
...

# Detect contradictions per tool
tool_contradictions = {
    "risk": ["strong_growth_despite_high_risk", ...] if ...,
    "tone": ["positive_tone_negative_news", ...] if ...,
    ...
}

# Build tool signals with contradictions
tool_signals = build_tool_signals_from_components(
    risk_avg=risk_avg,
    risk_evidence_count=len(risk_evidence),
    ...
    contradiction_map=tool_contradictions,
)

# Aggregate with dynamic weighting
score = compute_final_signal_dynamic(tools=tool_signals)
```

---

## Aggregation Formula

### Old Approach (Fixed Weights)
```
score = Σ(normalized_score[i] × base_weight[i])
confidence = 0.35 + 0.08 × evidence_count  (global)
```

**Problem**: All tools weighted the same regardless of confidence

### New Approach (Dynamic Weighting)

```
# Step 1: Adjust confidences for contradictions
adjusted_confidence[i] = base_confidence[i] - 0.08 × num_contradictions[i]

# Step 2: Compute dynamic weights
effective_weight[i] = base_weight[i] × adjusted_confidence[i]
total_weight = Σ(effective_weight[i])

# Step 3: Aggregate score
score = Σ(normalized_score[i] × effective_weight[i]) / total_weight

# Step 4: Aggregate confidence
confidence = Σ(effective_weight[i] × adjusted_confidence[i]) / total_weight
```

**Benefit**: High-confidence tools get more influence; contradictions reduce influence

---

## Contradiction Detection

The server now detects tool-level contradictions before aggregation:

| Contradiction | Condition | Tools Affected |
|---|---|---|
| `strong_growth_despite_high_risk` | risk_avg > 0.6 AND growth > 15% | Risk |
| `attractive_valuation_despite_high_risk` | risk_avg > 0.6 AND valuation_gap > 20% | Risk |
| `positive_tone_negative_news` | tone_delta > 0.1 AND news < -15% | Tone |
| `positive_tone_overvalued` | tone_delta > 0.1 AND valuation_gap < -20% | Tone |
| `undervalued_high_risk` | valuation_gap > 20% AND risk > 55% | Valuation |
| `undervalued_declining_growth` | valuation_gap > 20% AND growth < 0% | Valuation |
| `strong_growth_negative_tone` | growth > 20% AND tone < -10% | Growth |
| `strong_growth_negative_news` | growth > 20% AND news < -15% | Growth |
| `positive_news_high_risk` | news > 15% AND risk > 55% | News |
| `positive_news_negative_growth` | news > 15% AND growth < 0% | News |

Each contradiction applies **-0.08 penalty** to the tool's confidence.

---

## Confidence Scaling

### Base Confidence (Per Tool)
```python
base_confidence = min(0.35 + 0.08 × evidence_count, 0.95)

# Examples:
evidence_count=0:  conf = 0.35 (minimum)
evidence_count=8:  conf = 0.95 (maximum)
evidence_count=1:  conf = 0.43
evidence_count=4:  conf = 0.67
```

### Adjusted Confidence (After Contradictions)
```python
adjusted_confidence = base_confidence - (0.08 × num_contradictions)

# Examples:
base=0.85, contradictions=0  → adjusted = 0.85
base=0.85, contradictions=1  → adjusted = 0.77
base=0.85, contradictions=2  → adjusted = 0.69
```

---

## Test Results

### TEST 1: Equal Confidence (All Tools High Confidence)
- Risk: -0.30 (conf=0.85) → weight = 0.30
- Tone: +0.20 (conf=0.80) → weight = 0.16
- Valuation: +0.40 (conf=0.85) → weight = 0.21
- Growth: +0.30 (conf=0.75) → weight = 0.08
- News: +0.10 (conf=0.60) → weight = 0.06

**Result**: Final Score = +0.0699, Decision = NO_ACT

### TEST 2: Low Confidence Tool (News Has Only 1 Article)
- News has low confidence (0.43)
- Even though score=-0.80, effective weight = 0.10 × 0.43 = 0.04
- News negative signal is muted compared to fixed-weight approach

**Result**: Final Score = +0.0201 (less negative), Decision = NO_ACT

### TEST 3: Contradictions Reduce Confidence
- Risk detected: "strong_growth_despite_high_risk"
- Risk confidence: 0.85 → 0.77 (penalized by 0.08)
- Risk effective weight: 0.35 × 0.77 = 0.27 (reduced from 0.35)

**Result**: Final Score = -0.0807, Confidence = 0.7830

### TEST 4: Building from Raw Components
- Automatically detects contradictions from raw metrics
- Adjusts confidence before aggregation
- Provides tool_details with weighted_rows for UI transparency

### TEST 5: Fixed vs Dynamic Comparison
- **Dynamic**: Score = -0.0158, Confidence = 0.8592
- **Fixed**: Score = +0.0600, Confidence = 0.9500
- **Difference**: Dynamic gives less weight to tools with medium confidence

---

## Example: Before vs After

### Scenario: NVDA with Mixed Signals
- Risk: -0.7 (high-risk language, conf=0.90)
- Tone: +0.3 (positive tone, conf=0.85)
- Valuation: +0.5 (undervalued, conf=0.88)
- Growth: +0.4 (strong growth, conf=0.80)
- News: +0.2 (positive news, conf=0.70)

**With Fixed Weights (Old)**:
```
score = (-0.7×0.35) + (+0.3×0.20) + (+0.5×0.25) + (+0.4×0.10) + (+0.2×0.10)
      = -0.245 + 0.060 + 0.125 + 0.040 + 0.020
      = 0.000
Decision: NO_ACT
```

**With Dynamic Weights (New)**:
```
effective_weights = [0.315, 0.170, 0.220, 0.080, 0.070]
score = (-0.7×0.315 + 0.3×0.170 + 0.5×0.220 + 0.4×0.080 + 0.2×0.070) / total
      = 0.033
confidence = weighted average of 0.90, 0.85, 0.88, 0.80, 0.70 = 0.85

Decision: NO_ACT (but with more nuance)
```

---

## Integration with UI

The response now includes:

```json
{
  "hackathon_signal_score": {
    "signal_score": 0.0699,
    "confidence": 0.8121,
    "label": "HOLD",
    "component_scores": {"risk": -0.30, "tone": 0.20, ...},
    "component_confidences": {"risk": 0.85, "tone": 0.80, ...},
    "tool_details": {
      "weighted_rows": [
        {
          "name": "risk",
          "base_weight": 0.35,
          "adjusted_confidence": 0.85,
          "effective_weight": 0.30,
          "weighted_contribution": -0.0892,
          ...
        },
        ...
      ],
      "total_effective_weight": 0.81,
      "aggregation_method": "confidence_weighted_dynamic"
    }
  }
}
```

**UI Can Now**:
- Show per-tool confidence scores
- Highlight tools with low confidence
- Display effective weights (why some tools matter less)
- Explain contradictions affecting each tool

---

## Files Modified

| File | Changes |
|------|---------|
| `signal_scoring.py` | +150 lines: New classes, dynamic weighting function |
| `server.py` | +50 lines: Contradiction detection, dynamic aggregation |
| `test_dynamic_weighting.py` | +280 lines: Comprehensive test suite |

---

## Backward Compatibility

✅ **Old `compute_final_signal()` still works** (used in legacy code)
✅ **New `compute_final_signal_dynamic()` is the production path** (used in `/api/decision`)
✅ **Both return same `SignalScore` structure** (with new fields populated differently)

To revert to fixed weights, change one line in server.py:
```python
# Old (fixed weights)
score = compute_final_signal(...)

# New (dynamic weights) - CURRENT
score = compute_final_signal_dynamic(tools=tool_signals)
```

---

## Testing

Run the comprehensive test suite:
```bash
python test_dynamic_weighting.py
```

Tests cover:
- ✅ Equal confidence aggregation
- ✅ Low confidence tool muting
- ✅ Contradiction penalties
- ✅ Component building
- ✅ Fixed vs dynamic comparison

---

## Next Steps

1. **Run full decision mode tests** with real AAPL/NVDA/TSLA queries
2. **Monitor confidence distributions** to tune penalty values (currently 0.08 per contradiction)
3. **Collect user feedback** on decision quality
4. **Consider learned weights** if backtesting shows better results

---

## Design Rationale

**Why confidence-based weighting?**
- ✅ Adapts to data quality automatically
- ✅ No manual regime multipliers needed
- ✅ Contradictions have clear impact
- ✅ More Bayesian (allocate weight to quality)
- ✅ Transparent (tool_details show all calculations)

**Why per-tool contradiction detection?**
- ✅ Tool-level context (risk vs growth contradiction is different from tone vs news)
- ✅ Early penalization before aggregation
- ✅ Allows nuanced confidence adjustment
- ✅ Clearer audit trail for decisions
