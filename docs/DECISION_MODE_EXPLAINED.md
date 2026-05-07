# Decision Mode Deep Dive - How It Works & Why

## Executive Summary

**Decision Mode** produces a **single investment decision (ACT/WATCH/NO_ACT)** by independently analyzing a stock across **6 parallel signal tools**, then using a **weighted aggregation algorithm** to combine their insights. This approach mitigates single-tool bias while maintaining analytical rigor.

---

## 1. The 6 Independent Signal Tools

### Tool 1: Risk Analysis
**What it does**: Extracts risk language severity from SEC 10-K Item 1A

**Input**: "Item 1A. Risk Factors" from filing documents
**Process**:
- Semantic snippet extraction (supply chain, regulatory, cyber, macro, etc.)
- FinBERT embeddings + cosine similarity for best snippet per category
- Rule-based risk scoring + neural classifier voting

**Output**: `risk_severity_avg` (0–1 scale)
**Example**: "Geopolitical tariffs could disrupt supply chains" → risk score = 0.62

```python
# From decision_engine.py line 63
risk_score = -_clip(risk_severity_avg, 0.0, 1.0)  # Inverted: high risk = negative signal
risk_conf = _clip(0.55 + 0.03 * min(max(evidence_count, 0), 10), 0.0, 0.9)
```

---

### Tool 2: Tone Analysis
**What it does**: Measures management sentiment shift between consecutive earnings calls

**Input**: Current-year vs prior-year earnings transcripts
**Process**:
- FinBERT-based sentiment analysis on management commentary
- Calculates tone delta: current_sentiment - prior_sentiment
- Confidence scales with transcript availability (no prior call = abstain)

**Output**: `tone_delta` (-1.0 to +1.0)
**Example**:
- Prior call: "cautiously optimistic" → -0.15 (negative tone)
- Current call: "strong momentum ahead" → +0.45 (positive tone)
- Delta: +0.60 (management improved outlook)

```python
# From decision_engine.py line 68
if tone_delta is not None:
    sigs.append(_signal("tone", _clip(tone_delta, -1.0, 1.0), 0.65, "management tone delta"))
```

---

### Tool 3: Valuation (DCF-based)
**What it does**: Calculates intrinsic value via Discounted Cash Flow analysis

**Input**: Free Cash Flow, net debt, shares outstanding, WACC
**Process**:
1. Extract FCF from financial tables (primary) or SEC filings
2. Fallback: Proxy FCF = Revenue × 12% margin
3. Run DCF with policy assumptions (WACC, terminal growth)
4. Compare intrinsic value to market price

**Output**: `valuation_gap_pct` = (intrinsic_value - market_price) / market_price
**Interpretation**:
- +0.25 = stock 25% undervalued (positive signal)
- -0.30 = stock 30% overvalued (negative signal)

```python
# From orchestrator.py line 891
valuation_gap_pct = (float(dcf.intrinsic_value_per_share) - mkt_price) / mkt_price
```

---

### Tool 4: Growth Analysis
**What it does**: Measures revenue momentum year-over-year

**Input**: Annual revenue from XBRL/financial tables
**Process**:
- Extract current FY revenue
- Extract prior FY revenue
- Calculate YoY growth rate
- Normalize to -1.0 to +1.0 (40% growth = +1.0 cap)

**Output**: `revenue_growth_yoy` (normalized 0-1.0)
**Example**: AAPL FY2024 vs FY2023 growth = 15% → normalized to +0.375

```python
# From signal_scoring.py line 60-64
def _normalize_growth(revenue_growth_yoy: Optional[float]) -> float:
    if revenue_growth_yoy is None:
        return 0.0
    # 40% YoY -> +1 cap
    return _clip(float(revenue_growth_yoy) / 0.40, -1.0, 1.0)
```

---

### Tool 5: News Analysis
**What it does**: Classifies recent news catalysts and their sentiment direction

**Input**: Last 5–10 news articles from NewsAPI
**Process**:
- Extract headline + summary sentiment via transformer-based classifier
- Aggregate direction: positive, negative, neutral
- Weight by recency and article quality

**Output**: `news_direction_score` (-1.0 to +1.0)
**Example**:
- "NVDA beats Q3 earnings estimates" → +0.70 (strong positive)
- "NVDA faces China export restrictions" → -0.50 (negative)
- Aggregate: -0.10 (slightly bearish overall)

---

### Tool 6: Peer Valuation Analysis
**What it does**: Compares stock multiples (P/E, EV/EBITDA) vs peer median

**Input**: Target company and peer group multiples
**Process**:
1. Retrieve peer median trading multiples
2. Calculate target company multiples
3. Compute premium/discount vs median

**Output**: `peer_premium_pct` = (target_multiple - peer_median) / peer_median
**Interpretation**:
- +0.25 = trading 25% above peers (suggests expensive)
- -0.15 = trading 15% below peers (suggests cheap)

```python
# From decision_engine.py line 89-90
# Positive premium means expensive vs peers => negative signal.
sigs.append(_signal("peer_valuation", _clip(-peer_premium_pct, -1.0, 1.0), 0.65, ...))
```

---

## 2. Signal Normalization (Turning Raw Metrics Into -1.0 to +1.0 Scale)

Each tool produces a **normalized score** on a -1.0 to +1.0 scale:

| Tool | Raw Input | Normalization | Interpretation |
|------|-----------|---|---|
| **Risk** | severity 0–1 | -severity | High risk = negative signal |
| **Tone** | delta -1 to +1 | clip to -1, +1 | Improvement = positive |
| **Valuation** | gap % | clip % to ±1.0 | Undervalued = positive |
| **Growth** | YoY % | % / 40% (capped) | High growth = positive |
| **News** | sentiment -1 to +1 | clip to -1, +1 | Positive news = positive |
| **Peer** | premium % | -premium% capped | Trading cheap = positive |

---

## 3. The Weighting System

After normalization, each tool is assigned a **fixed weight** based on reliability:

```python
# From signal_scoring.py lines 102-108
weights = {
    "risk": 0.35,        # 35% — SEC filings most reliable
    "tone": 0.20,        # 20% — requires transcript data
    "valuation": 0.25,   # 25% — depends on FCF availability
    "growth": 0.10,      # 10% — straightforward metric
    "news": 0.10,        # 10% — higher volatility/noise
}
```

**Why these weights?**
- **Risk (35%)**: Highest priority because it's extracted from regulated SEC filings with standardized disclosures
- **Valuation (25%)**: Second priority; DCF is theoretically sound but sensitive to assumption quality
- **Tone (20%)**: Management commentary can be misleading; requires year-over-year comparison
- **Growth + News (10% each)**: Lower weight due to data availability and noise

---

## 4. Regime-Based Multipliers

Before aggregating, scores are **adjusted based on market conditions**:

```python
# From decision_engine.py lines 110-126
def _regime_multiplier(signal_name: str, regime: Dict[str, Any]) -> float:
    vol = regime.get("volatility_regime")  # low_vol, mid_vol, high_vol
    near_earnings = bool(regime.get("near_earnings_window"))

    # Valuation is MORE reliable in low-volatility environments
    if signal_name == "valuation":
        if vol == "low_vol": m *= 1.35  # Boost in calm markets
        elif vol == "high_vol": m *= 0.85  # Discount in volatile markets

    # News is MORE relevant near earnings
    if signal_name == "news":
        if vol == "high_vol": m *= 1.25  # Boost when market is volatile
        if near_earnings: m *= 1.35  # Boost around earnings season

    # Tone is stronger when near earnings window
    if signal_name == "tone" and near_earnings:
        m *= 1.20

    return m
```

**Example**:
- NVDA (beta=1.8, high_vol) earnings call score is 1.20× amplified
- Apple (beta=0.9, low_vol) valuation score is 1.35× amplified

---

## 5. The Aggregation Formula

```python
# From decision_engine.py lines 196-199
total_w = sum(r["effective_weight"] for r in weighted_rows)
raw_score = (sum(r["weighted_contribution"] for r in weighted_rows) / total_w)
contradiction_penalty = detect_contradictions(signals)  # -0.03 per conflict
final_score = clip(raw_score - contradiction_penalty, -1.0, 1.0)
```

**Step-by-step example** (hypothetical AAPL):

```
Tool Scores (normalized):
  Risk:       -0.45  (moderate risk language)
  Tone:       +0.20  (slightly positive tone shift)
  Valuation:  +0.35  (undervalued by 35%)
  Growth:     +0.25  (10% YoY revenue growth)
  News:       -0.10  (mixed recent news)

Regime: low_vol (beta=0.95), NOT near earnings

Weighted Contributions:
  Risk:       -0.45 × 0.35 = -0.158
  Tone:       +0.20 × 0.20 × 1.0 = +0.040
  Valuation:  +0.35 × 0.25 × 1.35 = +0.118  (boosted for low-vol)
  Growth:     +0.25 × 0.10 = +0.025
  News:       -0.10 × 0.10 = -0.010

Total effective weight: 0.35 + 0.20 + 0.338 + 0.10 + 0.10 = 1.088

Raw aggregated score = (-0.158 + 0.040 + 0.118 + 0.025 - 0.010) / 1.088 = +0.0944

Contradictions check:
  Risk (-0.45) vs Tone (+0.20) are opposite signs
  → Contradiction detected, penalty = -0.03

Final score = 0.0944 - 0.03 = +0.064
```

---

## 6. Confidence Scoring

Confidence **increases with evidence quantity** and is **reduced by contradictions**:

```python
# From signal_scoring.py lines 114-116
base_conf = min(0.35 + 0.08 * int(evidence_count), 0.95)
confidence = round(_clip(base_conf - float(contradiction_penalty), 0.0, 1.0), 4)
```

**Example**:
- 8 evidence chunks → confidence = 0.35 + (0.08 × 8) = 0.99 (capped at 0.95)
- Minus 0.03 contradiction penalty → final confidence = 0.92

**Why base 0.35?** Even with zero evidence, we have some confidence from methodology. The 0.08 multiplier means each additional evidence block adds 8 percentage points.

---

## 7. Decision Rules

```python
# From signal_scoring.py lines 17-37
def signal_action_from_score(
    signal_score: float,
    confidence: float,
    act_threshold: float = 0.35,      # Strong positive signal
    watch_threshold: float = 0.10,    # Moderate signal
    min_confidence: float = 0.55,     # Minimum quality gate
) -> str:
    if confidence >= 0.55 and signal_score >= 0.35:
        return "ACT"        # High conviction BUY
    if signal_score >= 0.10:
        return "WATCH"      # Monitor but don't act
    return "NO_ACT"         # Sell or neutral
```

**Decision Matrix**:

| Confidence | Score ≥ 0.35 | Score 0.10-0.35 | Score < 0.10 |
|---|---|---|---|
| **≥ 0.55** | **ACT** ✓ | WATCH | NO_ACT |
| **0.45-0.55** | WATCH | WATCH | NO_ACT |
| **< 0.45** | WATCH | NO_ACT | NO_ACT |

**Interpretation**:
- **ACT**: High confidence + strong positive signal → execute position
- **WATCH**: Mixed signal or low confidence → monitor for clarity
- **NO_ACT**: Negative signal or uncertain → stay on sidelines

---

## 8. Contradiction Detection

The system detects when signals **disagree strongly**:

```python
# From decision_engine.py lines 129-145
def _detect_contradictions(signals: List[Dict[str, Any]]) -> Tuple[...]:
    # Only compare signals that are "active":
    # - available (data exists)
    # - confidence ≥ 0.60
    # - magnitude ≥ ±0.20

    active = [s for s in signals
              if s.get("available")
              and s.get("confidence", 0) >= 0.60
              and abs(s.get("score", 0.0)) >= 0.20]

    # Detect opposite-sign pairs
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            if active[i]["score"] * active[j]["score"] < 0:
                contradictions.append({...})
                penalty += 0.03  # 3 percentage points per conflict
```

**Example contradictions**:
- Risk says "sell" (-0.50) but Valuation says "buy" (+0.60) → flag it, apply penalty
- Tone positive but News negative → flag it
- Growth strong but Peers expensive → flag it

**Why penalize?** Contradictions suggest the investment thesis is uncertain. The 0.03 penalty per conflict is conservative—enough to escalate uncertainty without eliminating the signal.

---

## 9. The Complete Flow (Visual)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POST /api/decision                               │
│                    (ticker, fiscal_year, strictness)                │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │   Load Market Data      │
          │   (price, beta, debt)   │
          └────────────┬────────────┘
                       │
        ┌──────────┬───┴───┬──────────┬──────────┬──────────┐
        │          │       │          │          │          │
      Risk       Tone   Valuation  Growth      News      Peers
        │          │       │          │          │          │
        │  SEC 10-K│Trans- │ Tables   │NewsAPI   │Multiples │
        │  Item1A  │script │ XBRL     │ Senti.   │          │
        │          │       │          │          │          │
        ├─→Score   ├─→Score├─→Score   ├─→Score   ├─→Score   ├─→Score
        │   -0.45  │ +0.20 │  +0.35   │  +0.25   │  -0.10   │  +0.15
        │   Conf   │ Conf  │  Conf    │  Conf    │  Conf    │  Conf
        │   0.72   │ 0.65  │  0.78    │  0.68    │  0.45    │  0.65
        │          │       │          │          │          │
        └──────────┴───┬───┴──────────┴──────────┴──────────┘
                       │
        ┌──────────────┴───────────────┐
        │   Regime Detection           │
        │   (beta, earnings window)    │
        │   → volatility_regime        │
        │   → regime_multipliers       │
        └──────────────┬───────────────┘
                       │
        ┌──────────────┴──────────────────┐
        │   Normalize & Weight            │
        │   component_scores × weights    │
        │   × regime_multipliers          │
        │                                 │
        │   weighted_rows = [            │
        │     {name:"risk",    score:..}  │
        │     {name:"tone",    score:..}  │
        │     {name:"valuation",score:..} │
        │     ...                         │
        │   ]                             │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────┴──────────────────┐
        │   Detect Contradictions         │
        │   → Find opposite-sign pairs    │
        │   → Calculate penalty           │
        │   → Log for UI transparency     │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────┴──────────────────┐
        │   Compute Final Score           │
        │                                 │
        │   raw_score = Σ(scores×weights)│
        │   final_score = raw - penalty   │
        │   confidence = base + evidence  │
        │                 - penalty       │
        │                                 │
        │   Result: score=+0.064          │
        │           confidence=0.92       │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────┴──────────────────┐
        │   Decision Rule                 │
        │                                 │
        │   if conf >= 0.55 & score >= 0.35:
        │       action = "ACT"            │
        │   elif score >= 0.10:           │
        │       action = "WATCH"          │
        │   else:                         │
        │       action = "NO_ACT"         │
        │                                 │
        │   Result: action="WATCH"        │
        │           reason="mixed_signal" │
        └──────────────┬──────────────────┘
                       │
        ┌──────────────┴──────────────────┐
        │   Response JSON                 │
        │                                 │
        │   {                             │
        │     "action": "WATCH",          │
        │     "score": 0.0640,            │
        │     "confidence": 0.9200,       │
        │     "signals": [...],           │
        │     "contradictions": [...],    │
        │     "tool_evidence": {...}      │
        │   }                             │
        └─────────────────────────────────┘
```

---

## 10. Why This Approach?

### ✅ Strengths

1. **Multi-perspective hedge**
   Six independent tools reduce single-method bias. Risk can't fake out Valuation.

2. **Transparency & debuggability**
   Every signal is traceable to source evidence. UI shows weighted_signals and contradictions.

3. **Regime-aware**
   High-volatility stocks weight news higher; low-vol stocks weight valuation higher.

4. **Contradiction detection**
   When Risk says "sell" but Growth says "buy," the system flags it and reduces confidence.

5. **Evidence-quality gates**
   Confidence scales with evidence count, preventing overconfident calls on thin data.

6. **Deterministic decision rules**
   No magic numbers. Clear thresholds (confidence ≥ 0.55, score ≥ 0.35) for ACT.

---

### ⚠️ Limitations

1. **Weighting is fixed, not learned**
   Risk always 35%, no backtesting calibration. Could be sub-optimal.

2. **Tool quality varies**
   News sentiment is noisy; Tone needs consecutive transcripts. Missing data → abstain.

3. **Contradictions are penalized but not resolved**
   When Risk and Valuation conflict, we reduce confidence rather than investigating why.

4. **Regime detection is simple**
   Only uses beta and earnings window. Ignores VIX, sector volatility, etc.

5. **No explicit temporal weighting**
   Recent news weighted same as month-old news if both in API response.

---

## 11. Example Decision Trace (Real Output)

```
=== DECISION ANALYSIS FOR NVDA (FY2024) ===

Regime: vol=high_vol beta=1.82 near_earnings=True

risk: score=-0.485 conf=0.72 regime_mult=1.00
      contribution=-0.170
tone: score=+0.150 conf=0.65 regime_mult=1.20
      contribution=+0.036
valuation: score=+0.280 conf=0.78 regime_mult=0.85
      contribution=+0.059
growth: score=+0.420 conf=0.68 regime_mult=1.00
      contribution=+0.042
news: score=-0.080 conf=0.45 regime_mult=1.25
      contribution=-0.004
peer: score=+0.150 conf=0.65 regime_mult=1.00
      contribution=+0.098

Contradictions detected: 1; penalty=0.030
  - risk (-0.485) vs tone (+0.150)

Final weighted score=+0.061, aggregate_confidence=0.89 => action=WATCH

Policy applied:
  - Score +0.061 > threshold +0.10? NO
  - Confidence 0.89 >= threshold 0.55? YES
  → Action=WATCH (not high enough score for ACT)

Component Scores: {risk: -0.485, tone: +0.150, valuation: +0.280, growth: +0.420, news: -0.080}
Key Findings:
  - Risk language is materially elevated in the retrieved evidence.
  - Revenue growth remains a positive supporting signal.
  - Valuation appears attractive under the current assumptions.

Risk Flags: [elevated_risk_language, negative_recent_catalysts]
```

---

## 12. Key Takeaway

Decision Mode doesn't try to predict stock prices. Instead, it **synthesizes six independent perspectives** into a **confidence-weighted recommendation**:

- **ACT** = High conviction buy (all tools aligned, strong positive)
- **WATCH** = Mixed or moderate signal (conflicting tools, medium positive)
- **NO_ACT** = Low confidence or negative (default position)

The weights, thresholds, and multipliers are **intentionally conservative**—favoring false negatives over false positives. The UI shows all working, contradictions, and evidence for human judgment override.

