# Decision Tab - Complete Workflow Explained

## User Interface Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DECISION TAB                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Ticker Selection:  [AAPL ▼]                                │
│  Strictness:       [========>] 70%                          │
│                                                              │
│                  [RUN ANALYSIS]                             │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  SIGNAL DECISION:  ⬤ WATCH                                 │
│  Strength: 0.064   Confidence: 92%                          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                   6 SIGNAL CARDS                        │ │
│  │                                                        │ │
│  │ [Risk]  [-0.30]  [Tone]  [+0.20]  [Valuation] [+0.40]│ │
│  │ conf:92% conf:80%       conf:85%                      │ │
│  │                                                        │ │
│  │ [Growth][+0.30]  [News]  [+0.10]  [Peers]    [+0.15] │ │
│  │ conf:75% conf:60%       conf:65%                      │ │
│  │                                                        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  KEY FINDINGS:                                               │
│  • Risk language is materially elevated                      │
│  • Valuation appears attractive                             │
│  • Revenue growth remains positive                          │
│                                                              │
│  [Click tool card to see evidence →]                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: User Input

### What the user controls:

**Ticker Selection**
- Dropdown: AAPL, NVDA, TSLA, META, GOOGL
- Default: AAPL
- Used to fetch company data

**Strictness Slider** (0-100%)
- **30%**: Lenient - accepts weaker evidence
  - Lower source quality threshold
  - More evidence blocks used
  - Higher tolerance for contradictions

- **70%**: Balanced (DEFAULT)
  - Standard evidence requirements
  - Normal filtering

- **95%**: Strict - requires strong evidence
  - Only SEC filings count
  - Minimum evidence blocks
  - Zero tolerance for contradictions

**Impact**: Strictness affects:
- Which evidence blocks are accepted
- Confidence calculation
- Contradiction penalties
- Final decision thresholds

---

## Step 2: Frontend Makes Request

```javascript
// From FinSightTerminal.jsx

const response = await fetch(`${API_BASE}/api/decision`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    ticker: 'AAPL',      // User selected
    fiscal_year: 2024,   // Auto: current year
    strictness: 70       // User set via slider
  })
});
```

---

## Step 3: Backend - The Decision Analysis Pipeline

### High-Level Flow

```
POST /api/decision {ticker, fiscal_year, strictness}
    ↓
TOOL 1: Risk Analysis
    ├─ Extract SEC 10-K Item 1A (Risk Factors)
    ├─ Identify risk categories
    ├─ Score severity
    └─ Evidence: 8-15 blocks
    ↓
TOOL 2: Tone Analysis
    ├─ Fetch earnings transcripts (current & prior year)
    ├─ Compare sentiment (delta)
    ├─ Fallback: filing MD&A + news
    └─ Evidence: 6-12 blocks
    ↓
TOOL 3: Valuation
    ├─ Extract FCF from tables
    ├─ Run DCF model
    ├─ Compare intrinsic vs market price
    └─ Evidence: 7-10 blocks
    ↓
TOOL 4: Growth
    ├─ Extract revenue (FY, FY-1)
    ├─ Calculate YoY growth
    └─ Evidence: 5-8 blocks
    ↓
TOOL 5: News
    ├─ Fetch recent news articles (last 30 days)
    ├─ Classify sentiment
    └─ Evidence: 3-6 blocks
    ↓
TOOL 6: Peer Valuation
    ├─ Get peer multiples
    ├─ Compare premium/discount
    └─ Evidence: 4-6 blocks
    ↓
AGGREGATE with Dynamic Weighting
    ├─ Analyze evidence quality
    ├─ Detect contradictions
    ├─ Calculate per-tool confidence
    ├─ Compute effective weights
    └─ Aggregate score & confidence
    ↓
DECISION RULE
    ├─ If conf ≥ 0.55 AND score ≥ 0.35 → ACT
    ├─ Else if score ≥ 0.10 → WATCH
    └─ Else → NO_ACT
    ↓
Return JSON response
```

---

## Step 4: Detailed Tool Execution

### TOOL 1: Risk Analysis (server.py:650-713)

```python
# Extract risk signals from 10-K Item 1A
risk_signals = extract_risk_signals_with_diagnostics(
    item1a_text=filing_data,
    use_advanced_model=True
)

# Categories detected:
# - supply_chain
# - regulatory
# - geopolitical
# - macro
# - cyber
# - competition
# - litigation
# - liquidity
# - customer_concentration

# Score calculation:
risk_avg = average of top 3 risk severities
risk_score = -risk_avg  # Negative: high risk bad for investment

# Evidence collection:
for each risk category:
    - Extract best snippet using FinBERT embeddings
    - Create evidence block with source attribution
    - Add to tool_evidence["risk"]

# Result:
tools_used["risk"] = {
    "score": -0.65,  # High risk
    "factors": [
        {"category": "supply_chain", "severity": 0.70, "reasoning": "..."},
        {"category": "regulatory", "severity": 0.60, "reasoning": "..."},
        ...
    ],
    "metadata": {...}
}
```

### TOOL 2: Tone Analysis (server.py:715-856)

```python
# Get current and prior earnings transcripts
current_text, prior_text = tc.get_current_and_prior_text(
    ticker=ticker,
    current_period="FY2024",
    prior_period="FY2023"
)

# Analyze sentiment
current_sentiment = analyze_tone(current_text)  # FinBERT
prior_sentiment = analyze_tone(prior_text)      # FinBERT

# Calculate delta
tone_delta = current_sentiment - prior_sentiment
# Example: 0.45 - 0.25 = +0.20 (improved tone)

# Fallback sources if no transcript pair:
if not prior_text:
    tone_delta = blend(
        filing_mda_tone (0.65 weight),
        news_tone (0.20 weight),
        press_release_tone (0.15 weight)
    )

# Result:
tools_used["tone"] = {
    "score": +0.20,  # Positive tone shift
    "factors": {
        "current_sentiment": 0.45,
        "prior_sentiment": 0.25,
        "delta": 0.20,
        "direction": "Positive"
    }
}
```

### TOOL 3: Valuation (server.py:876-1065)

```python
# Step 1: Get market data
price = market_data["price"]  # Current stock price
market_cap = market_data["market_cap"]
net_debt = market_data["net_debt"]

# Step 2: Extract FCF
fcf = extract_table_metric(
    context=filing_tables,
    metric="free_cash_flow"
) or extract_from_text(...)

# Fallback: FCF proxy = Revenue × 12% margin
if not fcf:
    revenue = extract_metric("revenue")
    fcf = revenue * 0.12

# Step 3: Run DCF
dcf = run_dcf(
    last_fcf=fcf,
    net_debt=net_debt,
    shares_outstanding=shares,
    assumptions=policy_assumptions  # WACC, terminal growth, etc.
)

# Step 4: Calculate gap
intrinsic_value = dcf.intrinsic_value_per_share
valuation_gap_pct = (intrinsic_value - price) / price
# Example: (150 - 120) / 120 = 0.25 (25% undervalued)

# Result:
tools_used["valuation"] = {
    "score": +0.25,  # Undervalued
    "valuation": {
        "method": "DCF",
        "intrinsic_value": 150.00,
        "market_price": 120.00,
        "gap_pct": 0.25,
        "assumptions": {...}
    }
}
```

### TOOL 4: Growth (server.py:1068-1152)

```python
# Extract revenue for current and prior year
current_revenue = extract_metric(
    metric="revenue",
    fiscal_year=2024
)
prior_revenue = extract_metric(
    metric="revenue",
    fiscal_year=2023
)

# Calculate YoY growth
revenue_growth_yoy = (current_revenue - prior_revenue) / prior_revenue
# Example: (100B - 85B) / 85B = 0.176 (17.6% growth)

# Normalize to -1.0 to +1.0 scale (40% = max)
growth_normalized = growth_yoy / 0.40
# 17.6% → 0.44 (capped at 1.0)

# Result:
tools_used["growth"] = {
    "score": +0.44,  # Strong growth
    "factors": {
        "current_revenue": 100_000_000_000,
        "prior_revenue": 85_000_000_000,
        "yoy_growth": 0.176,
        "normalized": 0.44
    }
}
```

### TOOL 5: News (server.py:1154-1280)

```python
# Fetch recent news
news_articles = NewsIngestionClient().fetch_recent_news(
    ticker="AAPL",
    max_results=10,
    days=30
)

# Classify sentiment of each article
sentiments = []
for article in news_articles:
    sentiment = classify_news_sentiment(article.title, article.summary)
    # Returns: -1.0 (very negative) to +1.0 (very positive)
    sentiments.append(sentiment)

# Aggregate
avg_news_score = mean(sentiments)
# Example: [+0.7, +0.3, -0.2, +0.5, +0.4] → avg = +0.34

# Result:
tools_used["news"] = {
    "score": +0.34,  # Positive recent news
    "factors": {
        "articles": 5,
        "sentiments": [0.7, 0.3, -0.2, 0.5, 0.4],
        "average": 0.34,
        "headlines": [
            "AAPL beats earnings estimates",
            "iPhone sales strong",
            ...
        ]
    }
}
```

### TOOL 6: Peer Valuation (server.py:1282-1494)

```python
# Get peer group (e.g., for AAPL: MSFT, GOOGL, TSLA)
peers = get_peer_group(ticker)

# Get multiples for all companies
peer_multiples = {
    peer: {
        "pe_ratio": ...,
        "ev_ebitda": ...,
        "price_to_sales": ...
    }
    for peer in peers
}

# Calculate median
median_pe = median([m["pe_ratio"] for m in peer_multiples.values()])

# Compare target
target_pe = price / eps
peer_premium_pct = (target_pe - median_pe) / median_pe
# Example: (28 - 25) / 25 = 0.12 (12% premium)

# Invert: expensive = negative signal
peer_valuation_score = -peer_premium_pct
# 12% premium → -0.12 (slightly overvalued vs peers)

# Result:
tools_used["peer"] = {
    "score": -0.12,
    "factors": {
        "target_pe": 28.0,
        "peer_median_pe": 25.0,
        "premium_pct": 0.12,
        "peers_used": ["MSFT", "GOOGL", "TSLA"]
    }
}
```

---

## Step 5: Aggregate with Dynamic Weighting

```python
# Collect all tool signals
tool_signals = {
    "risk": ToolSignal(score=-0.65, confidence=0.85, evidence=10, contradictions=[]),
    "tone": ToolSignal(score=+0.20, confidence=0.80, evidence=6, contradictions=[]),
    "valuation": ToolSignal(score=+0.25, confidence=0.88, evidence=8, contradictions=[]),
    "growth": ToolSignal(score=+0.44, confidence=0.75, evidence=5, contradictions=[]),
    "news": ToolSignal(score=+0.34, confidence=0.60, evidence=3, contradictions=[]),
    "peer": ToolSignal(score=-0.12, confidence=0.70, evidence=4, contradictions=[]),
}

# Dynamic weighting:
base_weights = {
    "risk": 0.35,
    "tone": 0.20,
    "valuation": 0.25,
    "growth": 0.10,
    "news": 0.10,
    "peer": 0.00,  # Fallback tool
}

# Calculate effective weights = base × confidence
effective_weights = {
    "risk": 0.35 × 0.85 = 0.298,
    "tone": 0.20 × 0.80 = 0.160,
    "valuation": 0.25 × 0.88 = 0.220,
    "growth": 0.10 × 0.75 = 0.075,
    "news": 0.10 × 0.60 = 0.060,
}

# Aggregate score
numerator = (-0.65×0.298) + (0.20×0.160) + (0.25×0.220) + (0.44×0.075) + (0.34×0.060)
         = -0.194 + 0.032 + 0.055 + 0.033 + 0.020
         = -0.054

denominator = 0.298 + 0.160 + 0.220 + 0.075 + 0.060 = 0.813

final_score = -0.054 / 0.813 = -0.066

# Aggregate confidence (weighted average of tool confidences)
agg_confidence = (0.298×0.85 + 0.160×0.80 + 0.220×0.88 + 0.075×0.75 + 0.060×0.60) / 0.813
              = 0.802
```

---

## Step 6: Apply Decision Rule

```python
final_score = -0.066      # Slightly negative
confidence = 0.802        # High confidence

# Decision rule:
if confidence >= 0.55 AND final_score >= 0.35:
    action = "ACT"           # Strong BUY
elif final_score >= 0.10:
    action = "WATCH"         # Monitor
else:
    action = "NO_ACT"        # Neutral/Hold

# Result: final_score (-0.066) < 0.10 → NO_ACT
```

---

## Step 7: Format Response

```json
{
  "ok": true,
  "action": "answer",
  "mode": "decision_analysis",
  "hackathon_signal_decision": {
    "action": "NO_ACT",
    "policy": "ACT if confidence>=0.55 AND score>=0.35; WATCH if score>=0.10; else NO_ACT"
  },
  "hackathon_signal_score": {
    "signal_score": -0.066,
    "confidence": 0.802,
    "label": "CAUTIOUS",
    "component_scores": {
      "risk": -0.65,
      "tone": 0.20,
      "valuation": 0.25,
      "growth": 0.44,
      "news": 0.34
    },
    "component_confidences": {
      "risk": 0.85,
      "tone": 0.80,
      "valuation": 0.88,
      "growth": 0.75,
      "news": 0.60
    },
    "key_findings": [
      "Risk language is materially elevated in the retrieved evidence.",
      "Management tone improved relative to the prior comparison period.",
      "Valuation appears attractive under the current assumptions.",
      "Revenue growth remains a positive supporting signal.",
      "Recent news flow is directionally supportive."
    ],
    "risk_flags": ["elevated_risk_language"]
  },
  "tool_evidence": {
    "risk": [
      {"text": "Supply chain risk HIGH", "source": "SEC 10-K", ...},
      ...
    ],
    "tone": [...],
    "valuation": [...],
    ...
  },
  "tools_used": {
    "risk": {...},
    "tone": {...},
    "valuation": {...},
    "growth": {...},
    "news": {...}
  }
}
```

---

## Step 8: Frontend Display

### The Decision Card
```
┌─────────────────────────────────────┐
│      SIGNAL DECISION: NO_ACT         │
│                                      │
│  Final Score:  -0.066  (CAUTIOUS)   │
│  Confidence:   80.2%   (HIGH)       │
│                                      │
│  Recommendation: Hold / Watch       │
│  (Don't initiate new position)      │
└─────────────────────────────────────┘
```

### The Tool Cards
```
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│   RISK TOOL    │  │   TONE TOOL    │  │ VALUATION TOOL │
│ Score: -0.65   │  │ Score: +0.20   │  │ Score: +0.25   │
│ Conf:  85%     │  │ Conf:  80%     │  │ Conf:  88%     │
│ Weight: 29.8%  │  │ Weight: 16.0%  │  │ Weight: 22.0%  │
│                │  │                │  │                │
│ HIGH RISK      │  │ IMPROVING TONE │  │ UNDERVALUED    │
└────────────────┘  └────────────────┘  └────────────────┘

┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  GROWTH TOOL   │  │   NEWS TOOL    │  │  PEER TOOL     │
│ Score: +0.44   │  │ Score: +0.34   │  │ Score: -0.12   │
│ Conf:  75%     │  │ Conf:  60%     │  │ Conf:  70%     │
│ Weight:  7.5%  │  │ Weight:  6.0%  │  │ Weight:  0.0%  │
│                │  │                │  │                │
│ STRONG GROWTH  │  │ POSITIVE NEWS  │  │ 12% PREMIUM    │
└────────────────┘  └────────────────┘  └────────────────┘
```

### Click Tool Card to See Evidence
```
RISK TOOL - Supply Chain Risk
────────────────────────────────
Evidence Block 1:
"Supply chain disruption represents material risk to operations..."
Source: SEC Form 10-K, Item 1A
Date: January 2024
Quality: SEC Filing (0.95)

Evidence Block 2:
"Geopolitical tensions may limit sourcing options..."
Source: SEC Form 10-K, Item 1A
Date: January 2024
Quality: SEC Filing (0.95)

[More evidence blocks...]

Analysis Summary:
- Base Confidence: 0.85 (from evidence quality & coherence)
- Internal Contradictions: None detected
- Final Confidence: 0.85
```

---

## Complete Timeline

```
User clicks "Run Analysis"
    ↓ [50ms] Frontend request
Server receives /api/decision
    ↓ [100ms] Load market data
    ↓ [200ms] Risk analysis (SEC extraction)
    ↓ [300ms] Tone analysis (transcript analysis)
    ↓ [400ms] Valuation (DCF model)
    ↓ [500ms] Growth extraction
    ↓ [600ms] News classification
    ↓ [700ms] Peer analysis
    ↓ [800ms] Dynamic weighting & aggregation
    ↓ [900ms] Format JSON response
Frontend receives response
    ↓ [1000ms] Display decision card
    ↓ [1100ms] Display 6 tool cards
    ↓ [1200ms] Ready for user interaction

Total: ~2-3 seconds
```

---

## Key Decision Drivers

**Why was the decision "NO_ACT"?**

1. **Risk dominates (29.8% weight)**
   - Score: -0.65 (high risk)
   - This is the strongest signal, pulling the decision negative

2. **Other tools are positive but weak**
   - Tone: +0.20 (modest improvement)
   - Valuation: +0.25 (attractive but 22% weight)
   - Growth: +0.44 (strong but only 7.5% weight)
   - News: +0.34 (positive but only 6% weight)

3. **Net result**
   - Positive signals can't overcome Risk's dominance
   - Final score: -0.066 (negative, below +0.10 threshold)
   - Decision: NO_ACT (don't buy)

**What would change to "WATCH"?**
- Risk score improves to -0.40 (medium risk)
- Then: final_score = +0.04 → WATCH

**What would change to "ACT"?**
- Risk score improves to -0.10 (low risk)
- OR news/valuation confidence increases
- Then: final_score = +0.25, confidence = 0.85 → ACT

---

## Summary

The Decision Tab workflow is:

1. **User Input** → Ticker + Strictness
2. **6 Independent Tools** → Run in parallel, each producing a score + evidence
3. **Evidence Quality Analysis** → Measure source quality, coherence, recency
4. **Dynamic Weighting** → Tool's influence = base_weight × confidence
5. **Aggregate** → Weighted average of all tool scores
6. **Decision Rule** → Apply thresholds (ACT/WATCH/NO_ACT)
7. **Display** → Show decision + 6 tools + evidence

The key insight: **A tool with low-confidence evidence gets less influence**, even if its raw score is extreme. This prevents overconfidence in weak signals.
