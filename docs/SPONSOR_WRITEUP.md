# Finding Signal in Noise: Applied NLP (Sponsor Write-Up)

## 1) Assumptions
- Input documents are consumed in timestamp order from the official feed/replay.
- Decision-time cutoff is enforced through `decision_time` (ISO timestamp) so retrieval does not use future news.
- Final decision is deterministic and rule-based (not free-form LLM control).

## 2) What We Built
- End-to-end pipeline: retrieval -> verification -> signal scoring -> explicit decision action.
- Signal output:
  - Numeric: `signal_score` in `[-1, 1]`
  - Categorical: `BUY/HOLD/CAUTIOUS/AVOID`
  - Action policy: `ACT/WATCH/NO_ACT`
- UI demo:
  - Query input, execution trace while running, evidence explorer tabs, source preview drawer, reasoning panel, and signal panel.

## 3) Explicit Decision Rule
- Implemented in `signal_scoring.signal_action_from_score(...)`.
- Policy:
  - `ACT` if `confidence >= 0.55` and `signal_score >= 0.35`
  - `WATCH` if `signal_score >= 0.10`
  - Else `NO_ACT`

## 4) Leakage Controls
- `decision_time` is plumbed through orchestrator and news ingestion path.
- News requests apply both `from` and `to` bounds relative to `decision_time`.
- Cache keys include `as_of/decision_time` to avoid serving future data for earlier replay timestamps.

## 5) Baseline vs. Signal Comparison
- Baseline: keyword-rule classifier.
- Advanced: weighted multi-signal scorer + deterministic action policy.
- Script: `tests/signal_baseline_eval.py`
- Output artifact: `tests/signal_baseline_report.json`
- Metrics reported:
  - Precision / Recall / F1
  - False alarm rate
  - Utility (TP, FP, FN weighted objective)

## 6) What Failed / Limitations
- Live external APIs can fail/rate-limit; cache and fallback paths are used but not perfect.
- Some valuation/relative-valuation cases require fallback proxies when denominator evidence is sparse.
- Label quality for evaluation depends on dataset labels; a stronger sponsor-provided ground truth improves rigor.

## 7) What Worked
- Verification gating and strictness scaling reduced unsupported answers.
- Multi-source signal blend (risk, tone, valuation, growth, news) is more informative than single-keyword rules.
- Cached API contexts significantly reduce redundant calls and improve repeat-run stability.

## 8) Interesting Failure Cases
- Conflicting short-term news vs. longer-term filing risk language can produce WATCH instead of ACT.
- Sparse transcript coverage weakens tone delta reliability.
- Ambiguous ticker queries can require clarify/abstain despite available raw text.

## 9) Demo Steps
1. Run UI: `python -m streamlit run userinterface.py --server.port 8502`
2. Ask a query (e.g., valuation/risk/comparative).
3. Observe:
   - execution trace (during run),
   - answer + signal recommendation/action,
   - evidence tabs + reasoning panel.
4. Run baseline comparison:
   - `python tests/signal_baseline_eval.py`
   - inspect `tests/signal_baseline_report.json`.
