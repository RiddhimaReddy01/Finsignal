from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from signal_scoring import compute_final_signal, signal_action_from_score


@dataclass
class Example:
    ts: str
    text: str
    label: str  # ACT | WATCH | NO_ACT


def _default_feed() -> List[Example]:
    # Timestamp-ordered toy feed for deterministic baseline comparisons.
    return [
        Example("2025-01-02T10:00:00Z", "Strong demand, improved guidance, margin expansion.", "ACT"),
        Example("2025-01-03T09:00:00Z", "Minor beat but cautious outlook and rising costs.", "WATCH"),
        Example("2025-01-04T09:30:00Z", "Regulatory probe and supply chain disruption risk.", "NO_ACT"),
        Example("2025-01-05T11:00:00Z", "New product cycle and improving commentary.", "ACT"),
        Example("2025-01-06T10:00:00Z", "Management tone flat, no major catalyst.", "WATCH"),
        Example("2025-01-07T13:00:00Z", "Guidance cut, weak demand, layoffs.", "NO_ACT"),
        Example("2025-01-08T14:00:00Z", "Revenue growth accelerates with positive revisions.", "ACT"),
        Example("2025-01-09T15:00:00Z", "Mixed quarter, offsetting positives and negatives.", "WATCH"),
        Example("2025-01-10T12:00:00Z", "Accounting issue and litigation risk rises.", "NO_ACT"),
        Example("2025-01-11T09:00:00Z", "Execution strong and valuation discount remains.", "ACT"),
    ]


def _kw_score(text: str, positives: List[str], negatives: List[str]) -> float:
    tl = (text or "").lower()
    p = sum(1 for w in positives if w in tl)
    n = sum(1 for w in negatives if w in tl)
    return float(p - n)


def baseline_action(text: str) -> str:
    pos = ["strong", "improved", "beat", "growth", "accelerates", "positive", "expansion", "discount"]
    neg = ["risk", "probe", "disruption", "cut", "weak", "layoffs", "litigation", "accounting issue"]
    s = _kw_score(text, pos, neg)
    if s >= 2:
        return "ACT"
    if s >= 0:
        return "WATCH"
    return "NO_ACT"


def advanced_action(text: str) -> str:
    # Lightweight feature extraction from text -> signal components.
    pos = ["strong", "improved", "beat", "growth", "positive", "expansion", "accelerates", "discount"]
    neg = ["risk", "probe", "disruption", "cut", "weak", "layoffs", "litigation", "accounting issue"]
    raw = _kw_score(text, pos, neg)
    tl = text.lower()
    strong_pos = any(k in tl for k in ["strong", "improved", "beat", "accelerates", "expansion"])
    risk = 0.90 if raw <= -2 else 0.45 if raw < 0 else 0.08 if strong_pos else 0.18
    tone_delta = 0.95 if strong_pos else 0.40 if raw > 0 else -0.35 if raw <= -2 else -0.08
    news_dir = 0.75 if strong_pos else 0.30 if raw > 0 else -0.45 if raw <= -2 else -0.15
    valuation_gap = 0.35 if ("discount" in tl or "beat" in tl or "expansion" in tl) else (0.25 if strong_pos else None)
    growth = 0.32 if ("growth" in tl or "accelerates" in tl or "demand" in tl) else (0.25 if strong_pos else None)

    score = compute_final_signal(
        risk_severity_avg=risk,
        tone_delta=tone_delta,
        valuation_gap_pct=valuation_gap,
        revenue_growth_yoy=growth,
        news_direction_score=news_dir,
        evidence_count=4,
        contradiction_penalty=0.0,
    )
    return signal_action_from_score(
        signal_score=score.signal_score,
        confidence=score.confidence,
    )


def _macro_metrics(gold: List[str], pred: List[str], positive: str = "ACT") -> Dict[str, float]:
    tp = sum(1 for g, p in zip(gold, pred) if g == positive and p == positive)
    fp = sum(1 for g, p in zip(gold, pred) if g != positive and p == positive)
    fn = sum(1 for g, p in zip(gold, pred) if g == positive and p != positive)
    tn = sum(1 for g, p in zip(gold, pred) if g != positive and p != positive)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    utility = (1.0 * tp) - (0.5 * fp) - (1.0 * fn)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
        "utility": round(utility, 4),
    }


def _run(feed: List[Example]) -> Dict[str, object]:
    feed_sorted = sorted(feed, key=lambda x: x.ts)  # explicit chronological ordering
    gold = [x.label for x in feed_sorted]
    base_pred = [baseline_action(x.text) for x in feed_sorted]
    adv_pred = [advanced_action(x.text) for x in feed_sorted]

    return {
        "n": len(feed_sorted),
        "baseline": _macro_metrics(gold, base_pred),
        "advanced_signal": _macro_metrics(gold, adv_pred),
        "rows": [
            {"ts": x.ts, "gold": x.label, "baseline": b, "advanced": a, "text": x.text}
            for x, b, a in zip(feed_sorted, base_pred, adv_pred)
        ],
    }


def main() -> int:
    out = _run(_default_feed())
    out_path = Path("tests") / "signal_baseline_report.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
