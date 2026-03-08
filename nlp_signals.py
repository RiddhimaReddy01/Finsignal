from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# ============================================================
# Data classes
# ============================================================

@dataclass
class ToneSignal:
    tone_score: float
    label: str
    positive_prob: float
    neutral_prob: float
    negative_prob: float
    hedge_hits: int
    confidence_score: float


@dataclass
class RiskSignal:
    category: str
    severity: float
    count: int
    snippets: List[str] = field(default_factory=list)


@dataclass
class NewsCatalyst:
    article_id: str
    title: str
    source_name: str
    published_at: str
    direction: str
    score: float
    rationale: str


# ============================================================
# Finance-domain model wrapper
# ============================================================

class FinBERTToneAnalyzer:
    """
    Finance-specific tone analyzer.

    Default model:
      yiyanghkust/finbert-tone

    Notes:
    - Runs on CPU by default if CUDA is not available.
    - Keep a single shared instance in app/runtime if possible.
    """

    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        device: Optional[int] = None,
        max_chars: int = 3500,
    ):
        self.model_name = model_name
        self.max_chars = max_chars

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        self.pipe = pipeline(
            task="text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True,
            truncation=True,
            device=device,
        )

    @staticmethod
    def _normalize_label(label: str) -> str:
        x = str(label).strip().lower()
        if "positive" in x or x == "1":
            return "positive"
        if "negative" in x or x == "2":
            return "negative"
        return "neutral"

    def predict(self, text: str) -> ToneSignal:
        text = str(text or "").strip()
        if not text:
            return ToneSignal(
                tone_score=0.0,
                label="neutral",
                positive_prob=0.0,
                neutral_prob=1.0,
                negative_prob=0.0,
                hedge_hits=0,
                confidence_score=0.0,
            )

        clean = text[: self.max_chars]
        scores = self.pipe(clean)[0]

        probs = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        for row in scores:
            probs[self._normalize_label(row["label"])] = float(row["score"])

        tone_score = probs["positive"] - probs["negative"]
        label = max(probs, key=probs.get)

        hedge_hits = count_hedge_words(clean)

        # confidence reduced by excessive hedging
        hedge_penalty = min(hedge_hits / 30.0, 0.75)
        confidence = max(probs.values()) * max(0.25, 1.0 - hedge_penalty)

        return ToneSignal(
            tone_score=round(float(tone_score), 4),
            label=label,
            positive_prob=round(probs["positive"], 4),
            neutral_prob=round(probs["neutral"], 4),
            negative_prob=round(probs["negative"], 4),
            hedge_hits=int(hedge_hits),
            confidence_score=round(float(confidence), 4),
        )


@lru_cache(maxsize=1)
def get_default_tone_analyzer() -> FinBERTToneAnalyzer:
    """
    Shared singleton-style analyzer to avoid reloading model repeatedly.
    """
    return FinBERTToneAnalyzer()


# ============================================================
# Deterministic finance helpers
# ============================================================

HEDGE_PATTERNS = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bcan\b",
    r"\bapproximately\b",
    r"\bgenerally\b",
    r"\bbelieve\b",
    r"\bexpect\b",
    r"\bintend\b",
    r"\bsubject to\b",
    r"\bestimate\b",
    r"\bpotentially\b",
    r"\buncertain\b",
    r"\buncertainty\b",
]

RISK_PATTERNS: Dict[str, List[str]] = {
    "supply_chain": [
        r"\bsupply chain\b",
        r"\bsupplier\b",
        r"\bmanufacturing\b",
        r"\bcomponent shortage\b",
        r"\bprocurement\b",
    ],
    "regulatory": [
        r"\bregulat",
        r"\bcompliance\b",
        r"\binvestigation\b",
        r"\blaw\b",
        r"\blegal proceeding\b",
    ],
    "competition": [
        r"\bcompetition\b",
        r"\bcompetitor\b",
        r"\bmarket share\b",
        r"\bcompetitive pressure\b",
    ],
    "macro": [
        r"\brecession\b",
        r"\binflation\b",
        r"\binterest rate\b",
        r"\bmacro\b",
        r"\beconomic slowdown\b",
    ],
    "geopolitical": [
        r"\bgeopolitical\b",
        r"\bsanction\b",
        r"\btariff\b",
        r"\bexport control\b",
        r"\btrade restriction\b",
    ],
    "cyber": [
        r"\bcyber\b",
        r"\bsecurity breach\b",
        r"\battack\b",
        r"\bdata breach\b",
        r"\binformation security\b",
    ],
    "liquidity": [
        r"\bliquidity\b",
        r"\bdebt\b",
        r"\bcash flow\b",
        r"\bcapital resources\b",
        r"\bcredit facility\b",
    ],
    "customer_concentration": [
        r"\bcustomer concentration\b",
        r"\bfew large customers\b",
        r"\bmajor customer\b",
        r"\bconcentration of customers\b",
    ],
    "litigation": [
        r"\blitigation\b",
        r"\blawsuit\b",
        r"\blegal claim\b",
        r"\bsettlement\b",
    ],
}

POSITIVE_HINT_WORDS = {
    "strong",
    "improved",
    "improving",
    "accelerating",
    "growth",
    "record",
    "confident",
    "resilient",
    "momentum",
    "opportunity",
    "expand",
    "healthy",
    "beat",
    "outperform",
    "tailwind",
}

NEGATIVE_HINT_WORDS = {
    "risk",
    "uncertain",
    "decline",
    "weakness",
    "pressure",
    "challenging",
    "slowdown",
    "exposure",
    "disruption",
    "litigation",
    "regulation",
    "loss",
    "volatile",
    "headwind",
    "constrained",
    "geopolitical",
    "supply chain",
}


def split_sentences(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def count_hedge_words(text: str) -> int:
    t = str(text or "").lower()
    return sum(len(re.findall(p, t)) for p in HEDGE_PATTERNS)


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z\-']+", str(text or "").lower())


def _fallback_tone_from_words(text: str) -> ToneSignal:
    toks = _simple_tokenize(text)
    if not toks:
        return ToneSignal(
            tone_score=0.0,
            label="neutral",
            positive_prob=0.0,
            neutral_prob=1.0,
            negative_prob=0.0,
            hedge_hits=0,
            confidence_score=0.0,
        )

    pos = sum(1 for t in toks if t in POSITIVE_HINT_WORDS)
    neg = sum(1 for t in toks if t in NEGATIVE_HINT_WORDS)
    hedge = count_hedge_words(text)

    raw = (pos - neg) / max(len(toks) ** 0.5, 1.0)
    raw = max(-1.0, min(1.0, raw))

    if raw > 0.08:
        label = "positive"
    elif raw < -0.08:
        label = "negative"
    else:
        label = "neutral"

    # heuristic pseudo-probs
    mag = min(abs(raw), 1.0)
    if label == "positive":
        positive_prob = 0.45 + 0.45 * mag
        negative_prob = 0.10
        neutral_prob = max(0.0, 1.0 - positive_prob - negative_prob)
    elif label == "negative":
        negative_prob = 0.45 + 0.45 * mag
        positive_prob = 0.10
        neutral_prob = max(0.0, 1.0 - positive_prob - negative_prob)
    else:
        neutral_prob = 0.70
        positive_prob = 0.15
        negative_prob = 0.15

    conf = max(0.2, 1.0 - min(hedge / 25.0, 0.8)) * 0.65

    return ToneSignal(
        tone_score=round(float(raw), 4),
        label=label,
        positive_prob=round(float(positive_prob), 4),
        neutral_prob=round(float(neutral_prob), 4),
        negative_prob=round(float(negative_prob), 4),
        hedge_hits=int(hedge),
        confidence_score=round(float(conf), 4),
    )


# ============================================================
# Public API
# ============================================================

def analyze_tone(
    text: str,
    analyzer: Optional[FinBERTToneAnalyzer] = None,
    use_fallback_on_error: bool = True,
) -> ToneSignal:
    """
    Finance-aware tone analysis.

    Prefers FinBERT. Falls back to a lightweight heuristic if model inference fails.
    """
    text = str(text or "").strip()
    if not text:
        return _fallback_tone_from_words(text)

    try:
        analyzer = analyzer or get_default_tone_analyzer()
        return analyzer.predict(text)
    except Exception:
        if use_fallback_on_error:
            return _fallback_tone_from_words(text)
        raise


def compare_tone(
    current_text: str,
    prior_text: str,
    analyzer: Optional[FinBERTToneAnalyzer] = None,
) -> Dict[str, Any]:
    cur = analyze_tone(current_text, analyzer=analyzer)
    prev = analyze_tone(prior_text, analyzer=analyzer)

    delta = round(cur.tone_score - prev.tone_score, 4)
    if delta > 0.05:
        direction = "improved"
    elif delta < -0.05:
        direction = "worsened"
    else:
        direction = "stable"

    return {
        "current": cur.__dict__,
        "prior": prev.__dict__,
        "delta": delta,
        "direction": direction,
    }


def extract_risk_signals(text: str, max_snippets_per_category: int = 3) -> List[RiskSignal]:
    """
    Deterministic risk-category extraction from filing text, especially Item 1A.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    scores: Dict[str, Dict[str, Any]] = {}

    for sent in sentences:
        s = sent.lower()
        for category, patterns in RISK_PATTERNS.items():
            hits = sum(1 for p in patterns if re.search(p, s))
            if hits <= 0:
                continue

            rec = scores.setdefault(category, {"count": 0, "snippets": []})
            rec["count"] += hits

            if len(rec["snippets"]) < max_snippets_per_category:
                rec["snippets"].append(sent[:320])

    out: List[RiskSignal] = []
    total_sentences = max(len(sentences), 1)

    for category, rec in scores.items():
        density = rec["count"] / total_sentences
        severity = min(1.0, math.log1p(rec["count"]) / 2.5 + density)

        out.append(
            RiskSignal(
                category=category,
                severity=round(float(severity), 4),
                count=int(rec["count"]),
                snippets=list(rec["snippets"]),
            )
        )

    out.sort(key=lambda x: (x.severity, x.count), reverse=True)
    return out


def detect_material_change(
    current_text: str,
    prior_text: str,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Lightweight textual change detector.
    """
    cur_sents = set(split_sentences(current_text))
    prev_sents = set(split_sentences(prior_text))

    added = list(cur_sents - prev_sents)
    removed = list(prev_sents - cur_sents)

    if keywords:
        kw = [k.lower() for k in keywords]
        added = [s for s in added if any(k in s.lower() for k in kw)]
        removed = [s for s in removed if any(k in s.lower() for k in kw)]

    return {
        "n_added": len(added),
        "n_removed": len(removed),
        "top_added": added[:5],
        "top_removed": removed[:5],
    }


def classify_news_catalysts(articles: List[Dict[str, Any]]) -> List[NewsCatalyst]:
    """
    Finance-aware news direction labeling.
    Uses title + description tone as a proxy.
    """
    out: List[NewsCatalyst] = []

    for art in articles:
        title = str(art.get("title") or "")
        desc = str(art.get("description") or "")
        source_name = str(art.get("source_name") or art.get("source") or "unknown")
        published_at = str(art.get("published_at") or "")
        article_id = str(art.get("article_id") or "")

        text = f"{title}. {desc}".strip()
        tone = analyze_tone(text)

        if tone.tone_score > 0.08:
            direction = "positive"
        elif tone.tone_score < -0.08:
            direction = "negative"
        else:
            direction = "neutral"

        rationale = (
            f"label={tone.label}, tone={tone.tone_score}, "
            f"pos={tone.positive_prob}, neu={tone.neutral_prob}, "
            f"neg={tone.negative_prob}, hedge={tone.hedge_hits}"
        )

        out.append(
            NewsCatalyst(
                article_id=article_id,
                title=title,
                source_name=source_name,
                published_at=published_at,
                direction=direction,
                score=float(tone.tone_score),
                rationale=rationale,
            )
        )

    out.sort(key=lambda x: abs(x.score), reverse=True)
    return out