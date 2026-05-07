from __future__ import annotations

import json
import logging
import os
import re
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertTokenizer, pipeline
except ImportError:
    torch = None # type: ignore
    pass

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_FINBERT = "ProsusAI/finbert"
DEFAULT_FALLBACK_TONE_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


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


@dataclass
class RiskSentenceLabel:
    sentence: str
    categories: List[str]
    rule_hits: int
    classifier_risk_prob: float
    hedge_hits: int


@dataclass
class RiskDiagnostics:
    sentence_labels: List[RiskSentenceLabel]
    category_rule_score: Dict[str, float]
    category_classifier_score: Dict[str, float]
    category_calibrated_score: Dict[str, float]
    contradictions: List[Dict[str, Any]]
    calibration_profile: Dict[str, Any]


# ============================================================
# Finance-domain model wrapper
# ============================================================

class FinBERTToneAnalyzer:
    """
    [DEPRECATED] Finance-specific tone analyzer using local transformers.
    Use LLMToneAnalyzer for better reliability in non-CUDA environments.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_PRIMARY_FINBERT,
        device: Optional[int] = None,
        max_chars: int = 3500,
    ):
        self.model_name = model_name
        self.max_chars = max_chars

        if torch is None:
            raise ImportError("transformers/torch not installed for FinBERTToneAnalyzer")

        # Some Windows/offline environments cannot build the fast tokenizer backend.
        # Fall back to slow tokenizer to keep the signal layer functional.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            except Exception:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
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


class LLMToneAnalyzer:
    """
    Tone analyzer using LLM (Gemini) for robust sentiment extraction.
    """
    def __init__(self, llm_client: Any):
        self.llm_client = llm_client

    def predict(self, text: str) -> ToneSignal:
        if not text.strip():
            return ToneSignal(0.0, "neutral", 0.0, 1.0, 0.0, 0, 0.0)
        
        system_prompt = (
            "Analyze the financial tone of the following text. "
            "Return JSON with: {\"tone_score\": float (-1 to 1), \"label\": \"positive\"|\"neutral\"|\"negative\", "
            "\"positive_prob\": float, \"neutral_prob\": float, \"negative_prob\": float, \"confidence\": float}"
        )
        user_prompt = f"Text: {text[:4000]}"
        
        try:
            res_text, _ = self.llm_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name="small"
            )
            data = json.loads(res_text)
            return ToneSignal(
                tone_score=round(float(data.get("tone_score", 0.0)), 4),
                label=data.get("label", "neutral"),
                positive_prob=round(float(data.get("positive_prob", 0.0)), 4),
                neutral_prob=round(float(data.get("neutral_prob", 1.0)), 4),
                negative_prob=round(float(data.get("negative_prob", 0.0)), 4),
                hedge_hits=count_hedge_words(text),
                confidence_score=round(float(data.get("confidence", 0.5)), 4)
            )
        except Exception as e:
            logger.warning("LLMToneAnalyzer failed: %s", e)
            return _fallback_tone_from_words(text)

@lru_cache(maxsize=4)
def get_tone_analyzer(model_name: str) -> FinBERTToneAnalyzer:
    logger.warning("get_tone_analyzer is deprecated. Use LLMToneAnalyzer.")
    return FinBERTToneAnalyzer(model_name=model_name)


@lru_cache(maxsize=1)
def get_default_tone_analyzer() -> FinBERTToneAnalyzer:
    """
    Shared singleton-style analyzer to avoid reloading model repeatedly.
    Tries a prioritized model list and returns the first loadable analyzer.
    """
    configured = str(os.environ.get("FINBERT_MODEL_CANDIDATES", "")).strip()
    candidates = [
        x.strip() for x in configured.split(",") if x.strip()
    ] or [
        DEFAULT_PRIMARY_FINBERT,
        "yiyanghkust/finbert-tone",
        DEFAULT_FALLBACK_TONE_MODEL,
    ]

    errs: List[str] = []
    for m in candidates:
        try:
            analyzer = get_tone_analyzer(m)
            logger.info("Loaded tone model: %s", m)
            return analyzer
        except Exception as e:
            errs.append(f"{m}:{type(e).__name__}")
            logger.warning("Tone model load failed for %s (%s)", m, e)
            continue

    raise RuntimeError("all tone models failed: " + " | ".join(errs))


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

DEFAULT_RISK_CALIBRATION_PROFILE: Dict[str, Any] = {
    "version": "demo_v1",
    "global_bias": 0.0,
    "global_scale": 1.0,
    "weights": {"rule": 0.62, "classifier": 0.38},
    "category_multipliers": {
        "supply_chain": 1.0,
        "regulatory": 1.05,
        "competition": 0.95,
        "macro": 1.0,
        "geopolitical": 1.08,
        "cyber": 1.02,
        "liquidity": 1.05,
        "customer_concentration": 0.95,
        "litigation": 1.03,
    },
}


def _load_risk_calibration_profile() -> Dict[str, Any]:
    p = Path("data/risk_calibration.json")
    if not p.exists():
        return dict(DEFAULT_RISK_CALIBRATION_PROFILE)
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return dict(DEFAULT_RISK_CALIBRATION_PROFILE)
        base = dict(DEFAULT_RISK_CALIBRATION_PROFILE)
        base.update({k: v for k, v in obj.items() if k in ("version", "global_bias", "global_scale", "weights", "category_multipliers")})
        return base
    except Exception:
        return dict(DEFAULT_RISK_CALIBRATION_PROFILE)


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
    analyzer: Optional[Any] = None,
    use_fallback_on_error: bool = True,
    llm_client: Optional[Any] = None,
) -> ToneSignal:
    """
    Finance-aware tone analysis.
    Favors LLM-based analysis if llm_client is provided, else falls back to local FinBERT or heuristics.
    """
    text = str(text or "").strip()
    if not text:
        return _fallback_tone_from_words(text)

    if llm_client:
        return LLMToneAnalyzer(llm_client).predict(text)

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
    analyzer: Optional[Any] = None,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compares tone between two text blocks.
    """
    cur = analyze_tone(current_text, analyzer=analyzer, llm_client=llm_client)
    prev = analyze_tone(prior_text, analyzer=analyzer, llm_client=llm_client)

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


def extract_risk_signals(
    text: str,
    max_snippets_per_category: int = 3,
    llm_client: Optional[Any] = None,
    use_advanced_model: bool = True,
) -> List[RiskSignal]:
    """
    Deterministic risk-category extraction from filing text, especially Item 1A.
    """
    result = extract_risk_signals_with_diagnostics(
        text=text,
        max_snippets_per_category=max_snippets_per_category,
        llm_client=llm_client,
        use_advanced_model=use_advanced_model,
    )
    return result["signals"]


def extract_risk_signals_with_diagnostics(
    text: str,
    max_snippets_per_category: int = 3,
    llm_client: Optional[Any] = None,
    use_advanced_model: bool = True,
) -> Dict[str, Any]:
    """
    Sentence-level multi-label finance risk classifier + calibrated ensemble.
    Returns both risk signals and diagnostics.
    """
    sentences = split_sentences(text)
    if not sentences:
        return {"signals": [], "diagnostics": RiskDiagnostics([], {}, {}, {}, [], _load_risk_calibration_profile()).__dict__}

    calib = _load_risk_calibration_profile()
    w_rule = float((calib.get("weights") or {}).get("rule", 0.62))
    w_cls = float((calib.get("weights") or {}).get("classifier", 0.38))
    g_bias = float(calib.get("global_bias", 0.0) or 0.0)
    g_scale = float(calib.get("global_scale", 1.0) or 1.0)
    multipliers = calib.get("category_multipliers") or {}

    # Sentence-level multi-label classification.
    sent_labels: List[RiskSentenceLabel] = []
    cat_rule_hits: Dict[str, int] = {}
    cat_snippets: Dict[str, List[str]] = {}
    cat_classifier_scores: Dict[str, List[float]] = {}
    total_sentences = max(len(sentences), 1)

    for sent in sentences:
        s = sent.lower()
        matched_categories: List[str] = []
        hit_total = 0
        for category, patterns in RISK_PATTERNS.items():
            hits = sum(1 for p in patterns if re.search(p, s))
            if hits > 0:
                matched_categories.append(category)
                hit_total += hits
                cat_rule_hits[category] = cat_rule_hits.get(category, 0) + hits
                arr = cat_snippets.setdefault(category, [])
                if len(arr) < max_snippets_per_category:
                    arr.append(sent[:320])

        cls_risk = 0.0
        if use_advanced_model and sent.strip():
            try:
                tone = analyze_tone(sent, llm_client=llm_client if llm_client else None)
                cls_risk = max(0.0, float(tone.negative_prob) - float(tone.positive_prob))
            except Exception:
                cls_risk = 0.0

        for c in matched_categories:
            cat_classifier_scores.setdefault(c, []).append(cls_risk)

        sent_labels.append(
            RiskSentenceLabel(
                sentence=sent[:320],
                categories=matched_categories,
                rule_hits=hit_total,
                classifier_risk_prob=round(float(cls_risk), 4),
                hedge_hits=count_hedge_words(sent),
            )
        )

    rule_score: Dict[str, float] = {}
    cls_score: Dict[str, float] = {}
    calibrated: Dict[str, float] = {}
    out: List[RiskSignal] = []
    contradictions: List[Dict[str, Any]] = []

    for category, hits in cat_rule_hits.items():
        density = hits / total_sentences
        r_score = min(1.0, math.log1p(hits) / 2.5 + density)
        c_vals = cat_classifier_scores.get(category, [])
        c_score = sum(c_vals) / max(len(c_vals), 1) if c_vals else 0.0
        raw = (w_rule * r_score) + (w_cls * c_score)
        mult = float(multipliers.get(category, 1.0) or 1.0)
        sev = max(0.0, min(1.0, ((raw + g_bias) * g_scale) * mult))

        if r_score >= 0.65 and c_score <= 0.20:
            contradictions.append({
                "type": "rule_vs_classifier_mismatch",
                "category": category,
                "severity": "medium",
                "detail": {"rule_score": round(r_score, 4), "classifier_score": round(c_score, 4)},
            })
        if r_score <= 0.20 and c_score >= 0.55:
            contradictions.append({
                "type": "classifier_only_risk",
                "category": category,
                "severity": "low",
                "detail": {"rule_score": round(r_score, 4), "classifier_score": round(c_score, 4)},
            })

        rule_score[category] = round(float(r_score), 4)
        cls_score[category] = round(float(c_score), 4)
        calibrated[category] = round(float(sev), 4)
        out.append(
            RiskSignal(
                category=category,
                severity=round(float(sev), 4),
                count=int(hits),
                snippets=list(cat_snippets.get(category, [])),
            )
        )

    out.sort(key=lambda x: (x.severity, x.count), reverse=True)
    diagnostics = RiskDiagnostics(
        sentence_labels=sent_labels,
        category_rule_score=rule_score,
        category_classifier_score=cls_score,
        category_calibrated_score=calibrated,
        contradictions=contradictions,
        calibration_profile=calib,
    )
    return {"signals": out, "diagnostics": diagnostics.__dict__}


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


def classify_news_catalysts(articles: List[Dict[str, Any]], llm_client: Optional[Any] = None) -> List[NewsCatalyst]:
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
        tone = analyze_tone(text, llm_client=llm_client)

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
