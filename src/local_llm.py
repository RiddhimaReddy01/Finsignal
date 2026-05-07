"""
Gemini-backed LLM client with cost-aware routing and fallback.

Keeps the existing app-facing interface:
  - build_local_llm_client(...)
  - client.generate_json(system_prompt=..., user_prompt=..., model_name=...)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

DEFAULT_SMALL_MODEL = "gemini-2.5-flash"
DEFAULT_LARGE_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"

# Backward-compatible aliases consumed by main.py / userinterface.py.
DEFAULT_PRIMARY_MODEL = DEFAULT_SMALL_MODEL


@dataclass(frozen=True)
class RateLimits:
    rpm: int = 5
    tpm: int = 250_000
    rpd: int = 20


class _RateLimiter:
    """Simple in-process limiter for RPM/TPM/RPD."""

    def __init__(self, limits: RateLimits):
        self._limits = limits
        self._lock = threading.Lock()
        self._req_ts: Deque[float] = deque()
        self._tok_ts: Deque[Tuple[float, int]] = deque()
        self._day_start = time.time()
        self._day_count = 0

    def _prune(self, now: float) -> None:
        minute_ago = now - 60.0
        while self._req_ts and self._req_ts[0] < minute_ago:
            self._req_ts.popleft()
        while self._tok_ts and self._tok_ts[0][0] < minute_ago:
            self._tok_ts.popleft()
        if now - self._day_start >= 86_400:
            self._day_start = now
            self._day_count = 0

    def _minute_tokens(self) -> int:
        return sum(toks for _, toks in self._tok_ts)

    def acquire(self, estimated_tokens: int) -> None:
        estimated = max(1, int(estimated_tokens))
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)

                if self._day_count >= self._limits.rpd:
                    raise RuntimeError("Gemini daily request limit reached (RPD).")

                req_wait = 0.0
                tok_wait = 0.0

                if len(self._req_ts) >= self._limits.rpm:
                    req_wait = max(0.0, 60.0 - (now - self._req_ts[0]))

                if self._minute_tokens() + estimated > self._limits.tpm and self._tok_ts:
                    tok_wait = max(0.0, 60.0 - (now - self._tok_ts[0][0]))

                wait_s = max(req_wait, tok_wait)
                if wait_s <= 0:
                    self._req_ts.append(now)
                    self._tok_ts.append((now, estimated))
                    self._day_count += 1
                    return

            time.sleep(min(wait_s, 1.0))


def _rough_token_estimate(text: str) -> int:
    # ~4 chars/token heuristic.
    return max(1, int(len(text or "") / 4))


class GeminiLLMClient:
    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: int = 120,
        limits: Optional[RateLimits] = None,
    ):
        self.model_name = model_name
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_s = int(timeout_s)
        self.limiter = _RateLimiter(limits or RateLimits())
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        selected = (model_name or self.model_name).strip()
        prompt = f"{system_prompt}\n\n{user_prompt}"
        self.limiter.acquire(_rough_token_estimate(prompt))

        url = f"{self.base_url}/v1beta/models/{selected}:generateContent"
        params = {"key": self.api_key}
        payload = {
            "systemInstruction": {"role": "system", "parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        resp = requests.post(url, params=params, json=payload, timeout=self.timeout_s)
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")

        data = resp.json() or {}
        cands = data.get("candidates") or []
        text = ""
        if cands:
            parts = ((cands[0].get("content") or {}).get("parts") or [])
            text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        if not text.strip():
            raise RuntimeError("Gemini returned empty content.")

        usage_meta = data.get("usageMetadata") or {}
        usage: Dict[str, Any] = {
            "input_tokens": usage_meta.get("promptTokenCount"),
            "output_tokens": usage_meta.get("candidatesTokenCount"),
            "total_tokens": usage_meta.get("totalTokenCount"),
            "model": selected,
            "provider": "gemini",
        }
        return text, usage


class CostRouterGeminiLLMClient:
    """Model router with small/large choice and fallback on failure."""

    def __init__(
        self,
        *,
        small_client: GeminiLLMClient,
        large_client: GeminiLLMClient,
        fallback_client: GeminiLLMClient,
    ):
        self.small_client = small_client
        self.large_client = large_client
        self.fallback_client = fallback_client

    def _is_complex(self, system_prompt: str, user_prompt: str) -> bool:
        txt = f"{system_prompt}\n{user_prompt}".lower()
        if len(txt) > 6_500:
            return True
        hard_terms = (
            "valuation",
            "discounted cash flow",
            "relative valuation",
            "comparative",
            "risk analysis",
            "framework",
        )
        return any(t in txt for t in hard_terms)

    def _pick(self, model_name: str, system_prompt: str, user_prompt: str) -> Tuple[GeminiLLMClient, str]:
        req = (model_name or "auto").strip().lower()
        if req == "small":
            return self.small_client, "small"
        if req == "large":
            return self.large_client, "large"
        if req in ("auto", ""):
            if self._is_complex(system_prompt, user_prompt):
                return self.large_client, "auto->large"
            return self.small_client, "auto->small"
        # Explicit model name.
        return self.small_client, f"explicit:{model_name}"

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        primary, route = self._pick(model_name, system_prompt, user_prompt)
        requested_model = model_name if model_name not in ("small", "large", "auto", "") else primary.model_name
        try:
            text, usage = primary.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=requested_model,
            )
            usage = dict(usage or {})
            usage["route"] = route
            return text, usage
        except Exception as exc:
            logger.warning("Primary Gemini route failed (%s), falling back to %s", exc, self.fallback_client.model_name)
            text, usage = self.fallback_client.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.fallback_client.model_name,
            )
            usage = dict(usage or {})
            usage["route"] = f"{route}|fallback:{self.fallback_client.model_name}"
            return text, usage


def build_local_llm_client(
    *,
    primary_model: str = DEFAULT_SMALL_MODEL,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout_s: int = 120,
) -> CostRouterGeminiLLMClient:
    api_key = os.environ.get("GEMINI_API_KEY", "")

    small_model = os.environ.get("GEMINI_SMALL_MODEL", primary_model or DEFAULT_SMALL_MODEL)
    large_model = os.environ.get("GEMINI_LARGE_MODEL", DEFAULT_LARGE_MODEL)
    fb_model = os.environ.get("GEMINI_FALLBACK_MODEL", fallback_model or DEFAULT_FALLBACK_MODEL)

    limits = RateLimits(
        rpm=int(os.environ.get("GEMINI_RPM", "5")),
        tpm=int(os.environ.get("GEMINI_TPM", "250000")),
        rpd=int(os.environ.get("GEMINI_RPD", "20")),
    )

    small = GeminiLLMClient(
        model_name=small_model,
        api_key=api_key,
        base_url=os.environ.get("GEMINI_BASE_URL", base_url),
        timeout_s=timeout_s,
        limits=limits,
    )
    large = GeminiLLMClient(
        model_name=large_model,
        api_key=api_key,
        base_url=os.environ.get("GEMINI_BASE_URL", base_url),
        timeout_s=timeout_s,
        limits=limits,
    )
    fallback = GeminiLLMClient(
        model_name=fb_model,
        api_key=api_key,
        base_url=os.environ.get("GEMINI_BASE_URL", base_url),
        timeout_s=timeout_s,
        limits=limits,
    )
    return CostRouterGeminiLLMClient(
        small_client=small,
        large_client=large,
        fallback_client=fallback,
    )
