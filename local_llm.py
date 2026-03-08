"""
local_llm.py

Local LLM clients using Ollama for inference.
Cost-aware defaults:
  - Qwen for low-cost / faster paths
  - Mistral for higher-quality / complex paths
with automatic fallback between them.
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import requests

logger = logging.getLogger(__name__)

DEFAULT_QWEN_MODEL = "qwen2.5:1.5b"
DEFAULT_MISTRAL_MODEL = "mistral:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Backward-compatible aliases
DEFAULT_PRIMARY_MODEL = DEFAULT_QWEN_MODEL
DEFAULT_FALLBACK_MODEL = DEFAULT_MISTRAL_MODEL


@dataclass(frozen=True)
class HardwareProfile:
    cpu_cores: int
    ram_gb: Optional[float]


def _detect_ram_gb() -> Optional[float]:
    try:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(stat.ullTotalPhys) / (1024.0 ** 3)
    except Exception:
        return None
    return None


def detect_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_cores=max(1, int(os.cpu_count() or 1)),
        ram_gb=_detect_ram_gb(),
    )


class OllamaLLMClient:
    """LLM client that talks to a local Ollama server."""

    def __init__(
        self,
        model_name: str = DEFAULT_PRIMARY_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_s: int = 300,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def _check_server(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _list_local_models(self) -> List[str]:
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            data = resp.json() or {}
            models = data.get("models") or []
            out: List[str] = []
            for m in models:
                if isinstance(m, dict):
                    name = str(m.get("name") or "").strip()
                    if name:
                        out.append(name)
            return out
        except Exception:
            return []

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        selected_model = model_name or self.model_name

        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096,
            },
        }

        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()

                text = data.get("message", {}).get("content", "")

                prompt_tokens = data.get("prompt_eval_count") or 0
                completion_tokens = data.get("eval_count") or 0
                usage: Dict[str, Any] = {
                    "input_tokens": prompt_tokens or None,
                    "output_tokens": completion_tokens or None,
                    "total_tokens": (prompt_tokens + completion_tokens) or None,
                    "cost_usd": 0.0,
                    "model": selected_model,
                    "local": True,
                }
                return text, usage

            except requests.exceptions.ConnectionError:
                raise RuntimeError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running: 'ollama serve'"
                )
            except requests.exceptions.Timeout:
                if attempt < 2:
                    logger.warning(
                        "Ollama timeout (attempt %d/3) for model %s",
                        attempt + 1,
                        selected_model,
                    )
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"Ollama timed out after {self.timeout_s}s for model {selected_model}"
                )
            except requests.exceptions.HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                if status == 404:
                    raise RuntimeError(
                        f"Model '{selected_model}' not found in Ollama. "
                        f"Pull it first: 'ollama pull {selected_model}'"
                    ) from exc
                raise

        raise RuntimeError(f"Ollama generation failed after 3 attempts for {selected_model}")


class FallbackLocalLLMClient:
    """Tries primary model first; falls back to secondary on failure."""

    def __init__(
        self,
        primary: OllamaLLMClient,
        fallback: Optional[OllamaLLMClient] = None,
    ):
        self.primary = primary
        self.fallback = fallback

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        try:
            return self.primary.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_name,
            )
        except Exception as exc:
            if self.fallback is None:
                raise
            logger.warning(
                "Primary model failed (%s); falling back to %s",
                exc,
                self.fallback.model_name,
            )
            return self.fallback.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.fallback.model_name,
            )


class CostRouterLocalLLMClient:
    """
    Hardware-aware cost router:
      - prefers Qwen for low-cost/fast paths
      - uses Mistral for harder prompts on capable hardware
      - always falls back to the other model on failure
    """

    def __init__(
        self,
        qwen: OllamaLLMClient,
        mistral: OllamaLLMClient,
        *,
        prefer_low_cost: bool = True,
    ):
        self.qwen = qwen
        self.mistral = mistral
        self.prefer_low_cost = bool(prefer_low_cost)
        self.hw = detect_hardware_profile()

    def _is_low_resource(self) -> bool:
        if self.hw.cpu_cores < 8:
            return True
        if self.hw.ram_gb is not None and self.hw.ram_gb < 16.0:
            return True
        return False

    def _is_complex(self, system_prompt: str, user_prompt: str) -> bool:
        txt = f"{system_prompt}\n{user_prompt}".lower()
        if len(txt) > 6500:
            return True
        hard_terms = (
            "valuation",
            "discounted cash flow",
            "dcf",
            "framework",
            "comparative",
            "relative valuation",
            "risk analysis",
        )
        return any(t in txt for t in hard_terms)

    def _pick_route(self, model_name: str, system_prompt: str, user_prompt: str) -> Tuple[OllamaLLMClient, OllamaLLMClient, str]:
        req = (model_name or "auto").strip().lower()
        # Explicit route hints used by orchestrator.
        if req == "small":
            return self.qwen, self.mistral, "small->qwen"
        if req == "large":
            if self._is_low_resource():
                return self.qwen, self.mistral, "large->qwen(low_resource)"
            return self.mistral, self.qwen, "large->mistral"
        if req in ("auto", ""):
            complex_prompt = self._is_complex(system_prompt, user_prompt)
            if complex_prompt and not self._is_low_resource():
                return self.mistral, self.qwen, "auto->mistral(complex)"
            return self.qwen, self.mistral, "auto->qwen"

        # Explicit model name passed by caller.
        if "mistral" in req:
            return self.mistral, self.qwen, f"explicit->{model_name}"
        if "qwen" in req:
            return self.qwen, self.mistral, f"explicit->{model_name}"
        # Unknown explicit name: try via qwen client first, then mistral client.
        return self.qwen, self.mistral, f"explicit_unknown->{model_name}"

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_name: str,
    ) -> Tuple[str, Dict[str, Any]]:
        primary, fallback, route_tag = self._pick_route(model_name, system_prompt, user_prompt)
        try:
            text, usage = primary.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=(model_name if model_name not in ("small", "large", "auto") else primary.model_name),
            )
            usage = dict(usage or {})
            usage["route"] = route_tag
            usage["hardware"] = {"cpu_cores": self.hw.cpu_cores, "ram_gb": self.hw.ram_gb}
            return text, usage
        except Exception as exc:
            logger.warning(
                "Primary routed model failed (%s); fallback to %s",
                exc,
                fallback.model_name,
            )
            text, usage = fallback.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=fallback.model_name,
            )
            usage = dict(usage or {})
            usage["route"] = f"{route_tag}|fallback->{fallback.model_name}"
            usage["hardware"] = {"cpu_cores": self.hw.cpu_cores, "ram_gb": self.hw.ram_gb}
            return text, usage


def build_local_llm_client(
    *,
    primary_model: str = DEFAULT_QWEN_MODEL,
    fallback_model: str = DEFAULT_MISTRAL_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout_s: int = 300,
) -> CostRouterLocalLLMClient:
    qwen = OllamaLLMClient(
        model_name=primary_model,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    mistral = OllamaLLMClient(
        model_name=fallback_model,
        base_url=base_url,
        timeout_s=timeout_s,
    )
    return CostRouterLocalLLMClient(qwen=qwen, mistral=mistral)
