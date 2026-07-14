"""Lazy Ollama/OpenAI client with mock, replay, cache, and safe provenance."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from autops.llm.cache import CACHE_KEY_SCHEMA, CacheEntry, ResponseCache, response_key


class LLMClient:
    """Synchronous provider adapter; optional packages are imported only on a live call."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.provider = str(cfg.get("llm_provider", "auto"))
        self.model = str(cfg.get("llm_model", "qwen3.6:35b"))
        self.temperature = float(cfg.get("llm_temperature", 0.0))
        self.mock_mode = bool(cfg.get("llm_mock", False))
        replay = cfg.get("llm_replay", ())
        if isinstance(replay, str) or not isinstance(replay, Sequence):
            raise TypeError("llm_replay must be a sequence of response strings")
        self._replay = [str(item) for item in replay]
        self._replay_index = 0
        self._retries = max(0, int(cfg.get("llm_retries", 0)))
        self._backoff_s = max(0.0, float(cfg.get("llm_retry_backoff_s", 1.0)))
        if "ollama_host" in cfg:
            raise ValueError("Ollama endpoints must be supplied through OLLAMA_HOST")
        self._ollama_host = os.getenv("OLLAMA_HOST", "")
        cache_dir = Path(str(cfg.get("llm_cache_dir", "artifacts/llm-cache")))
        self._cache = ResponseCache(cache_dir)
        self._cache_enabled = bool(cfg.get("llm_cache", True)) and not (
            self.mock_mode or self._replay
        )
        self._calls = 0
        self._live_calls = 0
        self._cache_hits = 0
        self._total_latency_s = 0.0
        self._providers_used: set[str] = set()

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return one completion or raise; never substitute a symbolic decision."""

        self._calls += 1
        if self._replay:
            if self._replay_index >= len(self._replay):
                raise RuntimeError("LLM replay exhausted")
            text = self._replay[self._replay_index]
            self._replay_index += 1
            self._providers_used.add("replay")
            return text
        if self.mock_mode:
            self._providers_used.add("mock")
            return _mock_response(user_prompt)

        actual_temperature = self.temperature if temperature is None else float(temperature)
        failures: list[str] = []
        for provider in self._provider_order():
            key = response_key(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                provider=provider,
                model=self.model,
                temperature=actual_temperature,
                json_mode=json_mode,
            )
            if self._cache_enabled and (cached := self._cache.get(key)) is not None:
                self._cache_hits += 1
                self._providers_used.add(f"cache:{cached.provider}")
                return cached.text
            for attempt in range(self._retries + 1):
                started = time.perf_counter()
                try:
                    text = self._call_provider(
                        provider,
                        system_prompt,
                        user_prompt,
                        actual_temperature,
                        json_mode,
                    )
                    self._total_latency_s += time.perf_counter() - started
                    self._live_calls += 1
                    self._providers_used.add(provider)
                    if not text.strip():
                        raise RuntimeError("provider returned an empty response")
                    if self._cache_enabled:
                        self._cache.put(key, CacheEntry(text, provider, self.model))
                    return text
                except Exception as exc:  # provider failures are summarized, not hidden
                    failures.append(f"{provider}: {type(exc).__name__}: {exc}")
                    if attempt < self._retries and self._backoff_s:
                        time.sleep(self._backoff_s * (2**attempt))
        detail = "; ".join(failures) or "no configured provider is available"
        raise RuntimeError(f"LLM generation failed: {detail}")

    def _provider_order(self) -> tuple[str, ...]:
        if self.provider == "ollama":
            return ("ollama",)
        if self.provider == "openai":
            return ("openai",)
        if self.provider != "auto":
            raise ValueError("llm_provider must be auto, ollama, or openai")
        providers: list[str] = []
        if self._ollama_host:
            providers.append("ollama")
        if os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
        return tuple(providers)

    def _call_provider(
        self,
        provider: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool,
    ) -> str:
        if provider == "ollama":
            return self._call_ollama(system_prompt, user_prompt, temperature, json_mode)
        return self._call_openai(system_prompt, user_prompt, temperature, json_mode)

    def _call_ollama(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        if not self._ollama_host:
            raise RuntimeError("OLLAMA_HOST is required for the Ollama provider")
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        response = requests.post(
            f"{self._ollama_host.rstrip('/')}/api/chat",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"])

    def _call_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        from openai import OpenAI

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = OpenAI().chat.completions.create(**kwargs)
        return str(response.choices[0].message.content or "")

    def metrics(self) -> dict[str, float]:
        return {
            "llm_calls": float(self._calls),
            "llm_live_calls": float(self._live_calls),
            "llm_cache_hits": float(self._cache_hits),
            "llm_total_latency_s": self._total_latency_s,
        }

    def provenance(self) -> dict[str, Any]:
        """Return reproducibility fields only: no endpoint, path, prompt, or secret."""

        return {
            "cache_key_schema": CACHE_KEY_SCHEMA,
            "configured_provider": self.provider,
            "configured_model": self.model,
            "mock": self.mock_mode,
            "replay": bool(self._replay),
            "providers_used": sorted(self._providers_used),
        }


def _mock_response(user_prompt: str) -> str:
    """Deterministic, explicitly non-scoring response used by CI and smoke tests."""

    match = re.search(r"PLAN THE NEXT\s+(\d+)\s+STEPS", user_prompt, re.IGNORECASE)
    gap = max(1, int(match.group(1))) if match else 1
    obc = re.search(r"OBC ready for downlink:\s*([0-9.]+)", user_prompt)
    contact = "Ground pass active now: YES" in user_prompt
    mode = "communication" if contact and obc and float(obc.group(1)) > 0 else "charging"
    schedule = [["charging", gap]]
    decision = {"mode": mode, "schedule": schedule, "rationale": "deterministic mock"}
    if "PLAN THE NEXT" in user_prompt or "tool budget is exhausted" in user_prompt:
        return json.dumps({"decision": decision})
    return json.dumps({"mode": mode, "rationale": "deterministic mock"})
