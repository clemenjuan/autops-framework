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
        self._stream = bool(cfg.get("llm_stream", True))
        self._connect_timeout_s = float(cfg.get("llm_connect_timeout_s", 15.0))
        self._read_timeout_s = float(cfg.get("llm_read_timeout_s", 90.0))
        self._hard_timeout_s = float(cfg.get("llm_hard_timeout_s", 300.0))
        decision_log = cfg.get("llm_decision_log")
        self._decision_log = Path(str(decision_log)) if decision_log else None
        think = cfg.get("llm_think")
        self._think = bool(think) if think is not None else None
        max_tokens = cfg.get("llm_max_tokens")
        self._max_tokens = int(max_tokens) if max_tokens is not None else None
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
                self._log_decision(event="cache_hit", provider=cached.provider, text=cached.text)
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
                    latency_s = time.perf_counter() - started
                    self._total_latency_s += latency_s
                    self._live_calls += 1
                    self._providers_used.add(provider)
                    if not text.strip():
                        raise RuntimeError("provider returned an empty response")
                    if self._cache_enabled:
                        self._cache.put(key, CacheEntry(text, provider, self.model))
                    self._log_decision(
                        event="success", provider=provider, text=text, latency_s=latency_s
                    )
                    return text
                except Exception as exc:  # provider failures are summarized, not hidden
                    failures.append(f"{provider}: {type(exc).__name__}: {exc}")
                    self._log_decision(
                        event="error",
                        provider=provider,
                        attempt=attempt + 1,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if attempt < self._retries and self._backoff_s:
                        time.sleep(self._backoff_s * (2**attempt))
        detail = "; ".join(failures) or "no configured provider is available"
        raise RuntimeError(f"LLM generation failed: {detail}")

    def _log_decision(self, **fields: Any) -> None:
        """Append one human-inspectable JSON line per call; opt-in via llm_decision_log.

        Independent of the response cache: this is ordered by wall-clock
        append time and scoped to one run, so it does not require sifting a
        content-addressed cache shared across every run that ever used it.
        """

        if self._decision_log is None:
            return
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": self.model, **fields}
        line = json.dumps(record, ensure_ascii=False)
        self._decision_log.parent.mkdir(parents=True, exist_ok=True)
        with self._decision_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

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
        """Call Ollama with a hard wall-clock timeout enforced from the outside.

        Some gateways can hold an HTTPS streaming connection open while
        emitting no chunks; requests/urllib3's ``timeout=`` does not enforce
        per-recv reads under ``stream=True``, and closing the socket from
        another thread does not unblock a stuck SSL read. Running the call
        in a daemon worker thread and bounding the wait with
        ``queue.Queue.get(timeout=...)`` is the only reliable escape. The
        worker is abandoned (not joined) on timeout; it is a daemon and
        writes only to a local queue, so it touches no shared state.
        """
        if not self._ollama_host:
            raise RuntimeError("OLLAMA_HOST is required for the Ollama provider")
        import queue
        import threading

        result_q: queue.Queue[tuple[str, Any]] = queue.Queue()

        def _worker() -> None:
            try:
                text = self._call_ollama_inner(system_prompt, user_prompt, temperature, json_mode)
                result_q.put(("ok", text))
            except Exception as exc:  # relayed to the calling thread, not hidden
                result_q.put(("err", exc))

        thread = threading.Thread(target=_worker, daemon=True, name="ollama-call")
        thread.start()
        try:
            status, payload = result_q.get(timeout=self._hard_timeout_s)
        except queue.Empty as exc:
            raise RuntimeError(
                f"Ollama call exceeded hard timeout of {self._hard_timeout_s:g}s "
                "(worker thread abandoned)"
            ) from exc
        if status == "err":
            raise payload
        return payload

    def _call_ollama_inner(
        self, system_prompt: str, user_prompt: str, temperature: float, json_mode: bool
    ) -> str:
        """Issue the actual HTTP call; streams by default. Runs in a worker thread."""

        import requests

        url = f"{self._ollama_host.rstrip('/')}/api/chat"
        options: dict[str, Any] = {"temperature": temperature}
        if self._max_tokens is not None:
            options["num_predict"] = self._max_tokens
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": self._stream,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": options,
        }
        if json_mode:
            payload["format"] = "json"
        if self._think is not None:
            payload["think"] = self._think
        request_timeout = (self._connect_timeout_s, self._read_timeout_s)

        if not self._stream:
            response = requests.post(url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            content = str(response.json()["message"]["content"])
            if not content:
                raise RuntimeError("Ollama non-streaming response empty")
            return content

        content_parts: list[str] = []
        saw_done = False
        with requests.post(url, json=payload, timeout=request_timeout, stream=True) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                content_parts.append((chunk.get("message") or {}).get("content", ""))
                if chunk.get("done"):
                    saw_done = True
                    break
        content = "".join(content_parts)
        if not content and not saw_done:
            raise RuntimeError("Ollama streaming response empty (no chunks from gateway)")
        return content

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
