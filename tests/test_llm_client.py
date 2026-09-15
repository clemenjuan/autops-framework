from __future__ import annotations

import json

import pytest

from autops.llm.cache import CACHE_KEY_SCHEMA, CacheEntry, ResponseCache, response_key
from autops.llm.client import LLMClient


def _key(prompt: str = "state") -> str:
    return response_key(
        system_prompt="system",
        user_prompt=prompt,
        provider="ollama",
        model="model",
        temperature=0.0,
        json_mode=True,
    )


def test_sha256_cache_identity_and_round_trip(tmp_path) -> None:
    assert len(_key()) == 64
    assert _key() == _key()
    assert _key("other") != _key()

    cache = ResponseCache(tmp_path / "cache")
    entry = CacheEntry('{"mode":"charging"}', "ollama", "model")
    assert cache.get(_key()) is None
    cache.put(_key(), entry)
    assert cache.get(_key()) == entry
    payload = json.loads(next((tmp_path / "cache").rglob("*.json")).read_text())
    assert payload["schema"] == CACHE_KEY_SCHEMA
    assert "prompt" not in payload


def test_mock_is_deterministic_and_provenance_redacts_paths_and_endpoints(tmp_path) -> None:
    client = LLMClient(
        {
            "llm_mock": True,
            "llm_cache_dir": str(tmp_path / "secret-location"),
        }
    )
    prompt = "PLAN THE NEXT 4 STEPS\nGround pass active now: YES\nOBC ready for downlink: 2.0"
    first = client.generate("system", prompt, json_mode=True)
    second = client.generate("system", prompt, json_mode=True)
    assert first == second
    assert json.loads(first)["decision"]["mode"] == "communication"
    serialized = json.dumps(client.provenance(), sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "secret-location" not in serialized
    assert client.metrics()["llm_live_calls"] == 0.0


def test_replay_consumes_exact_responses_without_provider_imports() -> None:
    client = LLMClient({"llm_replay": ['{"one":1}', '{"two":2}']})
    assert client.generate("s", "u") == '{"one":1}'
    assert client.generate("s", "u") == '{"two":2}'
    with pytest.raises(RuntimeError, match="replay exhausted"):
        client.generate("s", "u")
    assert client.provenance()["providers_used"] == ["replay"]


def test_auto_provider_without_environment_fails_explicitly(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient({"llm_provider": "auto", "llm_cache": False})
    with pytest.raises(RuntimeError, match="no configured provider"):
        client.generate("system", "state")


def test_openai_receives_declared_seed_and_completion_budget(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(
            OpenAI=lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
        ),
    )
    client = LLMClient({"llm_max_tokens": 321, "llm_cache": False})
    assert client._call_provider("openai", "system", "user", 1.0, True, 2, 73) == "ok"
    assert captured["seed"] == 73
    assert captured["max_completion_tokens"] == 321


def test_ollama_transport_retry_preserves_cache_key_seed(monkeypatch) -> None:
    from types import SimpleNamespace

    requests = pytest.importorskip("requests")

    captured = []

    def post(_url, **kwargs):
        captured.append(kwargs["json"])
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"message": {"content": "ok"}},
        )

    monkeypatch.setattr(requests, "post", post)
    client = LLMClient({"llm_stream": False, "llm_cache": False})
    for attempt in (0, 1):
        assert client._call_ollama_inner("system", "user", 1.0, True, attempt, 73) == "ok"
    assert [payload["options"]["seed"] for payload in captured] == [73, 73]
