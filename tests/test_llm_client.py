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
