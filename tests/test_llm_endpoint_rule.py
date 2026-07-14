from __future__ import annotations

import pytest

from autops.llm.client import LLMClient


def test_ollama_endpoint_is_environment_only() -> None:
    with pytest.raises(ValueError, match="OLLAMA_HOST"):
        LLMClient({"ollama_host": "https://example.invalid"})
