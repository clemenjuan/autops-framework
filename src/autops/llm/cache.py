"""Content-addressed response cache for reproducible LLM decisions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

CACHE_KEY_SCHEMA = "autops-llm-decision-v2"


def response_key(
    *,
    system_prompt: str,
    user_prompt: str,
    provider: str,
    model: str,
    temperature: float,
    json_mode: bool,
    seed: int | None = None,
    think: bool | None = None,
    max_tokens: int | None = None,
) -> str:
    """Return a stable SHA256 identity without retaining prompt text.

    Every input that changes what the provider generates belongs in the key:
    reasoning mode and generation budget alter the response as decisively as
    the prompt does, and the seed is what makes a sampled decision replayable.
    """

    payload = {
        "schema": CACHE_KEY_SCHEMA,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "provider": provider,
        "model": model,
        "temperature": float(temperature),
        "json_mode": bool(json_mode),
        "seed": None if seed is None else int(seed),
        "think": None if think is None else bool(think),
        "max_tokens": None if max_tokens is None else int(max_tokens),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    text: str
    provider: str
    model: str


class ResponseCache:
    """Small JSON-file cache; paths never enter result provenance."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def get(self, key: str) -> CacheEntry | None:
        path = self.root / key[:2] / f"{key}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if value.get("schema") != CACHE_KEY_SCHEMA or not isinstance(value.get("text"), str):
            return None
        return CacheEntry(
            text=value["text"],
            provider=str(value.get("provider", "unknown")),
            model=str(value.get("model", "unknown")),
        )

    def put(self, key: str, entry: CacheEntry) -> None:
        directory = self.root / key[:2]
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CACHE_KEY_SCHEMA,
            "text": entry.text,
            "provider": entry.provider,
            "model": entry.model,
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".llm-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            os.replace(temporary, directory / f"{key}.json")
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
