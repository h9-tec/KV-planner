"""Detect which local LLM runtimes are reachable and list their installed models.

Currently supports:
* **Ollama** — HTTP API on ``localhost:11434``
* **LM Studio** — HTTP API on ``localhost:1234`` (OpenAI-compatible)
* **vLLM** — HTTP API on ``localhost:8000`` if a user spawned one
* **llama-server** (llama.cpp) — HTTP API on ``localhost:8080``

All probes use stdlib ``urllib`` with short timeouts so the detector doesn't
hang when a port is firewalled. Every probe fails silent and returns an empty
list rather than raising.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeProbe:
    name: str
    reachable: bool
    version: str = ""
    models: list[str] = field(default_factory=list)
    endpoint: str = ""
    error: str = ""


def _get_json(url: str, timeout: float = 1.2) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def probe_ollama(host: str = "127.0.0.1", port: int = 11434) -> RuntimeProbe:
    base = f"http://{host}:{port}"
    tags = _get_json(f"{base}/api/tags")
    if tags is None:
        return RuntimeProbe(name="ollama", reachable=False, endpoint=base)
    models = sorted({m.get("name", "") for m in tags.get("models", [])})
    ver = _get_json(f"{base}/api/version") or {}
    return RuntimeProbe(
        name="ollama",
        reachable=True,
        version=str(ver.get("version", "")),
        models=models,
        endpoint=base,
    )


def probe_openai_compatible(
    name: str, host: str = "127.0.0.1", port: int = 1234
) -> RuntimeProbe:
    base = f"http://{host}:{port}"
    data = _get_json(f"{base}/v1/models")
    if data is None:
        return RuntimeProbe(name=name, reachable=False, endpoint=base)
    models = sorted({m.get("id", "") for m in data.get("data", [])})
    return RuntimeProbe(
        name=name,
        reachable=True,
        models=models,
        endpoint=base,
    )


def probe_lmstudio() -> RuntimeProbe:
    return probe_openai_compatible("lmstudio", port=1234)


def probe_vllm() -> RuntimeProbe:
    return probe_openai_compatible("vllm", port=8000)


def probe_llamacpp() -> RuntimeProbe:
    # llama-server uses /v1/models too (OpenAI compat shim).
    return probe_openai_compatible("llama.cpp", port=8080)


def probe_all() -> list[RuntimeProbe]:
    return [
        probe_ollama(),
        probe_lmstudio(),
        probe_vllm(),
        probe_llamacpp(),
    ]
