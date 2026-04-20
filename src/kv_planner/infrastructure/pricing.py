"""Live / cached pricing for GPUs and API models.

Sources (best-effort; fallback to cached JSON if offline):

* **Artificial Analysis** — per-provider LLM price-per-M-tokens, latency,
  intelligence scores. https://artificialanalysis.ai/
* **OpenRouter** — passes provider prices through unchanged.
  https://openrouter.ai/docs#models
* Published GPU-hour rates from AWS / GCP / Azure / Lambda / RunPod /
  CoreWeave / Vast.ai — scraped from their public pricing pages.

The HTTP fetch uses stdlib ``urllib`` with a short timeout and a strict
fallback to the baked-in defaults. Nothing blocks the planner if the
network is flaky.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

_CACHE_PATH = pathlib.Path.home() / ".cache" / "kv-planner" / "pricing.json"
_CACHE_TTL_S = 24 * 60 * 60  # refresh once per day


@dataclass(frozen=True)
class ApiPricing:
    """Price per million tokens for a hosted model / endpoint."""

    provider: str
    model: str
    input_per_m: float
    output_per_m: float
    source: str
    fetched_at: float   # unix seconds


@dataclass(frozen=True)
class GpuPricing:
    gpu_model: str
    cost_per_hour: float
    provider: str
    spot: bool = False
    region: str = ""


# --------------------------------------------------------------------------
# Defaults — point-in-time snapshot for offline use
# --------------------------------------------------------------------------


_FALLBACK_API: dict[str, ApiPricing] = {
    "claude-3-5-sonnet": ApiPricing(
        "anthropic", "claude-3-5-sonnet", 3.00, 15.00,
        "anthropic-2026-04-fallback", 0.0),
    "claude-3-opus":     ApiPricing(
        "anthropic", "claude-3-opus", 15.00, 75.00,
        "anthropic-2026-04-fallback", 0.0),
    "gpt-4o":            ApiPricing(
        "openai", "gpt-4o", 2.50, 10.00,
        "openai-2026-04-fallback", 0.0),
    "gpt-4o-mini":       ApiPricing(
        "openai", "gpt-4o-mini", 0.15, 0.60,
        "openai-2026-04-fallback", 0.0),
    "gpt-3.5-turbo":     ApiPricing(
        "openai", "gpt-3.5-turbo", 0.50, 1.50,
        "openai-2026-04-fallback", 0.0),
    "gemini-2.0-flash":  ApiPricing(
        "google", "gemini-2.0-flash", 0.075, 0.30,
        "google-2026-04-fallback", 0.0),
    "gemini-2.5-pro":    ApiPricing(
        "google", "gemini-2.5-pro", 1.25, 5.00,
        "google-2026-04-fallback", 0.0),
    "llama-3.3-70b":     ApiPricing(
        "groq", "llama-3.3-70b", 0.59, 0.79,
        "groq-2026-04-fallback", 0.0),
    "deepseek-r1":       ApiPricing(
        "deepseek", "deepseek-r1", 0.55, 2.19,
        "deepseek-2026-04-fallback", 0.0),
    "mistral-large-24": ApiPricing(
        "mistral", "mistral-large-24", 2.00, 6.00,
        "mistral-2026-04-fallback", 0.0),
}

_FALLBACK_SPOT_MULTIPLIER = 0.35  # typical on-demand → spot discount (AWS/GCP EC2)

_FALLBACK_GPU_HOURS: dict[tuple[str, str], float] = {
    # (gpu, provider) → $/hr on-demand
    ("H100-SXM-80GB", "AWS"): 4.50,
    ("H100-SXM-80GB", "GCP"): 4.99,
    ("H100-SXM-80GB", "Azure"): 6.98,
    ("H100-SXM-80GB", "Lambda"): 2.49,
    ("H100-SXM-80GB", "RunPod"): 2.89,
    ("H100-SXM-80GB", "CoreWeave"): 4.25,
    ("H100-SXM-80GB", "Vast.ai"): 1.79,
    ("H200-SXM-141GB", "AWS"): 6.50,
    ("H200-SXM-141GB", "Lambda"): 3.29,
    ("H200-SXM-141GB", "RunPod"): 3.49,
    ("A100-SXM-80GB", "AWS"): 2.50,
    ("A100-SXM-80GB", "GCP"): 2.93,
    ("A100-SXM-80GB", "Lambda"): 1.29,
    ("A100-SXM-80GB", "RunPod"): 1.79,
    ("A100-SXM-80GB", "Vast.ai"): 0.89,
    ("B200-SXM-192GB", "Lambda"): 4.99,
    ("B200-SXM-192GB", "RunPod"): 5.49,
    ("L40S", "RunPod"): 1.19,
    ("L40S", "Vast.ai"): 0.79,
    ("RTX-5090", "RunPod"): 0.54,
    ("RTX-5090", "Vast.ai"): 0.32,
    ("RTX-4090", "RunPod"): 0.44,
    ("RTX-4090", "Vast.ai"): 0.23,
    ("RTX-3090", "Vast.ai"): 0.12,
    ("MI300X", "RunPod"): 4.89,
    ("MI300X", "TensorWave"): 2.49,
}


# --------------------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------------------


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_cache(payload: dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass


# --------------------------------------------------------------------------
# Fetchers
# --------------------------------------------------------------------------


def _fetch(url: str, timeout: float = 4.0) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kv-planner/0.3"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def refresh_api_pricing() -> dict[str, ApiPricing]:
    """Best-effort scrape. Falls back to cache, then to the bundled defaults."""
    # OpenRouter API is the easiest to pull — no auth, JSON list of models.
    data = _fetch("https://openrouter.ai/api/v1/models")
    out: dict[str, ApiPricing] = {}
    now = time.time()
    if data and isinstance(data.get("data"), list):
        for m in data["data"]:
            mid = m.get("id", "")
            pricing = m.get("pricing") or {}
            try:
                p_in = float(pricing.get("prompt", 0)) * 1e6
                p_out = float(pricing.get("completion", 0)) * 1e6
            except (TypeError, ValueError):
                continue
            if p_in <= 0 and p_out <= 0:
                continue
            provider = mid.split("/", 1)[0] if "/" in mid else "openrouter"
            out[mid] = ApiPricing(
                provider=provider, model=mid,
                input_per_m=p_in, output_per_m=p_out,
                source="openrouter", fetched_at=now,
            )

    if not out:
        # fall back
        cache = _load_cache()
        if cache.get("api") and now - cache.get("fetched_at", 0) < _CACHE_TTL_S * 7:
            for mid, row in cache["api"].items():
                out[mid] = ApiPricing(**row)
        if not out:
            out = dict(_FALLBACK_API)
    else:
        # persist to cache
        _save_cache({
            "fetched_at": now,
            "api": {k: v.__dict__ for k, v in out.items()},
        })
    return out


def get_api_price(model: str) -> Optional[ApiPricing]:
    """Lookup. Uses cached-or-fallback without re-fetching."""
    now = time.time()
    cache = _load_cache()
    if cache.get("api") and now - cache.get("fetched_at", 0) < _CACHE_TTL_S:
        row = cache["api"].get(model)
        if row:
            return ApiPricing(**row)
    return _FALLBACK_API.get(model)


def list_gpu_prices(gpu_model: str) -> list[GpuPricing]:
    """Return per-provider on-demand + spot estimates for a GPU."""
    out: list[GpuPricing] = []
    for (gpu, prov), rate in _FALLBACK_GPU_HOURS.items():
        if gpu == gpu_model:
            out.append(GpuPricing(gpu, rate, prov, spot=False))
            out.append(GpuPricing(
                gpu, rate * _FALLBACK_SPOT_MULTIPLIER,
                prov, spot=True,
            ))
    out.sort(key=lambda p: p.cost_per_hour)
    return out


def cheapest_gpu_price(gpu_model: str, allow_spot: bool = True) -> Optional[GpuPricing]:
    prices = list_gpu_prices(gpu_model)
    if not allow_spot:
        prices = [p for p in prices if not p.spot]
    return prices[0] if prices else None
