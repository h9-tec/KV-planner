"""MoE regression fixtures — active vs total params, decode scaling."""

from __future__ import annotations

import pytest

from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.model_catalog import by_slug


def test_mixtral_has_moe_flag():
    e = by_slug("mixtral-8x7b")
    assert e is not None
    assert e.config.is_moe
    assert e.config.num_experts == 8
    assert e.config.num_experts_per_token == 2


def test_mixtral_active_is_about_13b():
    """Mistral publishes Mixtral-8x7B as '12.9B active'."""
    e = by_slug("mixtral-8x7b")
    active = e.config.active_params()
    assert 11.5e9 < active < 14e9, f"expected ~13B active, got {active/1e9:.1f}B"


def test_mixtral_total_is_about_47b():
    """Mistral publishes Mixtral-8x7B as '46.7B total' ."""
    e = by_slug("mixtral-8x7b")
    total = e.config.total_params()
    assert 45e9 < total < 50e9, f"expected ~47B total, got {total/1e9:.1f}B"


def test_qwen3_30b_a3b_ratio_matches_name():
    """Qwen3-30B-A3B has 30B total / ~3B active — matches its 'A3B' marketing."""
    e = by_slug("qwen3-30b-a3b")
    total = e.config.total_params() / 1e9
    active = e.config.active_params() / 1e9
    assert 28 < total < 32, f"total {total}"
    assert 2 < active < 4, f"active {active}"


def test_dense_models_active_equals_total(llama3_8b):
    """Dense models: active == total."""
    assert llama3_8b.active_params() == llama3_8b.total_params()


def test_mixtral_decode_faster_than_40b_dense():
    """MoE decode bandwidth = active params, so Mixtral should NOT decode
    like a 47B dense model — it should decode like a ~13B dense model.

    This is the canonical MoE capacity-planning insight and the reason
    the physics engine distinguishes active vs total."""
    ra = RooflineAnalyzer()
    gpu = GPUDatabase.to_hardware_spec("H100-SXM-80GB")

    mixtral_cfg = by_slug("mixtral-8x7b").config
    # llama-3-70b approximates "what decode would look like if Mixtral
    # behaved like a dense model of its total-param count"
    llama_70b_cfg = by_slug("llama-3-70b").config

    ra_m = ra.predict_latency(
        mixtral_cfg, gpu, batch_size=1,
        input_length=512, output_length=128, precision="fp16",
    )
    ra_d = ra.predict_latency(
        llama_70b_cfg, gpu, batch_size=1,
        input_length=512, output_length=128, precision="fp16",
    )
    assert ra_m.decode_latency_ms < ra_d.decode_latency_ms, (
        f"MoE should decode faster than its total-params dense equivalent; "
        f"Mixtral {ra_m.decode_latency_ms:.0f} ms vs 70B dense {ra_d.decode_latency_ms:.0f} ms"
    )


def test_four_moe_models_in_catalog():
    slugs = {"mixtral-8x7b", "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2-lite"}
    from kv_planner.infrastructure.model_catalog import CATALOG
    catalog_moe = {e.slug for e in CATALOG if e.config.is_moe}
    assert slugs <= catalog_moe, f"missing: {slugs - catalog_moe}"
