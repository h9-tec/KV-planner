"""Tests for speculative decoding, reasoning KV planner, carbon, waterfall, pricing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from kv_planner.core.cost.carbon import estimate_carbon, get_grid_intensity
from kv_planner.core.explain.waterfall import build_waterfall
from kv_planner.core.performance.reasoning import PROFILES, plan_reasoning
from kv_planner.core.performance.speculative import (
    expected_accepted_tokens,
    plan as spec_plan,
    speedup_over_autoregressive,
)
from kv_planner.application.diagnose import diagnose
from kv_planner.infrastructure import pricing
from kv_planner.infrastructure.hardware_db import GPUDatabase


# ---- speculative decoding ------------------------------------------------


def test_expected_accepted_tokens_bounds():
    assert expected_accepted_tokens(0.0, 5) == pytest.approx(1.0)
    assert expected_accepted_tokens(1.0, 5) == pytest.approx(6.0)
    v = expected_accepted_tokens(0.8, 5)
    assert 1.0 < v < 6.0


def test_speedup_zero_accept_is_no_speedup():
    assert speedup_over_autoregressive(0.0, 5, 0.02) == pytest.approx(1.0)


def test_speedup_high_accept_large_window_gives_multiple_x():
    s = speedup_over_autoregressive(0.9, 8, 0.02)
    assert s > 2.0


def test_eagle3_plan_llama8b():
    # Llama-3 8B target: ~8 GFLOPs/token, ~128 KiB KV
    r = spec_plan(
        method="eagle3",
        target_model_params=8_000_000_000,
        draft_model_params=0,   # EAGLE head is tiny, we use the 0.02 ratio
        target_tpot_ms=25.0,
        target_kv_bytes_per_token=131072,
    )
    assert r.method == "eagle3"
    assert r.acceptance_rate == pytest.approx(0.80)
    assert r.speedup > 1.5, f"EAGLE-3 should give ≥1.5× speedup, got {r.speedup}"


# ---- reasoning plan ------------------------------------------------------


def test_reasoning_p99_greater_than_mean(llama3_8b):
    wk = PROFILES["deepseek-r1-math"]
    p = plan_reasoning(llama3_8b, wk, prompt_tokens=500, batch_size=1, precision="fp16")
    assert p.kv_bytes_p99_per_seq > p.kv_bytes_mean_per_seq
    assert p.p99_over_mean_ratio > 1.0


def test_reasoning_scales_linearly_with_batch(llama3_8b):
    wk = PROFILES["balanced"]
    p1 = plan_reasoning(llama3_8b, wk, prompt_tokens=500, batch_size=1)
    p4 = plan_reasoning(llama3_8b, wk, prompt_tokens=500, batch_size=4)
    assert p4.kv_gb_p99_batch == pytest.approx(p1.kv_gb_p99_batch * 4, rel=0.01)


# ---- memory waterfall ---------------------------------------------------


def test_waterfall_terms_sum_to_total(llama3_8b):
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    w = build_waterfall(llama3_8b, h, batch_size=32,
                       input_length=2048, output_length=512, precision="fp16")
    assert sum(t.bytes_ for t in w.terms) == w.total_bytes


def test_waterfall_every_term_cites_a_url(llama3_8b):
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    w = build_waterfall(llama3_8b, h, 1, 512, 128, "fp16")
    # The fragmentation term is 0 bytes but still has a citation.
    assert all(t.citation or t.bytes_ == 0 for t in w.terms)


def test_waterfall_fits_llama8b_on_h100(llama3_8b):
    h = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    w = build_waterfall(llama3_8b, h, 32, 2048, 512, "fp16")
    assert w.fits is True


# ---- diagnose -----------------------------------------------------------


def test_diagnose_finds_culprit_on_oversized_config(llama3_70b):
    h = GPUDatabase.to_hardware_spec("RTX-4090")
    d = diagnose(llama3_70b, h, batch_size=16,
                 input_length=2048, output_length=512, precision="fp16")
    assert not d.waterfall.fits
    assert d.culprit_term is not None
    assert d.overflow_gb > 0


def test_diagnose_proposes_at_least_one_fix(llama3_8b):
    h = GPUDatabase.to_hardware_spec("RTX-3090")  # 24 GB
    d = diagnose(llama3_8b, h, batch_size=32,
                 input_length=4096, output_length=1024, precision="fp16")
    assert len(d.suggestions) >= 1
    # at least one of the proposed fixes fits after quantize / shrink
    assert any(s.fits for s in d.suggestions), \
        f"no fix fits: {[(s.change, s.fits, s.new_memory_gb) for s in d.suggestions]}"


# ---- carbon -------------------------------------------------------------


def test_grid_intensity_iceland_is_lowest():
    assert get_grid_intensity("iceland") < get_grid_intensity("india")
    assert get_grid_intensity("france") < get_grid_intensity("global")


def test_carbon_gco2e_is_positive_and_finite():
    c = estimate_carbon(
        throughput_tok_s=5000, tdp_watts=700, mfu=0.4, mbu=0.6,
        region="us-east", num_gpus=1,
    )
    assert 0 < c.g_co2e_per_million_tokens < 10000  # sanity band


# ---- pricing ------------------------------------------------------------


def test_pricing_lists_multiple_providers_per_gpu():
    rows = pricing.list_gpu_prices("H100-SXM-80GB")
    providers = {r.provider for r in rows}
    assert "AWS" in providers
    assert "RunPod" in providers
    # Spot variants included
    assert any(r.spot for r in rows)
    assert any(not r.spot for r in rows)


def test_pricing_spot_cheaper_than_on_demand():
    rows = pricing.list_gpu_prices("H100-SXM-80GB")
    for provider in {r.provider for r in rows}:
        on_demand = next((r for r in rows if r.provider == provider and not r.spot), None)
        spot = next((r for r in rows if r.provider == provider and r.spot), None)
        if on_demand and spot:
            assert spot.cost_per_hour < on_demand.cost_per_hour


def test_api_pricing_fallback_has_entries():
    p = pricing.get_api_price("gpt-4o")
    assert p is not None
    assert p.provider == "openai"
    assert p.input_per_m > 0


# ---- MCP server integration ---------------------------------------------


def _mcp_rpc(*messages: dict) -> list[dict]:
    """Send messages to `kv-planner mcp` on stdin, return parsed stdout lines."""
    root = Path(__file__).resolve().parents[3]
    venv_py = root / ".venv" / "bin" / "python"
    if not venv_py.exists():
        pytest.skip("venv python not found")
    payload = "\n".join(json.dumps(m) for m in messages) + "\n"
    proc = subprocess.run(
        [str(venv_py), "-m", "kv_planner.cli.main", "mcp"],
        input=payload, capture_output=True, text=True, timeout=30,
    )
    out = []
    for line in proc.stdout.strip().splitlines():
        if line.startswith("{"):
            out.append(json.loads(line))
    return out


def test_mcp_initialize_returns_protocol_version():
    r = _mcp_rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r and r[0]["result"]["protocolVersion"] == "2025-11-25"
    assert r[0]["result"]["serverInfo"]["name"] == "kv-planner"


def test_mcp_tools_list_has_nine_tools():
    r = _mcp_rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tools = r[1]["result"]["tools"]
    names = {t["name"] for t in tools}
    assert len(tools) == 9
    assert {"system_info", "plan_deployment", "memory_waterfall",
            "speculative_decode", "reasoning_plan", "carbon_estimate"} <= names


def test_mcp_tools_call_memory_waterfall():
    r = _mcp_rpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
         "params": {"name": "memory_waterfall",
                    "arguments": {"model": "llama-3-8b",
                                  "gpu": "H100-SXM-80GB",
                                  "batch": 1, "input_length": 512,
                                  "output_length": 128, "precision": "fp16"}}},
    )
    assert r[1].get("result"), f"expected result, got {r[1]}"
    content = r[1]["result"]["content"][0]["text"]
    parsed = json.loads(content)
    assert parsed["model"] == "llama-3-8b"
    assert len(parsed["terms"]) >= 4
