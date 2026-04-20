"""MCP stdio server — JSON-RPC 2.0 over stdin/stdout.

Zero deps. Implements just enough of the MCP 2025-11-25 spec to register
tools and respond to tools/list + tools/call. Streaming responses (SSE)
are not implemented — tool outputs here are fast and fit in one response.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable


PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "kv-planner"
SERVER_VERSION = "0.3.0"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: Callable[[dict], Any],
    ) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def as_listing(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


_TOOLS: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _TOOLS[tool.name] = tool


# ---------------------------------------------------------------------------
# Tool handlers — each imports lazily to keep startup fast
# ---------------------------------------------------------------------------


def _tool_system_info(_args: dict) -> dict:
    from kv_planner.infrastructure.hw_detect import detect
    from kv_planner.infrastructure.runtime_probe import probe_all

    hw = detect()
    runtimes = probe_all()
    return {
        "cpu_cores": hw.cpu_cores,
        "ram_total_gb": round(hw.ram_total_gb, 1),
        "gpu": hw.gpu_matched_db_key,
        "vram_gb": round(hw.gpu_vram_gb, 1),
        "runtimes": [
            {"name": r.name, "reachable": r.reachable, "models": r.models}
            for r in runtimes
        ],
    }


def _tool_plan(args: dict) -> dict:
    from kv_planner.application import DeploymentPlanner
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args["model"])
    if entry is None:
        return {"error": f"unknown model slug {args['model']}"}
    plan = DeploymentPlanner().create_plan(
        model=entry.config,
        hardware=args.get("gpu", "H100-SXM-80GB"),
        target_rps=float(args.get("rps", 10.0)),
        input_length=int(args.get("input_length", 2048)),
        output_length=int(args.get("output_length", 512)),
        optimization_goal=args.get("goal", "balanced"),  # type: ignore[arg-type]
    )
    return {
        "model": plan.model.name,
        "gpu": plan.hardware.gpu_model,
        "precision": plan.recommended_precision,
        "batch_size": plan.recommended_batch_size,
        "throughput_tok_s": round(plan.performance.throughput_tokens_per_sec, 0),
        "total_latency_ms": round(plan.performance.total_latency_ms, 0),
        "memory_gb": round(plan.total_memory_gb, 2),
        "device_memory_gb": plan.hardware.gpu_memory_gb,
        "cost_per_m_tokens": round(plan.cost.cost_per_million_tokens, 3),
    }


def _tool_recommend(args: dict) -> dict:
    from kv_planner.application.recommender import Recommender
    from kv_planner.infrastructure.hardware_db import GPUDatabase
    from kv_planner.infrastructure.hw_detect import detect

    gpu = args.get("gpu") or detect().gpu_matched_db_key or "H100-SXM-80GB"
    spec = GPUDatabase.to_hardware_spec(gpu)
    recs = Recommender().top_n(
        spec, n=int(args.get("limit", 5)),
        use_case=args.get("use_case", "general"),  # type: ignore[arg-type]
    )
    return {
        "gpu": gpu,
        "use_case": args.get("use_case", "general"),
        "models": [
            {
                "slug": r.entry.slug,
                "composite_score": round(r.score_composite, 1),
                "throughput_tok_s": round(r.throughput_tok_s, 0),
                "memory_gb": round(r.memory_gb, 2),
                "precision": r.precision,
            }
            for r in recs
        ],
    }


def _tool_size_fleet(args: dict) -> dict:
    from kv_planner.application.fleet import FleetPlanner
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args["model"])
    if entry is None:
        return {"error": f"unknown model slug {args['model']}"}
    gpus = args.get("gpu_candidates") or [
        "H100-SXM-80GB", "A100-SXM-80GB", "RTX-5090",
    ]
    designs = FleetPlanner().design(
        model=entry.config,
        target_rps=float(args["target_rps"]),
        input_length=int(args.get("input_length", 2048)),
        output_length=int(args.get("output_length", 512)),
        gpu_candidates=gpus,
        p99_latency_ms=float(args.get("p99_latency_ms", 3000)),
    )[:5]
    return {
        "designs": [
            {
                "gpu": d.gpu_model, "tp": d.tp_size, "replicas": d.replicas,
                "precision": d.precision,
                "cost_per_hour": round(d.cost_per_hour, 2),
                "cost_per_million_tokens": round(d.cost_per_million_tokens, 3),
                "p99_latency_ms": round(d.p99_latency_ms, 0),
                "meets_slo": d.meets_slo,
            }
            for d in designs
        ],
    }


def _tool_explain(args: dict) -> dict:
    from kv_planner.application.rationale import explain
    from kv_planner.application.recommender import Recommender
    from kv_planner.infrastructure.hardware_db import GPUDatabase
    from kv_planner.infrastructure.hw_detect import detect

    gpu = args.get("gpu") or detect().gpu_matched_db_key or "H100-SXM-80GB"
    spec = GPUDatabase.to_hardware_spec(gpu)
    recs = Recommender().recommend(
        spec, use_case=args.get("use_case", "general"),  # type: ignore[arg-type]
    )
    r = next((r for r in recs if r.entry.slug == args["slug"]), None)
    if r is None:
        return {"error": f"unknown slug {args['slug']}"}
    rat = explain(r, spec)
    return {
        "slug": r.entry.slug, "gpu": gpu,
        "verdict": rat.verdict,
        "bullets": rat.bullets,
        "caveats": rat.caveats,
        "scores": {
            "quality": r.score_quality, "fit": r.score_fit,
            "speed": r.score_speed, "context": r.score_context,
            "composite": round(r.score_composite, 1),
        },
    }


def _tool_memory_waterfall(args: dict) -> dict:
    from kv_planner.core.explain.waterfall import build_waterfall
    from kv_planner.infrastructure.hardware_db import GPUDatabase
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args["model"])
    if entry is None:
        return {"error": f"unknown model slug {args['model']}"}
    hw = GPUDatabase.to_hardware_spec(args.get("gpu", "H100-SXM-80GB"))
    w = build_waterfall(
        entry.config, hw,
        batch_size=int(args.get("batch", 1)),
        input_length=int(args.get("input_length", 2048)),
        output_length=int(args.get("output_length", 512)),
        precision=args.get("precision", "fp16"),  # type: ignore[arg-type]
    )
    return {
        "model": entry.slug,
        "gpu": hw.gpu_model,
        "total_gb": round(w.total_gb, 2),
        "budget_gb": round(w.budget_gb, 2),
        "fits": w.fits,
        "headroom_gb": round(w.headroom_gb, 2),
        "terms": [
            {
                "label": t.label,
                "bytes": t.bytes_,
                "gb": round(t.gb, 3),
                "formula": t.formula,
                "citation": t.citation,
                "note": t.note,
            }
            for t in w.terms
        ],
    }


def _tool_speculative(args: dict) -> dict:
    from kv_planner.core.performance.speculative import plan as spec_plan
    from kv_planner.infrastructure.model_catalog import by_slug

    target = by_slug(args["target"])
    if target is None:
        return {"error": f"unknown target {args['target']}"}
    draft = by_slug(args.get("draft", ""))
    draft_params = draft.config.total_params() if draft else 0
    r = spec_plan(
        method=args.get("method", "eagle3"),
        target_model_params=target.config.total_params(),
        draft_model_params=draft_params,
        target_tpot_ms=float(args.get("target_tpot_ms", 30.0)),
        target_kv_bytes_per_token=target.config.kv_cache_bytes_per_token("fp16"),
    )
    return {
        "method": r.method, "K": r.K,
        "acceptance_rate": r.acceptance_rate,
        "expected_tokens_per_verify_step": round(r.expected_tokens_per_verify_step, 2),
        "speedup": round(r.speedup, 2),
        "percent_faster": round(r.percent_faster, 1),
        "effective_tpot_ms": round(r.effective_tpot_ms, 2),
    }


def _tool_reasoning(args: dict) -> dict:
    from kv_planner.core.performance.reasoning import PROFILES, plan_reasoning
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args["model"])
    if entry is None:
        return {"error": f"unknown model {args['model']}"}
    profile = PROFILES.get(args.get("thinking_profile", "balanced"), PROFILES["balanced"])
    p = plan_reasoning(
        entry.config, profile,
        prompt_tokens=int(args.get("prompt_tokens", 500)),
        batch_size=int(args.get("batch", 1)),
        precision=args.get("precision", "fp16"),  # type: ignore[arg-type]
    )
    return {
        "model": p.model, "precision": p.precision,
        "p99_context_tokens": p.p99_context_tokens,
        "kv_gb_mean": round(p.kv_gb_mean_batch, 3),
        "kv_gb_p99": round(p.kv_gb_p99_batch, 3),
        "p99_over_mean_ratio": round(p.p99_over_mean_ratio, 2),
        "thinking_profile": args.get("thinking_profile", "balanced"),
    }


def _tool_carbon(args: dict) -> dict:
    from kv_planner.core.cost.carbon import estimate_carbon
    from kv_planner.infrastructure.hardware_db import GPUDatabase

    gpu_model = args.get("gpu", "H100-SXM-80GB")
    gpu = GPUDatabase.get(gpu_model)
    tdp = gpu.typical_tdp_w if gpu else 700
    c = estimate_carbon(
        throughput_tok_s=float(args["throughput_tok_s"]),
        tdp_watts=tdp,
        mfu=float(args.get("mfu", 0.4)),
        mbu=float(args.get("mbu", 0.6)),
        region=args.get("region", "us-east"),
        num_gpus=int(args.get("num_gpus", 1)),
    )
    return {
        "gpu_watts_avg": round(c.gpu_watts_avg, 0),
        "kwh_per_million_tokens": round(c.kwh_per_million_tokens, 4),
        "grid_intensity_g_per_kwh": c.grid_intensity_g_per_kwh,
        "g_co2e_per_million_tokens": round(c.g_co2e_per_million_tokens, 2),
        "g_co2e_per_request": round(c.g_co2e_per_request, 4),
        "region": c.region,
    }


def _register_all() -> None:
    register(Tool(
        "system_info",
        "Detect the local CPU / RAM / GPU and probe reachable LLM runtimes (Ollama, LM Studio, vLLM, llama.cpp).",
        {"type": "object", "properties": {}},
        _tool_system_info,
    ))
    register(Tool(
        "plan_deployment",
        "Create a deployment plan (precision + batch + latency + cost) for one model on one GPU.",
        {
            "type": "object",
            "required": ["model"],
            "properties": {
                "model": {"type": "string", "description": "catalog slug (e.g., llama-3-8b)"},
                "gpu": {"type": "string", "default": "H100-SXM-80GB"},
                "rps": {"type": "number", "default": 10},
                "input_length": {"type": "integer", "default": 2048},
                "output_length": {"type": "integer", "default": 512},
                "goal": {"type": "string", "enum": ["balanced", "cost", "latency", "throughput", "quality"]},
            },
        },
        _tool_plan,
    ))
    register(Tool(
        "recommend_models",
        "Top-N models for a given GPU and use case, ranked by physics-based composite score.",
        {
            "type": "object",
            "properties": {
                "gpu": {"type": "string"},
                "use_case": {"type": "string", "enum": ["general", "coding", "reasoning", "chat", "multimodal", "agent"]},
                "limit": {"type": "integer", "default": 5},
            },
        },
        _tool_recommend,
    ))
    register(Tool(
        "size_fleet",
        "Design the cheapest cluster (GPU × TP × precision × replicas) for a target RPS and p99 latency SLO.",
        {
            "type": "object",
            "required": ["model", "target_rps"],
            "properties": {
                "model": {"type": "string"},
                "target_rps": {"type": "number"},
                "p99_latency_ms": {"type": "number", "default": 3000},
                "input_length": {"type": "integer", "default": 2048},
                "output_length": {"type": "integer", "default": 512},
                "gpu_candidates": {"type": "array", "items": {"type": "string"}},
            },
        },
        _tool_size_fleet,
    ))
    register(Tool(
        "explain_model",
        "Physics-grounded rationale for why one model ranks where it does on a given GPU/use-case.",
        {
            "type": "object",
            "required": ["slug"],
            "properties": {
                "slug": {"type": "string"},
                "gpu": {"type": "string"},
                "use_case": {"type": "string"},
            },
        },
        _tool_explain,
    ))
    register(Tool(
        "memory_waterfall",
        "Memory decomposition — weights + KV + activations + workspace, each with formula and source URL.",
        {
            "type": "object",
            "required": ["model"],
            "properties": {
                "model": {"type": "string"},
                "gpu": {"type": "string"},
                "batch": {"type": "integer"},
                "input_length": {"type": "integer"},
                "output_length": {"type": "integer"},
                "precision": {"type": "string"},
            },
        },
        _tool_memory_waterfall,
    ))
    register(Tool(
        "speculative_decode",
        "Model EAGLE-3 / Medusa / Lookahead / draft-model speculative decoding speedup for a target.",
        {
            "type": "object",
            "required": ["target"],
            "properties": {
                "target": {"type": "string"},
                "draft": {"type": "string"},
                "method": {"type": "string", "enum": ["eagle3", "medusa", "lookahead", "draft_model"]},
                "target_tpot_ms": {"type": "number"},
            },
        },
        _tool_speculative,
    ))
    register(Tool(
        "reasoning_plan",
        "KV-memory plan for a reasoning model — mean and p99 thinking-token distributions.",
        {
            "type": "object",
            "required": ["model"],
            "properties": {
                "model": {"type": "string"},
                "thinking_profile": {"type": "string", "enum": ["balanced", "deepseek-r1-math", "o3-mini-chat", "qwq-code"]},
                "precision": {"type": "string"},
                "prompt_tokens": {"type": "integer"},
                "batch": {"type": "integer"},
            },
        },
        _tool_reasoning,
    ))
    register(Tool(
        "carbon_estimate",
        "gCO2e per million tokens given throughput, GPU TDP, MFU/MBU utilisation, and regional grid intensity.",
        {
            "type": "object",
            "required": ["throughput_tok_s"],
            "properties": {
                "throughput_tok_s": {"type": "number"},
                "gpu": {"type": "string"},
                "region": {"type": "string", "description": "us-east / us-west / france / iceland / etc."},
                "num_gpus": {"type": "integer"},
                "mfu": {"type": "number"},
                "mbu": {"type": "number"},
            },
        },
        _tool_carbon,
    ))


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 dispatcher
# ---------------------------------------------------------------------------


def _dispatch(msg: dict) -> dict | None:
    method = msg.get("method")
    mid = msg.get("id")

    if method == "initialize":
        return _resp(mid, {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {"tools": {"listChanged": False}},
        })
    if method == "notifications/initialized":
        # Notifications (no id) get no response.
        return None
    if method == "tools/list":
        return _resp(mid, {
            "tools": [t.as_listing() for t in _TOOLS.values()],
        })
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        tool = _TOOLS.get(name)
        if tool is None:
            return _err(mid, -32601, f"unknown tool: {name}")
        args = params.get("arguments") or {}
        try:
            out = tool.handler(args)
        except Exception as e:
            tb = traceback.format_exc(limit=3)
            return _err(mid, -32000, f"{type(e).__name__}: {e}", data={"traceback": tb})
        return _resp(mid, {"content": [{"type": "text", "text": json.dumps(out, indent=2)}]})
    if method == "ping":
        return _resp(mid, {})
    return _err(mid, -32601, f"method not found: {method}")


def _resp(mid, result) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _err(mid, code, message, data=None) -> dict:
    out = {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}
    if data is not None:
        out["error"]["data"] = data
    return out


# ---------------------------------------------------------------------------
# stdio loop
# ---------------------------------------------------------------------------


def main() -> int:
    _register_all()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = _dispatch(msg)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0
