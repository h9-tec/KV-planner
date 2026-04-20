#!/usr/bin/env python3
"""
kv-planner CLI - Command-line interface for LLM deployment planning.

Usage:
    kv-planner plan --model MODEL --gpu GPU [OPTIONS]
    kv-planner compare --model MODEL --gpus GPU1,GPU2,GPU3 [OPTIONS]
    kv-planner list-gpus [--filter FILTER]
    kv-planner benchmark --model MODEL --gpu GPU [OPTIONS]
    kv-planner validate --plan-file PLAN --benchmark-file BENCHMARK
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from kv_planner.application import DeploymentPlanner, export
from kv_planner.application.fleet import FleetPlanner
from kv_planner.application.rationale import explain as explain_rec
from kv_planner.application.recommender import Recommender
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    PredictionValidator,
)
from kv_planner.infrastructure.benchmarks.runner import create_config_from_plan
from kv_planner.infrastructure.hw_detect import detect as detect_hardware
from kv_planner.infrastructure.model_catalog import CATALOG, match_ollama_name
from kv_planner.infrastructure.runtime_probe import probe_all as probe_runtimes


def _positive_float(value: str) -> float:
    """argparse ``type=`` callable: strictly positive float."""
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return f


def _positive_int(value: str) -> int:
    """argparse ``type=`` callable: strictly positive int."""
    i = int(value)
    if i <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return i


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="kv-planner",
        description="KV cache memory and throughput planner for LLM deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Plan command
    plan_parser = subparsers.add_parser(
        "plan",
        help="Create deployment plan",
        description="Generate complete deployment plan with recommendations",
    )
    plan_parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., meta-llama/Llama-3-8b-hf)",
    )
    plan_parser.add_argument(
        "--gpu",
        required=True,
        help="GPU model (e.g., RTX-5090, H100-80GB)",
    )
    plan_parser.add_argument(
        "--rps",
        type=_positive_float,
        default=10.0,
        help="Target requests per second (default: 10.0)",
    )
    plan_parser.add_argument(
        "--input-length",
        type=_positive_int,
        default=2048,
        help="Average input length in tokens (default: 2048)",
    )
    plan_parser.add_argument(
        "--output-length",
        type=_positive_int,
        default=512,
        help="Average output length in tokens (default: 512)",
    )
    plan_parser.add_argument(
        "--optimization-goal",
        "--goal",
        dest="goal",
        choices=["cost", "latency", "throughput", "quality", "balanced"],
        default="balanced",
        help="Optimization goal (default: balanced)",
    )
    plan_parser.add_argument(
        "--max-memory",
        type=float,
        help="Maximum memory budget in GB",
    )
    plan_parser.add_argument(
        "--min-quality",
        choices=["none", "minimal", "slight", "moderate"],
        default="minimal",
        help="Minimum quality requirement (default: minimal)",
    )
    plan_parser.add_argument(
        "--no-caching",
        action="store_true",
        help="Disable prefix caching",
    )
    plan_parser.add_argument(
        "--system-prompt-length",
        type=int,
        default=512,
        help="System prompt length for caching (default: 512)",
    )
    plan_parser.add_argument(
        "--output",
        "-o",
        help="Output file (supports .json, .yaml, .md)",
    )
    plan_parser.add_argument(
        "--format",
        choices=["text", "json", "yaml"],
        default="text",
        help="Output format (default: text)",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple GPUs",
        description="Compare deployment plans across different GPUs",
    )
    compare_parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., meta-llama/Llama-3-8b-hf)",
    )
    compare_parser.add_argument(
        "--gpus",
        required=True,
        help="Comma-separated GPU models (e.g., RTX-5090,RTX-4090,RTX-3090)",
    )
    compare_parser.add_argument(
        "--rps",
        type=float,
        default=10.0,
        help="Target requests per second (default: 10.0)",
    )
    compare_parser.add_argument(
        "--input-length",
        type=int,
        default=2048,
        help="Average input length in tokens (default: 2048)",
    )
    compare_parser.add_argument(
        "--output-length",
        type=int,
        default=512,
        help="Average output length in tokens (default: 512)",
    )
    compare_parser.add_argument(
        "--optimization-goal",
        "--goal",
        dest="goal",
        choices=["cost", "latency", "throughput", "quality", "balanced"],
        default="balanced",
        help="Optimization goal (default: balanced)",
    )
    compare_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # ----- recommend ------------------------------------------------------
    rec_parser = subparsers.add_parser(
        "recommend",
        help="Recommend models for the detected (or specified) hardware",
        description="Rank popular open-weight models by physics-based fit + "
                    "quality + speed + context score.",
    )
    rec_parser.add_argument("--gpu", help="GPU key (default: auto-detect)")
    rec_parser.add_argument("--use-case", default="general",
                            choices=["general", "coding", "reasoning",
                                     "chat", "multimodal", "agent"])
    rec_parser.add_argument("-n", "--limit", type=_positive_int, default=10)
    rec_parser.add_argument("--input-length", type=_positive_int, default=2048)
    rec_parser.add_argument("--output-length", type=_positive_int, default=512)
    rec_parser.add_argument("--batch-size", type=_positive_int, default=1)
    rec_parser.add_argument("--include-unfit", action="store_true",
                            help="Include models that won't fit")
    rec_parser.add_argument("--json", action="store_true",
                            help="Emit machine-readable JSON")

    # ----- system ---------------------------------------------------------
    sys_parser = subparsers.add_parser(
        "system",
        help="Show detected CPU / RAM / GPU and runtime providers",
    )
    sys_parser.add_argument("--json", action="store_true")

    # ----- installed ------------------------------------------------------
    inst_parser = subparsers.add_parser(
        "installed",
        help="List locally installed models across Ollama / LM Studio / vLLM / llama-server",
    )
    inst_parser.add_argument("--json", action="store_true")

    # ----- fleet ----------------------------------------------------------
    fleet_parser = subparsers.add_parser(
        "fleet",
        help="Cluster-sizing: given RPS + latency SLO, design the cheapest fleet",
        description=(
            "Ranks (gpu × tensor_parallel × precision × replicas) combinations "
            "by $/M tokens. Each design meets a p99 latency SLO or is marked."
        ),
    )
    fleet_parser.add_argument("--model", required=True,
                              help="Catalog slug (llama-3-8b, qwen2.5-14b, …)")
    fleet_parser.add_argument("--rps", type=_positive_float, required=True,
                              help="Target requests/sec across the fleet")
    fleet_parser.add_argument("--input-length", type=_positive_int, default=2048)
    fleet_parser.add_argument("--output-length", type=_positive_int, default=512)
    fleet_parser.add_argument("--p99-latency-ms", type=_positive_float, default=2000.0,
                              help="Latency SLO (default 2000 ms)")
    fleet_parser.add_argument("--gpus", default="H100-SXM-80GB,H200-SXM-141GB,A100-SXM-80GB,L40S,RTX-5090",
                              help="Comma-separated GPU candidates")
    fleet_parser.add_argument("--tp", default="1,2,4,8",
                              help="TP sizes to consider (default: 1,2,4,8)")
    fleet_parser.add_argument("--precisions", default="fp16,fp8,int4")
    fleet_parser.add_argument("-n", "--limit", type=_positive_int, default=10)
    fleet_parser.add_argument("--include-missing", action="store_true",
                              help="Include designs that miss the SLO")
    fleet_parser.add_argument("--json", action="store_true")

    # ----- explain --------------------------------------------------------
    exp_parser = subparsers.add_parser(
        "explain",
        help="Physics-grounded rationale for a recommended model",
    )
    exp_parser.add_argument("--model", required=True,
                            help="Catalog slug to explain")
    exp_parser.add_argument("--gpu", help="GPU key (default: auto-detect)")
    exp_parser.add_argument("--use-case", default="general",
                            choices=["general", "coding", "reasoning",
                                     "chat", "multimodal", "agent"])
    exp_parser.add_argument("--input-length", type=_positive_int, default=2048)
    exp_parser.add_argument("--output-length", type=_positive_int, default=512)
    exp_parser.add_argument("--batch-size", type=_positive_int, default=1)
    exp_parser.add_argument("--json", action="store_true")

    # ----- why (memory waterfall) ----------------------------------------
    why_parser = subparsers.add_parser(
        "why",
        help="Memory waterfall — show every term, its formula, and a citation URL",
    )
    why_parser.add_argument("--model", required=True, help="Catalog slug")
    why_parser.add_argument("--gpu", help="GPU key (default: auto-detect)")
    why_parser.add_argument("--batch", type=_positive_int, default=1)
    why_parser.add_argument("--input-length", type=_positive_int, default=2048)
    why_parser.add_argument("--output-length", type=_positive_int, default=512)
    why_parser.add_argument("--precision", default="fp16",
                            choices=["fp32", "fp16", "bf16", "fp8", "int8", "int4"])
    why_parser.add_argument("--tp", type=_positive_int, default=None)
    why_parser.add_argument("--json", action="store_true")

    # ----- diagnose (OOM post-mortem) -------------------------------------
    diag_parser = subparsers.add_parser(
        "diagnose",
        help="OOM post-mortem — which term overflowed and the three cheapest fixes",
    )
    diag_parser.add_argument("--model", required=True)
    diag_parser.add_argument("--gpu", required=True)
    diag_parser.add_argument("--batch", type=_positive_int, default=32)
    diag_parser.add_argument("--input-length", type=_positive_int, default=2048)
    diag_parser.add_argument("--output-length", type=_positive_int, default=512)
    diag_parser.add_argument("--precision", default="fp16")
    diag_parser.add_argument("--tp", type=_positive_int, default=None)
    diag_parser.add_argument("--vllm-cmdline",
                             help="Optional: paste a vLLM command line to parse args from")
    diag_parser.add_argument("--json", action="store_true")

    # ----- specdec (speculative decoding physics) ------------------------
    spec_parser = subparsers.add_parser(
        "specdec",
        help="Speculative-decoding speedup estimate (EAGLE-3/Medusa/Lookahead/draft)",
    )
    spec_parser.add_argument("--target", required=True)
    spec_parser.add_argument("--draft", help="Catalog slug for draft model (draft_model method)")
    spec_parser.add_argument("--method", default="eagle3",
                             choices=["eagle3", "medusa", "lookahead", "draft_model"])
    spec_parser.add_argument("--target-tpot-ms", type=float, default=30.0)
    spec_parser.add_argument("-K", "--spec-k", type=_positive_int, default=None)
    spec_parser.add_argument("--acceptance", type=float, default=None)
    spec_parser.add_argument("--json", action="store_true")

    # ----- reasoning (p99 KV planning for thinking models) ---------------
    reason_parser = subparsers.add_parser(
        "reasoning",
        help="Reasoning-model KV planning — p99 thinking-token distribution",
    )
    reason_parser.add_argument("--model", required=True)
    reason_parser.add_argument("--profile", default="balanced",
                               choices=["balanced", "deepseek-r1-math", "o3-mini-chat", "qwq-code"])
    reason_parser.add_argument("--prompt-tokens", type=_positive_int, default=500)
    reason_parser.add_argument("--batch", type=_positive_int, default=1)
    reason_parser.add_argument("--precision", default="fp16")
    reason_parser.add_argument("--gpu", help="GPU key (for fit check)")
    reason_parser.add_argument("--json", action="store_true")

    # ----- carbon (gCO2e per M tokens) -----------------------------------
    carbon_parser = subparsers.add_parser(
        "carbon",
        help="Carbon footprint — gCO2e per million tokens for a deployment",
    )
    carbon_parser.add_argument("--model", required=True)
    carbon_parser.add_argument("--gpu", required=True)
    carbon_parser.add_argument("--region", default="us-east",
                               help="us-east / us-west / france / iceland / india / etc.")
    carbon_parser.add_argument("--num-gpus", type=_positive_int, default=1)
    carbon_parser.add_argument("--batch", type=_positive_int, default=32)
    carbon_parser.add_argument("--input-length", type=_positive_int, default=2048)
    carbon_parser.add_argument("--output-length", type=_positive_int, default=512)
    carbon_parser.add_argument("--precision", default="fp16")
    carbon_parser.add_argument("--json", action="store_true")

    # ----- pricing (live GPU + API rates) --------------------------------
    price_parser = subparsers.add_parser(
        "pricing",
        help="Live / cached GPU $/hr + API $/M-tokens rates across providers",
    )
    price_parser.add_argument("--gpu", help="Show GPU providers for a specific key")
    price_parser.add_argument("--api-model", help="Show API rate for a specific hosted model")
    price_parser.add_argument("--refresh", action="store_true",
                              help="Force refresh from OpenRouter")
    price_parser.add_argument("--json", action="store_true")

    # ----- mcp (Model Context Protocol server) ---------------------------
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run an MCP stdio server so Claude / Cursor / Cline can call kv-planner as a tool",
    )

    # ----- loadtest (concurrent HTTP load generator) ----------------------
    lt_parser = subparsers.add_parser(
        "loadtest",
        help="Concurrent load test against an OpenAI-compat or Ollama endpoint",
        description=(
            "Fires N concurrent streaming requests, measures TTFT / TPOT / "
            "E2E p50/p95/p99, aggregate tok/s, and goodput vs joint SLO."
        ),
    )
    lt_parser.add_argument("--endpoint", default="http://127.0.0.1:11434",
                           help="Base URL (default: Ollama on localhost)")
    lt_parser.add_argument("--model", required=True,
                           help="Model id (e.g., 'llama3.2:3b' or 'gpt-4o-mini')")
    lt_parser.add_argument("--api", choices=["ollama", "openai"], default="ollama")
    lt_parser.add_argument("--concurrency", type=_positive_int, default=8)
    lt_parser.add_argument("--num-requests", type=_positive_int, default=32)
    lt_parser.add_argument("--num-predict", type=_positive_int, default=128,
                           help="Max output tokens per request")
    lt_parser.add_argument("--prompt", default="Explain attention mechanisms in transformers.")
    lt_parser.add_argument("--prompt-file", help="Read prompts from a file (one per line)")
    lt_parser.add_argument("--api-key", help="API key for OpenAI-compat endpoints")
    lt_parser.add_argument("--ttft-slo-ms", type=float, default=None)
    lt_parser.add_argument("--tpot-slo-ms", type=float, default=None)
    lt_parser.add_argument("--e2e-slo-ms", type=float, default=None)
    lt_parser.add_argument("--timeout", type=float, default=120.0)
    lt_parser.add_argument("--json", action="store_true")

    # ----- sweep (concurrency ladder) -------------------------------------
    sw_parser = subparsers.add_parser(
        "sweep",
        help="Run loadtest at increasing concurrency; find the throughput knee",
    )
    sw_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    sw_parser.add_argument("--model", required=True)
    sw_parser.add_argument("--api", choices=["ollama", "openai"], default="ollama")
    sw_parser.add_argument("--concurrencies", default="1,2,4,8,16",
                           help="Comma-separated list (default 1,2,4,8,16)")
    sw_parser.add_argument("--num-requests-per-step", type=_positive_int, default=16)
    sw_parser.add_argument("--num-predict", type=_positive_int, default=128)
    sw_parser.add_argument("--prompt", default="Explain attention mechanisms in transformers.")
    sw_parser.add_argument("--api-key", help="For OpenAI-compat endpoints")
    sw_parser.add_argument("--json", action="store_true")

    # ----- calibrate (back-solve MBU from a measured run) -----------------
    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Run a loadtest and derive the MBU that matches measured tok/s",
        description=(
            "Runs a concurrent loadtest then back-solves the runtime's "
            "achieved memory_efficiency (MBU). Persist to ~/.config/kv-planner/calibration.json "
            "so subsequent `plan` calls use it automatically."
        ),
    )
    cal_parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    cal_parser.add_argument("--model", required=True,
                            help="Either an Ollama tag (llama3.2:3b) or a catalog slug (llama-3.2-3b)")
    cal_parser.add_argument("--gpu", help="GPU key (default: auto-detect)")
    cal_parser.add_argument("--api", choices=["ollama", "openai"], default="ollama")
    cal_parser.add_argument("--runtime", default="auto",
                            help="Runtime name for calibration key (auto detects from --api)")
    cal_parser.add_argument("--concurrency", type=_positive_int, default=4)
    cal_parser.add_argument("--num-requests", type=_positive_int, default=16)
    cal_parser.add_argument("--num-predict", type=_positive_int, default=128)
    cal_parser.add_argument("--persist", action="store_true",
                            help="Save the calibrated MBU to ~/.config/kv-planner/calibration.json")
    cal_parser.add_argument("--json", action="store_true")

    # ----- serve (dashboard only) -----------------------------------------
    serve_parser = subparsers.add_parser(
        "serve",
        help="Run the REST dashboard on :8787 (llmfit-compatible endpoints)",
        description=(
            "Starts a FastAPI server mounting the same endpoints llmfit "
            "exposes, plus kv-planner-specific ones (/api/v1/fleet, "
            "/api/v1/training-plan)."
        ),
    )
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=_positive_int, default=8787)

    # ----- tui ------------------------------------------------------------
    tui_parser = subparsers.add_parser(
        "tui",
        help="Launch the full-screen llmfit-style TUI (same as running bare `kv-planner`)",
    )
    tui_parser.add_argument("--no-dashboard", action="store_true",
                            help="Don't auto-start the REST dashboard alongside the TUI")
    tui_parser.add_argument("--dashboard-host", default="127.0.0.1")
    tui_parser.add_argument("--dashboard-port", type=_positive_int, default=8787)

    # ----- train ----------------------------------------------------------
    train_parser = subparsers.add_parser(
        "train",
        help="Plan or run a fine-tune (Unsloth + TRL backend)",
        description=(
            "Pre-flight capacity plan, then optional launch of an "
            "Unsloth-accelerated SFT / DPO / GRPO fine-tune with 2× speed "
            "and ~70%% less VRAM than vanilla transformers. "
            "Requires `pip install -e .[train]` for --run."
        ),
    )
    train_parser.add_argument(
        "--model", required=True,
        help="Catalog slug (llama-3-8b, qwen2.5-7b, …) or full HF model id",
    )
    train_parser.add_argument("--gpu", help="GPU key (default: auto-detect)")
    train_parser.add_argument(
        "--method", choices=["lora", "qlora", "full_ft"], default="qlora",
    )
    train_parser.add_argument(
        "--pipeline", choices=["sft", "dpo", "grpo"], default="sft",
        help="Training pipeline (default: sft)",
    )
    train_parser.add_argument(
        "--precision", choices=["bf16", "fp16"], default="bf16",
    )
    train_parser.add_argument("--dataset", help="Dataset path (.jsonl) or HF id")
    train_parser.add_argument("--output-dir", default="./ft-out")
    train_parser.add_argument("--batch-size", type=_positive_int, default=2)
    train_parser.add_argument("--grad-accum", type=_positive_int, default=4)
    train_parser.add_argument("--sequence-length", type=_positive_int, default=2048)
    train_parser.add_argument("--num-epochs", type=float, default=1.0)
    train_parser.add_argument("--max-steps", type=int, default=-1)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--lora-rank", type=_positive_int, default=16)
    train_parser.add_argument("--lora-alpha", type=_positive_int, default=16)
    train_parser.add_argument(
        "--chat-template", default=None,
        help="Chat template name (e.g., llama-3.1, qwen-2.5, chatml)",
    )
    train_parser.add_argument(
        "--dataset-tokens", type=_positive_int, default=1_000_000,
        help="Approx total tokens in the dataset (for pre-flight time estimate)",
    )
    train_parser.add_argument(
        "--run", action="store_true",
        help="Actually launch the training run (requires [train] extras)",
    )
    train_parser.add_argument(
        "--push-to-hub", default=None,
        help="Optional HF repo id to push the adapter to after training",
    )
    train_parser.add_argument("--report-to", default="none",
                              choices=["none", "tensorboard", "wandb"])
    train_parser.add_argument("--json", action="store_true")

    # List GPUs command
    list_parser = subparsers.add_parser(
        "list-gpus",
        help="List available GPUs",
        description="List all GPUs in the database",
    )
    list_parser.add_argument(
        "--filter",
        help="Filter by name (e.g., RTX, H100)",
    )
    list_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run vLLM benchmark",
        description="Run vLLM benchmark and save results",
    )
    benchmark_parser.add_argument(
        "--model",
        required=True,
        help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)",
    )
    benchmark_parser.add_argument(
        "--gpu",
        required=True,
        help="GPU model (e.g., RTX-5090)",
    )
    benchmark_parser.add_argument(
        "--type",
        choices=["latency", "throughput"],
        default="throughput",
        help="Benchmark type (default: throughput)",
    )
    benchmark_parser.add_argument(
        "--input-length",
        type=int,
        default=512,
        help="Input length in tokens (default: 512)",
    )
    benchmark_parser.add_argument(
        "--output-length",
        type=int,
        default=128,
        help="Output length in tokens (default: 128)",
    )
    benchmark_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    benchmark_parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts for throughput benchmark (default: 100)",
    )
    benchmark_parser.add_argument(
        "--dtype",
        default="auto",
        help="Model precision (default: auto)",
    )
    benchmark_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file for benchmark results (JSON)",
    )
    benchmark_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate predictions against benchmark",
        description="Compare kv-planner predictions against vLLM benchmark results",
    )
    validate_parser.add_argument(
        "--plan-file",
        required=True,
        help="Deployment plan file (JSON/YAML)",
    )
    validate_parser.add_argument(
        "--benchmark-file",
        required=True,
        help="Benchmark results file (JSON)",
    )
    validate_parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="Error tolerance percentage (default: 20.0)",
    )
    validate_parser.add_argument(
        "--output",
        "-o",
        help="Output file for validation results (JSON)",
    )

    return parser


def cmd_plan(args: argparse.Namespace) -> int:
    """Execute plan command."""
    try:
        planner = DeploymentPlanner()

        print(f"Creating deployment plan for {args.model} on {args.gpu}...")

        plan = planner.create_plan(
            model=args.model,
            hardware=args.gpu,
            target_rps=args.rps,
            input_length=args.input_length,
            output_length=args.output_length,
            optimization_goal=args.goal,  # type: ignore
            max_memory_budget_gb=args.max_memory,
            min_quality=args.min_quality,  # type: ignore
            enable_caching=not args.no_caching,
            system_prompt_length=args.system_prompt_length,
        )

        # Output
        if args.format == "json":
            output = export.to_json(plan)
            print(output)
        elif args.format == "yaml":
            output = export.to_yaml(plan)
            print(output)
        else:  # text
            print("\n" + plan.summary)
            print("\nKey Metrics:")
            print(f"  • Precision: {plan.recommended_precision.upper()}")
            print(f"  • Batch Size: {plan.recommended_batch_size}")
            print(f"  • Throughput: {plan.performance.throughput_tokens_per_sec:,.0f} tokens/sec")
            print(f"  • Cost: ${plan.cost.cost_per_million_tokens:.2f}/M tokens")
            print(f"  • Monthly: ${plan.cost.monthly_cost_usd:,.2f}")

            if plan.enable_prefix_caching and plan.caching:
                print(f"\nPrefix Caching:")
                print(f"  • Latency reduction: {plan.caching.latency_reduction_pct:.0f}%")
                print(f"  • Memory savings: {plan.caching.memory_savings_pct:.0f}%")

            print(f"\nvLLM Command:")
            print(f"  python -m vllm.entrypoints.openai.api_server \\")
            print(f"    --model {plan.vllm_config['model']} \\")
            print(f"    --dtype {plan.vllm_config['dtype']} \\")
            print(f"    --max-model-len {plan.vllm_config['max_model_len']} \\")
            print(f"    --max-num-seqs {plan.vllm_config['max_num_seqs']} \\")
            print(f"    --gpu-memory-utilization {plan.vllm_config['gpu_memory_utilization']}")

        # Save to file if requested
        if args.output:
            export.save(plan, args.output)
            print(f"\n✓ Saved to {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute compare command."""
    try:
        planner = DeploymentPlanner()
        gpu_list = [gpu.strip() for gpu in args.gpus.split(",")]

        print(f"Comparing GPUs for {args.model}...")

        plans = planner.compare_options(
            model=args.model,
            hardware_options=gpu_list,
            target_rps=args.rps,
            input_length=args.input_length,
            output_length=args.output_length,
            optimization_goal=args.goal,  # type: ignore
        )

        if args.format == "json":
            data = [
                {
                    "gpu": p.hardware.gpu_model,
                    "precision": p.recommended_precision,
                    "batch_size": p.recommended_batch_size,
                    "cost_per_million_tokens": p.cost.cost_per_million_tokens,
                    "throughput_tokens_per_sec": p.performance.throughput_tokens_per_sec,
                    "monthly_cost_usd": p.cost.monthly_cost_usd,
                }
                for p in plans
            ]
            print(json.dumps(data, indent=2))
        else:  # table
            print(f"\n{'GPU':<15} {'Precision':<10} {'Batch':<8} {'$/M tokens':<15} {'Throughput':<15} {'Monthly $':<12}")
            print("-" * 90)
            for p in plans:
                print(
                    f"{p.hardware.gpu_model:<15} "
                    f"{p.recommended_precision:<10} "
                    f"{p.recommended_batch_size:<8} "
                    f"${p.cost.cost_per_million_tokens:<14.2f} "
                    f"{p.performance.throughput_tokens_per_sec:<15,.0f} "
                    f"${p.cost.monthly_cost_usd:<11,.2f}"
                )

            best = plans[0]
            print(f"\n✨ Best value: {best.hardware.gpu_model} (${best.cost.cost_per_million_tokens:.2f}/M tokens)")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_list_gpus(args: argparse.Namespace) -> int:
    """Execute list-gpus command."""
    try:
        db = GPUDatabase()
        gpus = db.list_gpus()

        # Filter if requested
        if args.filter:
            filter_lower = args.filter.lower()
            gpus = [gpu for gpu in gpus if filter_lower in gpu.model.lower()]

        if args.format == "json":
            data = [
                {
                    "model": gpu.model,
                    "memory_gb": gpu.memory_gb,
                    "peak_tflops_fp16": gpu.peak_tflops_fp16,
                    "bandwidth_gb_s": gpu.memory_bandwidth_gb_s,
                    "architecture": gpu.architecture,
                }
                for gpu in gpus
            ]
            print(json.dumps(data, indent=2))
        else:  # table
            print(f"\n{'GPU Model':<20} {'Memory':<12} {'TFLOPS':<12} {'Bandwidth':<15} {'Architecture':<15}")
            print("-" * 80)
            for gpu in gpus:
                print(
                    f"{gpu.model:<20} "
                    f"{gpu.memory_gb:<12.0f} GB "
                    f"{gpu.peak_tflops_fp16:<12.1f} "
                    f"{gpu.memory_bandwidth_gb_s:<15.0f} GB/s "
                    f"{gpu.architecture:<15}"
                )

            print(f"\nTotal: {len(gpus)} GPUs")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark command."""
    try:
        print(f"Running {args.type} benchmark for {args.model}...")
        print()

        # Create benchmark config
        config = BenchmarkConfig(
            model_name=args.model,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            num_prompts=args.num_prompts,
            dtype=args.dtype,
        )

        # Run benchmark
        runner = BenchmarkRunner()

        if args.type == "latency":
            results = runner.run_latency_benchmark(config, verbose=args.verbose)
        else:  # throughput
            results = runner.run_throughput_benchmark(config, verbose=args.verbose)

        if not results.success:
            print(f"❌ Benchmark failed: {results.error_message}", file=sys.stderr)
            return 1

        # Display results
        print("✅ Benchmark completed successfully")
        print()
        print("Results:")
        print(f"  Benchmark type: {results.benchmark_type}")

        if results.throughput_tokens_per_sec:
            print(f"  Throughput: {results.throughput_tokens_per_sec:,.0f} tokens/sec")

        if results.mean_latency_ms:
            print(f"  Mean latency: {results.mean_latency_ms:,.1f} ms")

        if results.p50_latency_ms:
            print(f"  P50 latency: {results.p50_latency_ms:,.1f} ms")

        if results.p95_latency_ms:
            print(f"  P95 latency: {results.p95_latency_ms:,.1f} ms")

        if results.p99_latency_ms:
            print(f"  P99 latency: {results.p99_latency_ms:,.1f} ms")

        if results.time_to_first_token_ms:
            print(f"  TTFT: {results.time_to_first_token_ms:,.1f} ms")

        if results.time_per_output_token_ms:
            print(f"  TPOT: {results.time_per_output_token_ms:,.1f} ms")

        # Save results
        output_path = Path(args.output)
        runner.save_results(results, output_path)
        print()
        print(f"✓ Saved results to {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Execute validate command."""
    try:
        print("Loading files...")

        # Load plan
        plan_path = Path(args.plan_file)
        if not plan_path.exists():
            print(f"❌ Plan file not found: {plan_path}", file=sys.stderr)
            return 1

        # Load plan from JSON/YAML
        with open(plan_path, "r") as f:
            if plan_path.suffix == ".json":
                plan_data = json.load(f)
            elif plan_path.suffix in [".yaml", ".yml"]:
                import yaml
                plan_data = yaml.safe_load(f)
            else:
                print(f"❌ Unsupported plan file format: {plan_path.suffix}", file=sys.stderr)
                return 1

        # Load benchmark
        benchmark_path = Path(args.benchmark_file)
        if not benchmark_path.exists():
            print(f"❌ Benchmark file not found: {benchmark_path}", file=sys.stderr)
            return 1

        runner = BenchmarkRunner()
        benchmark_results = runner.load_results(benchmark_path)

        print(f"✓ Loaded plan from {plan_path}")
        print(f"✓ Loaded benchmark from {benchmark_path}")
        print()

        # For now, we need to recreate the plan from data
        # This is a simplified version - in production, we'd serialize the full plan
        planner = DeploymentPlanner()

        plan = planner.create_plan(
            model=plan_data["model"]["name"],
            hardware=plan_data["hardware"]["gpu_model"],
            target_rps=plan_data.get("target_rps", 10.0),
            input_length=benchmark_results.config.input_length,
            output_length=benchmark_results.config.output_length,
            optimization_goal=plan_data.get("optimization_goal", "balanced"),
        )

        # Validate
        validator = PredictionValidator(tolerance_pct=args.tolerance)
        validation = validator.validate(plan, benchmark_results)

        # Print summary
        print(validation.summary())

        # Save if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump({
                    "overall_accuracy": validation.overall_accuracy,
                    "mean_error_pct": validation.mean_error_pct,
                    "max_error_pct": validation.max_error_pct,
                    "systematic_bias": validation.systematic_bias,
                    "passed": validation.passed,
                    "comparisons": [
                        {
                            "metric": c.metric_name,
                            "predicted": c.predicted,
                            "actual": c.actual,
                            "error_pct": c.error_pct,
                            "within_tolerance": c.within_tolerance,
                        }
                        for c in validation.comparisons
                    ],
                    "tuning_suggestions": validation.tuning_suggestions,
                }, f, indent=2)
            print(f"\n✓ Saved validation results to {output_path}")

        return 0 if validation.passed else 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def _launch_tui(no_dashboard: bool = False, host: str = "127.0.0.1", port: int = 8787) -> int:
    try:
        from kv_planner.tui.app import main as tui_main
    except ImportError as e:
        print(
            f"TUI extras not installed. pip install -e '.[tui]'  ({e})",
            file=sys.stderr,
        )
        return 1
    dashboard_url = None
    if not no_dashboard:
        try:
            from kv_planner.tui.dashboard import run_background
            import os as _os
            h = _os.environ.get("KVP_DASHBOARD_HOST") or \
                _os.environ.get("LLMFIT_DASHBOARD_HOST") or host
            p = int(
                _os.environ.get("KVP_DASHBOARD_PORT")
                or _os.environ.get("LLMFIT_DASHBOARD_PORT")
                or port
            )
            dashboard_url = run_background(h, p)
        except Exception as e:
            print(f"(dashboard failed to start: {e})", file=sys.stderr)
    return tui_main(dashboard_url=dashboard_url)


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # Bare `kv-planner` → TUI (if stdout is a tty); otherwise print help.
        if sys.stdout.isatty() and sys.stdin.isatty():
            return _launch_tui()
        parser.print_help()
        return 1

    if args.command == "plan":
        return cmd_plan(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "list-gpus":
        return cmd_list_gpus(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "recommend":
        return cmd_recommend(args)
    elif args.command == "system":
        return cmd_system(args)
    elif args.command == "installed":
        return cmd_installed(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "fleet":
        return cmd_fleet(args)
    elif args.command == "explain":
        return cmd_explain(args)
    elif args.command == "serve":
        from kv_planner.tui.dashboard import run_foreground
        run_foreground(args.host, args.port)
        return 0
    elif args.command == "tui":
        return _launch_tui(
            no_dashboard=args.no_dashboard,
            host=args.dashboard_host,
            port=args.dashboard_port,
        )
    elif args.command == "loadtest":
        return cmd_loadtest(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "calibrate":
        return cmd_calibrate(args)
    elif args.command == "why":
        return cmd_why(args)
    elif args.command == "diagnose":
        return cmd_diagnose(args)
    elif args.command == "specdec":
        return cmd_specdec(args)
    elif args.command == "reasoning":
        return cmd_reasoning(args)
    elif args.command == "carbon":
        return cmd_carbon(args)
    elif args.command == "pricing":
        return cmd_pricing(args)
    elif args.command == "mcp":
        from kv_planner.mcp import main as mcp_main
        return mcp_main()
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


# ============================================================================
# Commands inspired by llmfit: system, recommend, installed
# ============================================================================


def cmd_system(args: argparse.Namespace) -> int:
    """Print detected CPU / RAM / GPU and which runtimes are reachable."""
    hw = detect_hardware()
    runtimes = probe_runtimes()

    if args.json:
        print(json.dumps({
            "cpu": {"model": hw.cpu_model, "cores": hw.cpu_cores},
            "ram": {
                "total_gb": round(hw.ram_total_gb, 1),
                "available_gb": round(hw.ram_available_gb, 1),
            },
            "gpu": {
                "vendor": hw.gpu_vendor,
                "name_raw": hw.gpu_name_raw,
                "vram_gb": round(hw.gpu_vram_gb, 2),
                "matched_db_key": hw.gpu_matched_db_key,
                "num_gpus": hw.num_gpus,
            },
            "runtimes": [
                {"name": r.name, "reachable": r.reachable,
                 "endpoint": r.endpoint, "models": r.models,
                 "version": r.version}
                for r in runtimes
            ],
        }, indent=2))
        return 0

    print()
    print("  SYSTEM")
    print(f"    CPU   {hw.cpu_model or 'unknown'}  ({hw.cpu_cores} cores)")
    print(f"    RAM   {hw.ram_total_gb:.1f} GB total  ·  {hw.ram_available_gb:.1f} GB free")
    if hw.gpu_vendor == "cpu-only":
        print("    GPU   not detected (CPU-only)")
    else:
        matched = hw.gpu_matched_db_key or "not in kv-planner DB"
        print(f"    GPU   {hw.gpu_name_raw} × {hw.num_gpus}  ·  "
              f"{hw.gpu_vram_gb:.1f} GB  ·  db: {matched}")
    print()
    print("  RUNTIMES")
    for r in runtimes:
        status = "reachable" if r.reachable else "not running"
        extras = f"  ({len(r.models)} models)" if r.reachable else ""
        print(f"    {r.name:<10} {status:<12}  {r.endpoint}{extras}")
    print()
    return 0


def cmd_installed(args: argparse.Namespace) -> int:
    """List installed models across local runtimes, cross-referenced with the catalog."""
    runtimes = probe_runtimes()

    rows = []
    for r in runtimes:
        if not r.reachable:
            continue
        for name in r.models:
            catalog_entry = match_ollama_name(name) if r.name == "ollama" else None
            rows.append({
                "runtime": r.name,
                "model": name,
                "in_catalog": catalog_entry is not None,
                "catalog_slug": catalog_entry.slug if catalog_entry else None,
                "quality_0_100": catalog_entry.quality_0_100 if catalog_entry else None,
                "use_cases": list(catalog_entry.use_cases) if catalog_entry else [],
            })

    if args.json:
        print(json.dumps({"installed": rows}, indent=2))
        return 0

    if not rows:
        print("  No installed models detected.  (Is Ollama/LM Studio/vLLM running?)")
        return 0
    print()
    print(f"  {'Runtime':<10} {'Model':<40} {'In catalog':<12} {'Quality':<8} {'Use cases'}")
    print("  " + "-" * 92)
    for r in rows:
        uc = ",".join(r["use_cases"]) if r["use_cases"] else "-"
        q = f"{r['quality_0_100']}" if r["quality_0_100"] is not None else "-"
        mark = "yes" if r["in_catalog"] else "no"
        print(f"  {r['runtime']:<10} {r['model'][:40]:<40} {mark:<12} {q:<8} {uc}")
    print()
    return 0


def cmd_fleet(args: argparse.Namespace) -> int:
    """Cluster sizing for a target RPS + latency SLO."""
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args.model)
    if entry is None:
        print(f"Unknown model slug '{args.model}'. Try `kv-planner recommend` for the catalog.",
              file=sys.stderr)
        return 1

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    tps = tuple(int(x) for x in args.tp.split(","))
    precs = tuple(p.strip() for p in args.precisions.split(","))

    designs = FleetPlanner().design(
        model=entry.config,
        target_rps=args.rps,
        input_length=args.input_length,
        output_length=args.output_length,
        gpu_candidates=gpus,
        p99_latency_ms=args.p99_latency_ms,
        precisions=precs,  # type: ignore[arg-type]
        tp_candidates=tps,
    )

    if not args.include_missing:
        designs = [d for d in designs if d.meets_slo]
    designs = designs[: args.limit]

    if args.json:
        print(json.dumps({
            "model": args.model,
            "target_rps": args.rps,
            "p99_latency_ms": args.p99_latency_ms,
            "designs": [
                {
                    "gpu": d.gpu_model, "tp": d.tp_size, "replicas": d.replicas,
                    "total_gpus": d.total_gpus, "precision": d.precision,
                    "per_replica_tok_s": round(d.per_replica_throughput_tok_s, 0),
                    "aggregate_tok_s": round(d.aggregate_throughput_tok_s, 0),
                    "p99_latency_ms": round(d.p99_latency_ms, 0),
                    "meets_slo": d.meets_slo,
                    "cost_per_hour": round(d.cost_per_hour, 2),
                    "cost_per_million_tokens": round(d.cost_per_million_tokens, 3),
                    "batch_per_replica": d.batch_per_replica,
                    "per_replica_memory_gb": round(d.per_replica_memory_gb, 2),
                    "notes": d.notes,
                }
                for d in designs
            ],
        }, indent=2))
        return 0

    print()
    print(f"  FLEET DESIGN — {entry.slug}  ·  {args.rps:.0f} RPS  ·  p99 SLO {args.p99_latency_ms:.0f} ms")
    print(f"  Workload: {args.input_length} in / {args.output_length} out tokens")
    print("  " + "-" * 104)
    print(f"  {'#':<3} {'GPU':<20} {'TP':>3} {'reps':>5} {'total':>6} {'prec':>5} "
          f"{'batch':>6} {'tok/s':>9} {'latency':>9} {'$/hr':>8} {'$/M':>8} {'slo':>5}")
    print("  " + "-" * 104)
    if not designs:
        print("  No valid design found. Try --include-missing, more GPU candidates, or a looser SLO.")
        return 0
    for i, d in enumerate(designs, 1):
        slo = "OK" if d.meets_slo else "miss"
        print(
            f"  {i:<3} {d.gpu_model:<20} {d.tp_size:>3} {d.replicas:>5} {d.total_gpus:>6} "
            f"{d.precision:>5} {d.batch_per_replica:>6} "
            f"{d.aggregate_throughput_tok_s:>9,.0f} "
            f"{d.p99_latency_ms:>7.0f}ms "
            f"${d.cost_per_hour:>6.2f} ${d.cost_per_million_tokens:>6.2f} {slo:>5}"
        )
    print("  " + "-" * 104)
    print()
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Physics-grounded rationale for a single catalog model."""
    from kv_planner.application.recommender import Recommender

    hw_key = args.gpu or detect_hardware().gpu_matched_db_key
    if not hw_key:
        print("Could not detect a supported GPU. Pass --gpu.", file=sys.stderr)
        return 1
    hw = GPUDatabase.to_hardware_spec(hw_key)

    # Compute the recommendation row for this specific model (borrow the
    # recommender to get Fit/Speed/Context scores).
    rec_list = Recommender().recommend(
        hw,
        use_case=args.use_case,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
    )
    target = next((r for r in rec_list if r.entry.slug == args.model), None)
    if target is None:
        print(f"Unknown slug '{args.model}'.", file=sys.stderr)
        return 1

    rat = explain_rec(
        target, hw,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
    )

    if args.json:
        print(json.dumps({
            "model": args.model, "gpu": hw_key, "use_case": args.use_case,
            "verdict": rat.verdict,
            "bullets": rat.bullets,
            "caveats": rat.caveats,
            "scores": {
                "quality": target.score_quality, "fit": target.score_fit,
                "speed": target.score_speed, "context": target.score_context,
                "composite": round(target.score_composite, 1),
            },
        }, indent=2))
        return 0

    print()
    print(f"  EXPLANATION — {args.model}  on  {hw_key}  for  {args.use_case}")
    print("  " + "-" * 88)
    print(f"  Verdict: {rat.verdict}  (composite {target.score_composite:.1f}/100)")
    print()
    for i, b in enumerate(rat.bullets, 1):
        import textwrap
        wrapped = textwrap.fill(b, width=86, subsequent_indent="     ")
        print(f"  {i}. {wrapped}")
    if rat.caveats:
        print()
        print("  Caveats:")
        for c in rat.caveats:
            print(f"    • {c}")
    print()
    return 0


def cmd_why(args: argparse.Namespace) -> int:
    """Memory waterfall — every term, formula, citation."""
    from kv_planner.core.explain.waterfall import build_waterfall
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args.model)
    if entry is None:
        print(f"Unknown model slug '{args.model}'.", file=sys.stderr)
        return 1
    gpu = args.gpu or detect_hardware().gpu_matched_db_key or "H100-SXM-80GB"
    hw = GPUDatabase.to_hardware_spec(gpu, tensor_parallel_size=args.tp or 1,
                                      num_gpus=args.tp or 1)
    w = build_waterfall(
        entry.config, hw,
        batch_size=args.batch, input_length=args.input_length,
        output_length=args.output_length, precision=args.precision,
        tensor_parallel_size=args.tp,
    )

    if args.json:
        print(json.dumps({
            "model": entry.slug, "gpu": gpu,
            "total_gb": round(w.total_gb, 3),
            "budget_gb": round(w.budget_gb, 2),
            "fits": w.fits,
            "headroom_gb": round(w.headroom_gb, 3),
            "terms": [
                {"label": t.label, "gb": round(t.gb, 4),
                 "formula": t.formula, "citation": t.citation, "note": t.note}
                for t in w.terms
            ],
        }, indent=2))
        return 0

    status = "FITS" if w.fits else "OVERFLOWS"
    print(f"\n  {entry.slug}  on  {gpu}  ·  {args.precision}  ·  "
          f"batch={args.batch}  ctx={args.input_length}+{args.output_length}\n")
    print(f"  {'term':<30} {'GB':>7}  formula")
    print("  " + "─" * 96)
    for t in w.terms:
        print(f"  {t.label:<30} {t.gb:>7.3f}  {t.formula}")
        if t.citation:
            print(f"  {'':<30} {'':>7}  └─ {t.citation}")
    print("  " + "─" * 96)
    print(f"  {'TOTAL':<30} {w.total_gb:>7.3f} GB")
    print(f"  {'device budget':<30} {w.budget_gb:>7.2f} GB  ({status})")
    if w.fits:
        print(f"  {'headroom':<30} {w.headroom_gb:>7.2f} GB")
    print()
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    """OOM post-mortem."""
    from kv_planner.application.diagnose import diagnose, parse_vllm_cmdline
    from kv_planner.infrastructure.model_catalog import by_slug

    # Optional vLLM cmdline parse to fill in missing args
    if args.vllm_cmdline:
        flags = parse_vllm_cmdline(args.vllm_cmdline)
        if "max-num-seqs" in flags and args.batch == 32:
            args.batch = int(flags["max-num-seqs"])
        if "max-model-len" in flags and args.input_length == 2048:
            # assume half in / half out
            mlen = int(flags["max-model-len"])
            args.input_length = mlen // 2
            args.output_length = mlen - args.input_length

    entry = by_slug(args.model)
    if entry is None:
        print(f"Unknown model slug '{args.model}'.", file=sys.stderr)
        return 1
    hw = GPUDatabase.to_hardware_spec(args.gpu, tensor_parallel_size=args.tp or 1,
                                     num_gpus=args.tp or 1)
    d = diagnose(
        entry.config, hw,
        batch_size=args.batch, input_length=args.input_length,
        output_length=args.output_length, precision=args.precision,
        tensor_parallel_size=args.tp,
    )

    if args.json:
        print(json.dumps({
            "model": entry.slug, "gpu": args.gpu,
            "fits": d.waterfall.fits,
            "total_gb": round(d.waterfall.total_gb, 3),
            "budget_gb": round(d.waterfall.budget_gb, 2),
            "overflow_gb": round(d.overflow_gb, 3),
            "culprit_term": d.culprit_term,
            "suggestions": [
                {"change": s.change, "rationale": s.rationale,
                 "new_memory_gb": round(s.new_memory_gb, 3), "fits": s.fits}
                for s in d.suggestions
            ],
        }, indent=2))
        return 0

    print(f"\n  diagnose · {entry.slug} on {args.gpu}")
    if d.waterfall.fits:
        print(f"  ✓ fits ({d.waterfall.total_gb:.2f} / {d.waterfall.budget_gb:.2f} GB)")
        return 0

    print(f"  ✗ overflows by {d.overflow_gb:.2f} GB")
    print(f"  culprit (largest term): {d.culprit_term}")
    print(f"\n  fixes (tried in order of cost):")
    for s in d.suggestions:
        mark = "✓" if s.fits else "✗"
        print(f"    {mark}  {s.change:<40}  → {s.new_memory_gb:6.2f} GB  [{s.rationale}]")
    print()
    return 0 if any(s.fits for s in d.suggestions) else 1


def cmd_specdec(args: argparse.Namespace) -> int:
    """Speculative decoding speedup estimate."""
    from kv_planner.core.performance.speculative import plan as spec_plan
    from kv_planner.infrastructure.model_catalog import by_slug

    target = by_slug(args.target)
    if target is None:
        print(f"Unknown target model '{args.target}'.", file=sys.stderr)
        return 1
    draft_params = 0
    if args.draft:
        draft = by_slug(args.draft)
        if draft is None:
            print(f"Unknown draft model '{args.draft}'.", file=sys.stderr)
            return 1
        draft_params = draft.config.total_params()

    r = spec_plan(
        method=args.method,
        target_model_params=target.config.total_params(),
        draft_model_params=draft_params,
        target_tpot_ms=args.target_tpot_ms,
        target_kv_bytes_per_token=target.config.kv_cache_bytes_per_token("fp16"),
        K=args.spec_k,
        acceptance_rate=args.acceptance,
    )

    if args.json:
        print(json.dumps({
            "method": r.method, "K": r.K,
            "acceptance_rate": r.acceptance_rate,
            "draft_cost_ratio": round(r.draft_cost_ratio, 4),
            "expected_tokens_per_verify_step": round(r.expected_tokens_per_verify_step, 2),
            "speedup": round(r.speedup, 2),
            "percent_faster": round(r.percent_faster, 1),
            "effective_tpot_ms": round(r.effective_tpot_ms, 2),
        }, indent=2))
        return 0

    print(f"\n  specdec  target={args.target}  method={r.method}  K={r.K}")
    print(f"  acceptance α      {r.acceptance_rate:.2f}")
    print(f"  draft cost ratio  {r.draft_cost_ratio:.3f}")
    print(f"  E[accepted/verify] {r.expected_tokens_per_verify_step:.2f} tokens")
    print(f"  net speedup       {r.speedup:.2f}×  ({r.percent_faster:+.0f}%)")
    print(f"  effective TPOT    {r.effective_tpot_ms:.2f} ms/token "
          f"(from {args.target_tpot_ms:.1f} baseline)")
    print()
    return 0


def cmd_reasoning(args: argparse.Namespace) -> int:
    """Reasoning-model KV planner."""
    from kv_planner.core.performance.reasoning import PROFILES, plan_reasoning
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args.model)
    if entry is None:
        print(f"Unknown model slug '{args.model}'.", file=sys.stderr)
        return 1
    profile = PROFILES[args.profile]
    p = plan_reasoning(
        entry.config, profile,
        prompt_tokens=args.prompt_tokens, batch_size=args.batch,
        precision=args.precision,
    )

    gpu_info = ""
    fits = None
    if args.gpu:
        hw = GPUDatabase.get(args.gpu)
        if hw:
            budget = hw.memory_gb * 0.9
            fits = p.kv_gb_p99_batch < budget
            gpu_info = (f"  GPU budget: {budget:.1f} GB · "
                        f"{'FITS p99 KV' if fits else 'OVERFLOWS at p99'}")

    if args.json:
        print(json.dumps({
            "model": p.model, "profile": args.profile, "precision": p.precision,
            "prompt_tokens": p.prompt_tokens,
            "think_mean": profile.think_mean, "think_p99": profile.think_p99,
            "p99_context_tokens": p.p99_context_tokens,
            "kv_gb_mean_per_seq": round(p.kv_bytes_mean_per_seq / 1e9, 4),
            "kv_gb_p99_per_seq": round(p.kv_bytes_p99_per_seq / 1e9, 4),
            "kv_gb_mean_batch": round(p.kv_gb_mean_batch, 4),
            "kv_gb_p99_batch": round(p.kv_gb_p99_batch, 4),
            "p99_over_mean": round(p.p99_over_mean_ratio, 2),
            "fits_p99": fits,
        }, indent=2))
        return 0

    print(f"\n  reasoning plan  ·  {entry.slug}  ·  profile={args.profile}  ·  {args.precision}")
    print(f"  prompt             {p.prompt_tokens:>6} tokens")
    print(f"  think mean         {profile.think_mean:>6} tokens  (answer {profile.answer_tokens})")
    print(f"  think p99          {profile.think_p99:>6} tokens   "
          f"(p99/mean × {p.p99_over_mean_ratio:.1f})")
    print(f"  p99 context total  {p.p99_context_tokens:>6} tokens")
    print()
    print(f"  KV at mean context   {p.kv_gb_mean_batch:>6.2f} GB  (batch={args.batch})")
    print(f"  KV at p99 context    {p.kv_gb_p99_batch:>6.2f} GB  ← plan VRAM for this")
    if gpu_info:
        print(gpu_info)
    print()
    return 0


def cmd_carbon(args: argparse.Namespace) -> int:
    """Carbon footprint estimate."""
    from kv_planner.application import DeploymentPlanner
    from kv_planner.core.cost.carbon import estimate_carbon
    from kv_planner.infrastructure.model_catalog import by_slug

    entry = by_slug(args.model)
    if entry is None:
        print(f"Unknown model '{args.model}'.", file=sys.stderr)
        return 1

    hw = GPUDatabase.get(args.gpu)
    if hw is None:
        print(f"Unknown GPU '{args.gpu}'.", file=sys.stderr)
        return 1

    plan = DeploymentPlanner().create_plan(
        model=entry.config, hardware=args.gpu,
        target_rps=10.0,
        input_length=args.input_length, output_length=args.output_length,
        optimization_goal="balanced",
    )
    c = estimate_carbon(
        throughput_tok_s=plan.performance.throughput_tokens_per_sec,
        tdp_watts=hw.typical_tdp_w,
        mfu=plan.performance.mfu,
        mbu=plan.performance.mbu,
        region=args.region,
        num_gpus=args.num_gpus,
    )
    dollars = plan.cost.cost_per_million_tokens

    if args.json:
        print(json.dumps({
            "model": entry.slug, "gpu": args.gpu, "region": args.region,
            "throughput_tok_s": round(plan.performance.throughput_tokens_per_sec, 0),
            "tdp_w": hw.typical_tdp_w, "num_gpus": args.num_gpus,
            "grid_intensity_g_per_kwh": c.grid_intensity_g_per_kwh,
            "kwh_per_million_tokens": round(c.kwh_per_million_tokens, 4),
            "g_co2e_per_million_tokens": round(c.g_co2e_per_million_tokens, 2),
            "cost_per_million_tokens": round(dollars, 3),
        }, indent=2))
        return 0

    print(f"\n  carbon  ·  {entry.slug} on {args.num_gpus}× {args.gpu}  ·  region={args.region}")
    print(f"  throughput              {plan.performance.throughput_tokens_per_sec:>8.0f} tok/s")
    print(f"  avg GPU power           {c.gpu_watts_avg:>8.0f} W")
    print(f"  grid intensity          {c.grid_intensity_g_per_kwh:>8.0f} gCO2e/kWh")
    print(f"  energy / M tokens       {c.kwh_per_million_tokens:>8.3f} kWh")
    print(f"  emissions / M tokens    {c.g_co2e_per_million_tokens:>8.1f} gCO2e")
    print(f"  cost / M tokens         ${dollars:>7.3f}")
    print()
    return 0


def cmd_pricing(args: argparse.Namespace) -> int:
    """Live / cached pricing."""
    from kv_planner.infrastructure import pricing

    if args.refresh:
        refreshed = pricing.refresh_api_pricing()
        if not args.json:
            print(f"  refreshed {len(refreshed)} API entries from OpenRouter")

    if args.gpu:
        rows = pricing.list_gpu_prices(args.gpu)
        if args.json:
            print(json.dumps([
                {"gpu": r.gpu_model, "provider": r.provider,
                 "cost_per_hour": r.cost_per_hour, "spot": r.spot}
                for r in rows
            ], indent=2))
            return 0
        print(f"\n  GPU pricing  ·  {args.gpu}\n")
        print(f"  {'provider':<15} {'$/hr':>7}  mode")
        for r in rows:
            print(f"  {r.provider:<15} ${r.cost_per_hour:>6.2f}  {'spot' if r.spot else 'on-demand'}")
        print()
        return 0

    if args.api_model:
        p = pricing.get_api_price(args.api_model)
        if p is None:
            print(f"no pricing for '{args.api_model}'", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(p.__dict__, indent=2))
            return 0
        print(f"\n  {p.model}  ({p.provider})  [{p.source}]")
        print(f"  input  ${p.input_per_m:>6.2f} / M tokens")
        print(f"  output ${p.output_per_m:>6.2f} / M tokens")
        print()
        return 0

    # Default: print a summary
    for model, p in list(pricing._FALLBACK_API.items())[:12]:
        print(f"  {model:<22} ${p.input_per_m:>6.2f} / ${p.output_per_m:>6.2f} per M ({p.provider})")
    return 0


def cmd_loadtest(args: argparse.Namespace) -> int:
    """Concurrent load test producing TTFT/TPOT/E2E distributions + goodput."""
    from kv_planner.loadtest import LoadTester, SloTargets

    prompts: list[str]
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.rstrip() for line in f if line.strip()]
        if not prompts:
            print("prompt file is empty", file=sys.stderr)
            return 1
    else:
        prompts = [args.prompt]

    slo = SloTargets(
        ttft_ms=args.ttft_slo_ms,
        tpot_ms=args.tpot_slo_ms,
        e2e_ms=args.e2e_slo_ms,
    )

    print(
        f"\n  loadtest {args.endpoint}  ·  {args.model}  ·  "
        f"concurrency={args.concurrency}  requests={args.num_requests}  "
        f"num_predict={args.num_predict}\n"
    )
    lt = LoadTester()

    # Progress bar (carriage-return based)
    def on_progress(done: int, total: int) -> None:
        pct = done / total * 100
        bar = "█" * int(pct // 3) + "░" * (33 - int(pct // 3))
        sys.stderr.write(f"\r  {bar}  {done}/{total} ({pct:.0f}%)")
        sys.stderr.flush()

    result = lt.run(
        endpoint=args.endpoint, model=args.model,
        prompt=prompts[0],  # only single prompt per run for now
        api=args.api,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        num_predict=args.num_predict,
        api_key=args.api_key,
        timeout=args.timeout,
        slo=slo,
        on_progress=None if args.json else on_progress,
    )
    if not args.json:
        sys.stderr.write("\n\n")

    if args.json:
        print(json.dumps({
            "endpoint": result.endpoint, "model": result.model, "api": result.api,
            "concurrency": result.concurrency, "num_requests": result.num_requests,
            "wall_s": round(result.wall_s, 3),
            "goodput_pct": round(result.goodput_pct, 1),
            "errors": result.error_count,
            "ttft_ms": {"p50": round(result.ttft_p50 * 1000, 1),
                        "p95": round(result.ttft_p95 * 1000, 1),
                        "p99": round(result.ttft_p99 * 1000, 1)},
            "tpot_ms": {"p50": round(result.tpot_p50 * 1000, 2),
                        "p95": round(result.tpot_p95 * 1000, 2),
                        "p99": round(result.tpot_p99 * 1000, 2)},
            "e2e_ms":  {"p50": round(result.e2e_p50 * 1000, 1),
                        "p95": round(result.e2e_p95 * 1000, 1),
                        "p99": round(result.e2e_p99 * 1000, 1)},
            "aggregate_tok_s": round(result.aggregate_tok_s, 1),
            "per_request_tok_s_mean": round(result.mean_per_request_tok_s, 1),
            "total_output_tokens": result.total_output_tokens,
        }, indent=2))
        return 0

    print(f"  wall clock       {result.wall_s:>8.2f} s   total output {result.total_output_tokens:>6} tokens")
    print(f"  aggregate        {result.aggregate_tok_s:>8.0f} tok/s")
    print(f"  per-request mean {result.mean_per_request_tok_s:>8.0f} tok/s")
    print(f"  errors           {result.error_count:>8}")
    if slo.ttft_ms or slo.tpot_ms or slo.e2e_ms:
        print(f"  goodput          {result.goodput_pct:>8.0f} %   ({result.pass_count}/{result.num_requests} pass joint SLO)")
    print()
    print(f"  {'metric':<6} {'p50':>10} {'p95':>10} {'p99':>10}")
    print(f"  {'-'*40}")
    print(f"  {'TTFT':<6} {result.ttft_p50*1000:>8.0f}ms {result.ttft_p95*1000:>8.0f}ms {result.ttft_p99*1000:>8.0f}ms")
    print(f"  {'TPOT':<6} {result.tpot_p50*1000:>8.2f}ms {result.tpot_p95*1000:>8.2f}ms {result.tpot_p99*1000:>8.2f}ms")
    print(f"  {'E2E':<6}  {result.e2e_p50*1000:>8.0f}ms {result.e2e_p95*1000:>8.0f}ms {result.e2e_p99*1000:>8.0f}ms")
    print()
    return 0 if result.error_count == 0 else 1


def cmd_sweep(args: argparse.Namespace) -> int:
    """Concurrency ladder — find the throughput knee."""
    from kv_planner.loadtest import LoadTester

    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    lt = LoadTester()

    if not args.json:
        print(f"\n  sweep {args.endpoint}  ·  {args.model}  ·  "
              f"concurrencies={concurrencies}  "
              f"requests/step={args.num_requests_per_step}  "
              f"num_predict={args.num_predict}\n")
        print(f"  {'c':>3} {'wall':>6} {'agg tok/s':>10} {'TTFT p95':>9} "
              f"{'TPOT p95':>9} {'E2E p95':>9} {'errors':>7}")
        print("  " + "-" * 66)

    def on_step(res) -> None:
        if args.json:
            return
        print(
            f"  {res.concurrency:>3} {res.wall_s:>5.1f}s "
            f"{res.aggregate_tok_s:>10.0f} "
            f"{res.ttft_p95*1000:>7.0f}ms "
            f"{res.tpot_p95*1000:>7.1f}ms "
            f"{res.e2e_p95*1000:>7.0f}ms "
            f"{res.error_count:>7}"
        )

    sweep = lt.sweep(
        endpoint=args.endpoint, model=args.model, api=args.api,
        concurrencies=concurrencies,
        num_requests_per_step=args.num_requests_per_step,
        num_predict=args.num_predict,
        prompt=args.prompt,
        api_key=args.api_key,
        on_step=on_step,
    )

    if args.json:
        print(json.dumps({
            "endpoint": sweep.endpoint, "model": sweep.model,
            "knee_concurrency": sweep.knee_concurrency,
            "points": [
                {
                    "concurrency": p.concurrency, "wall_s": round(p.wall_s, 2),
                    "aggregate_tok_s": round(p.aggregate_tok_s, 1),
                    "ttft_p95_ms": round(p.ttft_p95 * 1000, 1),
                    "tpot_p95_ms": round(p.tpot_p95 * 1000, 2),
                    "e2e_p95_ms": round(p.e2e_p95 * 1000, 1),
                    "errors": p.error_count,
                }
                for p in sweep.points
            ],
        }, indent=2))
        return 0

    print()
    if sweep.knee_concurrency is not None:
        print(f"  knee at concurrency = {sweep.knee_concurrency}  "
              f"(throughput gain < 10 % beyond this)")
    print()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Loadtest then back-solve MBU so kv-planner's predicted tok/s matches reality."""
    from pathlib import Path

    from kv_planner.domain import bytes_per_element
    from kv_planner.infrastructure.hardware_db import GPUDatabase
    from kv_planner.infrastructure.hw_detect import detect as detect_hw
    from kv_planner.infrastructure.model_catalog import by_slug, match_ollama_name
    from kv_planner.loadtest import LoadTester

    # ---- resolve model ----
    entry = by_slug(args.model) or match_ollama_name(args.model)
    if entry is None:
        print(f"Unknown model '{args.model}'. Pass a catalog slug or Ollama tag "
              f"that maps to one (see `kv-planner recommend`).", file=sys.stderr)
        return 1

    # ---- resolve GPU ----
    gpu_key = args.gpu or detect_hw().gpu_matched_db_key
    if not gpu_key:
        print("Could not detect a supported GPU. Pass --gpu.", file=sys.stderr)
        return 1
    spec = GPUDatabase.get(gpu_key)
    if spec is None:
        print(f"Unknown GPU '{gpu_key}'.", file=sys.stderr)
        return 1

    # For Ollama we pass the Ollama tag to the runner but catalog stays for physics.
    ollama_tag = args.model
    if not any(t == args.model for t in entry.ollama_tags):
        if entry.ollama_tags:
            ollama_tag = entry.ollama_tags[0]

    print(f"\n  calibrate {entry.slug}  ·  {gpu_key}  "
          f"({args.runtime if args.runtime != 'auto' else args.api})\n")

    lt = LoadTester()
    result = lt.run(
        endpoint=args.endpoint,
        model=ollama_tag,
        api=args.api,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        num_predict=args.num_predict,
    )

    # ---- Back-solve MBU from measured per-request tok/s ----
    # Decode per-token time ≈ (weight_bytes + kv_bytes) / (peak_bw × mbu)
    # ⇒ mbu = (weight_bytes + kv_bytes) / (peak_bw × measured_tpot_s)
    # Use per-request mean throughput for this, not aggregate (which includes queuing).
    precision = entry.recommended_quant  # int4 for Q4_K_M catalog entries
    bpe = bytes_per_element(precision)
    weight_bytes = entry.config.total_params() * bpe
    ctx_len_est = args.num_predict // 2 + 32   # mid-decode context
    kv_bytes = entry.config.kv_cache_bytes_per_token(precision) * ctx_len_est

    # Prefer measured TPOT; fall back to per-request mean throughput.
    if result.tpot_p50 > 0:
        measured_tpot_s = result.tpot_p50
    elif result.mean_per_request_tok_s > 0:
        measured_tpot_s = 1.0 / result.mean_per_request_tok_s
    else:
        print("  no successful requests — cannot calibrate", file=sys.stderr)
        return 1

    achieved_bw = (weight_bytes + kv_bytes) / measured_tpot_s
    peak_bw = spec.memory_bandwidth_gb_s * 1e9
    derived_mbu = achieved_bw / peak_bw if peak_bw > 0 else 0.0
    derived_mbu = max(0.05, min(0.98, derived_mbu))

    runtime_key = args.runtime if args.runtime != "auto" else args.api

    payload = {
        "model_slug": entry.slug,
        "gpu_key": gpu_key,
        "runtime": runtime_key,
        "precision": precision,
        "measured_tpot_ms": round(measured_tpot_s * 1000, 3),
        "measured_per_request_tok_s": round(result.mean_per_request_tok_s, 2),
        "measured_aggregate_tok_s": round(result.aggregate_tok_s, 2),
        "weight_bytes": weight_bytes,
        "kv_bytes_per_decode_step": kv_bytes,
        "achieved_memory_bandwidth_gb_s": round(achieved_bw / 1e9, 1),
        "peak_memory_bandwidth_gb_s": spec.memory_bandwidth_gb_s,
        "calibrated_mbu": round(derived_mbu, 3),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"  measured TPOT p50      {measured_tpot_s*1000:>8.2f} ms/token")
        print(f"  measured per-req tok/s {result.mean_per_request_tok_s:>8.1f}")
        print(f"  bytes streamed per step {(weight_bytes + kv_bytes)/1e9:>8.2f} GB")
        print(f"  achieved HBM bandwidth {achieved_bw/1e9:>8.1f} GB/s  "
              f"(peak {spec.memory_bandwidth_gb_s:.0f} GB/s)")
        print(f"  calibrated MBU         {derived_mbu:>8.3f}  "
              f"(default 0.75 is vLLM-tuned; Ollama/llama.cpp typically 0.30–0.45)")
        print()

    if args.persist:
        config_dir = Path.home() / ".config" / "kv-planner"
        config_dir.mkdir(parents=True, exist_ok=True)
        calib_file = config_dir / "calibration.json"
        existing: dict = {}
        if calib_file.exists():
            try:
                existing = json.loads(calib_file.read_text())
            except json.JSONDecodeError:
                existing = {}
        key = f"{entry.slug}::{gpu_key}::{runtime_key}"
        existing[key] = payload
        calib_file.write_text(json.dumps(existing, indent=2))
        if not args.json:
            print(f"  saved to {calib_file}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Pre-flight training plan (always). Optional ``--run`` to launch Unsloth."""
    from kv_planner.core.training import TrainingPlanner
    from kv_planner.infrastructure.model_catalog import by_slug

    # ------ resolve model into a ModelConfig -----------------------------
    entry = by_slug(args.model)
    if entry is not None:
        model_cfg = entry.config
        display_name = entry.slug
    else:
        # Unknown slug — require the user to pass architecture inline, or
        # just warn. For pre-flight without a ModelConfig we can't plan.
        print(
            f"Unknown catalog slug '{args.model}'. For now pass one of: "
            + ", ".join(e.slug for e in __import__(
                'kv_planner.infrastructure.model_catalog', fromlist=['CATALOG']
            ).CATALOG),
            file=sys.stderr,
        )
        return 1

    # ------ resolve hardware ---------------------------------------------
    hw_key = args.gpu or detect_hardware().gpu_matched_db_key
    if not hw_key:
        print("Could not detect a supported GPU. Pass --gpu explicitly.",
              file=sys.stderr)
        return 1
    hw = GPUDatabase.to_hardware_spec(hw_key)

    # ------ pre-flight plan ----------------------------------------------
    plan = TrainingPlanner().plan(
        model=model_cfg,
        hardware=hw,
        method=args.method,
        precision=args.precision,  # type: ignore[arg-type]
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_epochs=int(args.num_epochs) if args.num_epochs.is_integer() else max(1, int(args.num_epochs)),
        dataset_tokens=args.dataset_tokens,
        lora_rank=args.lora_rank,
    )

    if args.json and not args.run:
        print(json.dumps({
            "model": display_name,
            "gpu": hw_key,
            "method": plan.method,
            "precision": plan.weight_precision,
            "memory": {
                "model_gb": round(plan.model_weight_gb, 2),
                "gradients_gb": round(plan.gradient_gb, 2),
                "optimizer_gb": round(plan.optimizer_state_gb, 2),
                "activations_gb": round(plan.activation_gb, 2),
                "total_gb": round(plan.total_memory_gb, 2),
                "budget_gb": hw.gpu_memory_gb,
                "fits": plan.fits_per_gpu,
            },
            "compute": {
                "tokens_per_step": plan.tokens_per_step,
                "step_time_sec": round(plan.step_time_sec, 3),
                "tokens_per_sec": round(plan.tokens_per_second, 1),
                "estimated_hours": round(plan.est_training_hours, 2),
                "estimated_cost_usd": round(plan.est_cost_usd, 2),
            },
            "trainable_params": plan.trainable_params,
        }, indent=2))
        return 0

    # ------ pretty-print plan --------------------------------------------
    print()
    print(f"  TRAINING PLAN — {display_name} on {hw_key} via Unsloth ({args.method.upper()})")
    print("  " + "-" * 86)
    fit_tag = "fits" if plan.fits_per_gpu else "WILL NOT FIT"
    print(f"  Memory per GPU")
    print(f"    model weights      {plan.model_weight_gb:8.2f} GB  ({args.method}/{args.precision})")
    print(f"    gradients          {plan.gradient_gb:8.2f} GB")
    print(f"    optimizer states   {plan.optimizer_state_gb:8.2f} GB  (AdamW; 8 B/param fp32)")
    print(f"    activations (SAC)  {plan.activation_gb:8.2f} GB")
    print(f"    TOTAL              {plan.total_memory_gb:8.2f} GB / {hw.gpu_memory_gb:.0f} GB device  ·  {fit_tag}")
    print()
    print(f"  Trainable params      {plan.trainable_params/1e6:8.2f} M  ({plan.trainable_params / model_cfg.total_params() * 100:.3f}% of total)")
    print(f"  Tokens per step       {plan.tokens_per_step:>10,}")
    print(f"  Step time             {plan.step_time_sec * 1000:>8.1f} ms  ·  {plan.tokens_per_second:,.0f} tok/s")
    print(f"  Estimated wall-clock  {plan.est_training_hours:>8.2f} hours")
    print(f"  Estimated cost        ${plan.est_cost_usd:,.2f}")
    print("  " + "-" * 86)
    print("  Unsloth ships pre-quantized 4-bit versions of most catalog models that")
    print("  typically reduce VRAM by ~70% and add ~2× speed over vanilla transformers.")
    print()

    if not args.run:
        if args.dataset is None:
            print("  (Dry run: pass --run --dataset data.jsonl to launch training)")
        else:
            print("  (Dry run: pass --run to launch)")
        return 0

    # ------ launch --------------------------------------------------------
    if args.dataset is None:
        print("--run requires --dataset path/to/data.jsonl (or HF dataset id)",
              file=sys.stderr)
        return 1

    if not plan.fits_per_gpu:
        print("\n  Refusing to launch: estimated memory exceeds VRAM. "
              "Reduce batch/seq, switch to qlora, or pick a bigger GPU.",
              file=sys.stderr)
        return 1

    try:
        from kv_planner.core.training.unsloth_runner import (
            TrainArgs, run_training,
        )
    except ImportError as e:
        print(f"\n  {e}", file=sys.stderr)
        return 1

    def on_log(logs: dict) -> None:
        step = logs.get("step", "?")
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        bits = [f"step {step}"]
        if loss is not None:
            bits.append(f"loss {loss:.4f}")
        if lr is not None:
            bits.append(f"lr {lr:.2e}")
        print("  " + "  ·  ".join(bits))

    targs = TrainArgs(
        model_slug_or_id=display_name,
        dataset=args.dataset,
        output_dir=args.output_dir,
        pipeline=args.pipeline,
        method=args.method,
        max_seq_length=args.sequence_length,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        chat_template=args.chat_template,
        push_to_hub_id=args.push_to_hub,
        report_to=args.report_to,
    )

    print("  Launching training … (press Ctrl-C to interrupt)\n")
    result = run_training(targs, progress_cb=on_log)
    print()
    print(f"  Training complete. Adapter saved to {result.output_dir}/adapter")
    print(f"  Final loss: {result.final_train_loss}  ·  Wall clock: {result.seconds_elapsed:.1f} s")
    if result.pushed_to_hub:
        print(f"  Pushed to https://huggingface.co/{result.pushed_to_hub}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """Physics-scored recommendation of catalog models for the given hardware."""
    hw = detect_hardware()
    gpu_key = args.gpu or hw.gpu_matched_db_key
    if not gpu_key:
        print("Could not detect a supported GPU. Pass --gpu explicitly, or run",
              "`kv-planner list-gpus` to see options.", file=sys.stderr)
        return 1

    spec = GPUDatabase.to_hardware_spec(gpu_key)
    rec = Recommender()
    results = rec.top_n(
        spec,
        n=args.limit,
        use_case=args.use_case,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        include_unfit=args.include_unfit,
    )

    if args.json:
        print(json.dumps({
            "hardware": {"gpu": gpu_key, "vram_gb": spec.gpu_memory_gb},
            "use_case": args.use_case,
            "models": [
                {
                    "slug": r.entry.slug,
                    "name": r.entry.config.name,
                    "provider": r.entry.provider,
                    "precision": r.precision,
                    "throughput_tok_s": round(r.throughput_tok_s, 1),
                    "memory_gb": round(r.memory_gb, 2),
                    "memory_util_pct": round(r.memory_util_pct, 1),
                    "fits": r.fits,
                    "score_quality": r.score_quality,
                    "score_fit": r.score_fit,
                    "score_speed": r.score_speed,
                    "score_context": r.score_context,
                    "score_composite": round(r.score_composite, 1),
                    "license": r.entry.license,
                    "ollama_tags": list(r.entry.ollama_tags),
                }
                for r in results
            ],
        }, indent=2))
        return 0

    print()
    print(f"  Recommended models for {gpu_key} ({spec.gpu_memory_gb:.0f} GB), "
          f"use case: {args.use_case}")
    print("  " + "-" * 96)
    print(f"  {'#':<3} {'Model':<30} {'Prov':<9} {'Prec':<5} "
          f"{'tok/s':>8} {'GB':>6} {'Util':>6} {'Q':>4} {'F':>4} {'S':>4} {'C':>4} {'SCORE':>6}")
    print("  " + "-" * 96)
    for i, r in enumerate(results, 1):
        star = "★" if r.score_composite >= 70 else " "
        fits = "" if r.fits else "  (won't fit)"
        print(
            f"  {i:<3} {r.entry.slug[:30]:<30} {r.entry.provider[:9]:<9} "
            f"{r.precision:<5} {r.throughput_tok_s:>8.0f} "
            f"{r.memory_gb:>6.1f} {r.memory_util_pct:>5.0f}% "
            f"{r.score_quality:>4} {r.score_fit:>4} {r.score_speed:>4} "
            f"{r.score_context:>4} {star}{r.score_composite:>5.1f}{fits}"
        )
    print("  " + "-" * 96)
    print("  Legend: Q=Quality F=Fit S=Speed C=Context  ·  Composite = 0.35Q+0.25F+0.25S+0.15C")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
