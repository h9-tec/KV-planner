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
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    PredictionValidator,
)
from kv_planner.infrastructure.benchmarks.runner import create_config_from_plan


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
        type=float,
        default=10.0,
        help="Target requests per second (default: 10.0)",
    )
    plan_parser.add_argument(
        "--input-length",
        type=int,
        default=2048,
        help="Average input length in tokens (default: 2048)",
    )
    plan_parser.add_argument(
        "--output-length",
        type=int,
        default=512,
        help="Average output length in tokens (default: 512)",
    )
    plan_parser.add_argument(
        "--goal",
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
        "--goal",
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
                    "bandwidth_gb_s": gpu.hbm_bandwidth_gb_s,
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
                    f"{gpu.hbm_bandwidth_gb_s:<15.0f} GB/s "
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


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
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
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
