#!/usr/bin/env python3
"""
Example: Benchmark Validation

Demonstrates how to:
1. Create a deployment plan with kv-planner
2. Run actual vLLM benchmarks
3. Compare predictions vs actual results
4. Identify prediction accuracy and biases
5. Get tuning suggestions

Prerequisites:
- vLLM installed: pip install vllm
- GPU available
- Model downloaded or accessible via HuggingFace
"""

import sys
from pathlib import Path

from kv_planner.application import DeploymentPlanner
from kv_planner.infrastructure.benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    PredictionValidator,
)
from kv_planner.infrastructure.benchmarks.runner import create_config_from_plan


def example_basic_validation() -> None:
    """
    Example 1: Basic validation workflow.

    Creates a plan, runs a benchmark, and compares results.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Validation Workflow")
    print("=" * 80)
    print()

    # Step 1: Create deployment plan
    print("Step 1: Creating deployment plan...")
    planner = DeploymentPlanner()

    plan = planner.create_plan(
        model="meta-llama/Llama-3.2-1B-Instruct",  # Small model for testing
        hardware="RTX-4090",
        target_rps=10.0,
        input_length=512,
        output_length=128,
        optimization_goal="throughput",
    )

    print(f"✓ Created plan for {plan.model.name}")
    print(f"  Recommended precision: {plan.recommended_precision}")
    print(f"  Recommended batch size: {plan.recommended_batch_size}")
    print(f"  Predicted throughput: {plan.performance.throughput_tokens_per_sec:,.0f} tok/s")
    print()

    # Step 2: Run benchmark
    print("Step 2: Running vLLM throughput benchmark...")
    print("(This will download the model if not cached)")
    print()

    runner = BenchmarkRunner()

    # Create benchmark config matching the plan
    bench_config = create_config_from_plan(
        model=plan.model,
        hardware=plan.hardware,
        precision=plan.recommended_precision,
        batch_size=plan.recommended_batch_size,
        input_length=512,
        output_length=128,
    )

    # Run throughput benchmark
    bench_results = runner.run_throughput_benchmark(
        config=bench_config,
        verbose=True,
    )

    if not bench_results.success:
        print(f"❌ Benchmark failed: {bench_results.error_message}")
        print()
        print("Note: This example requires:")
        print("1. vLLM installed: pip install vllm")
        print("2. A GPU available")
        print("3. Sufficient memory to load the model")
        return

    print(f"✓ Benchmark completed successfully")
    print(f"  Actual throughput: {bench_results.throughput_tokens_per_sec:,.0f} tok/s")
    print()

    # Step 3: Validate predictions
    print("Step 3: Validating predictions...")
    validator = PredictionValidator(
        tolerance_pct=20.0,  # 20% tolerance
        strict_tolerance_pct=10.0,  # 10% for critical metrics
    )

    validation = validator.validate(plan, bench_results)

    print(validation.summary())
    print()

    # Step 4: Save results
    print("Step 4: Saving results...")
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    bench_file = results_dir / "benchmark_results.json"
    runner.save_results(bench_results, bench_file)
    print(f"✓ Saved benchmark results to {bench_file}")
    print()


def example_latency_validation() -> None:
    """
    Example 2: Latency benchmark validation.

    Focuses on latency metrics (TTFT, TPOT).
    """
    print("=" * 80)
    print("EXAMPLE 2: Latency Validation")
    print("=" * 80)
    print()

    # Create plan
    planner = DeploymentPlanner()

    plan = planner.create_plan(
        model="meta-llama/Llama-3.2-1B-Instruct",
        hardware="RTX-4090",
        target_rps=5.0,
        input_length=1024,
        output_length=256,
        optimization_goal="latency",
    )

    print(f"Created plan for low-latency deployment")
    print(f"  Predicted prefill latency: {plan.performance.prefill_latency_ms:,.0f} ms")
    print(f"  Predicted decode latency: {plan.performance.decode_latency_ms:,.0f} ms")
    print()

    # Run latency benchmark
    runner = BenchmarkRunner()

    bench_config = BenchmarkConfig(
        model_name=plan.model.name,
        input_length=1024,
        output_length=256,
        batch_size=1,  # Single request for latency
        dtype="float16" if plan.recommended_precision == "fp16" else plan.recommended_precision,
        tensor_parallel_size=plan.hardware.tensor_parallel_size,
        num_iters=10,
    )

    print("Running latency benchmark...")
    bench_results = runner.run_latency_benchmark(
        config=bench_config,
        verbose=True,
    )

    if not bench_results.success:
        print(f"❌ Benchmark failed: {bench_results.error_message}")
        return

    print(f"✓ Benchmark completed")
    if bench_results.time_to_first_token_ms:
        print(f"  TTFT: {bench_results.time_to_first_token_ms:.1f} ms")
    if bench_results.time_per_output_token_ms:
        print(f"  TPOT: {bench_results.time_per_output_token_ms:.1f} ms")
    print()

    # Validate
    validator = PredictionValidator()
    validation = validator.validate(plan, bench_results)

    print(validation.summary())
    print()


def example_multiple_gpu_validation() -> None:
    """
    Example 3: Validate across multiple GPU configurations.

    Compares predictions for different GPUs.
    """
    print("=" * 80)
    print("EXAMPLE 3: Multi-GPU Validation")
    print("=" * 80)
    print()

    planner = DeploymentPlanner()

    # Test different GPUs
    gpus = ["RTX-4090", "RTX-3090"]
    validations = []

    for gpu in gpus:
        print(f"Testing {gpu}...")

        # Create plan
        plan = planner.create_plan(
            model="meta-llama/Llama-3.2-1B-Instruct",
            hardware=gpu,
            target_rps=10.0,
            input_length=512,
            output_length=128,
        )

        print(f"  Predicted throughput: {plan.performance.throughput_tokens_per_sec:,.0f} tok/s")

        # Note: In a real scenario, you would run benchmarks on each GPU
        # For this example, we'll just create the config
        bench_config = create_config_from_plan(
            model=plan.model,
            hardware=plan.hardware,
            precision=plan.recommended_precision,
            batch_size=plan.recommended_batch_size,
            input_length=512,
            output_length=128,
        )

        print(f"  Benchmark config: {bench_config.dtype}, batch={bench_config.batch_size}")
        print()

    print("To run full validation:")
    print("1. Run benchmark on each GPU")
    print("2. Collect results")
    print("3. Use validator.validate_multiple() to compare")
    print("4. Use validator.aggregate_results() for summary statistics")
    print()


def example_prefix_caching_validation() -> None:
    """
    Example 4: Validate prefix caching benefits.

    Compares with/without caching enabled.
    """
    print("=" * 80)
    print("EXAMPLE 4: Prefix Caching Validation")
    print("=" * 80)
    print()

    planner = DeploymentPlanner()

    # Plan with caching enabled
    plan_with_cache = planner.create_plan(
        model="meta-llama/Llama-3.2-1B-Instruct",
        hardware="RTX-4090",
        target_rps=10.0,
        enable_caching=True,
        system_prompt_length=512,
    )

    # Plan without caching
    plan_no_cache = planner.create_plan(
        model="meta-llama/Llama-3.2-1B-Instruct",
        hardware="RTX-4090",
        target_rps=10.0,
        enable_caching=False,
    )

    print("Predicted benefits of prefix caching:")
    print(f"  Without caching: {plan_no_cache.performance.throughput_tokens_per_sec:,.0f} tok/s")
    print(f"  With caching: {plan_with_cache.performance.throughput_tokens_per_sec:,.0f} tok/s")

    if plan_with_cache.caching:
        print(f"  Latency reduction: {plan_with_cache.caching.latency_reduction_pct:.0f}%")
        print(f"  Memory savings: {plan_with_cache.caching.memory_savings_pct:.0f}%")
    print()

    # Create benchmark configs
    config_with_cache = create_config_from_plan(
        model=plan_with_cache.model,
        hardware=plan_with_cache.hardware,
        precision=plan_with_cache.recommended_precision,
        batch_size=plan_with_cache.recommended_batch_size,
    )
    config_with_cache.enable_prefix_caching = True

    config_no_cache = create_config_from_plan(
        model=plan_no_cache.model,
        hardware=plan_no_cache.hardware,
        precision=plan_no_cache.recommended_precision,
        batch_size=plan_no_cache.recommended_batch_size,
    )
    config_no_cache.enable_prefix_caching = False

    print("To validate caching benefits:")
    print("1. Run benchmark with --enable-prefix-caching")
    print("2. Run benchmark without prefix caching")
    print("3. Compare throughput and latency differences")
    print("4. Validate against predicted caching benefits")
    print()


def main() -> None:
    """Run all examples."""
    examples = [
        ("Basic Validation", example_basic_validation),
        ("Latency Validation", example_latency_validation),
        ("Multi-GPU Validation", example_multiple_gpu_validation),
        ("Prefix Caching Validation", example_prefix_caching_validation),
    ]

    print("\nBenchmark Validation Examples")
    print("=" * 80)
    print()
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print("  0. Run all examples")
    print()

    try:
        choice = input("Select example (0-4, or q to quit): ").strip()

        if choice.lower() == 'q':
            return

        choice_num = int(choice)

        if choice_num == 0:
            # Run all examples
            for name, func in examples:
                print()
                func()
                print()
        elif 1 <= choice_num <= len(examples):
            # Run selected example
            _, func = examples[choice_num - 1]
            func()
        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
