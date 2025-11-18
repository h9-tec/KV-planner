"""
Benchmark runner for vLLM performance validation.

This module provides a wrapper around vLLM's benchmark CLI to:
- Run latency benchmarks (vllm bench latency)
- Run throughput benchmarks (vllm bench throughput)
- Run serving benchmarks (vllm bench serve)
- Parse and structure benchmark results
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from kv_planner.domain import HardwareSpec, ModelConfig


@dataclass(frozen=True)
class BenchmarkConfig:
    """
    Configuration for running benchmarks.

    Attributes:
        model_name: HuggingFace model name
        input_length: Number of input tokens
        output_length: Number of output tokens
        batch_size: Batch size for latency benchmarks
        num_prompts: Number of prompts for throughput/serving benchmarks
        dtype: Model precision (float16, bfloat16, float8_e4m3fn, etc.)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Fraction of GPU memory to use
        enable_prefix_caching: Enable prefix caching
        enforce_eager: Disable CUDA graphs (for testing)
        num_warmup_iters: Number of warmup iterations
        num_iters: Number of measurement iterations
    """

    model_name: str
    input_length: int = 2048
    output_length: int = 512
    batch_size: int = 1
    num_prompts: int = 100
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    num_warmup_iters: int = 5
    num_iters: int = 10


@dataclass
class BenchmarkResults:
    """
    Results from a benchmark run.

    Attributes:
        benchmark_type: Type of benchmark (latency, throughput, serving)
        config: Configuration used
        success: Whether benchmark completed successfully
        error_message: Error message if benchmark failed

        # Latency metrics (batch processing)
        mean_latency_ms: Mean latency in milliseconds
        p50_latency_ms: Median latency
        p95_latency_ms: 95th percentile latency
        p99_latency_ms: 99th percentile latency

        # Throughput metrics
        throughput_tokens_per_sec: Total throughput in tokens/sec
        requests_per_sec: Requests per second (serving only)

        # Token-level metrics
        time_to_first_token_ms: TTFT - time to first token
        time_per_output_token_ms: TPOT - time per output token
        inter_token_latency_ms: ITL - time between tokens

        # Memory metrics
        peak_gpu_memory_gb: Peak GPU memory usage

        # Raw outputs
        raw_output: Raw stdout from benchmark
        raw_stderr: Raw stderr from benchmark
    """

    benchmark_type: Literal["latency", "throughput", "serving"]
    config: BenchmarkConfig
    success: bool = False
    error_message: Optional[str] = None

    # Latency metrics
    mean_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None

    # Throughput metrics
    throughput_tokens_per_sec: Optional[float] = None
    requests_per_sec: Optional[float] = None

    # Token-level metrics
    time_to_first_token_ms: Optional[float] = None
    time_per_output_token_ms: Optional[float] = None
    inter_token_latency_ms: Optional[float] = None

    # Memory metrics
    peak_gpu_memory_gb: Optional[float] = None

    # Raw outputs
    raw_output: str = ""
    raw_stderr: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "benchmark_type": self.benchmark_type,
            "config": {
                "model_name": self.config.model_name,
                "input_length": self.config.input_length,
                "output_length": self.config.output_length,
                "batch_size": self.config.batch_size,
                "num_prompts": self.config.num_prompts,
                "dtype": self.config.dtype,
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "enable_prefix_caching": self.config.enable_prefix_caching,
            },
            "success": self.success,
            "error_message": self.error_message,
            "metrics": {
                "mean_latency_ms": self.mean_latency_ms,
                "p50_latency_ms": self.p50_latency_ms,
                "p95_latency_ms": self.p95_latency_ms,
                "p99_latency_ms": self.p99_latency_ms,
                "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
                "requests_per_sec": self.requests_per_sec,
                "time_to_first_token_ms": self.time_to_first_token_ms,
                "time_per_output_token_ms": self.time_per_output_token_ms,
                "inter_token_latency_ms": self.inter_token_latency_ms,
                "peak_gpu_memory_gb": self.peak_gpu_memory_gb,
            },
        }


class BenchmarkRunner:
    """
    Runner for vLLM benchmarks.

    Wraps vLLM CLI commands to run benchmarks and collect results.
    """

    def __init__(self, vllm_path: str = "vllm") -> None:
        """
        Initialize benchmark runner.

        Args:
            vllm_path: Path to vllm executable (default: "vllm" from PATH)
        """
        self.vllm_path = vllm_path

    def run_latency_benchmark(
        self,
        config: BenchmarkConfig,
        verbose: bool = False,
    ) -> BenchmarkResults:
        """
        Run latency benchmark.

        Measures latency for a single batch using `vllm bench latency`.

        Args:
            config: Benchmark configuration
            verbose: Print verbose output

        Returns:
            BenchmarkResults with latency metrics
        """
        cmd = [
            self.vllm_path, "bench", "latency",
            "--model", config.model_name,
            "--input-len", str(config.input_length),
            "--output-len", str(config.output_length),
            "--batch-size", str(config.batch_size),
            "--dtype", config.dtype,
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--num-iters-warmup", str(config.num_warmup_iters),
            "--num-iters", str(config.num_iters),
        ]

        if config.max_model_len:
            cmd.extend(["--max-model-len", str(config.max_model_len)])

        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        if config.enforce_eager:
            cmd.append("--enforce-eager")

        return self._run_command(cmd, "latency", config, verbose)

    def run_throughput_benchmark(
        self,
        config: BenchmarkConfig,
        verbose: bool = False,
    ) -> BenchmarkResults:
        """
        Run throughput benchmark.

        Measures offline inference throughput using `vllm bench throughput`.

        Args:
            config: Benchmark configuration
            verbose: Print verbose output

        Returns:
            BenchmarkResults with throughput metrics
        """
        cmd = [
            self.vllm_path, "bench", "throughput",
            "--model", config.model_name,
            "--input-len", str(config.input_length),
            "--output-len", str(config.output_length),
            "--num-prompts", str(config.num_prompts),
            "--dtype", config.dtype,
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        ]

        if config.max_model_len:
            cmd.extend(["--max-model-len", str(config.max_model_len)])

        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")

        if config.enforce_eager:
            cmd.append("--enforce-eager")

        return self._run_command(cmd, "throughput", config, verbose)

    def run_serving_benchmark(
        self,
        config: BenchmarkConfig,
        server_url: str = "http://localhost:8000",
        verbose: bool = False,
    ) -> BenchmarkResults:
        """
        Run serving benchmark.

        Measures online serving performance using `vllm bench serve`.
        Requires a running vLLM server.

        Args:
            config: Benchmark configuration
            server_url: URL of running vLLM server
            verbose: Print verbose output

        Returns:
            BenchmarkResults with serving metrics
        """
        # Parse server URL
        from urllib.parse import urlparse
        parsed = urlparse(server_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8000

        cmd = [
            self.vllm_path, "bench", "serve",
            "--model", config.model_name,
            "--host", host,
            "--port", str(port),
            "--random-input-len", str(config.input_length),
            "--random-output-len", str(config.output_length),
            "--num-prompts", str(config.num_prompts),
        ]

        return self._run_command(cmd, "serving", config, verbose)

    def _run_command(
        self,
        cmd: list[str],
        benchmark_type: Literal["latency", "throughput", "serving"],
        config: BenchmarkConfig,
        verbose: bool,
    ) -> BenchmarkResults:
        """
        Run benchmark command and parse results.

        Args:
            cmd: Command to run
            benchmark_type: Type of benchmark
            config: Benchmark configuration
            verbose: Print verbose output

        Returns:
            BenchmarkResults
        """
        if verbose:
            print(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if verbose:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

            # Parse results
            benchmark_results = BenchmarkResults(
                benchmark_type=benchmark_type,
                config=config,
                success=result.returncode == 0,
                raw_output=result.stdout,
                raw_stderr=result.stderr,
            )

            if result.returncode == 0:
                # Parse metrics from output
                self._parse_metrics(benchmark_results, result.stdout)
            else:
                benchmark_results.error_message = result.stderr

            return benchmark_results

        except subprocess.TimeoutExpired:
            return BenchmarkResults(
                benchmark_type=benchmark_type,
                config=config,
                success=False,
                error_message="Benchmark timed out after 1 hour",
            )
        except Exception as e:
            return BenchmarkResults(
                benchmark_type=benchmark_type,
                config=config,
                success=False,
                error_message=str(e),
            )

    def _parse_metrics(self, results: BenchmarkResults, output: str) -> None:
        """
        Parse metrics from benchmark output.

        Args:
            results: BenchmarkResults to populate
            output: Raw stdout from benchmark
        """
        lines = output.split("\n")

        for line in lines:
            line_lower = line.lower()

            # Latency metrics
            if "mean latency" in line_lower or "avg latency" in line_lower:
                results.mean_latency_ms = self._extract_float(line)
            elif "p50" in line_lower or "median" in line_lower:
                results.p50_latency_ms = self._extract_float(line)
            elif "p95" in line_lower:
                results.p95_latency_ms = self._extract_float(line)
            elif "p99" in line_lower:
                results.p99_latency_ms = self._extract_float(line)

            # Throughput metrics
            elif "throughput" in line_lower and "token" in line_lower:
                results.throughput_tokens_per_sec = self._extract_float(line)
            elif "request" in line_lower and ("per sec" in line_lower or "rps" in line_lower):
                results.requests_per_sec = self._extract_float(line)

            # Token-level metrics
            elif "ttft" in line_lower or "time to first token" in line_lower:
                results.time_to_first_token_ms = self._extract_float(line)
            elif "tpot" in line_lower or "time per output token" in line_lower:
                results.time_per_output_token_ms = self._extract_float(line)
            elif "itl" in line_lower or "inter token latency" in line_lower:
                results.inter_token_latency_ms = self._extract_float(line)

            # Memory metrics
            elif "peak" in line_lower and "memory" in line_lower:
                results.peak_gpu_memory_gb = self._extract_float(line)

    def _extract_float(self, line: str) -> Optional[float]:
        """
        Extract first float from a line.

        Args:
            line: Line to parse

        Returns:
            Extracted float or None
        """
        import re

        # Look for numbers (including decimals and scientific notation)
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def save_results(self, results: BenchmarkResults, filepath: Path) -> None:
        """
        Save benchmark results to JSON file.

        Args:
            results: Benchmark results
            filepath: Output file path
        """
        with open(filepath, "w") as f:
            json.dump(results.to_dict(), f, indent=2)

    def load_results(self, filepath: Path) -> BenchmarkResults:
        """
        Load benchmark results from JSON file.

        Args:
            filepath: Input file path

        Returns:
            BenchmarkResults
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        config = BenchmarkConfig(**data["config"])

        results = BenchmarkResults(
            benchmark_type=data["benchmark_type"],
            config=config,
            success=data["success"],
            error_message=data.get("error_message"),
        )

        # Load metrics
        metrics = data.get("metrics", {})
        for key, value in metrics.items():
            if value is not None:
                setattr(results, key, value)

        return results


def create_config_from_plan(
    model: ModelConfig,
    hardware: HardwareSpec,
    precision: str,
    batch_size: int,
    input_length: int = 2048,
    output_length: int = 512,
) -> BenchmarkConfig:
    """
    Create benchmark config from kv-planner deployment plan.

    Args:
        model: Model configuration
        hardware: Hardware specification
        precision: Precision (fp16, fp8, int8, int4)
        batch_size: Batch size
        input_length: Input length
        output_length: Output length

    Returns:
        BenchmarkConfig
    """
    # Map precision to vLLM dtype
    dtype_map = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp8": "float8_e4m3fn",
        "int8": "int8",
        "int4": "int4",
    }
    dtype = dtype_map.get(precision.lower(), "auto")

    return BenchmarkConfig(
        model_name=model.name,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        dtype=dtype,
        tensor_parallel_size=hardware.tensor_parallel_size,
        max_model_len=model.max_sequence_length,
        gpu_memory_utilization=0.9,
    )
