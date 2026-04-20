"""Fleet / cluster capacity planner.

llmfit optimises for "what fits on my laptop". Production LLM serving lives
at a different scale: *given a target RPS and a p99 latency SLO, design a
cluster*. This module takes:

* workload (model, input/output length, target RPS, latency SLO)
* GPU candidates to consider (e.g., H100, H200, RTX-5090)
* parallelism strategies (TP ∈ {1,2,4,8}, replicas)

… and returns a ranked list of *cluster designs*, each with:

* TP × replicas chosen so each replica fits in VRAM
* predicted steady-state throughput per replica
* number of replicas needed to hit the RPS target
* aggregate $/hour and $/M tokens
* whether the latency SLO is met

Ranking = cheapest cluster that meets SLO.
"""

from __future__ import annotations

from dataclasses import dataclass

from kv_planner.core.cost import CostAnalyzer
from kv_planner.core.performance import RooflineAnalyzer, RooflineConfig
from kv_planner.domain import HardwareSpec, ModelConfig, PrecisionType, bytes_per_element
from kv_planner.infrastructure.hardware_db import GPUDatabase


# Common production TP configurations (must divide num_kv_heads).
DEFAULT_TP_CANDIDATES: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class FleetDesign:
    gpu_model: str
    tp_size: int
    replicas: int
    total_gpus: int
    precision: PrecisionType
    per_replica_throughput_tok_s: float
    aggregate_throughput_tok_s: float
    p99_latency_ms: float
    meets_slo: bool
    cost_per_hour: float
    cost_per_million_tokens: float
    batch_per_replica: int
    per_replica_memory_gb: float
    notes: str = ""


class FleetPlanner:
    """Physics-based multi-GPU cluster sizing."""

    def __init__(self, roofline: RooflineAnalyzer | None = None) -> None:
        self._rl = roofline or RooflineAnalyzer(
            config=RooflineConfig(compute_efficiency=0.50, memory_efficiency=0.75)
        )

    def _fit_in_gpu(
        self,
        model: ModelConfig,
        gpu: HardwareSpec,
        precision: PrecisionType,
        input_len: int,
        output_len: int,
    ) -> int:
        """Max batch that fits in VRAM (weights + KV, simple model)."""
        bpe = bytes_per_element(precision)
        weight_bytes = model.total_params() * bpe / gpu.tensor_parallel_size
        budget_bytes = gpu.gpu_memory_gb * gpu.gpu_memory_utilization * 1e9
        kv_per_seq = model.kv_cache_bytes_per_token(precision) * (input_len + output_len)
        available = max(0.0, budget_bytes - weight_bytes)
        if kv_per_seq <= 0:
            return 1
        return max(1, int(available / kv_per_seq))

    def design(
        self,
        model: ModelConfig,
        target_rps: float,
        input_length: int,
        output_length: int,
        gpu_candidates: list[str],
        p99_latency_ms: float = 2000.0,
        precisions: tuple[PrecisionType, ...] = ("fp16", "fp8", "int4"),
        tp_candidates: tuple[int, ...] = DEFAULT_TP_CANDIDATES,
    ) -> list[FleetDesign]:
        """Rank all valid (gpu, tp, precision) designs cheapest-first."""
        designs: list[FleetDesign] = []
        tokens_per_req = input_length + output_length

        for gkey in gpu_candidates:
            base = GPUDatabase.to_hardware_spec(gkey)

            for tp in tp_candidates:
                # TP must divide KV heads (required for attention parallelism).
                if model.num_key_value_heads % tp != 0:
                    continue

                # Build a TP×1 hardware spec for the replica.
                try:
                    hw = GPUDatabase.to_hardware_spec(
                        gkey, num_gpus=tp, tensor_parallel_size=tp
                    )
                except Exception:
                    continue

                for prec in precisions:
                    # Will it even load?
                    max_bs = self._fit_in_gpu(model, hw, prec, input_length, output_length)
                    if max_bs < 1:
                        continue

                    # Pick a modest batch that gives good throughput.
                    batch = min(max_bs, 32)
                    try:
                        perf = self._rl.predict_latency(
                            model=model, hardware=hw,
                            batch_size=batch,
                            input_length=input_length,
                            output_length=output_length,
                            precision=prec,
                        )
                    except Exception:
                        continue

                    meets_slo = perf.total_latency_ms <= p99_latency_ms

                    # Per-replica RPS.
                    per_replica_rps = (
                        perf.throughput_tokens_per_sec / tokens_per_req
                        if tokens_per_req else 0
                    )
                    if per_replica_rps <= 0:
                        continue

                    replicas_needed = max(1, int((target_rps / per_replica_rps) + 0.999))
                    total_gpus = replicas_needed * tp

                    # Cost.
                    try:
                        cost = CostAnalyzer(roofline_analyzer=self._rl).analyze_cost(
                            model=model, hardware=hw,
                            batch_size=batch,
                            input_length=input_length,
                            output_length=output_length,
                            requests_per_second=per_replica_rps,
                            precision=prec,
                        )
                        cost_per_hour_single_replica = cost.cost_per_hour
                    except Exception:
                        continue

                    cost_per_hour_total = cost_per_hour_single_replica * replicas_needed
                    total_tokens_per_hour = target_rps * tokens_per_req * 3600
                    cost_per_mtok = (
                        (cost_per_hour_total / total_tokens_per_hour) * 1e6
                        if total_tokens_per_hour > 0 else float("inf")
                    )

                    notes = ""
                    if tp > 1:
                        notes = f"TP={tp} requires NVLink/NVSwitch for decent perf"
                    if not meets_slo:
                        notes = (notes + "; " if notes else "") + "misses latency SLO"

                    designs.append(FleetDesign(
                        gpu_model=gkey,
                        tp_size=tp,
                        replicas=replicas_needed,
                        total_gpus=total_gpus,
                        precision=prec,
                        per_replica_throughput_tok_s=perf.throughput_tokens_per_sec,
                        aggregate_throughput_tok_s=perf.throughput_tokens_per_sec * replicas_needed,
                        p99_latency_ms=perf.total_latency_ms,
                        meets_slo=meets_slo,
                        cost_per_hour=cost_per_hour_total,
                        cost_per_million_tokens=cost_per_mtok,
                        batch_per_replica=batch,
                        per_replica_memory_gb=(
                            (model.total_params() * bytes_per_element(prec) / tp) / 1e9
                            + (model.kv_cache_bytes_per_token(prec) * (input_length + output_length) * batch) / 1e9
                        ),
                        notes=notes,
                    ))

        # Rank: valid SLO-meeting designs, cheapest first.
        meeting = [d for d in designs if d.meets_slo]
        missing = [d for d in designs if not d.meets_slo]
        meeting.sort(key=lambda d: d.cost_per_million_tokens)
        missing.sort(key=lambda d: d.p99_latency_ms)
        return meeting + missing
