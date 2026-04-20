"""Physics-based model recommender — llmfit-style workflow on a Roofline core.

``Recommender.top_n()`` ranks catalog models for a given (hardware, use-case,
workload) triple. Each candidate gets four sub-scores in [0, 100]:

* **Quality** — from :mod:`model_catalog` (use-case-aware).
* **Fit** — how snugly the model + KV cache fit the GPU's VRAM.
  100 = ≤ 50 % utilised; 0 = won't fit.
* **Speed** — predicted decode throughput scaled against a reference
  (the fastest runnable candidate).
* **Context** — matches the model's ``max_position_embeddings`` against
  the user's requested context.

Composite score = 0.35·Q + 0.25·F + 0.25·S + 0.15·C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kv_planner.core.performance import RooflineAnalyzer, RooflineConfig
from kv_planner.domain import HardwareSpec, PrecisionType, bytes_per_element
from kv_planner.infrastructure.model_catalog import (
    CATALOG,
    CatalogEntry,
    UseCase,
)


@dataclass(frozen=True)
class Recommendation:
    entry: CatalogEntry
    precision: PrecisionType
    throughput_tok_s: float
    latency_ms: float
    memory_gb: float
    memory_util_pct: float
    fits: bool
    # 0-100 sub-scores
    score_quality: int
    score_fit: int
    score_speed: int
    score_context: int
    score_composite: float


class Recommender:
    def __init__(self, roofline: RooflineAnalyzer | None = None) -> None:
        # A conservative MBU — good for mixed-runtime (Ollama/llama.cpp) users.
        self._rl = roofline or RooflineAnalyzer(
            config=RooflineConfig(compute_efficiency=0.50, memory_efficiency=0.50)
        )

    # ---- scoring helpers --------------------------------------------------
    @staticmethod
    def _fit_score(memory_gb: float, budget_gb: float) -> int:
        if memory_gb > budget_gb:
            return 0
        ratio = memory_gb / budget_gb
        if ratio <= 0.5:
            return 100
        if ratio <= 0.7:
            return 85
        if ratio <= 0.85:
            return 70
        if ratio <= 0.95:
            return 50
        return 25

    @staticmethod
    def _context_score(model_ctx: int, want_ctx: int) -> int:
        if model_ctx >= want_ctx * 4:
            return 100
        if model_ctx >= want_ctx * 2:
            return 90
        if model_ctx >= want_ctx:
            return 75
        if model_ctx >= want_ctx // 2:
            return 40
        return 10

    # ---- public API -------------------------------------------------------
    def recommend(
        self,
        hardware: HardwareSpec,
        use_case: UseCase = "general",
        input_length: int = 2048,
        output_length: int = 512,
        batch_size: int = 1,
        precision: PrecisionType | Literal["auto"] = "auto",
        max_candidates: int = len(CATALOG),
    ) -> list[Recommendation]:
        """Rank every catalog model; return Recommendations sorted best-first."""
        want_ctx = input_length + output_length
        budget_gb = hardware.gpu_memory_gb * hardware.gpu_memory_utilization

        prelim: list[tuple[CatalogEntry, PrecisionType, dict]] = []
        speeds: list[float] = []

        for entry in CATALOG[:max_candidates]:
            cfg = entry.config
            # Pick precision. fp16 by default; fall back to int4 if weights
            # don't fit.
            prec_candidates: list[PrecisionType]
            if precision == "auto":
                prec_candidates = ["fp16", "int8", "int4"]
            else:
                prec_candidates = [precision]

            picked_prec: PrecisionType | None = None
            metrics = None
            for p in prec_candidates:
                weight_gb = cfg.total_params() * bytes_per_element(p) / 1e9
                if weight_gb > budget_gb * 0.95:
                    continue
                try:
                    m = self._rl.predict_latency(
                        model=cfg, hardware=hardware, batch_size=batch_size,
                        input_length=input_length, output_length=output_length,
                        precision=p,
                    )
                except Exception:
                    continue
                picked_prec = p
                metrics = m
                break

            if picked_prec is None or metrics is None:
                # Model doesn't fit at any precision — still rank it, but as "won't fit"
                prec_fallback: PrecisionType = (
                    "int4" if precision == "auto" else precision
                )
                weight_gb = cfg.total_params() * bytes_per_element(prec_fallback) / 1e9
                kv_gb = (
                    cfg.kv_cache_bytes_per_token(prec_fallback)
                    * want_ctx * batch_size / 1e9
                )
                prelim.append((entry, prec_fallback, {
                    "throughput": 0.0, "latency_ms": 0.0,
                    "weight_gb": weight_gb, "kv_gb": kv_gb, "fits": False,
                }))
                speeds.append(0.0)
                continue

            weight_gb = cfg.total_params() * bytes_per_element(picked_prec) / 1e9
            kv_gb = (
                cfg.kv_cache_bytes_per_token(picked_prec)
                * want_ctx * batch_size / 1e9
            )
            total_gb = weight_gb + kv_gb
            fits = total_gb <= budget_gb

            prelim.append((entry, picked_prec, {
                "throughput": metrics.throughput_tokens_per_sec,
                "latency_ms": metrics.total_latency_ms,
                "weight_gb": weight_gb, "kv_gb": kv_gb, "fits": fits,
            }))
            if fits:
                speeds.append(metrics.throughput_tokens_per_sec)

        speed_ref = max(speeds) if speeds else 1.0

        out: list[Recommendation] = []
        for entry, prec, m in prelim:
            total_gb = m["weight_gb"] + m["kv_gb"]
            q = entry.score_for(use_case)
            f = self._fit_score(total_gb, budget_gb)
            s = (
                int(100 * min(1.0, m["throughput"] / speed_ref))
                if (speed_ref > 0 and m["fits"]) else 0
            )
            c = self._context_score(entry.config.max_position_embeddings, want_ctx)

            composite = 0.35 * q + 0.25 * f + 0.25 * s + 0.15 * c

            out.append(Recommendation(
                entry=entry,
                precision=prec,
                throughput_tok_s=m["throughput"],
                latency_ms=m["latency_ms"],
                memory_gb=total_gb,
                memory_util_pct=(total_gb / hardware.gpu_memory_gb * 100),
                fits=m["fits"],
                score_quality=q,
                score_fit=f,
                score_speed=s,
                score_context=c,
                score_composite=composite,
            ))

        out.sort(key=lambda r: r.score_composite, reverse=True)
        return out

    # Convenience: "top N" that skips won't-fit models by default.
    def top_n(
        self,
        hardware: HardwareSpec,
        n: int = 5,
        use_case: UseCase = "general",
        input_length: int = 2048,
        output_length: int = 512,
        batch_size: int = 1,
        include_unfit: bool = False,
    ) -> list[Recommendation]:
        all_recs = self.recommend(
            hardware, use_case=use_case, input_length=input_length,
            output_length=output_length, batch_size=batch_size,
        )
        if not include_unfit:
            all_recs = [r for r in all_recs if r.fits]
        return all_recs[:n]
