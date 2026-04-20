"""Physics-grounded rationale generator.

llmfit reports a score. Nobody knows WHY model X scored 82. kv-planner's
edge: we have every formula, so we can *explain* the score — "this is
compute-bound because AI=380 exceeds the ridge 295, so FP8 doubles
effective throughput without improving memory-bound decode".

Takes a :class:`Recommendation` and produces a short (~5 bullet) physics-
grounded rationale. All numbers in the rationale come from the roofline
itself, not from a template.
"""

from __future__ import annotations

from dataclasses import dataclass

from kv_planner.application.recommender import Recommendation
from kv_planner.core.performance import RooflineAnalyzer
from kv_planner.domain import HardwareSpec, bytes_per_element


@dataclass(frozen=True)
class Rationale:
    """A physics-grounded explanation of a recommendation."""

    bullets: list[str]
    verdict: str  # one-sentence summary
    caveats: list[str]


def explain(
    rec: Recommendation,
    hardware: HardwareSpec,
    input_length: int = 2048,
    output_length: int = 512,
    batch_size: int = 1,
    analyzer: RooflineAnalyzer | None = None,
) -> Rationale:
    """Emit a rationale for why ``rec`` is / isn't a good pick."""
    ra = analyzer or RooflineAnalyzer()
    cfg = rec.entry.config
    prec = rec.precision

    bullets: list[str] = []
    caveats: list[str] = []

    # --- 1. Memory fit -------------------------------------------------------
    bpe = bytes_per_element(prec)
    weight_gb = cfg.total_params() * bpe / 1e9
    kv_per_tok_kib = cfg.kv_cache_bytes_per_token(prec) / 1024
    kv_gb = cfg.kv_cache_bytes_per_token(prec) * (input_length + output_length) * batch_size / 1e9
    util = rec.memory_util_pct

    fit_hint = (
        "comfortable headroom" if util < 50 else
        "reasonable fit" if util < 75 else
        "tight fit — reduce batch/context if you need burst headroom"
        if util < 95 else "at the ceiling — a single big prompt could OOM"
    )
    bullets.append(
        f"Memory: {weight_gb:.1f} GB weights + {kv_gb:.2f} GB KV at "
        f"{kv_per_tok_kib:.1f} KiB/token = {rec.memory_gb:.1f} GB "
        f"({util:.0f}% of {hardware.gpu_memory_gb:.0f} GB) — {fit_hint}."
    )

    # --- 2. Roofline regime --------------------------------------------------
    ridge = ra.get_hardware_balance_point(hardware, prec)
    ai_decode = ra.calculate_arithmetic_intensity_decode(
        cfg, batch_size, input_length + output_length, prec
    )
    ai_prefill = ra.calculate_arithmetic_intensity_prefill(
        cfg, batch_size, input_length, prec
    )
    decode_regime = "compute-bound" if ai_decode > ridge else "memory-bound"
    prefill_regime = "compute-bound" if ai_prefill > ridge else "memory-bound"
    bullets.append(
        f"Roofline: hardware ridge = {ridge:.0f} FLOPs/byte. "
        f"Prefill AI = {ai_prefill:.0f} ({prefill_regime}), "
        f"decode AI = {ai_decode:.1f} ({decode_regime})."
    )

    # --- 3. Why this precision? ---------------------------------------------
    if prec in ("int4", "int8", "fp8"):
        memory_gain = {
            "int4": "4× smaller KV cache + 4× smaller weights vs fp16",
            "int8": "2× smaller KV cache + 2× smaller weights vs fp16",
            "fp8": "2× smaller weights with Hopper+ transformer engine",
        }[prec]
        bullets.append(
            f"Precision {prec.upper()} chosen: {memory_gain}. "
            f"Memory-bound decode halves with halved bytes streamed, "
            f"so tok/s roughly scales with 1/bytes_per_element."
        )
    elif prec in ("fp16", "bf16"):
        bullets.append(
            "Precision FP16/BF16 chosen: quantization not needed at this "
            "size — the model fits comfortably and FP16 preserves full quality."
        )

    # --- 4. Throughput + speed score -----------------------------------------
    bullets.append(
        f"Predicted throughput: {rec.throughput_tok_s:.0f} tok/s end-to-end "
        f"({rec.score_speed}/100 relative to the fastest runnable candidate). "
        f"Total latency ~{rec.latency_ms:.0f} ms at batch 1."
    )

    # --- 5. Composite score commentary ---------------------------------------
    q, f, s, c = (
        rec.score_quality, rec.score_fit, rec.score_speed, rec.score_context
    )
    driver = max(
        [("quality", 0.35 * q), ("fit", 0.25 * f),
         ("speed", 0.25 * s), ("context", 0.15 * c)],
        key=lambda x: x[1],
    )[0]
    weakest = min(
        [("quality", q), ("fit", f), ("speed", s), ("context", c)],
        key=lambda x: x[1],
    )[0]
    bullets.append(
        f"Score {rec.score_composite:.1f}/100 driven mostly by {driver}"
        + (f"; weakest axis is {weakest}." if weakest != driver else ".")
    )

    # --- Caveats -------------------------------------------------------------
    if not rec.fits:
        caveats.append("Will NOT fit this GPU — consider a larger VRAM tier or a smaller model.")
    if rec.entry.license.startswith("CC-BY-NC"):
        caveats.append(
            f"{rec.entry.license} license: non-commercial use only — check policy before shipping."
        )
    if util > 90 and rec.fits:
        caveats.append("VRAM utilisation > 90 % — no headroom for long-prompt bursts.")
    if decode_regime == "memory-bound" and prec == "fp16":
        caveats.append(
            "Decode is memory-bound — quantizing to FP8 or INT8 would typically give a ~1.5–2× speedup."
        )

    # --- Verdict -------------------------------------------------------------
    if rec.score_composite >= 80:
        verdict = f"Strong pick for this GPU and use case."
    elif rec.score_composite >= 65:
        verdict = f"Solid option; trade-offs worth checking."
    elif rec.score_composite >= 45:
        verdict = f"Usable but not optimal — consider the top 3 first."
    else:
        verdict = f"Not recommended on this hardware for {rec.entry.use_cases}."

    return Rationale(bullets=bullets, verdict=verdict, caveats=caveats)
