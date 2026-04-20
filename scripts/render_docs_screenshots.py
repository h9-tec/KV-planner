"""Render each GUI chart to a PNG in docs/screenshots/ without launching GTK.

Uses the same Cairo renderers the GUI uses internally, just skips the
Gtk.Picture wrapping and writes the surface to disk.
"""

from __future__ import annotations

import io
import pathlib
import sys

import gi
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
try:
    gi.require_foreign("cairo")
except Exception:
    pass

import cairo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from kv_planner.application.recommender import Recommender  # noqa: E402
from kv_planner.core.memory import PagedMemoryCalculator  # noqa: E402
from kv_planner.core.performance import RooflineAnalyzer  # noqa: E402
from kv_planner.domain import ModelConfig, bytes_per_element  # noqa: E402
from kv_planner.gui.charts import (  # noqa: E402
    PAL_ACCENT, PAL_LIME, PAL_ROSE, PAL_TEAL, PAL_VIOLET,
    Bar, BarChart, LineChart, RooflineChart, Series, StackedBar, StackSegment, Workload,
)
from kv_planner.gui.presets import PRESETS  # noqa: E402
from kv_planner.infrastructure.hardware_db import GPUDatabase  # noqa: E402


OUTDIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUTDIR.mkdir(parents=True, exist_ok=True)


def render_chart(chart, width: int, height: int) -> cairo.ImageSurface:
    """Render a kv-planner chart instance to a cairo surface at the given size."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    chart._on_draw(cr, width, height)
    return surface


def save(chart, name: str, width: int = 1200, height: int = 500) -> None:
    """Render + save to docs/screenshots/<name>.png."""
    surface = render_chart(chart, width, height)
    out = OUTDIR / name
    surface.write_to_png(str(out))
    print(f"  ✓ {out.relative_to(pathlib.Path(__file__).resolve().parent.parent)}  "
          f"({width}×{height})")


# ---------------------------------------------------------------------------
# Build the same scenario the GUI shows by default
# ---------------------------------------------------------------------------


def main() -> None:
    model = PRESETS["llama-3-8b"].config
    gpu = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    ra = RooflineAnalyzer()
    batch, in_len, out_len, precision = 32, 2048, 512, "fp16"

    print(f"\n  Rendering charts for Llama-3 8B on H100-SXM-80GB "
          f"(batch={batch}, ctx={in_len}+{out_len}, {precision})")
    print(f"  Output: {OUTDIR}\n")

    perf = ra.predict_latency(
        model=model, hardware=gpu, batch_size=batch,
        input_length=in_len, output_length=out_len, precision=precision,
    )

    # ---- 1. Memory breakdown ------------------------------------------------
    bpe = bytes_per_element(precision)
    weight_gb = model.total_params() * bpe / 1e9
    kv_bytes = PagedMemoryCalculator().calculate_kv_cache_size(
        batch_size=batch, sequence_length=in_len + out_len,
        model=model, precision=precision,
    )
    kv_gb = kv_bytes / 1e9
    free_gb = max(0.0, gpu.gpu_memory_gb - weight_gb - kv_gb)

    mem = StackedBar(title="MEMORY BREAKDOWN (per-GPU, at this batch & context)", unit="GB")
    mem.set_data([
        StackSegment("weights", weight_gb, PAL_VIOLET),
        StackSegment("KV cache", kv_gb, PAL_ACCENT),
        StackSegment("free headroom", free_gb, (0.30, 0.32, 0.38)),
    ], total_budget=gpu.gpu_memory_gb)
    save(mem, "gui-02-memory.png", 1200, 260)

    # ---- 2. Roofline --------------------------------------------------------
    ai_pre = ra.calculate_arithmetic_intensity_prefill(model, batch, in_len, precision)
    ai_dec = ra.calculate_arithmetic_intensity_decode(
        model, batch, in_len + out_len // 2, precision,
    )
    rf = RooflineChart()
    rf.set_data(
        peak_tflops=gpu.peak_tflops_for(precision),
        peak_bw_gb_s=gpu.memory_bandwidth_gb_s,
        workloads=[
            Workload("prefill", ai_pre, perf.prefill_tflops, PAL_TEAL),
            Workload("decode", ai_dec, perf.decode_tflops, PAL_ROSE),
        ],
        title=f"ROOFLINE — H100-SXM-80GB @ {precision}  ·  ridge = peak / bw",
    )
    save(rf, "gui-03-roofline.png", 1200, 520)

    # ---- 3. Latency waterfall ----------------------------------------------
    lat = BarChart(title="LATENCY WATERFALL (ms)", unit="ms",
                   value_format=lambda v: f"{v:,.1f}")
    lat.set_bars([
        Bar("prefill", perf.prefill_latency_ms,
            subtitle=f"{in_len} prompt tokens", colour=PAL_TEAL),
        Bar("decode (total)", perf.decode_latency_ms,
            subtitle=f"{out_len} generated tokens", colour=PAL_ROSE),
        Bar("per-token decode",
            perf.decode_latency_ms / max(1, out_len),
            subtitle="ms / generated token", colour=PAL_ACCENT),
    ])
    save(lat, "gui-04-latency.png", 1200, 280)

    # ---- 4. Batch sweep ----------------------------------------------------
    batch_pts = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]:
        p = ra.predict_latency(
            model=model, hardware=gpu, batch_size=bs,
            input_length=in_len, output_length=out_len, precision=precision,
        )
        batch_pts.append((bs, p.throughput_tokens_per_sec))
    bs_chart = LineChart(
        title="BATCH SWEEP — throughput vs batch size",
        x_label="batch size", y_label="tokens / sec",
        x_format=lambda v: f"{int(v)}",
        y_format=lambda v: f"{int(v):,}",
    )
    bs_chart.set_series([Series("throughput", batch_pts, colour=PAL_ACCENT)])
    save(bs_chart, "gui-05-batch-sweep.png", 1200, 440)

    # ---- 5. Context scaling (the hero bug-fix viz) -------------------------
    ctx_pts = []
    for ctx in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        ms, _, _ = ra.predict_decode_latency(
            model=model, hardware=gpu, batch_size=batch,
            sequence_length=ctx, precision=precision,
        )
        ctx_pts.append((ctx, ms))
    ctx_chart = LineChart(
        title="CONTEXT SWEEP — decode latency vs context length",
        x_label="context length (tokens)", y_label="ms / token",
        x_format=lambda v: f"{int(v)}",
        y_format=lambda v: f"{v:.1f}",
    )
    ctx_chart.set_series([Series("ms/token", ctx_pts, colour=PAL_ROSE)])
    save(ctx_chart, "gui-06-context-scaling.png", 1200, 440)

    # ---- 6. Precision comparison -------------------------------------------
    prec_bars = []
    colours = [PAL_VIOLET, PAL_TEAL, PAL_ACCENT, PAL_ROSE, PAL_LIME]
    for i, prec in enumerate(["fp16", "bf16", "fp8", "int8", "int4"]):
        try:
            p = ra.predict_latency(
                model=model, hardware=gpu, batch_size=batch,
                input_length=in_len, output_length=out_len, precision=prec,
            )
            prec_bars.append(Bar(
                prec, p.throughput_tokens_per_sec,
                subtitle=f"{p.total_latency_ms:.0f} ms total",
                colour=colours[i % len(colours)],
            ))
        except Exception:
            continue
    prec_chart = BarChart(
        title="QUANTIZATION — throughput by precision",
        unit="tok/s", value_format=lambda v: f"{v:,.0f}",
    )
    prec_chart.set_bars(prec_bars)
    save(prec_chart, "gui-07-precision.png", 1200, 320)

    # ---- 7. Recommendations (physics-scored top-8) -------------------------
    rec = Recommender().top_n(
        gpu, n=8, use_case="general",
        input_length=in_len, output_length=out_len, batch_size=1,
    )
    rec_chart = BarChart(
        title="TOP MODELS — composite score for this GPU & use case",
        unit="", value_format=lambda v: f"{v:.1f}",
    )
    rec_chart.set_bars([
        Bar(r.entry.slug, r.score_composite,
            subtitle=(f"{r.precision}  ·  {r.throughput_tok_s:.0f} tok/s  ·  "
                      f"{r.memory_gb:.1f} GB  ·  "
                      f"Q{r.score_quality}·F{r.score_fit}·S{r.score_speed}·C{r.score_context}"),
            colour=colours[i % len(colours)])
        for i, r in enumerate(rec)
    ])
    save(rec_chart, "gui-08-recommend.png", 1200, 480)

    # ---- 8. GPU comparison (throughput) + GPU comparison (cost) ------------
    from kv_planner.core.cost import CostAnalyzer
    gpu_bars = []
    cost_bars = []
    for gkey in ["H100-SXM-80GB", "H200-SXM-141GB", "B200-SXM-192GB",
                 "A100-SXM-80GB", "L40S", "RTX-5090", "RTX-4090",
                 "RTX-3090-Ti", "RTX-3090", "MI300X"]:
        try:
            h = GPUDatabase.to_hardware_spec(gkey)
            if model.total_params() * 2 / 1e9 > h.gpu_memory_gb * 0.95:
                continue
            p = ra.predict_latency(
                model=model, hardware=h, batch_size=batch,
                input_length=in_len, output_length=out_len, precision=precision,
            )
            c = CostAnalyzer(roofline_analyzer=ra).analyze_cost(
                model=model, hardware=h, batch_size=batch,
                input_length=in_len, output_length=out_len,
                requests_per_second=10.0, precision=precision,
            )
            gpu_bars.append(Bar(gkey, p.throughput_tokens_per_sec, colour=PAL_ACCENT))
            cost_bars.append(Bar(gkey, c.cost_per_million_tokens, colour=PAL_LIME))
        except Exception:
            continue
    gpu_chart = BarChart(
        title="GPU COMPARISON — throughput on this workload",
        unit="tok/s", value_format=lambda v: f"{v:,.0f}",
    )
    gpu_chart.set_bars(gpu_bars)
    save(gpu_chart, "gui-09-gpu-throughput.png", 1200, 440)

    cost_chart = BarChart(
        title="GPU COMPARISON — cost per million tokens",
        unit="$/M", value_format=lambda v: f"{v:,.2f}",
    )
    cost_chart.set_bars(cost_bars)
    save(cost_chart, "gui-10-gpu-cost.png", 1200, 440)

    print(f"\n  Done. {len(list(OUTDIR.glob('gui-*.png')))} PNGs in {OUTDIR}")


if __name__ == "__main__":
    main()
