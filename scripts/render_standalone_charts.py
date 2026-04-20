"""Render 8 GUI-look-alike charts to PNG using pure cairo (no Pango).

Purpose: documentation. The live GUI uses PangoCairo for text which requires
a python3-gi-cairo system package not installed here. These standalone renders
use cairo's built-in text API, producing the same colors and layout but with
a plainer (but still clean) typeface.
"""

from __future__ import annotations

import math
import pathlib
import sys

import cairo

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from kv_planner.application.recommender import Recommender  # noqa: E402
from kv_planner.core.cost import CostAnalyzer  # noqa: E402
from kv_planner.core.memory import PagedMemoryCalculator  # noqa: E402
from kv_planner.core.performance import RooflineAnalyzer  # noqa: E402
from kv_planner.domain import bytes_per_element  # noqa: E402
from kv_planner.gui.presets import PRESETS  # noqa: E402
from kv_planner.infrastructure.hardware_db import GPUDatabase  # noqa: E402

OUTDIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Palette (same as GUI)
BG = (0.098, 0.106, 0.133)
CARD = (0.141, 0.149, 0.180)
TEXT = (0.898, 0.898, 0.918)
MUTED = (0.56, 0.58, 0.62)
GRID = (0.224, 0.235, 0.263)
ACCENT = (0.980, 0.580, 0.235)
TEAL = (0.290, 0.745, 0.678)
ROSE = (0.922, 0.431, 0.533)
VIOLET = (0.604, 0.529, 0.929)
LIME = (0.639, 0.843, 0.361)


def _new(width: int, height: int) -> tuple[cairo.ImageSurface, cairo.Context]:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    cr = cairo.Context(surface)
    cr.set_source_rgb(*CARD)
    cr.rectangle(0, 0, width, height)
    cr.fill()
    return surface, cr


def _text(cr, text: str, x: float, y: float, *,
          size: float = 12, bold: bool = False, colour=TEXT, align="left") -> float:
    cr.select_font_face(
        "JetBrains Mono",
        cairo.FONT_SLANT_NORMAL,
        cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
    )
    cr.set_font_size(size)
    xb, yb, w, h, xa, ya = cr.text_extents(text)
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2
    cr.set_source_rgb(*colour)
    cr.move_to(x, y + size * 0.8)
    cr.show_text(text)
    return w


def _save(surface, name: str) -> None:
    out = OUTDIR / name
    surface.write_to_png(str(out))
    print(f"  ✓ {out.relative_to(pathlib.Path(__file__).resolve().parent.parent)}")


# ---------------------------------------------------------------------------
# 1. Stacked bar (memory breakdown)
# ---------------------------------------------------------------------------


def chart_memory(weight_gb: float, kv_gb: float, free_gb: float, budget_gb: float) -> None:
    W, H = 1200, 240
    surface, cr = _new(W, H)
    pad = 24
    _text(cr, "MEMORY BREAKDOWN · per-GPU", pad, 10, size=13, bold=True)

    bar_y = 56
    bar_h = 36
    bar_w = W - 2 * pad
    total = weight_gb + kv_gb + free_gb
    # Track
    cr.set_source_rgb(*GRID)
    cr.rectangle(pad, bar_y, bar_w, bar_h)
    cr.fill()
    # Segments
    x = pad
    for label, v, col in (
        ("weights", weight_gb, VIOLET),
        ("KV cache", kv_gb, ACCENT),
        ("free", free_gb, (0.30, 0.32, 0.38)),
    ):
        seg_w = v / budget_gb * bar_w
        cr.set_source_rgb(*col)
        cr.rectangle(x, bar_y, seg_w, bar_h)
        cr.fill()
        x += seg_w

    # Summary
    used = weight_gb + kv_gb
    pct = used / budget_gb * 100
    _text(cr, f"{used:.1f} / {budget_gb:.0f} GB  ({pct:.0f}% of device)",
          pad, bar_y + bar_h + 16, size=12, colour=MUTED)

    # Legend
    legend_y = bar_y + bar_h + 44
    x = pad
    for label, v, col in (
        ("weights", weight_gb, VIOLET),
        ("KV cache", kv_gb, ACCENT),
        ("free headroom", free_gb, (0.30, 0.32, 0.38)),
    ):
        cr.set_source_rgb(*col)
        cr.rectangle(x, legend_y + 2, 14, 14)
        cr.fill()
        w = _text(cr, f"{label}  {v:.2f} GB", x + 22, legend_y, size=12)
        x += w + 48
    _save(surface, "gui-02-memory.png")


# ---------------------------------------------------------------------------
# 2. Horizontal bars (latency, precision, recommend, gpu)
# ---------------------------------------------------------------------------


def chart_bars(title: str, bars: list[tuple[str, float, str, tuple]],
               unit: str, out: str, value_fmt=lambda v: f"{v:,.1f}") -> None:
    H = 80 + 40 * len(bars)
    W = 1200
    surface, cr = _new(W, H)
    pad_l, pad_r = 24, 24
    _text(cr, title, pad_l, 10, size=13, bold=True)

    y = 48
    max_v = max((b[1] for b in bars), default=1.0) or 1.0
    label_w = 230
    value_w = 150
    bar_left = pad_l + label_w
    bar_right = W - pad_r - value_w
    bar_width_avail = max(80, bar_right - bar_left)

    for label, val, subtitle, col in bars:
        # Label
        _text(cr, label, pad_l, y, size=12)
        if subtitle:
            _text(cr, subtitle, pad_l, y + 14, size=10, colour=MUTED)
        # Track
        cr.set_source_rgb(*GRID)
        cr.rectangle(bar_left, y + 6, bar_width_avail, 12)
        cr.fill()
        # Bar
        cr.set_source_rgb(*col)
        cr.rectangle(bar_left, y + 6, val / max_v * bar_width_avail, 12)
        cr.fill()
        # Value
        _text(cr, f"{value_fmt(val)} {unit}".strip(), W - pad_r, y,
              size=11, align="right")
        y += 36
    _save(surface, out)


# ---------------------------------------------------------------------------
# 3. Line chart (batch sweep, context scaling)
# ---------------------------------------------------------------------------


def chart_line(title: str, points: list[tuple[float, float]],
               x_label: str, y_label: str, out: str, colour=ACCENT,
               x_fmt=lambda v: f"{int(v)}", y_fmt=lambda v: f"{int(v):,}") -> None:
    W, H = 1200, 440
    surface, cr = _new(W, H)
    _text(cr, title, 24, 10, size=13, bold=True)

    pad_l, pad_r, pad_t, pad_b = 80, 24, 44, 56
    plot_x, plot_y = pad_l, pad_t
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs) or 1.0
    y_min, y_max = 0.0, (max(ys) or 1.0) * 1.1

    def xp(v): return plot_x + (v - x_min) / (x_max - x_min) * plot_w
    def yp(v): return plot_y + plot_h - (v - y_min) / (y_max - y_min) * plot_h

    # Grid
    cr.set_source_rgb(*GRID)
    cr.set_line_width(1.0)
    for i in range(6):
        gy = y_min + (y_max - y_min) * i / 5
        py = yp(gy)
        cr.move_to(plot_x, py)
        cr.line_to(plot_x + plot_w, py)
        _text(cr, y_fmt(gy), plot_x - 6, py - 9, size=10,
              align="right", colour=MUTED)
    cr.stroke()

    # X ticks
    for i in range(7):
        gx = x_min + (x_max - x_min) * i / 6
        px = xp(gx)
        _text(cr, x_fmt(gx), px, plot_y + plot_h + 8, size=10,
              align="center", colour=MUTED)

    # Frame
    cr.set_source_rgb(*MUTED)
    cr.set_line_width(1.0)
    cr.move_to(plot_x, plot_y)
    cr.line_to(plot_x, plot_y + plot_h)
    cr.line_to(plot_x + plot_w, plot_y + plot_h)
    cr.stroke()

    # Axis labels
    _text(cr, x_label, plot_x + plot_w / 2, H - 18, size=11,
          align="center", colour=MUTED)
    # Rotated y label
    cr.save()
    cr.translate(20, plot_y + plot_h / 2)
    cr.rotate(-math.pi / 2)
    _text(cr, y_label, 0, 0, size=11, align="center", colour=MUTED)
    cr.restore()

    # Series line
    cr.set_source_rgb(*colour)
    cr.set_line_width(2.6)
    first = True
    for x, y in points:
        if first:
            cr.move_to(xp(x), yp(y))
            first = False
        else:
            cr.line_to(xp(x), yp(y))
    cr.stroke()

    # Points
    for x, y in points:
        cr.arc(xp(x), yp(y), 3.5, 0, 2 * math.pi)
        cr.fill()

    _save(surface, out)


# ---------------------------------------------------------------------------
# 4. Roofline (log-log)
# ---------------------------------------------------------------------------


def chart_roofline(peak_tflops: float, peak_bw_gb_s: float,
                   workloads: list[tuple[str, float, float, tuple]]) -> None:
    W, H = 1200, 520
    surface, cr = _new(W, H)
    _text(cr, f"ROOFLINE — peak {peak_tflops:.0f} TFLOPS  ·  "
              f"{peak_bw_gb_s:.0f} GB/s  ·  ridge = peak ÷ bw",
          24, 10, size=13, bold=True)

    pad_l, pad_r, pad_t, pad_b = 80, 40, 56, 56
    plot_x, plot_y = pad_l, pad_t
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b

    ai_min, ai_max = 0.1, 10000.0
    t_min, t_max = max(peak_tflops * 0.005, 0.05), peak_tflops * 1.3
    lax = (math.log10(ai_min), math.log10(ai_max))
    lay = (math.log10(t_min), math.log10(t_max))

    def xp(ai):
        f = (math.log10(max(ai, ai_min)) - lax[0]) / (lax[1] - lax[0])
        return plot_x + f * plot_w

    def yp(t):
        f = (math.log10(max(t, t_min)) - lay[0]) / (lay[1] - lay[0])
        return plot_y + plot_h - f * plot_h

    # Log grid
    cr.set_source_rgb(*GRID)
    cr.set_line_width(0.7)
    for e in range(-1, 5):
        for step in range(1, 10):
            xi = xp(step * 10 ** e)
            if plot_x <= xi <= plot_x + plot_w:
                cr.move_to(xi, plot_y)
                cr.line_to(xi, plot_y + plot_h)
    for e in range(-2, 5):
        yi = yp(10 ** e)
        if plot_y <= yi <= plot_y + plot_h:
            cr.move_to(plot_x, yi)
            cr.line_to(plot_x + plot_w, yi)
    cr.stroke()

    # Tick labels
    for e in range(-1, 5):
        xi = xp(10 ** e)
        if plot_x <= xi <= plot_x + plot_w:
            _text(cr, f"10^{e}", xi, plot_y + plot_h + 8, size=10,
                  align="center", colour=MUTED)
    for e in range(-2, 5):
        yi = yp(10 ** e)
        if plot_y <= yi <= plot_y + plot_h:
            _text(cr, f"10^{e}", plot_x - 6, yi - 9, size=10,
                  align="right", colour=MUTED)

    # Axes
    cr.set_source_rgb(*MUTED)
    cr.move_to(plot_x, plot_y)
    cr.line_to(plot_x, plot_y + plot_h)
    cr.line_to(plot_x + plot_w, plot_y + plot_h)
    cr.stroke()

    _text(cr, "arithmetic intensity (FLOPs / byte) →",
          plot_x + plot_w / 2, H - 20, size=11, align="center", colour=MUTED)

    # Roofline
    ridge = peak_tflops * 1e12 / (peak_bw_gb_s * 1e9)
    x_ridge = xp(ridge)
    y_peak = yp(peak_tflops)
    t_left = ai_min * peak_bw_gb_s / 1000

    cr.set_source_rgb(*ACCENT)
    cr.set_line_width(3.0)
    cr.move_to(xp(ai_min), yp(t_left))
    cr.line_to(x_ridge, y_peak)
    cr.line_to(xp(ai_max), y_peak)
    cr.stroke()

    # Ridge dashed line
    cr.set_source_rgba(*ACCENT, 0.35)
    cr.set_line_width(1.5)
    cr.set_dash([6, 6])
    cr.move_to(x_ridge, plot_y)
    cr.line_to(x_ridge, plot_y + plot_h)
    cr.stroke()
    cr.set_dash([])
    _text(cr, f"ridge = {ridge:.0f}", x_ridge + 6, plot_y + 6, size=11,
          colour=ACCENT, bold=True)

    # Workloads
    for label, ai, tflops, col in workloads:
        if ai <= 0 or tflops <= 0:
            continue
        px, py = xp(ai), yp(tflops)
        cr.set_source_rgb(*col)
        cr.arc(px, py, 7, 0, 2 * math.pi)
        cr.fill()
        _text(cr, label, px + 11, py - 10, size=12, colour=col, bold=True)

    _save(surface, "gui-03-roofline.png")


# ---------------------------------------------------------------------------
# Build all charts for the docs scenario
# ---------------------------------------------------------------------------


def main() -> None:
    model = PRESETS["llama-3-8b"].config
    gpu = GPUDatabase.to_hardware_spec("H100-SXM-80GB")
    ra = RooflineAnalyzer()
    batch, in_len, out_len, precision = 32, 2048, 512, "fp16"

    print("\n  Rendering documentation charts "
          "(Llama-3 8B on H100-SXM-80GB)\n")

    perf = ra.predict_latency(
        model=model, hardware=gpu, batch_size=batch,
        input_length=in_len, output_length=out_len, precision=precision,
    )

    # 1. Memory
    bpe = bytes_per_element(precision)
    weight_gb = model.total_params() * bpe / 1e9
    kv_bytes = PagedMemoryCalculator().calculate_kv_cache_size(
        batch_size=batch, sequence_length=in_len + out_len,
        model=model, precision=precision,
    )
    kv_gb = kv_bytes / 1e9
    free_gb = max(0.0, gpu.gpu_memory_gb - weight_gb - kv_gb)
    chart_memory(weight_gb, kv_gb, free_gb, gpu.gpu_memory_gb)

    # 2. Roofline
    ai_pre = ra.calculate_arithmetic_intensity_prefill(model, batch, in_len, precision)
    ai_dec = ra.calculate_arithmetic_intensity_decode(
        model, batch, in_len + out_len // 2, precision)
    chart_roofline(
        peak_tflops=gpu.peak_tflops_for(precision),
        peak_bw_gb_s=gpu.memory_bandwidth_gb_s,
        workloads=[
            ("prefill", ai_pre, perf.prefill_tflops, TEAL),
            ("decode",  ai_dec, perf.decode_tflops, ROSE),
        ],
    )

    # 3. Latency waterfall
    chart_bars(
        "LATENCY WATERFALL",
        [
            ("prefill", perf.prefill_latency_ms,
             f"{in_len} prompt tokens", TEAL),
            ("decode (total)", perf.decode_latency_ms,
             f"{out_len} generated tokens", ROSE),
            ("per-token decode",
             perf.decode_latency_ms / max(1, out_len),
             "ms / generated token", ACCENT),
        ],
        "ms", "gui-04-latency.png",
    )

    # 4. Batch sweep
    batch_pts = []
    for bs in [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]:
        p = ra.predict_latency(
            model=model, hardware=gpu, batch_size=bs,
            input_length=in_len, output_length=out_len, precision=precision,
        )
        batch_pts.append((bs, p.throughput_tokens_per_sec))
    chart_line(
        "BATCH SWEEP — throughput vs batch size",
        batch_pts, "batch size", "tokens / sec",
        "gui-05-batch-sweep.png", colour=ACCENT,
    )

    # 5. Context scaling
    ctx_pts = []
    for ctx in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        ms, _, _ = ra.predict_decode_latency(
            model=model, hardware=gpu, batch_size=batch,
            sequence_length=ctx, precision=precision,
        )
        ctx_pts.append((ctx, ms))
    chart_line(
        "CONTEXT SCALING — decode latency vs context length (the fixed bug)",
        ctx_pts, "context length (tokens)", "ms per generated token",
        "gui-06-context-scaling.png", colour=ROSE,
        y_fmt=lambda v: f"{v:.1f}",
    )

    # 6. Precision comparison
    prec_bars = []
    colours = [VIOLET, TEAL, ACCENT, ROSE, LIME]
    for i, prec in enumerate(["fp16", "bf16", "fp8", "int8", "int4"]):
        try:
            p = ra.predict_latency(
                model=model, hardware=gpu, batch_size=batch,
                input_length=in_len, output_length=out_len, precision=prec,
            )
            prec_bars.append((prec, p.throughput_tokens_per_sec,
                              f"{p.total_latency_ms:.0f} ms total",
                              colours[i % len(colours)]))
        except Exception:
            pass
    chart_bars(
        "QUANTIZATION — throughput by precision",
        prec_bars, "tok/s", "gui-07-precision.png",
        value_fmt=lambda v: f"{v:,.0f}",
    )

    # 7. Recommendations
    rec = Recommender().top_n(
        gpu, n=8, use_case="general",
        input_length=in_len, output_length=out_len, batch_size=1,
    )
    rec_bars = [
        (r.entry.slug, r.score_composite,
         f"{r.precision}  ·  {r.throughput_tok_s:.0f} tok/s  ·  "
         f"{r.memory_gb:.1f} GB  ·  "
         f"Q{r.score_quality} F{r.score_fit} S{r.score_speed} C{r.score_context}",
         colours[i % len(colours)])
        for i, r in enumerate(rec)
    ]
    chart_bars(
        "RECOMMENDED MODELS — physics-scored composite for this GPU",
        rec_bars, "", "gui-08-recommend.png",
        value_fmt=lambda v: f"{v:.1f}",
    )

    # 8. GPU throughput comparison + cost comparison (two panels)
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
            gpu_bars.append((gkey, p.throughput_tokens_per_sec, "", ACCENT))
            cost_bars.append((gkey, c.cost_per_million_tokens, "", LIME))
        except Exception:
            continue
    chart_bars(
        "GPU COMPARISON — throughput on Llama-3 8B fp16",
        gpu_bars, "tok/s", "gui-09-gpu-throughput.png",
        value_fmt=lambda v: f"{v:,.0f}",
    )
    chart_bars(
        "GPU COMPARISON — $ per million tokens",
        cost_bars, "$/M", "gui-10-gpu-cost.png",
        value_fmt=lambda v: f"{v:,.2f}",
    )

    print(f"\n  Done. {len(list(OUTDIR.glob('gui-*.png')))} PNGs ready.\n")


if __name__ == "__main__":
    main()
