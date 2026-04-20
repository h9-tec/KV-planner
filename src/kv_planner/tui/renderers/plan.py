"""Plan mode — inverse hardware sizing for the current model."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from kv_planner.application.recommender import Recommender
from kv_planner.core.performance import RooflineAnalyzer, RooflineConfig
from kv_planner.domain import bytes_per_element
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.scoring import Row
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


_CANDIDATE_GPUS = [
    "RTX-5060-Laptop", "RTX-4060", "RTX-3090", "RTX-4070-Ti", "RTX-4080",
    "RTX-4090", "RTX-5080", "RTX-5090",
    "L40S", "A100-SXM-80GB", "H100-SXM-80GB", "H200-SXM-141GB", "B200-SXM-192GB",
    "MI300X",
]


def render(row: Row, state: AppState, theme: Theme, width: int) -> str:
    t = Table(expand=True, box=None, show_header=True, header_style="header")
    t.add_column("GPU", width=22, no_wrap=True)
    t.add_column("VRAM", width=6, justify="right")
    t.add_column("Precision", width=9)
    t.add_column("Weights", width=8, justify="right")
    t.add_column("Total", width=10, justify="right")
    t.add_column("Util", width=6, justify="right")
    t.add_column("tok/s", width=7, justify="right")
    t.add_column("Fit", width=10)

    for gpu_key in _CANDIDATE_GPUS:
        hw = GPUDatabase.to_hardware_spec(gpu_key)
        ra = RooflineAnalyzer(
            config=RooflineConfig(
                compute_efficiency=state.efficiency.compute_efficiency,
                memory_efficiency=state.efficiency.memory_efficiency,
            )
        )
        rec = Recommender(roofline=ra).recommend(
            hw,
            use_case=state.use_case,
            input_length=state.input_length,
            output_length=state.output_length,
            batch_size=state.batch_size,
        )
        matching = [r for r in rec if r.entry.slug == row.slug]
        if not matching:
            continue
        r = matching[0]
        prec = r.precision
        w_gb = r.entry.config.total_params() * bytes_per_element(prec) / 1e9

        fit_colour = "success" if r.fits and r.memory_util_pct <= 60 else (
            "accent2" if r.fits and r.memory_util_pct <= 80 else (
            "warning" if r.fits else "error"
            )
        )
        fit_text = ("Perfect" if r.memory_util_pct <= 50 else (
            "Good" if r.memory_util_pct <= 75 else (
            "Marginal" if r.memory_util_pct <= 95 else "Too tight"
            )
        )) if r.fits else "Won't fit"

        t.add_row(
            gpu_key, f"{hw.gpu_memory_gb:.0f}", prec,
            f"{w_gb:.1f} GB", f"{r.memory_gb:.1f} GB",
            f"{r.memory_util_pct:.0f}%",
            f"{r.throughput_tok_s:.0f}" if r.fits else "—",
            f"[{fit_colour}]{fit_text}[/]",
        )

    meta = (
        f"[b]{row.slug}[/]  ·  {row.params_b:.1f} B params  ·  "
        f"context {state.input_length + state.output_length} tokens  ·  "
        f"batch {state.batch_size}"
    )
    panel = Panel(
        t,
        title=f"[b {theme.accent}]Plan · which GPU for {row.slug}  (Esc to close)[/]",
        subtitle=meta,
        subtitle_align="left",
        title_align="left",
        border_style=theme.accent,
    )
    return render_to_ansi(panel, width=width, theme=theme)
