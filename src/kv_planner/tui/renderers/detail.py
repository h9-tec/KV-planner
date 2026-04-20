"""Detail pane — rationale + full metrics for the selected row."""

from __future__ import annotations

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kv_planner.application.rationale import explain
from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.scoring import Row, Snapshot, effective_hardware
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


def render(row: Row, snap: Snapshot, state: AppState, theme: Theme, width: int) -> str:
    hw = effective_hardware(snap, state)
    rationale = explain(
        row.rec, hw,
        input_length=state.input_length,
        output_length=state.output_length,
        batch_size=state.batch_size,
    )

    # ---- Scores sub-table -----------------------------------------------
    scores = Table(show_header=False, box=None, padding=(0, 1))
    scores.add_column(style="muted", width=10)
    scores.add_column(justify="right", width=5)
    scores.add_row("Quality", f"[success]{row.rec.score_quality}[/]")
    scores.add_row("Fit", f"[accent2]{row.rec.score_fit}[/]")
    scores.add_row("Speed", f"[accent]{row.rec.score_speed}[/]")
    scores.add_row("Context", f"[accent2]{row.rec.score_context}[/]")
    scores.add_row(
        "[b]Composite[/]",
        f"[b {theme.accent}]{row.rec.score_composite:.1f}[/]",
    )

    # ---- Metrics sub-table ----------------------------------------------
    metrics = Table(show_header=False, box=None, padding=(0, 1))
    metrics.add_column(style="muted", width=16)
    metrics.add_column(justify="right")
    metrics.add_row("Provider", row.provider)
    metrics.add_row("Params", f"{row.params_b:.2f} B")
    metrics.add_row("Precision", row.quant)
    metrics.add_row("KV/token", f"{row.rec.entry.config.kv_cache_bytes_per_token(row.rec.precision) / 1024:.1f} KiB")
    metrics.add_row("Memory", f"{row.memory_gb:.2f} / {hw.gpu_memory_gb:.0f} GB ({row.memory_pct:.0f}%)")
    metrics.add_row("Prefill", f"{row.rec.latency_ms * 0.2:.0f} ms (est.)")
    metrics.add_row("Decode", f"{row.throughput:.0f} tok/s")
    metrics.add_row("Context window", f"{row.context:,}")
    metrics.add_row("Release", row.release_date)
    metrics.add_row("License", row.license)
    tags = ", ".join(row.ollama_tags) if row.ollama_tags else "—"
    metrics.add_row("Ollama tags", tags)

    # ---- Rationale bullets ----------------------------------------------
    rat = Text()
    rat.append(f"{rationale.verdict}\n\n", style=f"bold {theme.accent}")
    for i, bullet in enumerate(rationale.bullets, 1):
        rat.append(f"{i}. ", style=f"bold {theme.accent2}")
        rat.append(bullet + "\n", style=theme.fg)
    if rationale.caveats:
        rat.append("\nCaveats\n", style=f"bold {theme.warning}")
        for c in rationale.caveats:
            rat.append(f"  • {c}\n", style=theme.warning)

    left = Panel(
        Columns([scores, metrics], expand=False, equal=False, padding=(0, 2)),
        title=f"[b {theme.accent}]{row.slug}[/]",
        title_align="left",
        border_style=theme.border,
    )
    right = Panel(
        rat,
        title=f"[b]rationale · physics[/]",
        title_align="left",
        border_style=theme.border,
    )

    return render_to_ansi(
        Columns([left, right], expand=True, padding=(0, 1)),
        width=width,
        theme=theme,
    )
