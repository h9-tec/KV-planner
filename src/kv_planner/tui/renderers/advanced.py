"""Advanced modal — tune efficiency knobs live."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


def render(state: AppState, theme: Theme, width: int, focus_field: int = 0) -> str:
    e = state.efficiency
    fields = [
        ("Memory eff. (MBU)",     e.memory_efficiency,
         "0.75 = vLLM+FlashAttention  ·  0.35 = llama.cpp"),
        ("Compute eff. (MFU)",    e.compute_efficiency,
         "0.50 = well-tuned prefill  ·  0.30 = unoptimised"),
        ("Run mode: GPU",         e.run_mode_gpu,           "pure-GPU inference"),
        ("Run mode: CPU offload", e.run_mode_cpu_offload,   "weights spill to system RAM"),
        ("Run mode: MoE",         e.run_mode_moe,           "MoE expert routing overhead"),
    ]

    t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    t.add_column("field", style="muted", width=22)
    t.add_column("value", width=8, justify="right", style=theme.fg)
    t.add_column("note", style="muted")

    for i, (label, val, note) in enumerate(fields):
        cur = f"[b {theme.accent}]→[/] " if i == focus_field else "  "
        val_s = f"{val:.2f}"
        val_cell = f"[b {theme.accent}]{val_s}[/]" if i == focus_field else val_s
        t.add_row(cur + label, val_cell, note)

    hint = (
        f"[{theme.muted}]Tab: cycle field  ·  +/- to tweak 0.05  ·  "
        f"digits to type  ·  Enter: apply  ·  Ctrl-R: reset  ·  Esc: cancel[/]"
    )

    panel = Panel(
        t,
        title=f"[b {theme.accent}]Advanced · efficiency knobs[/]",
        subtitle=hint,
        subtitle_align="left",
        title_align="left",
        border_style=theme.accent,
        padding=(1, 2),
    )
    return render_to_ansi(panel, width=min(width, 84), theme=theme)
