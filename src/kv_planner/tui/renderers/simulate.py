"""Simulate modal — override detected HW (VRAM / RAM / CPU)."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.scoring import Snapshot
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


def render(snap: Snapshot, state: AppState, theme: Theme, width: int, focus_field: int = 0) -> str:
    hw = snap.detected
    fields = [
        ("VRAM (GB)", state.sim_hw.vram_gb, f"detected: {hw.gpu_vram_gb:.1f}"),
        ("RAM (GB)", state.sim_hw.ram_gb, f"detected: {hw.ram_total_gb:.1f}"),
        ("CPU cores", state.sim_hw.cpu_cores, f"detected: {hw.cpu_cores}"),
    ]
    t = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    t.add_column("field", style="muted", width=14)
    t.add_column("value", style=theme.fg, width=14)
    t.add_column("note", style="muted")
    for i, (label, val, note) in enumerate(fields):
        cursor_pre = f"[b {theme.accent}]→[/] " if i == focus_field else "  "
        v = "" if val is None else str(val)
        val_cell = f"[b {theme.accent}]{v}|[/]" if i == focus_field else v or f"[{theme.muted}]—[/]"
        t.add_row(cursor_pre + label, val_cell, note)

    hint = (
        f"[{theme.muted}]Tab: cycle field  ·  digits/. to edit  ·  "
        f"Enter: apply  ·  Ctrl-R: reset  ·  Esc: cancel[/]"
    )

    panel = Panel(
        t,
        title=f"[b {theme.accent}]Simulate Hardware[/]",
        subtitle=hint,
        subtitle_align="left",
        title_align="left",
        border_style=theme.accent,
        padding=(1, 2),
    )
    return render_to_ansi(panel, width=min(width, 72), theme=theme)
