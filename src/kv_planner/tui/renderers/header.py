"""Top header ribbon — CPU / RAM / GPU / Runtimes."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.scoring import Snapshot
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


def render(snap: Snapshot, state: AppState, theme: Theme, width: int) -> str:
    hw = snap.detected

    gpu_line = (
        f"[b {theme.accent}]{hw.gpu_matched_db_key or hw.gpu_name_raw or 'no GPU'}[/]"
        f"  [b]{hw.gpu_vram_gb:.0f} GB[/]"
        f"  ·  [{theme.muted}]{hw.gpu_vendor}[/]"
    )
    if state.sim_hw.active:
        eff_vram = state.sim_hw.vram_gb or hw.gpu_vram_gb
        gpu_line += f"  [b {theme.warning}]SIM → {eff_vram:.0f} GB[/]"

    cpu_line = (
        f"[{theme.muted}]CPU[/] {hw.cpu_cores}C"
        f"  [{theme.muted}]RAM[/] {hw.ram_total_gb:.0f} GB"
        f"  [{theme.muted}]avail[/] {hw.ram_available_gb:.0f} GB"
    )

    rt_bits: list[str] = []
    for r in snap.runtimes:
        if r.reachable:
            rt_bits.append(f"[{theme.success}]{r.name}[/]:{len(r.models)}")
        else:
            rt_bits.append(f"[{theme.muted}]{r.name}[/]:off")
    rt_line = "  ".join(rt_bits)

    dash_line = ""
    if state.dashboard_url:
        dash_line = f"  [{theme.muted}]dashboard[/] [link]{state.dashboard_url}[/]"

    title = (
        f"[b {theme.accent}]kv-planner[/]  "
        f"[{theme.muted}]llmfit-style TUI  ·  physics-scored[/]"
    )

    body = Text.from_markup(
        f"{gpu_line}   {cpu_line}   {rt_line}{dash_line}"
    )

    panel = Panel(
        body,
        title=title,
        title_align="left",
        border_style=theme.border,
        padding=(0, 1),
    )
    return render_to_ansi(panel, width=width, theme=theme)
