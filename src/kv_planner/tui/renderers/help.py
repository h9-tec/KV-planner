"""Help modal — key reference (same layout as llmfit)."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.themes import Theme


_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Navigation", [
        ("j / k / ↓ ↑", "move cursor"),
        ("g / G",       "jump to top / bottom"),
        ("PgUp / PgDn", "page scroll"),
        ("Enter",        "toggle detail pane"),
    ]),
    ("Filters", [
        ("/ then type", "search (partial match across name, provider, use case)"),
        ("f",            "cycle fit filter (All → Runnable → Perfect → Good → Marginal)"),
        ("a",            "cycle availability (All → Installed)"),
        ("s",            "cycle sort (Score → Params → Mem% → Ctx → Date → Use Case)"),
        ("U",            "use-case popup"),
        ("P",            "provider popup"),
        ("i",            "toggle installed-first sorting"),
    ]),
    ("Modes", [
        ("v",  "Visual mode — range select for compare"),
        ("V",  "Select mode — column-based filter/sort"),
        ("p",  "Plan mode — inverse hardware sizing for current model"),
        ("c",  "Compare marked/selected models"),
        ("m",  "mark / unmark current row"),
        ("S",  "Simulate hardware (override VRAM / RAM / CPU)"),
        ("A",  "Advanced — tune efficiency knobs"),
    ]),
    ("Actions", [
        ("d",       "Download (ollama pull) for selected model"),
        ("t",       "cycle colour theme"),
        ("r",       "refresh installed-runtime probes"),
        ("?  /  h", "this help popup"),
        ("Esc",     "close popup / exit current mode"),
        ("q",       "quit"),
    ]),
]


def render(theme: Theme, width: int) -> str:
    bigtable = Table.grid(padding=(0, 2))
    bigtable.add_column()
    bigtable.add_column()

    columns: list[Table] = []
    for title, rows in _GROUPS:
        t = Table(box=None, show_header=False, padding=(0, 1), expand=False)
        t.add_column("k", style="accent", width=15, no_wrap=True)
        t.add_column("d", style=theme.fg, overflow="fold")
        t.add_row(f"[b {theme.accent2}]{title}[/]", "")
        for k, d in rows:
            t.add_row(k, d)
        columns.append(t)

    # 2-column layout
    from rich.columns import Columns
    grid = Columns(columns, expand=True, padding=(0, 4))

    panel = Panel(
        grid,
        title=f"[b {theme.accent}]kv-planner TUI · keybindings[/]",
        title_align="left",
        border_style=theme.accent,
        padding=(1, 2),
    )
    return render_to_ansi(panel, width=width, theme=theme)
