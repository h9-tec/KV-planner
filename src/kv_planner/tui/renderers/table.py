"""Main model table — the llmfit "fit" view.

Columns: Ins | Model | Provider | Params | Score | tok/s | Mem% | Ctx | Date | Quant | Fit | Use Case
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from kv_planner.tui.renderers._util import fmt_context, fmt_params, fmt_tok_s, render_to_ansi
from kv_planner.tui.scoring import Row
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


_FIT_COLOUR = {
    "perfect": "success",
    "good": "accent2",
    "marginal": "warning",
    "too_tight": "error",
}


def _fit_cell(fit_tag: str, theme: Theme) -> str:
    colour = getattr(theme, _FIT_COLOUR.get(fit_tag, "muted"))
    label = {
        "perfect": "Perfect",
        "good": "Good",
        "marginal": "Marginal",
        "too_tight": "Too tight",
    }.get(fit_tag, "—")
    return f"[{colour}]{label}[/]"


def render(
    rows: list[Row],
    state: AppState,
    theme: Theme,
    width: int,
    height: int,
) -> str:
    table = Table(
        expand=True,
        box=None,
        header_style=f"b {theme.table_header}",
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
    )

    # Columns sized so llmfit-level 100-column view looks right; rich auto-expands.
    table.add_column("Ins", width=3, justify="center", no_wrap=True)
    table.add_column("Model", min_width=20, max_width=32, overflow="ellipsis")
    table.add_column("Prov", width=9, no_wrap=True)
    table.add_column("Params", width=7, justify="right", no_wrap=True)
    table.add_column("Score", width=5, justify="right", no_wrap=True)
    table.add_column("tok/s", width=5, justify="right", no_wrap=True)
    table.add_column("Mem%", width=5, justify="right", no_wrap=True)
    table.add_column("Ctx", width=5, justify="right", no_wrap=True)
    table.add_column("Date", width=7, no_wrap=True)
    table.add_column("Quant", width=6, no_wrap=True)
    table.add_column("Fit", width=9, no_wrap=True)
    table.add_column("Use Case", min_width=12, overflow="fold")

    # Window the rows so only a screenful renders
    n = len(rows)
    if n == 0:
        table.add_row(*(["—"] * 12))
        return render_to_ansi(table, width=width, height=height)

    # Crude viewport around cursor
    visible = max(1, height - 2)
    if state.cursor < state.visible_window_top:
        state.visible_window_top = state.cursor
    if state.cursor >= state.visible_window_top + visible:
        state.visible_window_top = state.cursor - visible + 1
    start = max(0, min(state.visible_window_top, max(0, n - visible)))
    end = min(n, start + visible)

    for i in range(start, end):
        r = rows[i]
        is_cursor = i == state.cursor
        is_marked = i in state.marked_rows

        inst = "[success]✓[/]" if r.installed else ""
        if is_marked:
            inst = f"[{theme.accent}]■[/]"

        score_col = theme.success if r.score >= 75 else (
            theme.accent if r.score >= 60 else theme.muted
        )

        tps = fmt_tok_s(r.throughput) if r.fits else f"[{theme.error}]  —[/]"
        mem = f"[{theme.error}]{r.memory_pct:>4.0f}%[/]" if not r.fits else (
            f"{r.memory_pct:>4.0f}%"
        )
        uc = ",".join(r.use_cases[:2]) if r.use_cases else "—"

        row_cells = [
            Text.from_markup(inst),
            Text.from_markup(f"[b]{r.slug}[/]" if is_cursor else r.slug),
            Text.from_markup(f"[{theme.muted}]{r.provider[:9]}[/]"),
            Text.from_markup(fmt_params(r.params_b)),
            Text.from_markup(f"[{score_col}]{r.score:>4.0f}[/]"),
            Text.from_markup(tps),
            Text.from_markup(mem),
            Text.from_markup(fmt_context(r.context)),
            Text.from_markup(f"[{theme.muted}]{r.release_date}[/]"),
            Text.from_markup(r.quant),
            Text.from_markup(_fit_cell(r.fit_tag, theme)),
            Text.from_markup(f"[{theme.muted}]{uc}[/]"),
        ]

        if is_cursor:
            table.add_row(
                *row_cells,
                style=f"on {theme.cursor_bg}",
            )
        else:
            table.add_row(*row_cells)

    return render_to_ansi(table, width=width, height=height, theme=theme)
