"""Multi-model compare — attribute × model matrix."""

from __future__ import annotations

from rich.panel import Panel
from rich.table import Table

from kv_planner.tui.renderers._util import fmt_context, fmt_params, fmt_tok_s, render_to_ansi
from kv_planner.tui.scoring import Row
from kv_planner.tui.themes import Theme


def render(rows: list[Row], theme: Theme, width: int) -> str:
    if not rows:
        panel = Panel(
            "[muted]No models selected. Press `m` to mark, `c` again to close.[/]",
            title=f"[b {theme.accent}]compare[/]",
            border_style=theme.border,
        )
        return render_to_ansi(panel, width=width, theme=theme)

    t = Table(expand=True, show_lines=True, box=None)
    t.add_column("Attribute", style="muted", width=16, no_wrap=True)
    for r in rows:
        t.add_column(r.slug, justify="right", style=theme.fg)

    def best(vals: list[float], higher_is_better: bool = True) -> int:
        if higher_is_better:
            best_v = max(vals)
        else:
            best_v = min(vals)
        return vals.index(best_v)

    def fmt_row(label: str, values: list[str], win_idx: int) -> None:
        styled = [
            f"[b {theme.success}]{v}[/]" if i == win_idx else v
            for i, v in enumerate(values)
        ]
        t.add_row(label, *styled)

    scores = [r.score for r in rows]
    fmt_row(
        "Composite", [f"{s:.1f}" for s in scores], best(scores),
    )
    fmt_row(
        "Quality", [str(r.quality) for r in rows],
        best([r.quality for r in rows]),
    )
    fmt_row(
        "Throughput", [fmt_tok_s(r.throughput) for r in rows],
        best([r.throughput for r in rows]),
    )
    fmt_row(
        "Memory %", [f"{r.memory_pct:.0f}%" for r in rows],
        best([r.memory_pct for r in rows], higher_is_better=False),
    )
    fmt_row(
        "Params", [fmt_params(r.params_b) for r in rows],
        best([r.params_b for r in rows], higher_is_better=False),
    )
    fmt_row(
        "Context", [fmt_context(r.context) for r in rows],
        best([r.context for r in rows]),
    )
    t.add_row("Quant", *[r.quant for r in rows])
    t.add_row("Provider", *[r.provider for r in rows])
    t.add_row("Released", *[r.release_date for r in rows])
    t.add_row("License", *[r.license for r in rows])
    t.add_row("Fit", *[r.fit_tag for r in rows])

    panel = Panel(
        t,
        title=f"[b {theme.accent}]compare · {len(rows)} models  (Esc to close)[/]",
        title_align="left",
        border_style=theme.accent,
    )
    return render_to_ansi(panel, width=width, theme=theme)
