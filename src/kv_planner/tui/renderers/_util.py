"""Shared renderer helpers — rich → ANSI string."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.theme import Theme as RichTheme

from kv_planner.tui.themes import Theme


def build_rich_theme(theme: Theme) -> RichTheme:
    """Turn a kv-planner palette into a rich.Theme so [success] etc. work."""
    return RichTheme({
        "success": theme.success,
        "warning": theme.warning,
        "error": theme.error,
        "muted": theme.muted,
        "accent": theme.accent,
        "accent2": theme.accent2,
        "compute_bound": theme.compute_bound,
        "memory_bound": theme.memory_bound,
        "header": f"bold {theme.table_header}",
    })


def render_to_ansi(
    renderable, width: int, height: int | None = None, theme: Theme | None = None
) -> str:
    """Render a rich renderable to an ANSI string suitable for prompt_toolkit."""
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system="truecolor",
        width=max(20, width),
        height=height,
        record=False,
        theme=build_rich_theme(theme) if theme else None,
    )
    console.print(renderable, end="")
    return buf.getvalue()


def fmt_params(params_b: float) -> str:
    if params_b < 1:
        return f"{params_b * 1000:4.0f} M"
    return f"{params_b:5.1f} B"


def fmt_context(tokens: int) -> str:
    if tokens >= 1_000_000:
        return f"{tokens // 1000}k"
    if tokens >= 1_000:
        return f"{tokens // 1000}k"
    return str(tokens)


def fmt_tok_s(v: float) -> str:
    if v <= 0:
        return "  —"
    if v >= 10_000:
        return f"{v/1000:4.1f}k"
    return f"{v:4.0f}"
