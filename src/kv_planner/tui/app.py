"""Full-screen prompt_toolkit Application wiring the TUI together."""

from __future__ import annotations

import os
import shutil
import sys
from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.layout import FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style

from kv_planner.tui.bindings import build_bindings
from kv_planner.tui.renderers import compare as compare_r
from kv_planner.tui.renderers import detail as detail_r
from kv_planner.tui.renderers import header as header_r
from kv_planner.tui.renderers import help as help_r
from kv_planner.tui.renderers import plan as plan_r
from kv_planner.tui.renderers import status as status_r
from kv_planner.tui.renderers import table as table_r
from kv_planner.tui.renderers import simulate as simulate_r
from kv_planner.tui.renderers import advanced as advanced_r
from kv_planner.tui.scoring import Row, apply_filters_and_sort, build_all_rows, snapshot
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import get_theme


def _term_width() -> int:
    try:
        return shutil.get_terminal_size((100, 30)).columns
    except OSError:
        return 100


def _term_height() -> int:
    try:
        return shutil.get_terminal_size((100, 30)).lines
    except OSError:
        return 30


class TUI:
    def __init__(self, dashboard_url: str | None = None) -> None:
        self.state = AppState()
        self.state.dashboard_url = dashboard_url
        self._snap = snapshot()
        self._all_rows = build_all_rows(self._snap, self.state)
        self._view: list[Row] = apply_filters_and_sort(self._all_rows, self.state)
        self._app: Application | None = None

    # ------------------------------------------------------------------
    # Data refresh
    # ------------------------------------------------------------------
    def _recompute(self) -> None:
        """Rebuild view after state changed."""
        # Only re-run the recommender if a state field that affects scoring changed.
        # For now, always rebuild — the Recommender is cheap (microseconds per model).
        self._snap = snapshot()
        self._all_rows = build_all_rows(self._snap, self.state)
        self._view = apply_filters_and_sort(self._all_rows, self.state)
        # Clamp cursor
        if self._view:
            self.state.cursor = max(0, min(self.state.cursor, len(self._view) - 1))
        else:
            self.state.cursor = 0

    def _view_rows(self) -> list[Row]:
        return self._view

    # ------------------------------------------------------------------
    # Renderers → ANSI strings consumed by FormattedTextControl
    # ------------------------------------------------------------------
    def _render_header(self) -> ANSI:
        self._recompute()
        theme = get_theme(self.state.theme)
        return ANSI(header_r.render(self._snap, self.state, theme, width=_term_width()))

    def _render_main(self) -> ANSI:
        theme = get_theme(self.state.theme)
        width = _term_width()
        height = max(10, _term_height() - 8)
        # Modal rendering takes over the main area.
        if self.state.mode == "help":
            return ANSI(help_r.render(theme, width=width))
        if self.state.mode == "compare":
            marked_rows = [self._view[i] for i in self.state.marked_rows
                           if 0 <= i < len(self._view)]
            return ANSI(compare_r.render(marked_rows, theme, width=width))
        if self.state.mode == "plan" and self._view:
            return ANSI(plan_r.render(
                self._view[min(self.state.cursor, len(self._view) - 1)],
                self.state, theme, width=width,
            ))
        if self.state.mode == "simulate":
            return ANSI(simulate_r.render(self._snap, self.state, theme, width=width))
        if self.state.mode == "advanced":
            return ANSI(advanced_r.render(self.state, theme, width=width))

        # Default: table (+ optional detail pane below)
        if self.state.show_detail and self._view:
            tbl_h = max(5, height - 14)
            ansi_table = table_r.render(self._view, self.state, theme, width=width, height=tbl_h)
            row = self._view[min(self.state.cursor, len(self._view) - 1)]
            ansi_detail = detail_r.render(row, self._snap, self.state, theme, width=width)
            return ANSI(ansi_table + "\n" + ansi_detail)
        return ANSI(table_r.render(self._view, self.state, theme, width=width, height=height))

    def _render_status(self) -> ANSI:
        theme = get_theme(self.state.theme)
        return ANSI(status_r.render(
            self.state, theme,
            n_rows=len(self._view), total_rows=len(self._all_rows),
            width=_term_width(),
        ))

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> Layout:
        header_window = Window(
            content=FormattedTextControl(self._render_header, focusable=False),
            height=Dimension(preferred=5, max=5),
            wrap_lines=False,
        )
        main_window = Window(
            content=FormattedTextControl(self._render_main, focusable=False),
            wrap_lines=False,
        )
        status_window = Window(
            content=FormattedTextControl(self._render_status, focusable=False),
            height=Dimension(preferred=1, max=1),
            wrap_lines=False,
        )

        root = HSplit([header_window, main_window, status_window])
        return Layout(FloatContainer(content=root, floats=[]))

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self) -> None:
        style = Style.from_dict({})
        layout = self._build_layout()
        invalidate: Callable[[], None] = lambda: self._app.invalidate() if self._app else None

        def exit_app() -> None:
            if self._app:
                self._app.exit()

        bindings = build_bindings(
            self.state,
            get_rows=self._view_rows,
            invalidate=invalidate,
            exit_app=exit_app,
        )

        self._app = Application(
            layout=layout,
            key_bindings=bindings,
            full_screen=True,
            mouse_support=True,
            style=style,
            refresh_interval=0.0,
        )
        self._app.run()


def main(dashboard_url: str | None = None) -> int:
    try:
        TUI(dashboard_url=dashboard_url).run()
    except KeyboardInterrupt:
        pass
    return 0
