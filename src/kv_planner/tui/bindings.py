"""Key bindings for the TUI — mode-aware dispatch via prompt_toolkit."""

from __future__ import annotations

import shlex
import subprocess
import sys
from typing import Callable

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from kv_planner.tui.scoring import (
    Row,
    apply_filters_and_sort,
    build_all_rows,
    snapshot,
)
from kv_planner.tui.state import AppState


def build_bindings(
    state: AppState,
    get_rows: Callable[[], list[Row]],
    invalidate: Callable[[], None],
    exit_app: Callable[[], None],
) -> KeyBindings:
    """Return the full prompt_toolkit :class:`KeyBindings`.

    ``get_rows()`` returns the currently filtered+sorted view (the app
    recomputes it on every redraw). ``invalidate()`` is the Application's
    ``invalidate`` method; call after state mutation to trigger a redraw.
    """
    kb = KeyBindings()

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def normal() -> bool:
        return state.mode == "normal"

    def in_modal() -> bool:
        return state.mode in ("help", "compare", "plan", "simulate", "advanced", "theme", "download")

    def search_mode() -> bool:
        return state.mode == "search"

    def editing_mode() -> bool:
        """Modal modes that consume text input (search/simulate/advanced)."""
        return state.mode in ("search", "simulate", "advanced")

    # ----------------------------------------------------------------
    # Navigation — active in Normal + Visual + Compare (for scroll)
    # ----------------------------------------------------------------
    @kb.add("j", filter=_c(lambda: state.mode in ("normal", "visual")))
    @kb.add(Keys.Down, filter=_c(lambda: state.mode in ("normal", "visual")))
    def _(event):
        state.move_cursor(1, len(get_rows()))
        invalidate()

    @kb.add("k", filter=_c(lambda: state.mode in ("normal", "visual")))
    @kb.add(Keys.Up, filter=_c(lambda: state.mode in ("normal", "visual")))
    def _(event):
        state.move_cursor(-1, len(get_rows()))
        invalidate()

    @kb.add("g", filter=_c(normal))
    def _(event):
        state.jump(0, len(get_rows()))
        invalidate()

    @kb.add("G", filter=_c(normal))
    def _(event):
        state.jump(10**6, len(get_rows()))
        invalidate()

    @kb.add(Keys.PageDown, filter=_c(lambda: state.mode in ("normal", "visual")))
    def _(event):
        state.move_cursor(10, len(get_rows()))
        invalidate()

    @kb.add(Keys.PageUp, filter=_c(lambda: state.mode in ("normal", "visual")))
    def _(event):
        state.move_cursor(-10, len(get_rows()))
        invalidate()

    # ----------------------------------------------------------------
    # Detail toggle
    # ----------------------------------------------------------------
    @kb.add(Keys.Enter, filter=_c(normal))
    def _(event):
        state.show_detail = not state.show_detail
        invalidate()

    # ----------------------------------------------------------------
    # Filter / sort cycles
    # ----------------------------------------------------------------
    @kb.add("f", filter=_c(normal))
    def _(event):
        state.cycle_fit(); invalidate()

    @kb.add("a", filter=_c(normal))
    def _(event):
        state.cycle_avail(); invalidate()

    @kb.add("s", filter=_c(normal))
    def _(event):
        state.cycle_sort(); invalidate()

    @kb.add("t", filter=_c(normal))
    def _(event):
        state.cycle_theme(); invalidate()

    # ----------------------------------------------------------------
    # Mode entries
    # ----------------------------------------------------------------
    @kb.add("/", filter=_c(normal))
    def _(event):
        state.enter_mode("search"); invalidate()

    @kb.add("v", filter=_c(normal))
    def _(event):
        state.enter_mode("visual"); invalidate()

    @kb.add("p", filter=_c(normal))
    def _(event):
        state.enter_mode("plan"); invalidate()

    @kb.add("c", filter=_c(normal))
    def _(event):
        state.enter_mode("compare"); invalidate()

    @kb.add("S", filter=_c(normal))
    def _(event):
        state.enter_mode("simulate"); invalidate()

    @kb.add("A", filter=_c(normal))
    def _(event):
        state.enter_mode("advanced"); invalidate()

    @kb.add("?", filter=_c(lambda: not search_mode()))
    @kb.add("h", filter=_c(lambda: state.mode == "normal"))
    def _(event):
        state.enter_mode("help"); invalidate()

    @kb.add("m", filter=_c(lambda: state.mode in ("normal", "visual")))
    def _(event):
        state.toggle_mark(); invalidate()

    @kb.add("x", filter=_c(normal))
    def _(event):
        state.clear_marks(); invalidate()

    # ----------------------------------------------------------------
    # Download (d) — shells out to `ollama pull`
    # ----------------------------------------------------------------
    @kb.add("d", filter=_c(normal))
    def _(event):
        rows = get_rows()
        if not rows:
            return
        row = rows[min(state.cursor, len(rows) - 1)]
        if not row.ollama_tags:
            state.status_message = f"no ollama tag known for {row.slug}"
            invalidate()
            return
        tag = row.ollama_tags[0]
        try:
            subprocess.Popen(
                ["ollama", "pull", tag],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            state.status_message = f"ollama pull {tag} (backgrounded)"
        except FileNotFoundError:
            state.status_message = "ollama binary not on PATH"
        invalidate()

    # refresh runtime probe
    @kb.add("r", filter=_c(normal))
    def _(event):
        snapshot(force_refresh=True)
        state.status_message = "runtimes refreshed"
        invalidate()

    # ----------------------------------------------------------------
    # Search mode — text input via key handlers
    # ----------------------------------------------------------------
    @kb.add(Keys.Any, filter=_c(search_mode))
    def _(event):
        ch = event.data
        if ch and ch.isprintable() and len(ch) == 1:
            state.search_query += ch
            invalidate()

    @kb.add(Keys.Backspace, filter=_c(search_mode))
    def _(event):
        state.search_query = state.search_query[:-1]
        invalidate()

    @kb.add("c-u", filter=_c(search_mode))
    def _(event):
        state.search_query = ""
        invalidate()

    @kb.add(Keys.Enter, filter=_c(search_mode))
    def _(event):
        state.exit_modal(); invalidate()

    @kb.add(Keys.Escape, filter=_c(lambda: state.mode != "normal"))
    def _(event):
        # Esc out of any modal / edit mode
        if state.mode == "search":
            # clear search too on Esc? llmfit clears; we keep query so users can iterate.
            pass
        state.exit_modal()
        invalidate()

    # ----------------------------------------------------------------
    # Quit
    # ----------------------------------------------------------------
    @kb.add("q", filter=_c(lambda: state.mode == "normal"))
    @kb.add("c-c")
    def _(event):
        state.quit_requested = True
        exit_app()

    return kb


def _c(predicate):
    """Wrap a bool-returning callable as a prompt_toolkit Filter."""
    from prompt_toolkit.filters import Condition
    return Condition(predicate)
