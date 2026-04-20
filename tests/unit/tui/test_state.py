"""Reactive-state transitions — tested without a real terminal."""

from __future__ import annotations

import pytest

from kv_planner.tui.state import (
    AVAIL_CYCLE, FIT_CYCLE, SORT_CYCLE, THEME_CYCLE,
    AppState, SimulatedHardware,
)


def test_mode_entries_and_exit():
    s = AppState()
    assert s.mode == "normal"
    s.enter_mode("search")
    assert s.mode == "search"
    assert s.prev_mode == "normal"
    s.exit_modal()
    assert s.mode == "normal"


def test_cycle_fit_wraps():
    s = AppState()
    for _ in range(len(FIT_CYCLE) + 1):
        s.cycle_fit()
    assert s.fit_filter in FIT_CYCLE


def test_cycle_avail_wraps():
    s = AppState()
    for _ in range(len(AVAIL_CYCLE) + 1):
        s.cycle_avail()
    assert s.avail_filter in AVAIL_CYCLE


def test_cycle_sort_wraps():
    s = AppState()
    for _ in range(len(SORT_CYCLE) + 1):
        s.cycle_sort()
    assert s.sort_key in SORT_CYCLE


def test_cycle_theme_wraps():
    s = AppState()
    for _ in range(len(THEME_CYCLE) + 1):
        s.cycle_theme()
    assert s.theme in THEME_CYCLE


def test_cursor_move_clamps():
    s = AppState()
    s.move_cursor(-5, n_rows=10)
    assert s.cursor == 0
    s.move_cursor(100, n_rows=10)
    assert s.cursor == 9
    s.move_cursor(-1, n_rows=0)
    assert s.cursor == 0


def test_jump_to_end():
    s = AppState()
    s.jump(10**6, n_rows=50)
    assert s.cursor == 49


def test_mark_toggle():
    s = AppState()
    s.cursor = 3
    s.toggle_mark()
    assert 3 in s.marked_rows
    s.toggle_mark()
    assert 3 not in s.marked_rows


def test_sim_hw_active_flag():
    s = SimulatedHardware()
    assert s.active is False
    s.vram_gb = 24.0
    # frozen? No — it's a mutable dataclass; just re-check with new instance
    s2 = SimulatedHardware(vram_gb=24.0)
    assert s2.active is True
