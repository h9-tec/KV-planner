"""Renderer smoke tests — each renderer produces non-empty ANSI output."""

from __future__ import annotations

import pytest

from kv_planner.tui.renderers import (
    advanced as advanced_r,
    compare as compare_r,
    detail as detail_r,
    header as header_r,
    help as help_r,
    plan as plan_r,
    simulate as simulate_r,
    status as status_r,
    table as table_r,
)
from kv_planner.tui.scoring import apply_filters_and_sort, build_all_rows, snapshot
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import get_theme


@pytest.fixture(scope="module")
def ctx():
    state = AppState()
    snap = snapshot()
    rows_all = build_all_rows(snap, state)
    rows = apply_filters_and_sort(rows_all, state)
    theme = get_theme("default")
    return state, snap, rows, theme


def test_header_renders(ctx):
    state, snap, _, theme = ctx
    s = header_r.render(snap, state, theme, width=100)
    assert len(s) > 50
    assert "kv-planner" in s


def test_table_renders_with_rows(ctx):
    state, _, rows, theme = ctx
    s = table_r.render(rows, state, theme, width=100, height=10)
    assert "Model" in s or "\x1b" in s   # ANSI escapes present


def test_table_handles_empty(ctx):
    state, _, _, theme = ctx
    s = table_r.render([], state, theme, width=80, height=8)
    assert isinstance(s, str)


def test_status_renders_mode_label(ctx):
    state, _, rows, theme = ctx
    s = status_r.render(state, theme, n_rows=len(rows), total_rows=len(rows), width=100)
    assert "NORMAL" in s


def test_help_renders(ctx):
    _, _, _, theme = ctx
    s = help_r.render(theme, width=100)
    assert "Navigation" in s or "j" in s


def test_detail_renders_for_first_row(ctx):
    state, snap, rows, theme = ctx
    if not rows:
        pytest.skip("no rows")
    s = detail_r.render(rows[0], snap, state, theme, width=100)
    assert rows[0].slug in s or rows[0].provider in s


def test_compare_renders(ctx):
    _, _, rows, theme = ctx
    s = compare_r.render(rows[:3], theme, width=120)
    assert "compare" in s.lower()


def test_plan_renders(ctx):
    state, _, rows, theme = ctx
    if not rows:
        pytest.skip("no rows")
    s = plan_r.render(rows[0], state, theme, width=120)
    assert "Plan" in s or "VRAM" in s


def test_simulate_renders(ctx):
    state, snap, _, theme = ctx
    s = simulate_r.render(snap, state, theme, width=100)
    assert "Simulate" in s


def test_advanced_renders(ctx):
    state, _, _, theme = ctx
    s = advanced_r.render(state, theme, width=100)
    assert "Advanced" in s or "MBU" in s
