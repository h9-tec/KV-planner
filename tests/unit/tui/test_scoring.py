"""TUI scoring adapter — filter / sort predicates tested end-to-end."""

from __future__ import annotations

import pytest

from kv_planner.tui.scoring import (
    apply_filters_and_sort, build_all_rows, snapshot,
)
from kv_planner.tui.state import AppState


@pytest.fixture(scope="module")
def rows():
    return build_all_rows(snapshot(), AppState())


def test_build_rows_nonempty(rows):
    assert len(rows) >= 10, "catalog shouldn't shrink below 10"


def test_search_matches_substring(rows):
    state = AppState()
    state.search_query = "coder"
    view = apply_filters_and_sort(rows, state)
    # There's at least one coder model in the catalog
    assert all("coder" in r.slug.lower() or "coder" in r.provider.lower()
               or "coder" in r.license.lower()
               for r in view), \
           f"search didn't filter: got {[r.slug for r in view]}"


def test_fit_runnable_filters_only_fits(rows):
    state = AppState()
    state.fit_filter = "runnable"
    view = apply_filters_and_sort(rows, state)
    assert all(r.fits for r in view)


def test_sort_by_params_descending(rows):
    state = AppState()
    state.sort_key = "params"
    view = apply_filters_and_sort(rows, state)
    for a, b in zip(view, view[1:]):
        assert a.params_b >= b.params_b


def test_sort_by_mem_pct_ascending(rows):
    state = AppState()
    state.sort_key = "mem_pct"
    view = apply_filters_and_sort(rows, state)
    for a, b in zip(view, view[1:]):
        assert a.memory_pct <= b.memory_pct


def test_default_sort_is_composite_desc(rows):
    state = AppState()
    view = apply_filters_and_sort(rows, state)
    for a, b in zip(view, view[1:]):
        assert a.score >= b.score
