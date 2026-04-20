"""Adapter layer: (catalog × detected-hw × state) → rendered rows.

Every TUI renderer pulls rows from :func:`build_rows` rather than touching
the physics engine directly. That keeps key handlers declarative — they
mutate :class:`state.AppState` and ask this module for the derived view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kv_planner.application.recommender import Recommendation, Recommender
from kv_planner.core.performance import RooflineAnalyzer, RooflineConfig
from kv_planner.domain import HardwareSpec
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.hw_detect import DetectedHardware, detect
from kv_planner.infrastructure.model_catalog import CATALOG, CatalogEntry, match_ollama_name
from kv_planner.infrastructure.runtime_probe import RuntimeProbe, probe_all
from kv_planner.tui.state import AppState, Mode


@dataclass(frozen=True)
class Row:
    """One row in the main model table, derived for the current state."""

    slug: str
    model_name: str
    provider: str
    params_b: float          # billions
    quality: int             # 0-100 use-case-aware
    score: float             # composite 0-100
    throughput: float        # tok/s
    memory_gb: float
    memory_pct: float
    context: int
    release_date: str        # YYYY-MM
    quant: str               # "int4" / "fp16"
    use_cases: tuple[str, ...]
    license: str
    ollama_tags: tuple[str, ...]
    installed: bool
    fits: bool
    fit_tag: str             # "perfect" / "good" / "marginal" / "too_tight" / "won't_fit"
    rec: Recommendation

    def __str__(self) -> str:  # handy in debug
        return f"{self.slug} {self.score:.0f}"


@dataclass(frozen=True)
class Snapshot:
    """Immutable view of hardware + runtimes at one redraw."""

    detected: DetectedHardware
    runtimes: list[RuntimeProbe]
    installed_slugs: frozenset[str]

    @property
    def ollama_models(self) -> list[str]:
        for r in self.runtimes:
            if r.name == "ollama" and r.reachable:
                return r.models
        return []


# ---------------------------------------------------------------------------
# Cached probes (refreshed on demand, not on every keystroke)
# ---------------------------------------------------------------------------

_cached_snapshot: Optional[Snapshot] = None


def snapshot(force_refresh: bool = False) -> Snapshot:
    """Return the current system snapshot. Cached until ``force_refresh=True``."""
    global _cached_snapshot
    if _cached_snapshot is not None and not force_refresh:
        return _cached_snapshot

    detected = detect()
    runtimes = probe_all()

    installed: set[str] = set()
    for r in runtimes:
        if not r.reachable:
            continue
        for name in r.models:
            entry = match_ollama_name(name)
            if entry:
                installed.add(entry.slug)

    _cached_snapshot = Snapshot(
        detected=detected,
        runtimes=runtimes,
        installed_slugs=frozenset(installed),
    )
    return _cached_snapshot


# ---------------------------------------------------------------------------
# Hardware reconciliation with Simulate overrides
# ---------------------------------------------------------------------------


def effective_hardware(snap: Snapshot, state: AppState) -> HardwareSpec:
    """Resolve the user's effective hardware after applying sim overrides."""
    gpu_key = snap.detected.gpu_matched_db_key
    if gpu_key is None:
        # No detected GPU; pick a sensible default so the table renders
        gpu_key = "RTX-5060-Laptop"
    spec = GPUDatabase.to_hardware_spec(gpu_key)

    # Apply VRAM override
    if state.sim_hw.vram_gb is not None:
        import dataclasses
        spec = dataclasses.replace(spec, gpu_memory_gb=float(state.sim_hw.vram_gb))
    return spec


# ---------------------------------------------------------------------------
# Fit categorisation (matches llmfit's bands)
# ---------------------------------------------------------------------------


def _fit_tag(pct: float, fits: bool) -> str:
    if not fits or pct > 100:
        return "too_tight"
    if pct <= 50:
        return "perfect"
    if pct <= 75:
        return "good"
    if pct <= 95:
        return "marginal"
    return "too_tight"


# ---------------------------------------------------------------------------
# Build rows
# ---------------------------------------------------------------------------


def _recommender_for(state: AppState) -> Recommender:
    return Recommender(
        roofline=RooflineAnalyzer(
            config=RooflineConfig(
                compute_efficiency=state.efficiency.compute_efficiency,
                memory_efficiency=state.efficiency.memory_efficiency,
            )
        )
    )


def build_all_rows(snap: Snapshot, state: AppState) -> list[Row]:
    """Return one :class:`Row` per catalog entry (pre-filter, pre-sort)."""
    hardware = effective_hardware(snap, state)
    rec = _recommender_for(state).recommend(
        hardware,
        use_case=state.use_case,
        input_length=state.input_length,
        output_length=state.output_length,
        batch_size=state.batch_size,
    )

    rows: list[Row] = []
    for r in rec:
        entry: CatalogEntry = r.entry
        rows.append(
            Row(
                slug=entry.slug,
                model_name=entry.config.name,
                provider=entry.provider,
                params_b=entry.config.total_params() / 1e9,
                quality=r.score_quality,
                score=r.score_composite,
                throughput=r.throughput_tok_s,
                memory_gb=r.memory_gb,
                memory_pct=r.memory_util_pct,
                context=entry.config.max_position_embeddings,
                release_date=entry.released or "—",
                quant=r.precision,
                use_cases=entry.use_cases,
                license=entry.license or "—",
                ollama_tags=entry.ollama_tags,
                installed=entry.slug in snap.installed_slugs,
                fits=r.fits,
                fit_tag=_fit_tag(r.memory_util_pct, r.fits),
                rec=r,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Filtering + sorting
# ---------------------------------------------------------------------------


def _match_search(row: Row, query: str) -> bool:
    if not query:
        return True
    q = query.lower()
    haystack = " ".join([
        row.slug, row.model_name, row.provider, row.quant, row.license,
        " ".join(row.use_cases),
    ]).lower()
    return q in haystack


def _match_fit(row: Row, fit_filter: str) -> bool:
    if fit_filter == "all":
        return True
    if fit_filter == "runnable":
        return row.fits
    return row.fit_tag == fit_filter


def _match_avail(row: Row, avail_filter: str) -> bool:
    if avail_filter == "all":
        return True
    if avail_filter == "installed":
        return row.installed
    return True


def _sort_key(row: Row, key: str) -> tuple:
    """Return a comparable tuple; earlier elements = primary sort."""
    if key == "score":
        return (-row.score, row.slug)
    if key == "params":
        return (-row.params_b, row.slug)
    if key == "mem_pct":
        return (row.memory_pct, row.slug)
    if key == "ctx":
        return (-row.context, row.slug)
    if key == "date":
        return (_date_sort(row.release_date), row.slug)
    if key == "use_case":
        return (row.use_cases[0] if row.use_cases else "", row.slug)
    return (-row.score, row.slug)


def _date_sort(s: str) -> tuple:
    # "2025-01" → (2025, 1); empty "—" sorts last
    try:
        y, m = s.split("-")
        return (-int(y), -int(m))
    except Exception:
        return (10**9, 0)


def apply_filters_and_sort(rows: list[Row], state: AppState) -> list[Row]:
    view = [
        r for r in rows
        if _match_search(r, state.search_query)
        and _match_fit(r, state.fit_filter)
        and _match_avail(r, state.avail_filter)
        and (state.provider_filter is None or r.provider == state.provider_filter)
    ]
    view.sort(key=lambda r: _sort_key(r, state.sort_key))
    return view
