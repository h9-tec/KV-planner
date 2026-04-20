"""Pure-Python reactive state for the TUI.

No terminal code here — just data + transitions. That makes it easy to
unit-test every key binding without spawning a pseudo-tty.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional

from kv_planner.domain import PrecisionType

Mode = Literal[
    "normal",
    "search",
    "visual",
    "select",
    "plan",
    "compare",
    "simulate",
    "advanced",
    "help",
    "theme",
    "download",
]

FitFilter = Literal["all", "runnable", "perfect", "good", "marginal"]
AvailFilter = Literal["all", "installed"]
SortKey = Literal["score", "params", "mem_pct", "ctx", "date", "use_case"]

FIT_CYCLE: tuple[FitFilter, ...] = ("all", "runnable", "perfect", "good", "marginal")
AVAIL_CYCLE: tuple[AvailFilter, ...] = ("all", "installed")
SORT_CYCLE: tuple[SortKey, ...] = ("score", "params", "mem_pct", "ctx", "date", "use_case")

UseCase = Literal["general", "coding", "reasoning", "chat", "multimodal", "agent", "embedding"]
Theme = Literal["default", "dracula", "nord", "solarized", "monokai", "gruvbox"]
THEME_CYCLE: tuple[Theme, ...] = (
    "default", "dracula", "nord", "solarized", "monokai", "gruvbox"
)


@dataclass
class SimulatedHardware:
    """Override fields for the Simulate modal. None = use detected value."""

    vram_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    cpu_cores: Optional[int] = None

    @property
    def active(self) -> bool:
        return any(v is not None for v in (self.vram_gb, self.ram_gb, self.cpu_cores))


@dataclass
class EfficiencyKnobs:
    """Tunable factors for the Advanced modal."""

    memory_efficiency: float = 0.75  # MBU ceiling
    compute_efficiency: float = 0.50  # MFU ceiling
    run_mode_gpu: float = 1.0
    run_mode_cpu_offload: float = 0.5
    run_mode_moe: float = 0.8


@dataclass
class AppState:
    """Everything the TUI cares about. Mutated by key handlers."""

    # --- Mode + cursor ----------------------------------------------------
    mode: Mode = "normal"
    prev_mode: Mode = "normal"  # for Esc-to-restore
    cursor: int = 0  # row index in the currently filtered+sorted view
    visible_window_top: int = 0  # for paging
    marked_rows: set[int] = field(default_factory=set)  # slug indices

    # --- Filters + sort ---------------------------------------------------
    search_query: str = ""
    fit_filter: FitFilter = "all"
    avail_filter: AvailFilter = "all"
    sort_key: SortKey = "score"
    use_case: UseCase = "general"
    provider_filter: Optional[str] = None  # None = all
    selected_column: int = 0  # for "select" mode

    # --- Workload parameters (used by Recommender) -----------------------
    input_length: int = 2048
    output_length: int = 512
    batch_size: int = 1

    # --- Simulated hardware + efficiency knobs ---------------------------
    sim_hw: SimulatedHardware = field(default_factory=SimulatedHardware)
    efficiency: EfficiencyKnobs = field(default_factory=EfficiencyKnobs)

    # --- Theme ------------------------------------------------------------
    theme: Theme = "default"

    # --- UI toggles -------------------------------------------------------
    show_detail: bool = False
    quit_requested: bool = False
    status_message: str = ""
    dashboard_url: Optional[str] = None

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def enter_mode(self, mode: Mode) -> None:
        if mode != self.mode:
            self.prev_mode = self.mode
            self.mode = mode

    def exit_modal(self) -> None:
        """Close any overlay; return to Normal mode."""
        if self.mode in ("normal",):
            return
        self.mode = "normal"

    def cycle_fit(self) -> None:
        idx = FIT_CYCLE.index(self.fit_filter)
        self.fit_filter = FIT_CYCLE[(idx + 1) % len(FIT_CYCLE)]

    def cycle_avail(self) -> None:
        idx = AVAIL_CYCLE.index(self.avail_filter)
        self.avail_filter = AVAIL_CYCLE[(idx + 1) % len(AVAIL_CYCLE)]

    def cycle_sort(self) -> None:
        idx = SORT_CYCLE.index(self.sort_key)
        self.sort_key = SORT_CYCLE[(idx + 1) % len(SORT_CYCLE)]

    def cycle_theme(self) -> None:
        idx = THEME_CYCLE.index(self.theme)
        self.theme = THEME_CYCLE[(idx + 1) % len(THEME_CYCLE)]

    def move_cursor(self, delta: int, n_rows: int) -> None:
        if n_rows <= 0:
            self.cursor = 0
            return
        self.cursor = max(0, min(n_rows - 1, self.cursor + delta))

    def jump(self, absolute: int, n_rows: int) -> None:
        if n_rows <= 0:
            self.cursor = 0
            return
        self.cursor = max(0, min(n_rows - 1, absolute))

    def toggle_mark(self) -> None:
        """Mark/unmark current row for compare."""
        if self.cursor in self.marked_rows:
            self.marked_rows.remove(self.cursor)
        else:
            self.marked_rows.add(self.cursor)

    def clear_marks(self) -> None:
        self.marked_rows.clear()

    def reset_sim(self) -> None:
        self.sim_hw = SimulatedHardware()

    def clone(self) -> "AppState":
        return dataclasses.replace(
            self,
            marked_rows=set(self.marked_rows),
            sim_hw=dataclasses.replace(self.sim_hw),
            efficiency=dataclasses.replace(self.efficiency),
        )
