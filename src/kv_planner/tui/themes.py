"""Colour palettes for the TUI. Subset of llmfit's themes (6 of 10)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    bg: str              # main background
    fg: str              # main foreground
    muted: str
    accent: str          # strong accent (headers, highlights)
    accent2: str         # secondary accent
    success: str         # "fits", "installed", OK status
    warning: str         # "marginal"
    error: str           # "won't fit"
    cursor_bg: str       # selected row background
    border: str
    table_header: str
    compute_bound: str
    memory_bound: str


DEFAULT = Theme(
    name="default",
    bg="#191b22", fg="#e5e5e9", muted="#908fa1",
    accent="#fa9450", accent2="#4abfad",
    success="#a2c761", warning="#e8c464", error="#eb6e88",
    cursor_bg="#2a2c36", border="#33363f", table_header="#fa9450",
    compute_bound="#4abfad", memory_bound="#eb6e88",
)

DRACULA = Theme(
    name="dracula",
    bg="#282a36", fg="#f8f8f2", muted="#6272a4",
    accent="#ff79c6", accent2="#8be9fd",
    success="#50fa7b", warning="#f1fa8c", error="#ff5555",
    cursor_bg="#44475a", border="#44475a", table_header="#bd93f9",
    compute_bound="#8be9fd", memory_bound="#ff79c6",
)

NORD = Theme(
    name="nord",
    bg="#2e3440", fg="#eceff4", muted="#7b88a1",
    accent="#88c0d0", accent2="#81a1c1",
    success="#a3be8c", warning="#ebcb8b", error="#bf616a",
    cursor_bg="#3b4252", border="#434c5e", table_header="#88c0d0",
    compute_bound="#88c0d0", memory_bound="#bf616a",
)

SOLARIZED = Theme(
    name="solarized",
    bg="#002b36", fg="#eee8d5", muted="#586e75",
    accent="#cb4b16", accent2="#2aa198",
    success="#859900", warning="#b58900", error="#dc322f",
    cursor_bg="#073642", border="#073642", table_header="#cb4b16",
    compute_bound="#2aa198", memory_bound="#dc322f",
)

MONOKAI = Theme(
    name="monokai",
    bg="#272822", fg="#f8f8f2", muted="#75715e",
    accent="#fd971f", accent2="#66d9ef",
    success="#a6e22e", warning="#e6db74", error="#f92672",
    cursor_bg="#3e3d32", border="#49483e", table_header="#fd971f",
    compute_bound="#66d9ef", memory_bound="#f92672",
)

GRUVBOX = Theme(
    name="gruvbox",
    bg="#282828", fg="#ebdbb2", muted="#928374",
    accent="#fe8019", accent2="#8ec07c",
    success="#b8bb26", warning="#fabd2f", error="#fb4934",
    cursor_bg="#3c3836", border="#504945", table_header="#fe8019",
    compute_bound="#8ec07c", memory_bound="#fb4934",
)


_THEMES: dict[str, Theme] = {
    t.name: t for t in (DEFAULT, DRACULA, NORD, SOLARIZED, MONOKAI, GRUVBOX)
}


def get_theme(name: str) -> Theme:
    return _THEMES.get(name, DEFAULT)
