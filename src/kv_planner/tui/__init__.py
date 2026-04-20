"""Terminal UI for kv-planner — llmfit-style, rich + prompt_toolkit.

The :func:`main` entry point is lazily loaded so that importing
``kv_planner.tui.state`` or ``kv_planner.tui.scoring`` does not require
having ``prompt_toolkit`` installed.
"""

from __future__ import annotations


def main() -> int:
    """Launch the TUI. Imports deferred to avoid hard dep on prompt_toolkit."""
    from kv_planner.tui.app import main as _main
    return _main()


__all__ = ["main"]
