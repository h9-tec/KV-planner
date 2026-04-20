"""Adw.Application entry point."""

from __future__ import annotations

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")
from gi.repository import Adw  # noqa: E402

from kv_planner.gui.window import KvpWindow


class KvpApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="com.heshamharoun.kvplanner")
        self.connect("activate", self._on_activate)

    def _on_activate(self, app: Adw.Application) -> None:
        # prefer dark mode for this aesthetic
        Adw.StyleManager.get_default().set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        win = KvpWindow(app)
        win.present()


def main() -> int:
    return KvpApplication().run([])
