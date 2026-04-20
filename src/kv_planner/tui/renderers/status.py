"""Bottom status bar — mode + filters summary + key hints."""

from __future__ import annotations

from rich.text import Text

from kv_planner.tui.renderers._util import render_to_ansi
from kv_planner.tui.state import AppState
from kv_planner.tui.themes import Theme


_MODE_HINTS: dict[str, str] = {
    "normal": "/:search  f:fit  a:avail  s:sort  v:visual  V:select  p:plan  c:compare  "
              "U:use-case  P:provider  S:sim  A:advanced  d:install  t:theme  ?:help  q:quit",
    "search": "type to filter  Enter/Esc:exit",
    "visual": "j/k:extend  c:compare  m:mark  Esc/v:exit",
    "select": "h/l:column  Enter:apply  Esc:exit",
    "plan": "Tab:field  edit numbers  Esc:exit",
    "compare": "Esc:close  arrows:scroll",
    "simulate": "Tab:field  edit numbers  Enter:apply  Ctrl-R:reset  Esc:cancel",
    "advanced": "Tab:field  edit  Enter:apply  Ctrl-R:reset  Esc:cancel",
    "help": "any key closes",
    "download": "y:confirm  n:cancel",
    "theme": "t:next  Esc:close",
}


def render(state: AppState, theme: Theme, n_rows: int, total_rows: int, width: int) -> str:
    mode_tag = f"[b reverse {theme.bg} on {theme.accent}] {state.mode.upper()} [/]"
    hint = _MODE_HINTS.get(state.mode, "")

    filters: list[str] = []
    if state.search_query:
        filters.append(f"/{state.search_query}")
    if state.fit_filter != "all":
        filters.append(f"fit:{state.fit_filter}")
    if state.avail_filter != "all":
        filters.append(f"avail:{state.avail_filter}")
    filters.append(f"sort:{state.sort_key}")
    filters.append(f"use:{state.use_case}")
    if state.provider_filter:
        filters.append(f"prov:{state.provider_filter}")
    filters.append(f"theme:{state.theme}")

    mid = f"[{theme.muted}]{'  '.join(filters)}[/]  "
    count = f"[{theme.muted}]{n_rows}/{total_rows}[/]"

    msg = ""
    if state.status_message:
        msg = f"  [{theme.success}]{state.status_message}[/]"

    line = Text.from_markup(f"{mode_tag}  {mid}  {count}  {hint}{msg}")
    if width > 20:
        plain = line.plain
        if len(plain) > width - 2:
            line = Text.from_markup(f"{mode_tag}  {mid}  {count}{msg}")
    return render_to_ansi(line, width=width, theme=theme)
