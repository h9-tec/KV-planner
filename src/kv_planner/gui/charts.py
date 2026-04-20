"""Cairo-backed chart widgets for GTK 4.

We draw everything by hand so we don't pull in matplotlib. Four primitives:

* :class:`BarChart` — horizontal/vertical bars, one colour per bar.
* :class:`RooflineChart` — log-log AI-vs-perf plot with workload markers.
* :class:`StackedBar` — single horizontal stack (good for memory breakdowns).
* :class:`LineChart` — x/y curves with multiple series.

All charts use the same palette so the app feels like one product.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import cairo
import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gdk, GLib, Gtk, Pango, PangoCairo  # noqa: E402


# ---------------------------------------------------------------------------
# Palette (warm, slightly-off-whites-and-deep-teals; not "AI-purple on white")
# ---------------------------------------------------------------------------

PAL_BG = (0.098, 0.106, 0.133)      # #191b22 — deep blue-gray
PAL_CARD = (0.141, 0.149, 0.180)    # slightly lighter card bg
PAL_TEXT = (0.898, 0.898, 0.918)    # off-white, slightly cool
PAL_MUTED = (0.56, 0.58, 0.62)
PAL_GRID = (0.224, 0.235, 0.263)
PAL_ACCENT = (0.980, 0.580, 0.235)  # warm amber — the "one unforgettable" accent
PAL_TEAL = (0.290, 0.745, 0.678)
PAL_ROSE = (0.922, 0.431, 0.533)
PAL_VIOLET = (0.604, 0.529, 0.929)
PAL_LIME = (0.639, 0.843, 0.361)

SERIES_COLOURS = (PAL_ACCENT, PAL_TEAL, PAL_ROSE, PAL_VIOLET, PAL_LIME)

TITLE_FONT = "JetBrains Mono"
AXIS_FONT = "JetBrains Mono"


# ---------------------------------------------------------------------------
# Shared chart base
# ---------------------------------------------------------------------------


class _ChartBase(Gtk.Picture):
    """Picture-based chart: renders to a Cairo ImageSurface, shows as Gdk.Texture.

    We don't use ``Gtk.DrawingArea.set_draw_func`` because that requires the
    ``python3-gi-cairo`` Ubuntu package (missing in this environment).
    Rendering to an offscreen surface and converting via ``Gdk.Texture`` is
    pure python-pygobject + pycairo, which we always have.
    """

    def __init__(self, min_height: int = 180) -> None:
        super().__init__()
        self._min_height = min_height
        self.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.set_can_shrink(True)
        self.set_size_request(-1, min_height)
        self.set_hexpand(True)
        self.set_vexpand(True)
        # Re-render when the allocated size changes so the chart stays sharp.
        self.connect("notify::default-width", lambda *_: self.refresh())
        self.connect("notify::default-height", lambda *_: self.refresh())

    def _on_draw(
        self, cr: cairo.Context, width: int, height: int
    ) -> None:  # noqa: D401
        """Override in subclasses."""
        raise NotImplementedError

    def set_content_height(self, h: int) -> None:
        self._min_height = h
        self.set_size_request(-1, h)

    def refresh(self) -> None:
        w = max(320, self.get_width() or 640)
        h = max(self._min_height, self.get_height() or self._min_height)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        cr = cairo.Context(surface)
        try:
            self._on_draw(cr, w, h)
        except Exception:  # don't kill the UI if a draw fails
            cr.set_source_rgb(*PAL_CARD)
            cr.rectangle(0, 0, w, h)
            cr.fill()
        buf = io.BytesIO()
        surface.write_to_png(buf)
        data = buf.getvalue()
        texture = Gdk.Texture.new_from_bytes(GLib.Bytes.new(data))
        self.set_paintable(texture)

    # ---- small helpers ------------------------------------------------------
    @staticmethod
    def _text(
        cr: cairo.Context,
        text: str,
        x: float,
        y: float,
        colour: tuple[float, float, float] = PAL_TEXT,
        size: int = 11,
        bold: bool = False,
        align: str = "left",
        font_family: str = AXIS_FONT,
    ) -> tuple[float, float]:
        layout = PangoCairo.create_layout(cr)
        desc = Pango.FontDescription()
        desc.set_family(font_family)
        desc.set_size(size * Pango.SCALE)
        if bold:
            desc.set_weight(Pango.Weight.BOLD)
        layout.set_font_description(desc)
        layout.set_text(text, -1)
        w, h = layout.get_pixel_size()
        if align == "right":
            x -= w
        elif align == "center":
            x -= w / 2
        cr.save()
        cr.set_source_rgb(*colour)
        cr.move_to(x, y)
        PangoCairo.show_layout(cr, layout)
        cr.restore()
        return w, h

    @staticmethod
    def _fill_bg(cr: cairo.Context, w: float, h: float) -> None:
        cr.set_source_rgb(*PAL_CARD)
        cr.rectangle(0, 0, w, h)
        cr.fill()


# ---------------------------------------------------------------------------
# Horizontal bar chart (for "GPU comparison", "batch sweep", etc.)
# ---------------------------------------------------------------------------


@dataclass
class Bar:
    label: str
    value: float
    subtitle: str = ""
    colour: tuple[float, float, float] = PAL_ACCENT


class BarChart(_ChartBase):
    """Horizontal bars, sorted by value descending, value label on the right."""

    def __init__(
        self,
        title: str = "",
        unit: str = "",
        min_height: int = 200,
        value_format: Callable[[float], str] = lambda v: f"{v:,.1f}",
    ) -> None:
        super().__init__(min_height=min_height)
        self.title = title
        self.unit = unit
        self._bars: list[Bar] = []
        self._value_format = value_format

    def set_bars(self, bars: Sequence[Bar]) -> None:
        self._bars = sorted(bars, key=lambda b: b.value, reverse=True)
        self.set_content_height(max(140, 34 * len(self._bars) + 60))
        self.refresh()

    def _on_draw(self, cr, w, h) -> None:
        self._fill_bg(cr, w, h)
        if not self._bars:
            self._text(
                cr, "no data", w / 2, h / 2, colour=PAL_MUTED, size=11, align="center"
            )
            return

        pad_left, pad_right = 14, 18
        y = 8
        if self.title:
            self._text(cr, self.title, pad_left, y, size=11, bold=True)
            y += 22

        max_v = max((b.value for b in self._bars), default=1.0) or 1.0
        label_col_w = 140
        value_col_w = 72
        bar_left = pad_left + label_col_w
        bar_right = w - pad_right - value_col_w
        bar_width_avail = max(40, bar_right - bar_left)

        row_h = 26
        for bar in self._bars:
            # label
            self._text(cr, bar.label, pad_left, y + 2, size=11)
            if bar.subtitle:
                self._text(
                    cr,
                    bar.subtitle,
                    pad_left,
                    y + 14,
                    colour=PAL_MUTED,
                    size=9,
                )

            # track
            cr.set_source_rgb(*PAL_GRID)
            cr.rectangle(bar_left, y + 6, bar_width_avail, 10)
            cr.fill()

            # bar
            filled = (bar.value / max_v) * bar_width_avail
            cr.set_source_rgb(*bar.colour)
            cr.rectangle(bar_left, y + 6, filled, 10)
            cr.fill()

            # value
            text = self._value_format(bar.value) + (" " + self.unit if self.unit else "")
            self._text(
                cr, text, w - pad_right, y + 2, size=10, align="right", colour=PAL_TEXT
            )

            y += row_h


# ---------------------------------------------------------------------------
# Stacked bar (for memory breakdown)
# ---------------------------------------------------------------------------


@dataclass
class StackSegment:
    label: str
    value: float
    colour: tuple[float, float, float]


class StackedBar(_ChartBase):
    """Single horizontal stacked bar with legend and labelled segments."""

    def __init__(self, title: str = "", unit: str = "GB") -> None:
        super().__init__(min_height=130)
        self.title = title
        self.unit = unit
        self._segments: list[StackSegment] = []
        self._total_budget: float | None = None

    def set_data(
        self, segments: Sequence[StackSegment], total_budget: float | None = None
    ) -> None:
        self._segments = list(segments)
        self._total_budget = total_budget
        self.refresh()

    def _on_draw(self, cr, w, h) -> None:
        self._fill_bg(cr, w, h)
        pad = 14
        y = pad
        if self.title:
            self._text(cr, self.title, pad, y, size=11, bold=True)
            y += 22

        if not self._segments:
            self._text(
                cr, "no data", w / 2, h / 2, colour=PAL_MUTED, align="center", size=11
            )
            return

        total = sum(s.value for s in self._segments)
        denom = self._total_budget if self._total_budget else total
        bar_w = w - 2 * pad
        bar_h = 24

        # subtle track (full budget)
        cr.set_source_rgb(*PAL_GRID)
        cr.rectangle(pad, y, bar_w, bar_h)
        cr.fill()

        # segments
        x = pad
        for seg in self._segments:
            seg_w = (seg.value / denom) * bar_w if denom > 0 else 0
            cr.set_source_rgb(*seg.colour)
            cr.rectangle(x, y, seg_w, bar_h)
            cr.fill()
            x += seg_w

        # budget-used text
        y += bar_h + 8
        if self._total_budget:
            pct = total / self._total_budget * 100
            self._text(
                cr,
                f"{total:.1f} / {self._total_budget:.1f} {self.unit}  ({pct:.0f}% of device)",
                pad,
                y,
                colour=PAL_MUTED,
                size=10,
            )
        else:
            self._text(cr, f"{total:.1f} {self.unit}", pad, y, colour=PAL_MUTED, size=10)

        # legend
        y += 18
        legend_x = pad
        for seg in self._segments:
            cr.set_source_rgb(*seg.colour)
            cr.rectangle(legend_x, y + 3, 10, 10)
            cr.fill()
            lw, _ = self._text(
                cr,
                f"{seg.label} {seg.value:.2f}{self.unit}",
                legend_x + 14,
                y,
                size=10,
            )
            legend_x += lw + 30


# ---------------------------------------------------------------------------
# Roofline chart (log-log, with workload marker)
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    label: str
    arithmetic_intensity: float
    achieved_tflops: float
    colour: tuple[float, float, float] = PAL_ACCENT


class RooflineChart(_ChartBase):
    """Williams-style log-log AI→perf with peak TFLOPS ceiling + BW ramp."""

    def __init__(self) -> None:
        super().__init__(min_height=260)
        self._peak_tflops = 0.0
        self._peak_bw_gb_s = 0.0
        self._workloads: list[Workload] = []
        self._title = "Roofline"

    def set_data(
        self,
        peak_tflops: float,
        peak_bw_gb_s: float,
        workloads: Sequence[Workload],
        title: str = "Roofline",
    ) -> None:
        self._peak_tflops = peak_tflops
        self._peak_bw_gb_s = peak_bw_gb_s
        self._workloads = list(workloads)
        self._title = title
        self.refresh()

    def _on_draw(self, cr, w, h) -> None:
        self._fill_bg(cr, w, h)
        if self._peak_tflops <= 0 or self._peak_bw_gb_s <= 0:
            self._text(
                cr, "no data", w / 2, h / 2, colour=PAL_MUTED, align="center", size=11
            )
            return

        # layout
        pad_l, pad_r, pad_t, pad_b = 54, 18, 34, 40
        plot_x0, plot_y0 = pad_l, pad_t
        plot_w = max(50, w - pad_l - pad_r)
        plot_h = max(50, h - pad_t - pad_b)

        # title
        self._text(cr, self._title, pad_l, 10, size=11, bold=True)

        # axes (log10)
        ai_min, ai_max = 0.1, 1000.0   # FLOPs/byte
        t_min, t_max = max(self._peak_tflops * 0.005, 0.05), self._peak_tflops * 1.2
        log_ai = (math.log10(ai_min), math.log10(ai_max))
        log_t = (math.log10(t_min), math.log10(t_max))

        def x_of(ai: float) -> float:
            f = (math.log10(max(ai, ai_min)) - log_ai[0]) / (log_ai[1] - log_ai[0])
            return plot_x0 + f * plot_w

        def y_of(t: float) -> float:
            f = (math.log10(max(t, t_min)) - log_t[0]) / (log_t[1] - log_t[0])
            return plot_y0 + plot_h - f * plot_h

        # grid
        cr.set_source_rgb(*PAL_GRID)
        cr.set_line_width(1.0)
        for e in range(-1, 4):
            for step in range(1, 10):
                xi = x_of(step * 10**e)
                if plot_x0 <= xi <= plot_x0 + plot_w:
                    cr.move_to(xi, plot_y0)
                    cr.line_to(xi, plot_y0 + plot_h)
        for e in range(-2, 5):
            yi = y_of(10**e)
            if plot_y0 <= yi <= plot_y0 + plot_h:
                cr.move_to(plot_x0, yi)
                cr.line_to(plot_x0 + plot_w, yi)
        cr.stroke()

        # axis tick labels (powers of 10 only)
        for e in range(-1, 4):
            xi = x_of(10**e)
            if plot_x0 <= xi <= plot_x0 + plot_w:
                self._text(
                    cr, f"1e{e}", xi, plot_y0 + plot_h + 6, size=9, align="center",
                    colour=PAL_MUTED,
                )
        for e in range(-2, 5):
            yi = y_of(10**e)
            if plot_y0 <= yi <= plot_y0 + plot_h:
                self._text(cr, f"1e{e}", plot_l_text(plot_x0), yi - 6, size=9,
                           align="right", colour=PAL_MUTED)

        # axes
        cr.set_source_rgb(*PAL_MUTED)
        cr.set_line_width(1.0)
        cr.move_to(plot_x0, plot_y0 + plot_h)
        cr.line_to(plot_x0 + plot_w, plot_y0 + plot_h)
        cr.move_to(plot_x0, plot_y0)
        cr.line_to(plot_x0, plot_y0 + plot_h)
        cr.stroke()

        self._text(
            cr, "Arithmetic intensity (FLOPs/byte) →", plot_x0 + plot_w / 2,
            plot_y0 + plot_h + 22, colour=PAL_MUTED, size=10, align="center",
        )

        # ridge point
        ridge = self._peak_tflops * 1e12 / (self._peak_bw_gb_s * 1e9)

        # roofline: y = min(peak, AI·bw)
        cr.set_source_rgb(*PAL_ACCENT)
        cr.set_line_width(2.2)
        # left (memory-bound) ramp
        x_ridge = x_of(ridge)
        y_peak = y_of(self._peak_tflops)
        # y = AI·bw/1e12 ; at ai_min: t = ai_min · peak_bw_gb_s/1000
        t_left = ai_min * self._peak_bw_gb_s / 1000
        cr.move_to(x_of(ai_min), y_of(t_left))
        cr.line_to(x_ridge, y_peak)
        # right (compute-bound) ceiling
        cr.line_to(x_of(ai_max), y_peak)
        cr.stroke()

        # ridge marker
        cr.set_source_rgba(*PAL_ACCENT, 0.3)
        cr.set_line_width(1.0)
        cr.set_dash((4, 4))
        cr.move_to(x_ridge, plot_y0)
        cr.line_to(x_ridge, plot_y0 + plot_h)
        cr.stroke()
        cr.set_dash(())
        self._text(
            cr, f"ridge = {ridge:.0f}", x_ridge + 4, plot_y0 + 4,
            colour=PAL_ACCENT, size=9,
        )

        # workload markers
        for wl in self._workloads:
            if wl.arithmetic_intensity <= 0 or wl.achieved_tflops <= 0:
                continue
            px, py = x_of(wl.arithmetic_intensity), y_of(wl.achieved_tflops)
            cr.set_source_rgb(*wl.colour)
            cr.arc(px, py, 5.0, 0, 2 * math.pi)
            cr.fill()
            self._text(cr, wl.label, px + 8, py - 8, colour=wl.colour, size=10, bold=True)


def plot_l_text(plot_x0: float) -> float:
    """Tiny helper — Cairo x coord for left-of-plot right-aligned tick labels."""
    return plot_x0 - 4


# ---------------------------------------------------------------------------
# Line chart with multiple series (for batch sweep / context sweep)
# ---------------------------------------------------------------------------


@dataclass
class Series:
    label: str
    points: list[tuple[float, float]]
    colour: tuple[float, float, float] = PAL_ACCENT


class LineChart(_ChartBase):
    """x/y line chart with N series, linear axes, auto-ranged."""

    def __init__(
        self,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
        min_height: int = 240,
        x_format: Callable[[float], str] = lambda v: f"{v:g}",
        y_format: Callable[[float], str] = lambda v: f"{v:g}",
    ) -> None:
        super().__init__(min_height=min_height)
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self._series: list[Series] = []
        self._x_format = x_format
        self._y_format = y_format

    def set_series(self, series: Sequence[Series]) -> None:
        self._series = list(series)
        self.refresh()

    def _on_draw(self, cr, w, h) -> None:
        self._fill_bg(cr, w, h)
        if not self._series or not any(s.points for s in self._series):
            self._text(
                cr, "no data", w / 2, h / 2, colour=PAL_MUTED, align="center", size=11
            )
            return

        pad_l, pad_r, pad_t, pad_b = 60, 16, 30, 46
        plot_x0, plot_y0 = pad_l, pad_t
        plot_w = max(50, w - pad_l - pad_r)
        plot_h = max(50, h - pad_t - pad_b)

        if self.title:
            self._text(cr, self.title, pad_l, 10, size=11, bold=True)

        all_pts = [p for s in self._series for p in s.points]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = 0.0, max(ys) * 1.1 or 1.0
        if x_max == x_min:
            x_max = x_min + 1.0

        def x_of(v: float) -> float:
            return plot_x0 + (v - x_min) / (x_max - x_min) * plot_w

        def y_of(v: float) -> float:
            return plot_y0 + plot_h - (v - y_min) / (y_max - y_min) * plot_h

        # grid
        cr.set_source_rgb(*PAL_GRID)
        cr.set_line_width(1.0)
        n_y = 5
        for i in range(n_y + 1):
            gy = y_min + (y_max - y_min) * i / n_y
            yi = y_of(gy)
            cr.move_to(plot_x0, yi)
            cr.line_to(plot_x0 + plot_w, yi)
            self._text(
                cr, self._y_format(gy), plot_x0 - 4, yi - 7, size=9,
                align="right", colour=PAL_MUTED,
            )
        cr.stroke()

        # x-axis ticks (~6)
        n_x = 6
        for i in range(n_x + 1):
            gx = x_min + (x_max - x_min) * i / n_x
            xi = x_of(gx)
            cr.set_source_rgb(*PAL_MUTED)
            cr.set_line_width(1.0)
            cr.move_to(xi, plot_y0 + plot_h)
            cr.line_to(xi, plot_y0 + plot_h + 3)
            cr.stroke()
            self._text(
                cr, self._x_format(gx), xi, plot_y0 + plot_h + 6, size=9,
                align="center", colour=PAL_MUTED,
            )

        # axis frame
        cr.set_source_rgb(*PAL_MUTED)
        cr.set_line_width(1.0)
        cr.move_to(plot_x0, plot_y0 + plot_h)
        cr.line_to(plot_x0 + plot_w, plot_y0 + plot_h)
        cr.move_to(plot_x0, plot_y0)
        cr.line_to(plot_x0, plot_y0 + plot_h)
        cr.stroke()

        # axis labels
        if self.x_label:
            self._text(
                cr, self.x_label, plot_x0 + plot_w / 2, plot_y0 + plot_h + 22,
                colour=PAL_MUTED, size=10, align="center",
            )
        if self.y_label:
            cr.save()
            cr.translate(14, plot_y0 + plot_h / 2)
            cr.rotate(-math.pi / 2)
            self._text(cr, self.y_label, 0, 0, colour=PAL_MUTED, size=10, align="center")
            cr.restore()

        # series
        for series in self._series:
            if not series.points:
                continue
            cr.set_source_rgb(*series.colour)
            cr.set_line_width(2.0)
            first = True
            for xv, yv in series.points:
                px, py = x_of(xv), y_of(yv)
                if first:
                    cr.move_to(px, py)
                    first = False
                else:
                    cr.line_to(px, py)
            cr.stroke()

            # data-point circles
            for xv, yv in series.points:
                cr.arc(x_of(xv), y_of(yv), 2.6, 0, 2 * math.pi)
                cr.fill()

        # legend (top-right of plot)
        lx, ly = plot_x0 + plot_w - 10, plot_y0 + 2
        for series in reversed(self._series):
            cr.set_source_rgb(*series.colour)
            label = series.label
            lw, lh = self._measure(label, size=10)
            cr.rectangle(lx - lw - 16, ly, 8, 8)
            cr.fill()
            self._text(cr, label, lx, ly - 1, colour=series.colour, size=10, align="right")
            ly += lh + 4

    def _measure(self, text: str, size: int = 10) -> tuple[float, float]:
        # Create a dummy cairo context to measure text.
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        cr = cairo.Context(surf)
        layout = PangoCairo.create_layout(cr)
        desc = Pango.FontDescription()
        desc.set_family(AXIS_FONT)
        desc.set_size(size * Pango.SCALE)
        layout.set_font_description(desc)
        layout.set_text(text, -1)
        return layout.get_pixel_size()
