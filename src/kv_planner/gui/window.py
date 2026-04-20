"""Main window — Adwaita-styled, split view, multi-tab."""

from __future__ import annotations

from typing import Callable

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk, Pango  # noqa: E402

from kv_planner.application.recommender import Recommender
from kv_planner.core.cost import CostAnalyzer
from kv_planner.core.memory import PagedMemoryCalculator
from kv_planner.core.performance import RooflineAnalyzer, RooflineConfig
from kv_planner.core.strategies import QuantizationEvaluator
from kv_planner.domain import ModelConfig, PrecisionType, bytes_per_element
from kv_planner.gui.charts import (
    PAL_ACCENT,
    PAL_LIME,
    PAL_ROSE,
    PAL_TEAL,
    PAL_VIOLET,
    BarChart,
    LineChart,
    RooflineChart,
    Series,
    StackedBar,
    StackSegment,
    Workload,
)
from kv_planner.gui.presets import PRESETS, preset_keys
from kv_planner.infrastructure.hardware_db import GPUDatabase
from kv_planner.infrastructure.hw_detect import detect as detect_hardware
from kv_planner.infrastructure.runtime_probe import probe_all as probe_runtimes


PRECISIONS: list[PrecisionType] = ["fp16", "bf16", "fp8", "int8", "int4"]


_CSS = b"""
window {
  background: #14161c;
  color: #e5e5e9;
  font-family: "JetBrains Mono", "Fira Code", monospace;
}
headerbar {
  background: #191b22;
  border-bottom: 1px solid #24272e;
}
.sidebar {
  background: #191b22;
  border-right: 1px solid #24272e;
  padding: 16px;
}
.kv-section-title {
  font-family: "JetBrains Mono";
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 2px;
  color: #908fa1;
  margin-top: 12px;
  margin-bottom: 4px;
}
.kv-metric-label {
  color: #908fa1;
  font-size: 10px;
  letter-spacing: 1.4px;
}
.kv-metric-value {
  font-family: "JetBrains Mono";
  font-weight: 700;
  font-size: 22px;
  color: #fa9450;
}
.kv-metric-sub {
  color: #908fa1;
  font-size: 10px;
}
.kv-card {
  background: #242630;
  border-radius: 10px;
  padding: 16px;
}
.kv-insight {
  font-size: 11px;
  color: #cdd0d9;
  font-family: "JetBrains Mono";
}
.kv-callout {
  background: #3a2c1e;
  border-left: 3px solid #fa9450;
  padding: 12px;
  border-radius: 4px;
  font-size: 11px;
}
.kv-compute-bound { color: #4abfad; }
.kv-memory-bound { color: #eb6e88; }
scale trough { background: #24272e; }
scale highlight { background: #fa9450; }
spinbutton, entry, dropdown { background: #24272e; color: #e5e5e9; border-radius: 6px; }
"""


def _load_css() -> None:
    provider = Gtk.CssProvider()
    provider.load_from_data(_CSS)
    display = Gtk.Widget.get_default_direction  # not needed; keep API simple
    from gi.repository import Gdk

    Gtk.StyleContext.add_provider_for_display(
        Gdk.Display.get_default(),
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


class _Sidebar(Gtk.Box):
    """Left panel: model/gpu/workload controls. Emits ``changed`` signal."""

    __gtype_name__ = "KvpSidebar"

    def __init__(self, on_change: Callable[[], None]) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.add_css_class("sidebar")
        self.set_size_request(320, -1)

        self._on_change = on_change

        # ------ Model --------------------------------------------------------
        self._section("MODEL")
        self.model_dropdown = self._dropdown(
            [PRESETS[k].label for k in preset_keys()],
            on_change=self._emit,
        )
        self.append(self.model_dropdown)
        self.model_desc = self._small_label("")
        self.append(self.model_desc)

        # ------ Hardware -----------------------------------------------------
        self._section("HARDWARE")
        gpu_names = GPUDatabase.list_models()
        # Push popular datacenter GPUs to the top for convenience.
        prio = [
            "H100-SXM-80GB", "H200-SXM-141GB", "B200-SXM-192GB",
            "A100-SXM-80GB", "RTX-5090", "RTX-4090", "RTX-3090",
            "MI300X", "RTX-5060-Laptop",
        ]
        ordered = [g for g in prio if g in gpu_names] + [
            g for g in gpu_names if g not in prio
        ]
        self.gpu_dropdown = self._dropdown(ordered, on_change=self._emit)
        self.append(self.gpu_dropdown)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row.append(self._small_label("TP"))
        self.tp_spin = Gtk.SpinButton.new_with_range(1, 16, 1)
        self.tp_spin.set_value(1)
        self.tp_spin.connect("value-changed", lambda *_: self._emit())
        row.append(self.tp_spin)
        row.append(self._small_label("×  GPUs"))
        self.n_spin = Gtk.SpinButton.new_with_range(1, 16, 1)
        self.n_spin.set_value(1)
        self.n_spin.connect("value-changed", lambda *_: self._emit())
        row.append(self.n_spin)
        self.append(row)

        # ------ Workload -----------------------------------------------------
        self._section("WORKLOAD")
        self.batch_spin = self._labeled_spin("Batch size", 1, 2048, 1, 32)
        self.input_spin = self._labeled_spin("Input length", 1, 131072, 64, 2048)
        self.output_spin = self._labeled_spin("Output length", 1, 65536, 32, 512)
        self.rps_spin = self._labeled_spin("Target RPS", 1, 10000, 1, 10)

        # ------ Precision & MBU ----------------------------------------------
        self._section("PRECISION")
        self.prec_dropdown = self._dropdown(PRECISIONS, on_change=self._emit)
        self.prec_dropdown.set_selected(0)  # fp16
        self.append(self.prec_dropdown)

        self._section("RUNTIME KNOBS")
        self.mbu_scale = self._labeled_scale("Memory efficiency (MBU)", 0.1, 0.95, 0.05, 0.75)
        self.compute_scale = self._labeled_scale("Compute efficiency (MFU)", 0.1, 0.95, 0.05, 0.50)

        # initial desc refresh
        self._refresh_desc()

    # ------- builders -------------------------------------------------------
    def _section(self, title: str) -> None:
        lbl = Gtk.Label(label=title, xalign=0)
        lbl.add_css_class("kv-section-title")
        self.append(lbl)

    def _small_label(self, text: str) -> Gtk.Label:
        lbl = Gtk.Label(label=text, xalign=0)
        lbl.add_css_class("kv-metric-sub")
        return lbl

    def _dropdown(self, items: list[str], on_change: Callable[[], None]) -> Gtk.DropDown:
        sl = Gtk.StringList.new(items)
        dd = Gtk.DropDown.new(sl, None)
        dd.set_hexpand(True)
        dd.connect("notify::selected", lambda *_: on_change())
        return dd

    def _labeled_spin(self, label: str, lo: int, hi: int, step: int, initial: int) -> Gtk.SpinButton:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        lbl = Gtk.Label(label=label, xalign=0)
        lbl.add_css_class("kv-metric-label")
        lbl.set_hexpand(True)
        box.append(lbl)
        spin = Gtk.SpinButton.new_with_range(lo, hi, step)
        spin.set_value(initial)
        spin.connect("value-changed", lambda *_: self._emit())
        box.append(spin)
        self.append(box)
        return spin

    def _labeled_scale(
        self, label: str, lo: float, hi: float, step: float, initial: float
    ) -> Gtk.Scale:
        lbl = Gtk.Label(label=label, xalign=0)
        lbl.add_css_class("kv-metric-label")
        self.append(lbl)
        scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, lo, hi, step)
        scale.set_value(initial)
        scale.set_draw_value(True)
        scale.set_digits(2)
        scale.set_hexpand(True)
        scale.connect("value-changed", lambda *_: self._emit())
        self.append(scale)
        return scale

    # ------- events ---------------------------------------------------------
    def _refresh_desc(self) -> None:
        key = preset_keys()[self.model_dropdown.get_selected()]
        self.model_desc.set_text(PRESETS[key].description)

    def _emit(self) -> None:
        self._refresh_desc()
        self._on_change()

    # ------- getters --------------------------------------------------------
    def model(self) -> ModelConfig:
        return PRESETS[preset_keys()[self.model_dropdown.get_selected()]].config

    def gpu_key(self) -> str:
        sl = self.gpu_dropdown.get_model()
        return sl.get_string(self.gpu_dropdown.get_selected())

    def precision(self) -> PrecisionType:
        return PRECISIONS[self.prec_dropdown.get_selected()]

    def batch(self) -> int:
        return int(self.batch_spin.get_value())

    def input_len(self) -> int:
        return int(self.input_spin.get_value())

    def output_len(self) -> int:
        return int(self.output_spin.get_value())

    def rps(self) -> float:
        return float(self.rps_spin.get_value())

    def mbu(self) -> float:
        return float(self.mbu_scale.get_value())

    def compute_eff(self) -> float:
        return float(self.compute_scale.get_value())

    def tp(self) -> int:
        return int(self.tp_spin.get_value())

    def num_gpus(self) -> int:
        return int(self.n_spin.get_value())


def _metric_card(label: str, value_widget: Gtk.Widget, sub: str = "") -> Gtk.Box:
    card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
    card.add_css_class("kv-card")
    card.set_hexpand(True)
    lab = Gtk.Label(label=label, xalign=0)
    lab.add_css_class("kv-metric-label")
    card.append(lab)
    card.append(value_widget)
    if sub:
        s = Gtk.Label(label=sub, xalign=0)
        s.add_css_class("kv-metric-sub")
        card.append(s)
    return card


class KvpWindow(Adw.ApplicationWindow):
    """Main window."""

    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="kv-planner")
        self.set_default_size(1320, 840)
        _load_css()

        # toolbar + header
        toolbar = Adw.ToolbarView()
        self.set_content(toolbar)
        header = Adw.HeaderBar()
        title = Gtk.Label()
        title.set_markup(
            '<span font_family="JetBrains Mono" weight="bold" size="13000">kv-planner</span>'
            '<span font_family="JetBrains Mono" size="9000" foreground="#908fa1">  '
            'roofline · paged-memory · quantization</span>'
        )
        header.set_title_widget(title)
        toolbar.add_top_bar(header)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_shrink_start_child(False)
        paned.set_resize_start_child(False)
        paned.set_position(320)
        toolbar.set_content(paned)

        self.sidebar = _Sidebar(on_change=self._recompute)
        paned.set_start_child(self.sidebar)

        self._main_area = self._build_main_area()
        paned.set_end_child(self._main_area)

        # initial computation
        self._recompute()

    # --- main area ----------------------------------------------------------
    def _build_main_area(self) -> Gtk.Widget:
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        outer.set_margin_start(16)
        outer.set_margin_end(16)
        outer.set_margin_top(16)
        outer.set_margin_bottom(16)
        outer.set_hexpand(True)
        outer.set_vexpand(True)

        # --- KPI strip ------------------------------------------------------
        kpis = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        kpis.set_hexpand(True)

        self.kpi_throughput = Gtk.Label(label="—", xalign=0)
        self.kpi_throughput.add_css_class("kv-metric-value")
        self.kpi_throughput_sub = Gtk.Label(label="", xalign=0)
        self.kpi_throughput_sub.add_css_class("kv-metric-sub")

        self.kpi_latency = Gtk.Label(label="—", xalign=0)
        self.kpi_latency.add_css_class("kv-metric-value")
        self.kpi_latency_sub = Gtk.Label(label="", xalign=0)
        self.kpi_latency_sub.add_css_class("kv-metric-sub")

        self.kpi_memory = Gtk.Label(label="—", xalign=0)
        self.kpi_memory.add_css_class("kv-metric-value")
        self.kpi_memory_sub = Gtk.Label(label="", xalign=0)
        self.kpi_memory_sub.add_css_class("kv-metric-sub")

        self.kpi_cost = Gtk.Label(label="—", xalign=0)
        self.kpi_cost.add_css_class("kv-metric-value")
        self.kpi_cost_sub = Gtk.Label(label="", xalign=0)
        self.kpi_cost_sub.add_css_class("kv-metric-sub")

        def wrap(v: Gtk.Label, sub: Gtk.Label) -> Gtk.Box:
            b = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            b.append(v)
            b.append(sub)
            return b

        kpis.append(_metric_card("THROUGHPUT", wrap(self.kpi_throughput, self.kpi_throughput_sub)))
        kpis.append(_metric_card("LATENCY",    wrap(self.kpi_latency,    self.kpi_latency_sub)))
        kpis.append(_metric_card("MEMORY",     wrap(self.kpi_memory,     self.kpi_memory_sub)))
        kpis.append(_metric_card("COST",       wrap(self.kpi_cost,       self.kpi_cost_sub)))
        outer.append(kpis)

        # --- insights / narrative block -------------------------------------
        self.insight_label = Gtk.Label(wrap=True, xalign=0, yalign=0)
        self.insight_label.add_css_class("kv-insight")
        insight_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        insight_box.add_css_class("kv-card")
        lbl = Gtk.Label(label="INSIGHTS", xalign=0)
        lbl.add_css_class("kv-section-title")
        insight_box.append(lbl)
        insight_box.append(self.insight_label)
        outer.append(insight_box)

        # --- tab view for detailed charts -----------------------------------
        stack = Adw.ViewStack()
        switcher = Adw.ViewSwitcher()
        switcher.set_stack(stack)
        switcher.set_policy(Adw.ViewSwitcherPolicy.WIDE)

        # Memory
        self.memory_bar = StackedBar(
            title="MEMORY BREAKDOWN (per-GPU, at this batch & context)",
            unit="GB",
        )
        mem_page = Gtk.ScrolledWindow()
        mem_page.set_child(self.memory_bar)
        stack.add_titled(mem_page, "memory", "Memory")

        # Roofline
        self.roofline = RooflineChart()
        rf_page = Gtk.ScrolledWindow()
        rf_page.set_child(self.roofline)
        stack.add_titled(rf_page, "roofline", "Roofline")

        # Latency waterfall
        self.latency_bar = BarChart(
            title="LATENCY WATERFALL (ms)",
            unit="ms",
            value_format=lambda v: f"{v:,.1f}",
        )
        lat_page = Gtk.ScrolledWindow()
        lat_page.set_child(self.latency_bar)
        stack.add_titled(lat_page, "latency", "Latency")

        # Batch sweep
        self.batch_chart = LineChart(
            title="BATCH SWEEP — throughput vs batch size",
            x_label="batch size",
            y_label="tokens / sec",
            x_format=lambda v: f"{int(v)}",
            y_format=lambda v: f"{int(v):,}",
        )
        batch_page = Gtk.ScrolledWindow()
        batch_page.set_child(self.batch_chart)
        stack.add_titled(batch_page, "batch", "Batch sweep")

        # Context sweep (the headline bug-fix viz)
        self.context_chart = LineChart(
            title="CONTEXT SWEEP — decode latency vs context length",
            x_label="context length (tokens)",
            y_label="ms / token",
            x_format=lambda v: f"{int(v)}",
            y_format=lambda v: f"{v:.1f}",
        )
        ctx_page = Gtk.ScrolledWindow()
        ctx_page.set_child(self.context_chart)
        stack.add_titled(ctx_page, "context", "Context scaling")

        # Precision comparison
        self.precision_chart = BarChart(
            title="QUANTIZATION — throughput by precision",
            unit="tok/s",
            value_format=lambda v: f"{v:,.0f}",
        )
        prec_page = Gtk.ScrolledWindow()
        prec_page.set_child(self.precision_chart)
        stack.add_titled(prec_page, "precision", "Precision")

        # Recommendations (llmfit-style)
        self.rec_chart = BarChart(
            title="TOP MODELS — composite score for this GPU & use case",
            unit="",
            value_format=lambda v: f"{v:.1f}",
        )
        self.rec_label = Gtk.Label(wrap=True, xalign=0)
        self.rec_label.add_css_class("kv-insight")
        rec_inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)

        # Use-case selector for recommendations
        uc_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        uc_row.append(Gtk.Label(label="Use case:", xalign=0))
        self.uc_dropdown = Gtk.DropDown.new(
            Gtk.StringList.new(["general", "coding", "reasoning", "chat", "agent"]),
            None,
        )
        self.uc_dropdown.connect("notify::selected", lambda *_: self._recompute())
        uc_row.append(self.uc_dropdown)
        auto_btn = Gtk.Button(label="Auto-detect hardware")
        auto_btn.connect("clicked", self._on_auto_detect)
        uc_row.append(auto_btn)
        rec_inner.append(uc_row)
        rec_inner.append(self.rec_chart)
        rec_inner.append(self.rec_label)

        rec_page = Gtk.ScrolledWindow()
        rec_page.set_child(rec_inner)
        stack.add_titled(rec_page, "recommend", "Recommend")

        # GPU comparison
        self.gpu_chart = BarChart(
            title="GPU COMPARISON — throughput on this workload",
            unit="tok/s",
            value_format=lambda v: f"{v:,.0f}",
        )
        self.gpu_cost_chart = BarChart(
            title="GPU COMPARISON — cost per million tokens",
            unit="$/M",
            value_format=lambda v: f"{v:,.2f}",
        )
        gpu_page_inner = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        gpu_page_inner.append(self.gpu_chart)
        gpu_page_inner.append(self.gpu_cost_chart)
        gpu_page = Gtk.ScrolledWindow()
        gpu_page.set_child(gpu_page_inner)
        stack.add_titled(gpu_page, "gpu", "GPU comparison")

        outer.append(switcher)
        outer.append(stack)

        # Expose for tests / screenshot automation
        self._stack = stack

        # Allow selecting an initial tab via env var KVP_GUI_TAB
        import os
        initial = os.environ.get("KVP_GUI_TAB")
        if initial:
            stack.set_visible_child_name(initial)

        # Force a second refresh after the widgets are allocated so charts
        # actually paint on first show. GLib.idle_add runs after the main
        # loop has processed the layout pass.
        from gi.repository import GLib
        def _post_show_refresh() -> bool:
            try:
                self._recompute()
                for chart_name in ("memory_bar", "roofline", "latency_bar",
                                   "batch_chart", "context_chart",
                                   "precision_chart", "rec_chart",
                                   "gpu_chart", "gpu_cost_chart"):
                    chart = getattr(self, chart_name, None)
                    if chart and hasattr(chart, "refresh"):
                        chart.refresh()
            except Exception:
                pass
            return False  # one-shot
        GLib.idle_add(_post_show_refresh)
        return outer

    def set_visible_tab(self, name: str) -> None:
        """Switch to a named tab — used by screenshot automation."""
        if hasattr(self, "_stack"):
            self._stack.set_visible_child_name(name)

    # --- computation --------------------------------------------------------
    def _recompute(self) -> None:
        try:
            self._do_recompute()
        except Exception as exc:  # never let a bad input kill the GUI
            self.insight_label.set_text(f"Error: {exc}")

    def _do_recompute(self) -> None:
        model = self.sidebar.model()
        gpu_key = self.sidebar.gpu_key()
        gpu = GPUDatabase.to_hardware_spec(
            gpu_key,
            num_gpus=self.sidebar.num_gpus(),
            tensor_parallel_size=self.sidebar.tp(),
        )
        precision = self.sidebar.precision()
        batch = self.sidebar.batch()
        in_len = self.sidebar.input_len()
        out_len = self.sidebar.output_len()
        rps = self.sidebar.rps()
        mbu = self.sidebar.mbu()
        cfu = self.sidebar.compute_eff()

        ra = RooflineAnalyzer(
            config=RooflineConfig(compute_efficiency=cfu, memory_efficiency=mbu)
        )
        perf = ra.predict_latency(
            model=model, hardware=gpu, batch_size=batch,
            input_length=in_len, output_length=out_len, precision=precision,
        )

        mem_calc = PagedMemoryCalculator(block_size=16)
        kv_bytes = mem_calc.calculate_kv_cache_size(
            batch_size=batch, sequence_length=in_len + out_len,
            model=model, precision=precision,
        )
        # Model weights per GPU (sharded by TP).
        bpe = bytes_per_element(precision)
        weight_bytes_total = model.total_params() * bpe
        weight_bytes_per_gpu = weight_bytes_total / gpu.tensor_parallel_size
        kv_per_gpu = kv_bytes / gpu.tensor_parallel_size
        total_per_gpu = weight_bytes_per_gpu + kv_per_gpu

        cost = CostAnalyzer(roofline_analyzer=ra).analyze_cost(
            model=model, hardware=gpu, batch_size=batch,
            input_length=in_len, output_length=out_len,
            requests_per_second=rps, precision=precision,
        )

        # ---- KPI strip ----
        self.kpi_throughput.set_text(f"{perf.throughput_tokens_per_sec:,.0f}")
        self.kpi_throughput_sub.set_text("tokens / sec (end-to-end)")

        self.kpi_latency.set_text(f"{perf.total_latency_ms:,.0f}")
        self.kpi_latency_sub.set_text(
            f"ms  (prefill {perf.prefill_latency_ms:.0f} + decode {perf.decode_latency_ms:.0f})"
        )

        device_mem = gpu.gpu_memory_gb
        self.kpi_memory.set_text(f"{total_per_gpu/1e9:.1f} / {device_mem:.0f}")
        self.kpi_memory_sub.set_text(
            f"GB per GPU  ({total_per_gpu/1e9/device_mem*100:.0f} % of device)"
        )

        self.kpi_cost.set_text(f"${cost.cost_per_million_tokens:,.2f}")
        self.kpi_cost_sub.set_text(
            f"/ M tokens · ${cost.cost_per_hour:,.2f}/h · util {cost.utilization_pct:.0f}%"
        )

        # ---- Narrative insights ----
        self.insight_label.set_markup(self._narrative(model, gpu, perf, precision,
                                                       total_per_gpu, batch, in_len, out_len))

        # ---- Memory breakdown chart ----
        free_per_gpu = max(0.0, device_mem * 1e9 - total_per_gpu)
        segments = [
            StackSegment("weights", weight_bytes_per_gpu / 1e9, PAL_VIOLET),
            StackSegment("KV cache", kv_per_gpu / 1e9, PAL_ACCENT),
            StackSegment("free headroom", free_per_gpu / 1e9, (0.30, 0.32, 0.38)),
        ]
        self.memory_bar.set_data(segments, total_budget=device_mem)

        # ---- Roofline ----
        peak = gpu.peak_tflops_for(precision)
        # Prefill workload point
        ai_pre = ra.calculate_arithmetic_intensity_prefill(model, batch, in_len, precision)
        ai_dec = ra.calculate_arithmetic_intensity_decode(
            model, batch, in_len + out_len // 2, precision
        )
        self.roofline.set_data(
            peak_tflops=peak,
            peak_bw_gb_s=gpu.memory_bandwidth_gb_s,
            workloads=[
                Workload("prefill", ai_pre, perf.prefill_tflops, PAL_TEAL),
                Workload("decode", ai_dec, perf.decode_tflops, PAL_ROSE),
            ],
            title=f"ROOFLINE — {gpu.gpu_model}  @  {precision}"
                  f"  ·  ridge = peak / bw",
        )

        # ---- Latency waterfall ----
        from kv_planner.gui.charts import Bar

        self.latency_bar.set_bars([
            Bar("prefill", perf.prefill_latency_ms,
                subtitle=f"{in_len} prompt tokens", colour=PAL_TEAL),
            Bar("decode (total)", perf.decode_latency_ms,
                subtitle=f"{out_len} generated tokens", colour=PAL_ROSE),
            Bar("per-token decode",
                perf.decode_latency_ms / max(1, out_len),
                subtitle="ms / generated token", colour=PAL_ACCENT),
        ])

        # ---- Batch sweep ----
        batch_pts: list[tuple[float, float]] = []
        latency_pts: list[tuple[float, float]] = []
        for bs in [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]:
            try:
                p = ra.predict_latency(
                    model=model, hardware=gpu, batch_size=bs,
                    input_length=in_len, output_length=out_len, precision=precision,
                )
                batch_pts.append((bs, p.throughput_tokens_per_sec))
                latency_pts.append((bs, p.total_latency_ms))
            except Exception:
                continue
        self.batch_chart.set_series([
            Series("throughput", batch_pts, colour=PAL_ACCENT),
        ])

        # ---- Context sweep (decode latency vs context length) ----
        ctx_pts: list[tuple[float, float]] = []
        for ctx in [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
            ms_per_tok, _, _ = ra.predict_decode_latency(
                model=model, hardware=gpu, batch_size=batch,
                sequence_length=ctx, precision=precision,
            )
            ctx_pts.append((ctx, ms_per_tok))
        self.context_chart.set_series([
            Series("ms per generated token", ctx_pts, colour=PAL_ROSE),
        ])

        # ---- Precision comparison ----
        prec_bars = []
        colours = [PAL_VIOLET, PAL_TEAL, PAL_ACCENT, PAL_ROSE, PAL_LIME]
        for i, prec in enumerate(PRECISIONS):
            try:
                p = ra.predict_latency(
                    model=model, hardware=gpu, batch_size=batch,
                    input_length=in_len, output_length=out_len, precision=prec,
                )
                prec_bars.append(Bar(
                    prec, p.throughput_tokens_per_sec,
                    subtitle=f"{p.total_latency_ms:.0f} ms total",
                    colour=colours[i % len(colours)],
                ))
            except Exception:
                continue
        self.precision_chart.set_bars(prec_bars)

        # ---- GPU comparison ----
        gpu_bars = []
        cost_bars = []
        cheap_candidates = [
            "H100-SXM-80GB", "H200-SXM-141GB", "B200-SXM-192GB",
            "A100-SXM-80GB", "L40S", "RTX-5090", "RTX-4090",
            "RTX-3090-Ti", "RTX-3090", "MI300X",
        ]
        for gkey in cheap_candidates:
            try:
                h = GPUDatabase.to_hardware_spec(gkey)
                weight_on_gpu = weight_bytes_total  # TP=1 for these comparisons
                if weight_on_gpu / 1e9 > h.gpu_memory_gb * h.gpu_memory_utilization:
                    continue  # won't fit
                p = ra.predict_latency(
                    model=model, hardware=h, batch_size=batch,
                    input_length=in_len, output_length=out_len, precision=precision,
                )
                c = CostAnalyzer(roofline_analyzer=ra).analyze_cost(
                    model=model, hardware=h, batch_size=batch,
                    input_length=in_len, output_length=out_len,
                    requests_per_second=rps, precision=precision,
                )
                gpu_bars.append(Bar(gkey, p.throughput_tokens_per_sec, colour=PAL_ACCENT))
                cost_bars.append(Bar(gkey, c.cost_per_million_tokens, colour=PAL_LIME))
            except Exception:
                continue
        self.gpu_chart.set_bars(gpu_bars)
        self.gpu_cost_chart.set_bars(cost_bars)

        # ---- Recommendations ----
        self._update_recommendations(gpu)

    # ------------------------------------------------------------------
    # Recommendations + hardware auto-detect
    # ------------------------------------------------------------------
    def _update_recommendations(self, gpu) -> None:
        from kv_planner.gui.charts import Bar

        uc_model = self.uc_dropdown.get_model()
        use_case = uc_model.get_string(self.uc_dropdown.get_selected()) or "general"

        try:
            rec = Recommender().top_n(
                gpu,
                n=8,
                use_case=use_case,
                input_length=self.sidebar.input_len(),
                output_length=self.sidebar.output_len(),
                batch_size=1,
                include_unfit=False,
            )
        except Exception as e:
            self.rec_label.set_text(f"Recommender failed: {e}")
            self.rec_chart.set_bars([])
            return

        colours = [PAL_ACCENT, PAL_TEAL, PAL_ROSE, PAL_VIOLET, PAL_LIME] * 3
        bars = [
            Bar(
                r.entry.slug,
                r.score_composite,
                subtitle=(
                    f"{r.precision}  ·  {r.throughput_tok_s:.0f} tok/s  ·  "
                    f"{r.memory_gb:.1f} GB  ·  Q{r.score_quality}·F{r.score_fit}·"
                    f"S{r.score_speed}·C{r.score_context}"
                ),
                colour=colours[i % len(colours)],
            )
            for i, r in enumerate(rec)
        ]
        self.rec_chart.set_bars(bars)

        if not rec:
            self.rec_label.set_text(
                "No catalog model fits this GPU at the requested context. "
                "Try a smaller batch/context, or a larger GPU."
            )
        else:
            top = rec[0]
            self.rec_label.set_markup(
                f"<b>Top pick</b>: <tt>{top.entry.slug}</tt> "
                f"({top.entry.provider}) at <tt>{top.precision}</tt>.  "
                f"Quality {top.score_quality}/100 for "
                f"<b>{use_case}</b>, fits at {top.memory_util_pct:.0f}% of "
                f"VRAM, predicted {top.throughput_tok_s:.0f} tok/s.  "
                f"Ollama tags: <tt>{', '.join(top.entry.ollama_tags) or '—'}</tt>"
            )

    def _on_auto_detect(self, _btn) -> None:
        hw = detect_hardware()
        if not hw.gpu_matched_db_key:
            self.rec_label.set_markup(
                f"<b>Auto-detect</b>: no GPU matched our DB "
                f"(nvidia-smi says: <tt>{hw.gpu_name_raw or 'none'}</tt>, "
                f"{hw.gpu_vram_gb:.1f} GB). Pick one manually in the sidebar."
            )
            return
        # Select the GPU in the sidebar dropdown
        sl = self.sidebar.gpu_dropdown.get_model()
        for i in range(sl.get_n_items()):
            if sl.get_string(i) == hw.gpu_matched_db_key:
                self.sidebar.gpu_dropdown.set_selected(i)
                break
        # Force recompute; recommendations will update with new GPU.
        self._recompute()

    # --- narrative ----------------------------------------------------------
    def _narrative(
        self, model: ModelConfig, gpu, perf, precision: str,
        total_per_gpu: float, batch: int, in_len: int, out_len: int,
    ) -> str:
        ridge = gpu.peak_tflops_for(precision) * 1e12 / (gpu.memory_bandwidth_gb_s * 1e9)
        ai = perf.arithmetic_intensity
        regime = (
            '<span foreground="#4abfad">compute-bound</span>'
            if ai > ridge
            else '<span foreground="#eb6e88">memory-bound</span>'
        )
        kv_per_token_kb = model.kv_cache_bytes_per_token(precision) / 1024
        params_b = model.total_params() / 1e9

        fit_status = (
            '<span foreground="#eb6e88">will NOT fit</span>'
            if total_per_gpu / 1e9 > gpu.gpu_memory_gb
            else '<span foreground="#4abfad">fits</span>'
        )

        # Rough headroom hint
        headroom_seqs = int(
            (gpu.gpu_memory_gb * gpu.gpu_memory_utilization * 1e9 - total_per_gpu)
            / max(1, model.kv_cache_bytes_per_token(precision) * (in_len + out_len))
        )

        parts = [
            f"• Model has <b>{params_b:.2f} B</b> params · KV cache costs "
            f"<b>{kv_per_token_kb:.1f} KiB / token</b> at {precision} · "
            f"one seq at {in_len + out_len} ctx = "
            f"<b>{model.kv_cache_bytes_per_token(precision) * (in_len + out_len) / 1e6:.1f} MB</b>.",

            f"• At batch={batch}, in={in_len}, out={out_len}: workload is "
            f"{regime} (AI={ai:.1f} FLOPs/byte, ridge={ridge:.0f}).",

            f"• Memory {fit_status}: "
            f"<b>{total_per_gpu/1e9:.1f} GB</b> per GPU vs "
            f"<b>{gpu.gpu_memory_gb:.0f} GB</b> device "
            f"(headroom ≈ <b>{max(0, headroom_seqs)}</b> extra sequences).",

            f"• Prefill runs {perf.prefill_latency_ms:.0f} ms @ "
            f"{perf.prefill_tflops:.0f} TFLOPS ({perf.mfu*100:.0f}% MFU). "
            f"Decode runs {perf.decode_latency_ms/max(1,out_len):.1f} ms/tok "
            f"({perf.mbu*100:.0f}% MBU).",
        ]
        return "\n".join(parts)
