"""
Laptop GPU sustained-performance adjustment.

**Why this module exists.** A laptop GPU with the same marketing name as a
desktop card isn't actually the same part: it usually has fewer SMs, lower
boost clocks, a much lower TGP (power budget), and severe sustained-load
thermal throttling. For identically-named SKUs ("RTX 5090 Laptop" vs
"RTX 5090 Desktop"), the measured sustained ratio is typically **50–65 %**
across reviewers — NOT 7 % as an earlier version of this module claimed.

The 7.2 % figure once enshrined here came from comparing a laptop RTX 5060
against a desktop RTX 5090 — **that's cross-tier, not a laptop penalty**.
It has been removed.

**What this module does now.** We do not estimate laptop-vs-desktop ratios
at all. We apply a pure sustained-thermal factor on top of whatever
per-SKU TFLOPS the caller already supplied (via
``gpu_specs.py:RTX-XXXX-Laptop``). Three profiles are exposed:

* ``thin-and-light`` — 0.6 × nominal (aggressive thermal/power limit)
* ``gaming`` — 0.75 × nominal (typical gaming laptop cooling)
* ``workstation`` — 0.85 × nominal (high-TDP laptop with good cooling)

Sources:

* Tom's Hardware RTX 5090 laptop review — ~50 % slower than desktop counterpart:
  https://www.tomshardware.com/pc-components/gpus/rtx-5090-laptop-review-claims-gpu-is-a-performance-dud-but-outshines-the-4090-in-power-efficiency
* Videocardz reviewers roundup — "50 % slower" across reviewers:
  https://videocardz.com/newz/reviewers-report-geforce-rtx-5090-for-laptops-is-50-slower-than-desktop-version
* Notebookcheck RTX 5090 Laptop benchmarks:
  https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-Laptop-Benchmarks-and-Specs.934947.0.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LaptopProfile = Literal["thin-and-light", "gaming", "workstation"]


@dataclass(frozen=True)
class LaptopThermalFactor:
    """Sustained-thermal factor applied on top of the laptop's own spec sheet.

    Interpret as: the fraction of peak FP16 TFLOPS a laptop of this class
    sustains under a 10-minute LLM-inference workload.
    """

    name: str
    factor: float
    notes: str


THIN_AND_LIGHT = LaptopThermalFactor(
    name="thin-and-light",
    factor=0.60,
    notes="Ultrabooks and thin creator laptops; aggressive TGP limit and short thermal transients.",
)

GAMING = LaptopThermalFactor(
    name="gaming",
    factor=0.75,
    notes="Typical 17-inch gaming laptops; dual-fan cooling, mid-range TGP.",
)

WORKSTATION = LaptopThermalFactor(
    name="workstation",
    factor=0.85,
    notes="High-TDP workstation/mobile-RTX-A laptops with vapor-chamber cooling.",
)

_PROFILES: dict[LaptopProfile, LaptopThermalFactor] = {
    "thin-and-light": THIN_AND_LIGHT,
    "gaming": GAMING,
    "workstation": WORKSTATION,
}


def is_laptop_gpu(gpu_model: str) -> bool:
    """True if ``gpu_model`` is a laptop / mobile variant."""
    model_upper = gpu_model.upper()
    return any(
        indicator in model_upper
        for indicator in ("-LAPTOP", "-MOBILE", " LAPTOP", " MOBILE", " MAX-Q")
    )


def get_laptop_thermal_factor(
    gpu_model: str, profile: LaptopProfile = "gaming"
) -> float:
    """Return the sustained-thermal factor for a laptop GPU (1.0 for desktops)."""
    if not is_laptop_gpu(gpu_model):
        return 1.0
    return _PROFILES[profile].factor


def adjust_performance_metrics(
    gpu_model: str,
    throughput_tokens_per_sec: float,
    latency_ms: float,
    profile: LaptopProfile = "gaming",
) -> tuple[float, float]:
    """Scale throughput down and latency up by the laptop thermal factor.

    Returns ``(adjusted_throughput, adjusted_latency_ms)``. Desktop GPUs
    pass through unchanged.
    """
    factor = get_laptop_thermal_factor(gpu_model, profile)
    return throughput_tokens_per_sec * factor, latency_ms / factor


def get_laptop_info(
    gpu_model: str, profile: LaptopProfile = "gaming"
) -> dict[str, object]:
    """Human-readable description — what the factor is and why."""
    if not is_laptop_gpu(gpu_model):
        return {
            "is_laptop": False,
            "thermal_factor": 1.0,
            "notes": "Desktop GPU — no adjustment.",
        }

    prof = _PROFILES[profile]
    return {
        "is_laptop": True,
        "profile": prof.name,
        "thermal_factor": prof.factor,
        "notes": prof.notes,
        "disclaimer": (
            "Factor applied on top of the laptop SKU's own spec-sheet "
            "TFLOPS, not a laptop-vs-desktop retention ratio. If you want "
            "a laptop-vs-desktop comparison, compare the two SKUs' peak "
            "TFLOPS numbers in gpu_specs.py directly."
        ),
    }
