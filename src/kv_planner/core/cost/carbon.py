"""Carbon / power model — gCO2e per million tokens.

Given GPU TDP, sustained utilization (a function of MFU+MBU), and the
grid carbon intensity of the region, compute gCO2e per M tokens alongside
$/M tokens. Aligns with 2024–2026 EU AI Act reporting requirements.

Carbon intensities (gCO2e / kWh) from Electricity Maps annual averages
(https://electricitymaps.com/). These are point-in-time defaults —
override via ``set_grid_intensity()``.
"""

from __future__ import annotations

from dataclasses import dataclass


# gCO2e per kWh, 2024 annual averages — override for your region.
_GRID_INTENSITY: dict[str, float] = {
    # Lowest-carbon regions
    "iceland": 25,
    "quebec": 40,
    "norway": 60,
    "france": 80,      # mostly nuclear
    "sweden": 40,
    # Medium-carbon
    "us-west": 220,    # Oregon, Washington
    "us-east": 390,    # Virginia (AWS us-east-1)
    "us-central": 420,
    "germany": 350,
    "uk": 240,
    "japan": 500,
    # High-carbon (mostly coal)
    "us-texas": 430,
    "india": 700,
    "china-north": 620,
    "poland": 680,
    "south-africa": 870,
    # Global average fallback
    "global": 480,
}


@dataclass(frozen=True)
class CarbonEstimate:
    gpu_watts_avg: float
    kwh_per_million_tokens: float
    grid_intensity_g_per_kwh: float
    g_co2e_per_million_tokens: float
    g_co2e_per_request: float
    region: str


def get_grid_intensity(region: str) -> float:
    return _GRID_INTENSITY.get(region.lower(), _GRID_INTENSITY["global"])


def set_grid_intensity(region: str, g_per_kwh: float) -> None:
    _GRID_INTENSITY[region.lower()] = g_per_kwh


def estimate_carbon(
    throughput_tok_s: float,
    tdp_watts: float,
    mfu: float,
    mbu: float,
    region: str = "us-east",
    num_gpus: int = 1,
    pue: float = 1.15,
    tokens_per_request: int = 2560,
) -> CarbonEstimate:
    """Emissions estimate given steady-state throughput and device specs.

    Power usage model: a GPU draws between ~30 % TDP at idle and TDP when
    saturated. Utilisation ≈ max(mfu, mbu) is a reasonable proxy for the
    fraction of TDP being drawn.

    ``pue`` = Power Usage Effectiveness of the datacenter (hyperscale ≈ 1.1,
    enterprise ≈ 1.5). Multiplies GPU power by this factor for cooling etc.
    """
    utilization = min(1.0, max(0.0, max(mfu, mbu)))
    avg_watts = tdp_watts * (0.3 + 0.7 * utilization) * num_gpus * pue

    # kWh per second of operation = W / 1000 / 3600
    kwh_per_second = avg_watts / 1000.0 / 3600.0
    # Tokens per kWh
    tokens_per_kwh = (throughput_tok_s / kwh_per_second) if kwh_per_second > 0 else 0.0
    kwh_per_million = 1e6 / tokens_per_kwh if tokens_per_kwh > 0 else float("inf")

    intensity = get_grid_intensity(region)
    g_per_million = kwh_per_million * intensity
    g_per_request = g_per_million * (tokens_per_request / 1e6)

    return CarbonEstimate(
        gpu_watts_avg=avg_watts,
        kwh_per_million_tokens=kwh_per_million,
        grid_intensity_g_per_kwh=intensity,
        g_co2e_per_million_tokens=g_per_million,
        g_co2e_per_request=g_per_request,
        region=region,
    )
