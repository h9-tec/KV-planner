"""Detect the local machine's GPU / CPU / RAM and match against our GPU DB.

Deliberately no hard dependency on NVML / pyCUDA / GPUtil — shells out to
``nvidia-smi`` (preferred) and falls back to ``/proc`` parsing. Works on
headless Linux, Ubuntu desktops, and Docker containers that expose the GPU.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from kv_planner.infrastructure.hardware_db import GPUDatabase


@dataclass(frozen=True)
class DetectedHardware:
    cpu_model: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpu_vendor: str  # "nvidia" | "amd" | "apple" | "cpu-only"
    gpu_name_raw: str  # the full string nvidia-smi reported
    gpu_vram_gb: float
    gpu_matched_db_key: Optional[str]  # key in GPUDatabase or None if not matched
    num_gpus: int

    def summary(self) -> str:
        gpu = self.gpu_matched_db_key or self.gpu_name_raw or "CPU-only"
        return (
            f"{self.cpu_cores}C CPU · "
            f"{self.ram_total_gb:.1f} GB RAM · "
            f"{gpu} ({self.gpu_vram_gb:.1f} GB)"
        )


# --------------------------------------------------------------------------
# CPU / RAM
# --------------------------------------------------------------------------


def _read_cpu() -> tuple[str, int]:
    model = ""
    cores = os.cpu_count() or 1
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return model, cores


def _read_ram_gb() -> tuple[float, float]:
    total_kb = avail_kb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
    except OSError:
        return 0.0, 0.0
    return total_kb / 1024 / 1024, avail_kb / 1024 / 1024


# --------------------------------------------------------------------------
# GPU — nvidia-smi
# --------------------------------------------------------------------------


def _nvidia_smi() -> list[tuple[str, float]]:
    """Return list of (gpu_name, total_vram_gb) for each NVIDIA GPU, or []."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=4,
        ).decode()
    except (subprocess.SubprocessError, OSError):
        return []
    gpus: list[tuple[str, float]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            name = parts[0]
            mib = float(parts[1])
            gpus.append((name, mib / 1024))
        except ValueError:
            continue
    return gpus


# --------------------------------------------------------------------------
# Match nvidia-smi name → GPUDatabase key
# --------------------------------------------------------------------------


def _match_gpu_name(raw: str, vram_gb: float) -> Optional[str]:
    """Best-effort match of an nvidia-smi GPU name to our DB."""
    r = raw.upper()
    db_keys = GPUDatabase.list_models()
    candidates: list[tuple[str, int]] = []  # (key, score)

    tokens = set(re.findall(r"[A-Z0-9]+", r))
    for key in db_keys:
        kt = set(re.findall(r"[A-Z0-9]+", key.upper()))
        score = len(kt & tokens)
        # penalize very mismatched sizes
        spec = GPUDatabase.get(key)
        if spec and abs(spec.memory_gb - vram_gb) > 8:
            score -= 2
        if score > 0:
            candidates.append((key, score))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def detect() -> DetectedHardware:
    cpu_model, cpu_cores = _read_cpu()
    total_ram, avail_ram = _read_ram_gb()

    nvidia_gpus = _nvidia_smi()
    if nvidia_gpus:
        name, vram = nvidia_gpus[0]
        matched = _match_gpu_name(name, vram)
        return DetectedHardware(
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_total_gb=total_ram,
            ram_available_gb=avail_ram,
            gpu_vendor="nvidia",
            gpu_name_raw=name,
            gpu_vram_gb=vram,
            gpu_matched_db_key=matched,
            num_gpus=len(nvidia_gpus),
        )

    # No GPU detected
    return DetectedHardware(
        cpu_model=cpu_model,
        cpu_cores=cpu_cores,
        ram_total_gb=total_ram,
        ram_available_gb=avail_ram,
        gpu_vendor="cpu-only",
        gpu_name_raw="",
        gpu_vram_gb=0.0,
        gpu_matched_db_key=None,
        num_gpus=0,
    )
