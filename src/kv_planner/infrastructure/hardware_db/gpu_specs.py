"""
GPU specifications database.

All TFLOPS numbers are **dense FP16 tensor-core** values sourced directly
from vendor whitepapers — NOT FP32 CUDA throughput and NOT structured-
sparsity numbers. Structured sparsity doubles the advertised TFLOPS on
Hopper/Ampere/Ada/Blackwell but requires 2:4-sparse weights that almost
no deployed LLM ships with, so it would systematically over-predict by 2×
for Roofline analysis.

Sources (every vendor value below is traceable to one of these):

* NVIDIA H100 datasheet — https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306
* NVIDIA Hopper architecture — https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
* NVIDIA H200 datasheet — https://www.nvidia.com/en-us/data-center/h200/
* NVIDIA Ampere GA102 whitepaper — https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
* NVIDIA Ada Lovelace whitepaper — https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
* NVIDIA RTX Blackwell whitepaper — https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
* NVIDIA Blackwell B200 (HGX) datasheet via Primeline — https://www.primeline-solutions.com/media/categories/server/nach-gpu/nvidia-hgx-h200/nvidia-blackwell-b200-datasheet.pdf
* NVIDIA A100 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
* AMD MI300X datasheet — https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from kv_planner.domain import HardwareSpec, PrecisionType


@dataclass(frozen=True)
class GPUSpec:
    """GPU specification — dense tensor-core FP16 as the compute ceiling.

    ``peak_tflops_by_precision`` optionally records per-precision peaks
    (e.g. 2× FP16 for FP8 on Hopper). Use for downstream Roofline on
    non-FP16 workloads.
    """

    model: str
    memory_gb: float
    peak_tflops_fp16: float  # dense tensor-core FP16
    memory_bandwidth_gb_s: float
    l2_cache_mb: float
    architecture: str
    launch_year: int
    typical_tdp_w: int
    memory_type: str = "HBM3"
    peak_tflops_by_precision: Mapping[PrecisionType, float] = field(default_factory=dict)


_NVIDIA_GPUS: dict[str, GPUSpec] = {
    # ---------- Hopper ----------
    "H100-SXM-80GB": GPUSpec(
        model="H100-SXM-80GB",
        memory_gb=80.0,
        peak_tflops_fp16=989.0,
        memory_bandwidth_gb_s=3350.0,
        l2_cache_mb=50.0,
        architecture="Hopper",
        launch_year=2022,
        typical_tdp_w=700,
        memory_type="HBM3",
        peak_tflops_by_precision={
            "fp16": 989.0,
            "bf16": 989.0,
            "fp8": 1979.0,
            "int8": 1979.0,
        },
    ),
    "H100-PCIe-80GB": GPUSpec(
        model="H100-PCIe-80GB",
        memory_gb=80.0,
        peak_tflops_fp16=756.0,
        memory_bandwidth_gb_s=2000.0,
        l2_cache_mb=50.0,
        architecture="Hopper",
        launch_year=2022,
        typical_tdp_w=350,
        memory_type="HBM2e",
        peak_tflops_by_precision={
            "fp16": 756.0,
            "bf16": 756.0,
            "fp8": 1513.0,
            "int8": 1513.0,
        },
    ),
    "H100-NVL-94GB": GPUSpec(
        # Single die of the dual-GPU H100-NVL card (per-die numbers).
        model="H100-NVL-94GB",
        memory_gb=94.0,
        peak_tflops_fp16=835.0,  # per die (1671 dual for the pair ÷ 2)
        memory_bandwidth_gb_s=3900.0,
        l2_cache_mb=50.0,
        architecture="Hopper",
        launch_year=2023,
        typical_tdp_w=400,
        memory_type="HBM3",
        peak_tflops_by_precision={
            "fp16": 835.0,
            "bf16": 835.0,
            "fp8": 1670.0,
            "int8": 1670.0,
        },
    ),
    "H200-SXM-141GB": GPUSpec(
        # Same compute die as H100 SXM5; gains come from memory capacity/bandwidth.
        model="H200-SXM-141GB",
        memory_gb=141.0,
        peak_tflops_fp16=989.0,
        memory_bandwidth_gb_s=4800.0,
        l2_cache_mb=50.0,
        architecture="Hopper",
        launch_year=2024,
        typical_tdp_w=700,
        memory_type="HBM3e",
        peak_tflops_by_precision={
            "fp16": 989.0,
            "bf16": 989.0,
            "fp8": 1979.0,
            "int8": 1979.0,
        },
    ),

    # ---------- Blackwell datacenter ----------
    "B200-SXM-192GB": GPUSpec(
        model="B200-SXM-192GB",
        memory_gb=192.0,
        peak_tflops_fp16=2250.0,  # dense FP16; NVIDIA headline 20 PFLOPS is FP4 sparse
        memory_bandwidth_gb_s=8000.0,
        l2_cache_mb=228.0,
        architecture="Blackwell",
        launch_year=2024,
        typical_tdp_w=1000,
        memory_type="HBM3e",
        peak_tflops_by_precision={
            "fp16": 2250.0,
            "bf16": 2250.0,
            "fp8": 4500.0,
            "int8": 4500.0,
        },
    ),
    "GB200-Superchip": GPUSpec(
        # Superchip = 1× Grace CPU + 2× B200 dies. Per-GPU numbers ×2.
        model="GB200-Superchip",
        memory_gb=384.0,  # 2× 192 GB HBM3e
        peak_tflops_fp16=4500.0,
        memory_bandwidth_gb_s=16000.0,
        l2_cache_mb=456.0,
        architecture="Blackwell",
        launch_year=2024,
        typical_tdp_w=2700,
        memory_type="HBM3e",
        peak_tflops_by_precision={
            "fp16": 4500.0,
            "bf16": 4500.0,
            "fp8": 9000.0,
            "int8": 9000.0,
        },
    ),

    # ---------- Ampere datacenter ----------
    "A100-SXM-80GB": GPUSpec(
        model="A100-SXM-80GB",
        memory_gb=80.0,
        peak_tflops_fp16=312.0,
        memory_bandwidth_gb_s=2039.0,
        l2_cache_mb=40.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=400,
        memory_type="HBM2e",
        peak_tflops_by_precision={"fp16": 312.0, "bf16": 312.0, "int8": 624.0},
    ),
    "A100-SXM-40GB": GPUSpec(
        model="A100-SXM-40GB",
        memory_gb=40.0,
        peak_tflops_fp16=312.0,
        memory_bandwidth_gb_s=1555.0,
        l2_cache_mb=40.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=400,
        memory_type="HBM2",
        peak_tflops_by_precision={"fp16": 312.0, "bf16": 312.0, "int8": 624.0},
    ),
    "A100-PCIe-80GB": GPUSpec(
        model="A100-PCIe-80GB",
        memory_gb=80.0,
        peak_tflops_fp16=312.0,
        memory_bandwidth_gb_s=1935.0,
        l2_cache_mb=40.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=300,
        memory_type="HBM2e",
        peak_tflops_by_precision={"fp16": 312.0, "bf16": 312.0, "int8": 624.0},
    ),
    "A10G": GPUSpec(
        model="A10G",
        memory_gb=24.0,
        peak_tflops_fp16=125.0,
        memory_bandwidth_gb_s=600.0,
        l2_cache_mb=6.0,
        architecture="Ampere",
        launch_year=2021,
        typical_tdp_w=300,
        memory_type="GDDR6",
    ),

    # ---------- Ada Lovelace datacenter ----------
    "L40S": GPUSpec(
        model="L40S",
        memory_gb=48.0,
        peak_tflops_fp16=362.0,
        memory_bandwidth_gb_s=864.0,
        l2_cache_mb=96.0,
        architecture="Ada Lovelace",
        launch_year=2023,
        typical_tdp_w=350,
        memory_type="GDDR6",
        peak_tflops_by_precision={"fp16": 362.0, "bf16": 362.0, "fp8": 733.0, "int8": 733.0},
    ),
    "L4": GPUSpec(
        model="L4",
        memory_gb=24.0,
        peak_tflops_fp16=121.0,
        memory_bandwidth_gb_s=300.0,
        l2_cache_mb=48.0,
        architecture="Ada Lovelace",
        launch_year=2023,
        typical_tdp_w=72,
        memory_type="GDDR6",
    ),

    "V100-SXM-32GB": GPUSpec(
        model="V100-SXM-32GB",
        memory_gb=32.0,
        peak_tflops_fp16=125.0,
        memory_bandwidth_gb_s=900.0,
        l2_cache_mb=6.0,
        architecture="Volta",
        launch_year=2017,
        typical_tdp_w=300,
        memory_type="HBM2",
    ),

    # ---------- Consumer RTX 50 (Blackwell) ----------
    "RTX-5090": GPUSpec(
        # Blackwell whitepaper: 1676 TFLOPS FP16 tensor core WITH 2:4 sparsity.
        # Dense is half, ~838. (NVIDIA marketing also quotes "3352 AI TOPS"
        # which is FP4 + sparsity; irrelevant for fp16 LLM roofline.)
        model="RTX-5090",
        memory_gb=32.0,
        peak_tflops_fp16=838.0,
        memory_bandwidth_gb_s=1792.0,
        l2_cache_mb=96.0,
        architecture="Blackwell",
        launch_year=2025,
        typical_tdp_w=575,
        memory_type="GDDR7",
        peak_tflops_by_precision={
            "fp16": 838.0, "bf16": 838.0, "fp8": 1676.0, "int8": 1676.0
        },
    ),
    "RTX-5080": GPUSpec(
        model="RTX-5080",
        memory_gb=16.0,
        peak_tflops_fp16=450.0,
        memory_bandwidth_gb_s=960.0,
        l2_cache_mb=64.0,
        architecture="Blackwell",
        launch_year=2025,
        typical_tdp_w=360,
        memory_type="GDDR7",
        peak_tflops_by_precision={
            "fp16": 450.0, "bf16": 450.0, "fp8": 900.0, "int8": 900.0
        },
    ),
    "RTX-5070-Ti": GPUSpec(
        model="RTX-5070-Ti",
        memory_gb=16.0,
        peak_tflops_fp16=290.0,
        memory_bandwidth_gb_s=896.0,
        l2_cache_mb=64.0,
        architecture="Blackwell",
        launch_year=2025,
        typical_tdp_w=300,
        memory_type="GDDR7",
        peak_tflops_by_precision={
            "fp16": 290.0, "bf16": 290.0, "fp8": 580.0, "int8": 580.0
        },
    ),
    "RTX-5070": GPUSpec(
        model="RTX-5070",
        memory_gb=12.0,
        peak_tflops_fp16=180.0,
        memory_bandwidth_gb_s=672.0,
        l2_cache_mb=48.0,
        architecture="Blackwell",
        launch_year=2025,
        typical_tdp_w=250,
        memory_type="GDDR7",
        peak_tflops_by_precision={
            "fp16": 180.0, "bf16": 180.0, "fp8": 360.0, "int8": 360.0
        },
    ),
    # Laptop SKU kept as-is for regression-test purposes; dense FP16 tensor-
    # core, not the old FP32 CUDA core number.
    "RTX-5060-Laptop": GPUSpec(
        model="RTX-5060-Laptop",
        memory_gb=8.0,
        peak_tflops_fp16=95.0,  # dense FP16 tensor (previous value 23.2 was FP32 CUDA)
        memory_bandwidth_gb_s=448.0,
        l2_cache_mb=32.0,
        architecture="Blackwell",
        launch_year=2025,
        typical_tdp_w=115,
        memory_type="GDDR7",
    ),

    # ---------- Consumer RTX 40 (Ada) ----------
    "RTX-4090": GPUSpec(
        # Ada whitepaper: 330.3 TFLOPS FP16 WITH sparsity; dense = 165.2.
        model="RTX-4090",
        memory_gb=24.0,
        peak_tflops_fp16=165.2,
        memory_bandwidth_gb_s=1008.0,
        l2_cache_mb=72.0,
        architecture="Ada Lovelace",
        launch_year=2022,
        typical_tdp_w=450,
        memory_type="GDDR6X",
        peak_tflops_by_precision={
            "fp16": 165.2, "bf16": 165.2, "fp8": 330.4, "int8": 330.4
        },
    ),
    "RTX-4080-Super": GPUSpec(
        model="RTX-4080-Super",
        memory_gb=16.0,
        peak_tflops_fp16=104.4,
        memory_bandwidth_gb_s=736.3,
        l2_cache_mb=64.0,
        architecture="Ada Lovelace",
        launch_year=2024,
        typical_tdp_w=320,
        memory_type="GDDR6X",
    ),
    "RTX-4080": GPUSpec(
        model="RTX-4080",
        memory_gb=16.0,
        peak_tflops_fp16=97.5,
        memory_bandwidth_gb_s=716.8,
        l2_cache_mb=64.0,
        architecture="Ada Lovelace",
        launch_year=2022,
        typical_tdp_w=320,
        memory_type="GDDR6X",
    ),
    "RTX-4070-Ti-Super": GPUSpec(
        model="RTX-4070-Ti-Super",
        memory_gb=16.0,
        peak_tflops_fp16=88.2,
        memory_bandwidth_gb_s=672.0,
        l2_cache_mb=48.0,
        architecture="Ada Lovelace",
        launch_year=2024,
        typical_tdp_w=285,
        memory_type="GDDR6X",
    ),
    "RTX-4070-Ti": GPUSpec(
        model="RTX-4070-Ti",
        memory_gb=12.0,
        peak_tflops_fp16=80.1,
        memory_bandwidth_gb_s=504.2,
        l2_cache_mb=48.0,
        architecture="Ada Lovelace",
        launch_year=2023,
        typical_tdp_w=285,
        memory_type="GDDR6X",
    ),
    "RTX-4070-Super": GPUSpec(
        model="RTX-4070-Super",
        memory_gb=12.0,
        peak_tflops_fp16=71.0,
        memory_bandwidth_gb_s=504.2,
        l2_cache_mb=48.0,
        architecture="Ada Lovelace",
        launch_year=2024,
        typical_tdp_w=220,
        memory_type="GDDR6X",
    ),
    "RTX-4070": GPUSpec(
        model="RTX-4070",
        memory_gb=12.0,
        peak_tflops_fp16=58.3,
        memory_bandwidth_gb_s=504.2,
        l2_cache_mb=36.0,
        architecture="Ada Lovelace",
        launch_year=2023,
        typical_tdp_w=200,
        memory_type="GDDR6X",
    ),
    "RTX-4060-Ti": GPUSpec(
        model="RTX-4060-Ti",
        memory_gb=16.0,
        peak_tflops_fp16=44.0,
        memory_bandwidth_gb_s=288.0,
        l2_cache_mb=32.0,
        architecture="Ada Lovelace",
        launch_year=2023,
        typical_tdp_w=160,
        memory_type="GDDR6",
    ),
    "RTX-4060": GPUSpec(
        model="RTX-4060",
        memory_gb=8.0,
        peak_tflops_fp16=30.0,
        memory_bandwidth_gb_s=272.0,
        l2_cache_mb=24.0,
        architecture="Ada Lovelace",
        launch_year=2023,
        typical_tdp_w=115,
        memory_type="GDDR6",
    ),

    # ---------- Consumer RTX 30 (Ampere) — dense tensor FP16 ----------
    "RTX-3090-Ti": GPUSpec(
        model="RTX-3090-Ti",
        memory_gb=24.0,
        peak_tflops_fp16=160.0,
        memory_bandwidth_gb_s=1008.0,
        l2_cache_mb=6.0,
        architecture="Ampere",
        launch_year=2022,
        typical_tdp_w=450,
        memory_type="GDDR6X",
    ),
    "RTX-3090": GPUSpec(
        model="RTX-3090",
        memory_gb=24.0,
        peak_tflops_fp16=142.0,
        memory_bandwidth_gb_s=936.2,
        l2_cache_mb=6.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=350,
        memory_type="GDDR6X",
    ),
    "RTX-3080-Ti": GPUSpec(
        model="RTX-3080-Ti",
        memory_gb=12.0,
        peak_tflops_fp16=136.0,
        memory_bandwidth_gb_s=912.4,
        l2_cache_mb=6.0,
        architecture="Ampere",
        launch_year=2021,
        typical_tdp_w=350,
        memory_type="GDDR6X",
    ),
    "RTX-3080-12GB": GPUSpec(
        model="RTX-3080-12GB",
        memory_gb=12.0,
        peak_tflops_fp16=122.0,
        memory_bandwidth_gb_s=912.0,
        l2_cache_mb=6.0,
        architecture="Ampere",
        launch_year=2022,
        typical_tdp_w=350,
        memory_type="GDDR6X",
    ),
    "RTX-3080": GPUSpec(
        model="RTX-3080",
        memory_gb=10.0,
        peak_tflops_fp16=119.0,
        memory_bandwidth_gb_s=760.3,
        l2_cache_mb=5.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=320,
        memory_type="GDDR6X",
    ),
    "RTX-3070-Ti": GPUSpec(
        model="RTX-3070-Ti",
        memory_gb=8.0,
        peak_tflops_fp16=87.0,
        memory_bandwidth_gb_s=608.0,
        l2_cache_mb=4.0,
        architecture="Ampere",
        launch_year=2021,
        typical_tdp_w=290,
        memory_type="GDDR6X",
    ),
    "RTX-3070": GPUSpec(
        model="RTX-3070",
        memory_gb=8.0,
        peak_tflops_fp16=81.2,
        memory_bandwidth_gb_s=448.0,
        l2_cache_mb=4.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=220,
        memory_type="GDDR6",
    ),
    "RTX-3060-Ti": GPUSpec(
        model="RTX-3060-Ti",
        memory_gb=8.0,
        peak_tflops_fp16=65.0,
        memory_bandwidth_gb_s=448.0,
        l2_cache_mb=4.0,
        architecture="Ampere",
        launch_year=2020,
        typical_tdp_w=200,
        memory_type="GDDR6",
    ),
}


_AMD_GPUS: dict[str, GPUSpec] = {
    "MI300X": GPUSpec(
        model="MI300X",
        memory_gb=192.0,
        peak_tflops_fp16=1307.0,
        memory_bandwidth_gb_s=5300.0,
        l2_cache_mb=256.0,
        architecture="CDNA 3",
        launch_year=2023,
        typical_tdp_w=750,
        memory_type="HBM3",
        peak_tflops_by_precision={
            "fp16": 1307.0, "bf16": 1307.0, "fp8": 2614.0, "int8": 2614.0
        },
    ),
    "MI250X": GPUSpec(
        model="MI250X",
        memory_gb=128.0,
        peak_tflops_fp16=383.0,
        memory_bandwidth_gb_s=3277.0,
        l2_cache_mb=16.0,
        architecture="CDNA 2",
        launch_year=2021,
        typical_tdp_w=560,
        memory_type="HBM2e",
    ),
    "MI210": GPUSpec(
        model="MI210",
        memory_gb=64.0,
        peak_tflops_fp16=181.0,
        memory_bandwidth_gb_s=1638.0,
        l2_cache_mb=8.0,
        architecture="CDNA 2",
        launch_year=2022,
        typical_tdp_w=300,
        memory_type="HBM2e",
    ),
}


# Legacy aliases — retained so old plans / configs keep resolving. Prefer
# the canonical names above.
_LEGACY_ALIASES = {
    "H100-80GB": "H100-SXM-80GB",
    "A100-80GB": "A100-SXM-80GB",
    "A100-40GB": "A100-SXM-40GB",
    "V100-32GB": "V100-SXM-32GB",
    "GB200-NVL72": "GB200-Superchip",  # not identical, but closest legacy target
}


class GPUDatabase:
    """Read-only registry of GPU specs."""

    _ALL_GPUS: dict[str, GPUSpec] = {**_NVIDIA_GPUS, **_AMD_GPUS}

    @classmethod
    def get(cls, model: str) -> Optional[GPUSpec]:
        if model in cls._ALL_GPUS:
            return cls._ALL_GPUS[model]
        # Try legacy alias.
        alias = _LEGACY_ALIASES.get(model)
        if alias:
            return cls._ALL_GPUS.get(alias)
        return None

    @classmethod
    def list_models(cls) -> list[str]:
        return sorted(cls._ALL_GPUS.keys())

    @classmethod
    def list_gpus(cls) -> list[GPUSpec]:
        return list(cls._ALL_GPUS.values())

    @classmethod
    def list_by_vendor(cls, vendor: str) -> list[str]:
        vendor = vendor.lower()
        if vendor == "nvidia":
            return sorted(_NVIDIA_GPUS.keys())
        if vendor == "amd":
            return sorted(_AMD_GPUS.keys())
        return []

    @classmethod
    def to_hardware_spec(
        cls,
        model: str,
        num_gpus: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        **kwargs: object,
    ) -> HardwareSpec:
        spec = cls.get(model)
        if spec is None:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown GPU model: {model}. Available models: {available}"
            )

        return HardwareSpec(
            gpu_model=spec.model,
            num_gpus=num_gpus,
            gpu_memory_gb=spec.memory_gb,
            peak_tflops=spec.peak_tflops_fp16,
            peak_tflops_by_precision=dict(spec.peak_tflops_by_precision),
            memory_bandwidth_gb_s=spec.memory_bandwidth_gb_s,
            l2_cache_mb=spec.l2_cache_mb,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **kwargs,  # type: ignore[arg-type]
        )


def get_gpu_spec(model: str) -> Optional[GPUSpec]:
    """Convenience wrapper."""
    return GPUDatabase.get(model)
