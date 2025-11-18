"""
GPU specifications database.

Comprehensive database of GPU specifications for common inference hardware.
Data sourced from official NVIDIA, AMD, and vendor specifications (2024-2025).
"""

from dataclasses import dataclass
from typing import Optional

from kv_planner.domain import HardwareSpec


@dataclass(frozen=True)
class GPUSpec:
    """
    GPU specification data.

    Attributes:
        model: GPU model identifier
        memory_gb: Memory per GPU in GB
        peak_tflops_fp16: Peak FP16 TFLOPS
        hbm_bandwidth_gb_s: HBM bandwidth in GB/s
        l2_cache_mb: L2 cache size in MB
        architecture: GPU architecture name
        launch_year: Year of launch
        typical_tdp_w: Typical thermal design power in watts
    """

    model: str
    memory_gb: float
    peak_tflops_fp16: float
    hbm_bandwidth_gb_s: float
    l2_cache_mb: float
    architecture: str
    launch_year: int
    typical_tdp_w: int


class GPUDatabase:
    """
    Database of GPU specifications.

    Data sources:
    - NVIDIA official specs (2024-2025)
    - AMD official specs (2024-2025)
    - Vendor datasheets
    """

    # NVIDIA GPUs
    _NVIDIA_GPUS = {
        # H100 Series (Hopper architecture, 2022-2023)
        "H100-80GB": GPUSpec(
            model="H100-80GB",
            memory_gb=80.0,
            peak_tflops_fp16=989.0,  # With Tensor Cores
            hbm_bandwidth_gb_s=3350.0,  # HBM3: 3.35 TB/s
            l2_cache_mb=60.0,
            architecture="Hopper",
            launch_year=2022,
            typical_tdp_w=700,
        ),
        "H100-NVL-94GB": GPUSpec(
            model="H100-NVL-94GB",
            memory_gb=94.0,
            peak_tflops_fp16=1979.0,  # Dual GPU
            hbm_bandwidth_gb_s=7800.0,  # Combined bandwidth
            l2_cache_mb=60.0,
            architecture="Hopper",
            launch_year=2023,
            typical_tdp_w=700,
        ),
        # A100 Series (Ampere architecture, 2020)
        "A100-80GB": GPUSpec(
            model="A100-80GB",
            memory_gb=80.0,
            peak_tflops_fp16=312.0,
            hbm_bandwidth_gb_s=2039.0,  # HBM2e: ~2 TB/s
            l2_cache_mb=40.0,
            architecture="Ampere",
            launch_year=2020,
            typical_tdp_w=400,
        ),
        "A100-40GB": GPUSpec(
            model="A100-40GB",
            memory_gb=40.0,
            peak_tflops_fp16=312.0,
            hbm_bandwidth_gb_s=1555.0,  # HBM2: 1.5 TB/s
            l2_cache_mb=40.0,
            architecture="Ampere",
            launch_year=2020,
            typical_tdp_w=400,
        ),
        # L40S (Ada Lovelace, 2023)
        "L40S": GPUSpec(
            model="L40S",
            memory_gb=48.0,
            peak_tflops_fp16=362.0,
            hbm_bandwidth_gb_s=864.0,
            l2_cache_mb=96.0,
            architecture="Ada Lovelace",
            launch_year=2023,
            typical_tdp_w=350,
        ),
        # L4 (Ada Lovelace, 2023)
        "L4": GPUSpec(
            model="L4",
            memory_gb=24.0,
            peak_tflops_fp16=242.0,
            hbm_bandwidth_gb_s=300.0,
            l2_cache_mb=48.0,
            architecture="Ada Lovelace",
            launch_year=2023,
            typical_tdp_w=72,
        ),
        # V100 (Volta architecture, 2017)
        "V100-32GB": GPUSpec(
            model="V100-32GB",
            memory_gb=32.0,
            peak_tflops_fp16=125.0,
            hbm_bandwidth_gb_s=900.0,
            l2_cache_mb=6.0,
            architecture="Volta",
            launch_year=2017,
            typical_tdp_w=300,
        ),
        # GB200 Grace Hopper (2024)
        "GB200-NVL72": GPUSpec(
            model="GB200-NVL72",
            memory_gb=192.0,  # Per GPU pair
            peak_tflops_fp16=2000.0,
            hbm_bandwidth_gb_s=8000.0,  # HBM3e: 8 TB/s
            l2_cache_mb=228.0,
            architecture="Blackwell",
            launch_year=2024,
            typical_tdp_w=1000,
        ),
        # RTX 50 Series (Blackwell architecture, 2025)
        "RTX-5090": GPUSpec(
            model="RTX-5090",
            memory_gb=32.0,
            peak_tflops_fp16=104.8,  # FP16 (1:1 with FP32 on Blackwell)
            hbm_bandwidth_gb_s=1792.0,  # GDDR7: 1.792 TB/s
            l2_cache_mb=96.0,
            architecture="Blackwell",
            launch_year=2025,
            typical_tdp_w=575,
        ),
        "RTX-5080": GPUSpec(
            model="RTX-5080",
            memory_gb=16.0,
            peak_tflops_fp16=56.3,  # FP16 estimated from 10,752 cores @ 2.62 GHz
            hbm_bandwidth_gb_s=960.0,  # GDDR7: 30 Gbps, 256-bit
            l2_cache_mb=64.0,
            architecture="Blackwell",
            launch_year=2025,
            typical_tdp_w=360,
        ),
        "RTX-5070-Ti": GPUSpec(
            model="RTX-5070-Ti",
            memory_gb=16.0,
            peak_tflops_fp16=44.0,  # FP16 (based on FP32 44.257 TFLOPS)
            hbm_bandwidth_gb_s=896.0,  # GDDR7: 28 Gbps, 256-bit
            l2_cache_mb=64.0,  # Sources report 48-64 MB
            architecture="Blackwell",
            launch_year=2025,
            typical_tdp_w=300,
        ),
        "RTX-5070": GPUSpec(
            model="RTX-5070",
            memory_gb=12.0,
            peak_tflops_fp16=31.0,  # FP16 (based on FP32 ~31 TFLOPS)
            hbm_bandwidth_gb_s=672.0,  # GDDR7: 28 Gbps, 192-bit
            l2_cache_mb=48.0,
            architecture="Blackwell",
            launch_year=2025,
            typical_tdp_w=250,
        ),
        "RTX-5060-Laptop": GPUSpec(
            model="RTX-5060-Laptop",
            memory_gb=8.0,
            peak_tflops_fp16=23.2,  # FP16 (3328 CUDA cores)
            hbm_bandwidth_gb_s=448.0,  # GDDR7: 128-bit bus
            l2_cache_mb=32.0,  # Estimated based on chip size
            architecture="Blackwell",
            launch_year=2025,
            typical_tdp_w=115,  # Typical laptop TDP (can range 50-115W)
        ),
        # RTX 40 Series (Ada Lovelace architecture, 2022-2024)
        "RTX-4090": GPUSpec(
            model="RTX-4090",
            memory_gb=24.0,
            peak_tflops_fp16=82.58,  # FP16 (1:1 with FP32)
            hbm_bandwidth_gb_s=1008.0,  # GDDR6X: 1.01 TB/s
            l2_cache_mb=72.0,
            architecture="Ada Lovelace",
            launch_year=2022,
            typical_tdp_w=450,
        ),
        "RTX-4080-Super": GPUSpec(
            model="RTX-4080-Super",
            memory_gb=16.0,
            peak_tflops_fp16=52.22,
            hbm_bandwidth_gb_s=736.3,  # GDDR6X
            l2_cache_mb=64.0,
            architecture="Ada Lovelace",
            launch_year=2024,
            typical_tdp_w=320,
        ),
        "RTX-4080": GPUSpec(
            model="RTX-4080",
            memory_gb=16.0,
            peak_tflops_fp16=48.74,
            hbm_bandwidth_gb_s=716.8,  # GDDR6X
            l2_cache_mb=64.0,
            architecture="Ada Lovelace",
            launch_year=2022,
            typical_tdp_w=320,
        ),
        "RTX-4070-Ti-Super": GPUSpec(
            model="RTX-4070-Ti-Super",
            memory_gb=16.0,
            peak_tflops_fp16=44.1,
            hbm_bandwidth_gb_s=672.0,  # GDDR6X, 256-bit bus
            l2_cache_mb=48.0,
            architecture="Ada Lovelace",
            launch_year=2024,
            typical_tdp_w=285,
        ),
        "RTX-4070-Ti": GPUSpec(
            model="RTX-4070-Ti",
            memory_gb=12.0,
            peak_tflops_fp16=40.0,
            hbm_bandwidth_gb_s=504.2,  # GDDR6X, 192-bit bus
            l2_cache_mb=48.0,
            architecture="Ada Lovelace",
            launch_year=2023,
            typical_tdp_w=285,
        ),
        "RTX-4070-Super": GPUSpec(
            model="RTX-4070-Super",
            memory_gb=12.0,
            peak_tflops_fp16=35.5,
            hbm_bandwidth_gb_s=504.2,  # GDDR6X
            l2_cache_mb=48.0,
            architecture="Ada Lovelace",
            launch_year=2024,
            typical_tdp_w=220,
        ),
        "RTX-4070": GPUSpec(
            model="RTX-4070",
            memory_gb=12.0,
            peak_tflops_fp16=29.15,
            hbm_bandwidth_gb_s=504.2,  # GDDR6X
            l2_cache_mb=36.0,
            architecture="Ada Lovelace",
            launch_year=2023,
            typical_tdp_w=200,
        ),
        # RTX 30 Series (Ampere architecture, 2020-2022)
        "RTX-3090-Ti": GPUSpec(
            model="RTX-3090-Ti",
            memory_gb=24.0,
            peak_tflops_fp16=40.0,  # FP16 (1:1 with FP32)
            hbm_bandwidth_gb_s=1008.0,  # GDDR6X: 21 Gbps
            l2_cache_mb=6.0,
            architecture="Ampere",
            launch_year=2022,
            typical_tdp_w=450,
        ),
        "RTX-3090": GPUSpec(
            model="RTX-3090",
            memory_gb=24.0,
            peak_tflops_fp16=35.58,
            hbm_bandwidth_gb_s=936.2,  # GDDR6X: 19.5 Gbps
            l2_cache_mb=6.0,
            architecture="Ampere",
            launch_year=2020,
            typical_tdp_w=350,
        ),
        "RTX-3080-Ti": GPUSpec(
            model="RTX-3080-Ti",
            memory_gb=12.0,
            peak_tflops_fp16=34.1,
            hbm_bandwidth_gb_s=912.4,  # GDDR6X
            l2_cache_mb=6.0,
            architecture="Ampere",
            launch_year=2021,
            typical_tdp_w=350,
        ),
        "RTX-3080-12GB": GPUSpec(
            model="RTX-3080-12GB",
            memory_gb=12.0,
            peak_tflops_fp16=30.6,  # Estimated based on 8960 CUDA cores
            hbm_bandwidth_gb_s=912.0,  # GDDR6X: 19 Gbps, 384-bit
            l2_cache_mb=6.0,
            architecture="Ampere",
            launch_year=2022,
            typical_tdp_w=350,
        ),
        "RTX-3080": GPUSpec(
            model="RTX-3080",
            memory_gb=10.0,
            peak_tflops_fp16=29.77,
            hbm_bandwidth_gb_s=760.3,  # GDDR6X: 320-bit
            l2_cache_mb=5.0,
            architecture="Ampere",
            launch_year=2020,
            typical_tdp_w=320,
        ),
        "RTX-3070-Ti": GPUSpec(
            model="RTX-3070-Ti",
            memory_gb=8.0,
            peak_tflops_fp16=21.75,
            hbm_bandwidth_gb_s=608.0,  # GDDR6X: 256-bit
            l2_cache_mb=4.0,
            architecture="Ampere",
            launch_year=2021,
            typical_tdp_w=290,
        ),
        "RTX-3070": GPUSpec(
            model="RTX-3070",
            memory_gb=8.0,
            peak_tflops_fp16=20.31,
            hbm_bandwidth_gb_s=448.0,  # GDDR6: 256-bit
            l2_cache_mb=4.0,
            architecture="Ampere",
            launch_year=2020,
            typical_tdp_w=220,
        ),
    }

    # AMD GPUs
    _AMD_GPUS = {
        "MI300X": GPUSpec(
            model="MI300X",
            memory_gb=192.0,
            peak_tflops_fp16=1307.0,
            hbm_bandwidth_gb_s=5300.0,  # HBM3: 5.3 TB/s
            l2_cache_mb=256.0,
            architecture="CDNA 3",
            launch_year=2023,
            typical_tdp_w=750,
        ),
        "MI250X": GPUSpec(
            model="MI250X",
            memory_gb=128.0,
            peak_tflops_fp16=383.0,
            hbm_bandwidth_gb_s=3277.0,
            l2_cache_mb=16.0,
            architecture="CDNA 2",
            launch_year=2021,
            typical_tdp_w=560,
        ),
    }

    # Combine all GPUs
    _ALL_GPUS = {**_NVIDIA_GPUS, **_AMD_GPUS}

    @classmethod
    def get(cls, model: str) -> Optional[GPUSpec]:
        """
        Get GPU specification by model name.

        Args:
            model: GPU model identifier (e.g., "H100-80GB", "A100-80GB")

        Returns:
            GPUSpec if found, None otherwise
        """
        return cls._ALL_GPUS.get(model)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all available GPU models."""
        return sorted(cls._ALL_GPUS.keys())

    @classmethod
    def list_gpus(cls) -> list[GPUSpec]:
        """List all GPU specifications."""
        return [spec for spec in cls._ALL_GPUS.values()]

    @classmethod
    def list_by_vendor(cls, vendor: str) -> list[str]:
        """
        List GPU models by vendor.

        Args:
            vendor: "nvidia" or "amd"

        Returns:
            List of model identifiers
        """
        vendor = vendor.lower()
        if vendor == "nvidia":
            return sorted(cls._NVIDIA_GPUS.keys())
        elif vendor == "amd":
            return sorted(cls._AMD_GPUS.keys())
        else:
            return []

    @classmethod
    def to_hardware_spec(
        cls,
        model: str,
        num_gpus: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        **kwargs
    ) -> HardwareSpec:
        """
        Create HardwareSpec from GPU model.

        Args:
            model: GPU model identifier
            num_gpus: Number of GPUs
            tensor_parallel_size: TP size
            pipeline_parallel_size: PP size
            **kwargs: Additional HardwareSpec parameters

        Returns:
            HardwareSpec instance

        Raises:
            ValueError: If model not found
        """
        spec = cls.get(model)
        if spec is None:
            available = ", ".join(cls.list_models())
            raise ValueError(
                f"Unknown GPU model: {model}. "
                f"Available models: {available}"
            )

        return HardwareSpec(
            gpu_model=model,
            num_gpus=num_gpus,
            gpu_memory_gb=spec.memory_gb,
            peak_tflops=spec.peak_tflops_fp16,
            hbm_bandwidth_gb_s=spec.hbm_bandwidth_gb_s,
            l2_cache_mb=spec.l2_cache_mb,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            **kwargs
        )


def get_gpu_spec(model: str) -> Optional[GPUSpec]:
    """
    Convenience function to get GPU specification.

    Args:
        model: GPU model identifier

    Returns:
        GPUSpec if found, None otherwise
    """
    return GPUDatabase.get(model)
