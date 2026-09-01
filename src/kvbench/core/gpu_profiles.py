"""
GPU Profile Definitions for KV-Bench.

This module provides GPU hardware profiles used for latency emulation.
Each profile contains the key specifications needed for roofline model calculations.

The profiles include:
- NVIDIA H100 (SXM and PCIe variants)
- NVIDIA H200
- NVIDIA A100
- NVIDIA L4
- NVIDIA L40S
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUProfile:
    """Hardware profile for a GPU.

    This dataclass contains the key specifications needed for latency
    calculations using the roofline model.

    Attributes:
        name: Human-readable name of the GPU.
        bf16_tflops: BF16 tensor core performance in TFLOPS.
        fp16_tflops: FP16 tensor core performance in TFLOPS.
        hbm_bandwidth_tb_s: HBM bandwidth in TB/s.
        hbm_capacity_gb: HBM capacity in GB.
        nvlink_bandwidth_gb_s: NVLink bandwidth in GB/s (if available).
        pcie_bandwidth_gb_s: PCIe bandwidth in GB/s.
        tdp_watts: Thermal design power in watts.
    """

    name: str
    bf16_tflops: float
    hbm_bandwidth_tb_s: float
    hbm_capacity_gb: int
    fp16_tflops: float | None = None
    nvlink_bandwidth_gb_s: float | None = None
    pcie_bandwidth_gb_s: float = 64.0  # PCIe Gen5 x16 default
    tdp_watts: int | None = None

    def __post_init__(self) -> None:
        """Set fp16_tflops to match bf16_tflops if not specified."""
        if self.fp16_tflops is None:
            # Use object.__setattr__ since the dataclass is frozen
            object.__setattr__(self, "fp16_tflops", self.bf16_tflops)

    @property
    def hbm_bandwidth_gb_s(self) -> float:
        """Return HBM bandwidth in GB/s for convenience."""
        return self.hbm_bandwidth_tb_s * 1000

    @property
    def arithmetic_intensity_threshold(self) -> float:
        """Calculate the arithmetic intensity threshold (ops/byte).

        This is the point where the GPU transitions from being memory-bound
        to compute-bound. Below this threshold, performance is limited by
        memory bandwidth. Above it, performance is limited by compute.

        Returns:
            Arithmetic intensity threshold in operations per byte.
        """
        # Convert TFLOPS to GFLOPS for consistent units with GB/s
        compute_gflops = self.bf16_tflops * 1000
        memory_gb_s = self.hbm_bandwidth_gb_s
        return compute_gflops / memory_gb_s


# GPU Profile Registry
# These profiles are based on official NVIDIA specifications.
# All TFLOPS values are DENSE (non-sparsity) tensor-core numbers so that
# cross-GPU comparisons are consistent; NVIDIA marketing numbers for
# Hopper/Ada are typically the 2:4 structured-sparsity figures (2x dense).
GPU_PROFILES: dict[str, GPUProfile] = {
    "B300_SXM": GPUProfile(
        name="NVIDIA B300 SXM (Blackwell Ultra)",
        bf16_tflops=2250.0,          # dense BF16 (HGX B300: 36 PF sparse / 8 GPUs / 2)
        hbm_bandwidth_tb_s=8.0,      # HBM3e
        hbm_capacity_gb=288,
        nvlink_bandwidth_gb_s=1800.0,  # NVLink 5
        tdp_watts=1400,
    ),
    "H100_SXM": GPUProfile(
        name="NVIDIA H100 SXM",
        bf16_tflops=989.5,
        hbm_bandwidth_tb_s=3.35,
        hbm_capacity_gb=80,
        nvlink_bandwidth_gb_s=900.0,
        tdp_watts=700,
    ),
    "H100_PCIe": GPUProfile(
        name="NVIDIA H100 PCIe",
        bf16_tflops=756.5,
        hbm_bandwidth_tb_s=2.0,
        hbm_capacity_gb=80,
        nvlink_bandwidth_gb_s=600.0,
        pcie_bandwidth_gb_s=128.0,  # PCIe Gen5 x16
        tdp_watts=350,
    ),
    "H200_SXM": GPUProfile(
        name="NVIDIA H200 SXM",
        bf16_tflops=989.5,
        hbm_bandwidth_tb_s=4.8,
        hbm_capacity_gb=141,
        nvlink_bandwidth_gb_s=900.0,
        tdp_watts=700,
    ),
    "A100_SXM": GPUProfile(
        name="NVIDIA A100 SXM",
        bf16_tflops=312.0,
        hbm_bandwidth_tb_s=2.0,
        hbm_capacity_gb=80,
        nvlink_bandwidth_gb_s=600.0,
        tdp_watts=400,
    ),
    "A100_PCIe": GPUProfile(
        name="NVIDIA A100 PCIe",
        bf16_tflops=312.0,
        hbm_bandwidth_tb_s=2.0,
        hbm_capacity_gb=80,
        pcie_bandwidth_gb_s=64.0,  # PCIe Gen4 x16
        tdp_watts=300,
    ),
    "L4": GPUProfile(
        name="NVIDIA L4",
        bf16_tflops=60.5,
        hbm_bandwidth_tb_s=0.3,
        hbm_capacity_gb=24,
        pcie_bandwidth_gb_s=64.0,
        tdp_watts=72,
    ),
    "L40S": GPUProfile(
        name="NVIDIA L40S",
        bf16_tflops=181.05,
        hbm_bandwidth_tb_s=0.864,
        hbm_capacity_gb=48,
        pcie_bandwidth_gb_s=64.0,
        tdp_watts=350,
    ),
}


def get_gpu_profile(name: str) -> GPUProfile:
    """Get a GPU profile by name.

    Args:
        name: Name of the GPU profile (e.g., "H100_SXM").

    Returns:
        The GPUProfile for the specified GPU.

    Raises:
        KeyError: If the GPU profile is not found.
    """
    if name not in GPU_PROFILES:
        available = ", ".join(sorted(GPU_PROFILES.keys()))
        raise KeyError(f"Unknown GPU profile: {name}. Available profiles: {available}")
    return GPU_PROFILES[name]


def list_gpu_profiles() -> list[str]:
    """List all available GPU profile names.

    Returns:
        List of GPU profile names.
    """
    return sorted(GPU_PROFILES.keys())


def get_gpu_profile_info(name: str) -> dict:
    """Get detailed information about a GPU profile.

    Args:
        name: Name of the GPU profile.

    Returns:
        Dictionary with GPU profile information.
    """
    profile = get_gpu_profile(name)
    return {
        "name": profile.name,
        "bf16_tflops": profile.bf16_tflops,
        "fp16_tflops": profile.fp16_tflops,
        "hbm_bandwidth_tb_s": profile.hbm_bandwidth_tb_s,
        "hbm_bandwidth_gb_s": profile.hbm_bandwidth_gb_s,
        "hbm_capacity_gb": profile.hbm_capacity_gb,
        "nvlink_bandwidth_gb_s": profile.nvlink_bandwidth_gb_s,
        "pcie_bandwidth_gb_s": profile.pcie_bandwidth_gb_s,
        "tdp_watts": profile.tdp_watts,
        "arithmetic_intensity_threshold": profile.arithmetic_intensity_threshold,
    }
