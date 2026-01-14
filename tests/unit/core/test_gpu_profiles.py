"""Unit tests for kvbench.core.gpu_profiles module."""

from __future__ import annotations

import pytest

from kvbench.core.gpu_profiles import (
    GPU_PROFILES,
    GPUProfile,
    get_gpu_profile,
    get_gpu_profile_info,
    list_gpu_profiles,
)


class TestGPUProfile:
    """Tests for GPUProfile dataclass."""

    def test_basic_profile_creation(self) -> None:
        """Test basic GPUProfile creation."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,
            hbm_bandwidth_tb_s=2.0,
            hbm_capacity_gb=80,
        )
        assert profile.name == "Test GPU"
        assert profile.bf16_tflops == 1000.0
        assert profile.hbm_bandwidth_tb_s == 2.0
        assert profile.hbm_capacity_gb == 80

    def test_fp16_defaults_to_bf16(self) -> None:
        """Test that fp16_tflops defaults to bf16_tflops."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=500.0,
            hbm_bandwidth_tb_s=1.0,
            hbm_capacity_gb=40,
        )
        assert profile.fp16_tflops == 500.0

    def test_fp16_custom_value(self) -> None:
        """Test that fp16_tflops can be set independently."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=500.0,
            fp16_tflops=600.0,
            hbm_bandwidth_tb_s=1.0,
            hbm_capacity_gb=40,
        )
        assert profile.bf16_tflops == 500.0
        assert profile.fp16_tflops == 600.0

    def test_hbm_bandwidth_gb_s_property(self) -> None:
        """Test hbm_bandwidth_gb_s conversion."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,
            hbm_bandwidth_tb_s=3.35,
            hbm_capacity_gb=80,
        )
        assert profile.hbm_bandwidth_gb_s == 3350.0

    def test_arithmetic_intensity_threshold(self) -> None:
        """Test arithmetic intensity threshold calculation."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,  # 1000 TFLOPS
            hbm_bandwidth_tb_s=1.0,  # 1000 GB/s
            hbm_capacity_gb=80,
        )
        # threshold = (1000 TFLOPS * 1000) / (1.0 TB/s * 1000 GB) = 1000 ops/byte
        assert profile.arithmetic_intensity_threshold == 1000.0

    def test_default_pcie_bandwidth(self) -> None:
        """Test default PCIe bandwidth."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,
            hbm_bandwidth_tb_s=2.0,
            hbm_capacity_gb=80,
        )
        assert profile.pcie_bandwidth_gb_s == 64.0

    def test_custom_pcie_bandwidth(self) -> None:
        """Test custom PCIe bandwidth."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,
            hbm_bandwidth_tb_s=2.0,
            hbm_capacity_gb=80,
            pcie_bandwidth_gb_s=128.0,
        )
        assert profile.pcie_bandwidth_gb_s == 128.0

    def test_optional_fields(self) -> None:
        """Test optional fields."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,
            hbm_bandwidth_tb_s=2.0,
            hbm_capacity_gb=80,
            nvlink_bandwidth_gb_s=900.0,
            tdp_watts=700,
        )
        assert profile.nvlink_bandwidth_gb_s == 900.0
        assert profile.tdp_watts == 700

    def test_profile_is_frozen(self) -> None:
        """Test that GPUProfile is immutable."""
        profile = GPUProfile(
            name="Test GPU",
            bf16_tflops=1000.0,
            hbm_bandwidth_tb_s=2.0,
            hbm_capacity_gb=80,
        )
        with pytest.raises(AttributeError):
            profile.name = "Modified"  # type: ignore[misc]


class TestGPUProfileRegistry:
    """Tests for GPU profile registry functions."""

    def test_gpu_profiles_dict_exists(self) -> None:
        """Test GPU_PROFILES dictionary exists and has entries."""
        assert isinstance(GPU_PROFILES, dict)
        assert len(GPU_PROFILES) >= 6  # At least 6 profiles as per spec

    def test_required_profiles_exist(self) -> None:
        """Test all required profiles exist."""
        required_profiles = [
            "H100_SXM",
            "H100_PCIe",
            "H200_SXM",
            "A100_SXM",
            "L4",
            "L40S",
        ]
        for profile_name in required_profiles:
            assert profile_name in GPU_PROFILES, f"Missing profile: {profile_name}"

    def test_get_gpu_profile_valid(self) -> None:
        """Test get_gpu_profile with valid profile name."""
        profile = get_gpu_profile("H100_SXM")
        assert isinstance(profile, GPUProfile)
        assert profile.name == "NVIDIA H100 SXM"

    def test_get_gpu_profile_invalid(self) -> None:
        """Test get_gpu_profile with invalid profile name."""
        with pytest.raises(KeyError, match="Unknown GPU profile"):
            get_gpu_profile("INVALID_GPU")

    def test_get_gpu_profile_error_message(self) -> None:
        """Test error message includes available profiles."""
        with pytest.raises(KeyError) as exc_info:
            get_gpu_profile("INVALID_GPU")
        assert "Available profiles:" in str(exc_info.value)

    def test_list_gpu_profiles(self) -> None:
        """Test list_gpu_profiles returns sorted list."""
        profiles = list_gpu_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) >= 6
        assert profiles == sorted(profiles)

    def test_list_gpu_profiles_contains_required(self) -> None:
        """Test list_gpu_profiles contains all required profiles."""
        profiles = list_gpu_profiles()
        assert "H100_SXM" in profiles
        assert "A100_SXM" in profiles


class TestGetGPUProfileInfo:
    """Tests for get_gpu_profile_info function."""

    def test_get_gpu_profile_info_returns_dict(self) -> None:
        """Test get_gpu_profile_info returns a dictionary."""
        info = get_gpu_profile_info("H100_SXM")
        assert isinstance(info, dict)

    def test_get_gpu_profile_info_keys(self) -> None:
        """Test get_gpu_profile_info returns expected keys."""
        info = get_gpu_profile_info("H100_SXM")
        expected_keys = [
            "name",
            "bf16_tflops",
            "fp16_tflops",
            "hbm_bandwidth_tb_s",
            "hbm_bandwidth_gb_s",
            "hbm_capacity_gb",
            "nvlink_bandwidth_gb_s",
            "pcie_bandwidth_gb_s",
            "tdp_watts",
            "arithmetic_intensity_threshold",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_get_gpu_profile_info_invalid(self) -> None:
        """Test get_gpu_profile_info with invalid profile."""
        with pytest.raises(KeyError):
            get_gpu_profile_info("INVALID_GPU")


class TestH100SXMProfile:
    """Tests for H100 SXM profile specifications."""

    def test_h100_sxm_specs(self) -> None:
        """Test H100 SXM profile has correct specifications."""
        profile = get_gpu_profile("H100_SXM")
        assert profile.bf16_tflops == 1979.0
        assert profile.hbm_bandwidth_tb_s == 3.35
        assert profile.hbm_capacity_gb == 80
        assert profile.nvlink_bandwidth_gb_s == 900.0
        assert profile.tdp_watts == 700


class TestH100PCIeProfile:
    """Tests for H100 PCIe profile specifications."""

    def test_h100_pcie_specs(self) -> None:
        """Test H100 PCIe profile has correct specifications."""
        profile = get_gpu_profile("H100_PCIe")
        assert profile.bf16_tflops == 1513.0
        assert profile.hbm_bandwidth_tb_s == 2.0
        assert profile.hbm_capacity_gb == 80
        assert profile.pcie_bandwidth_gb_s == 128.0  # PCIe Gen5


class TestH200SXMProfile:
    """Tests for H200 SXM profile specifications."""

    def test_h200_sxm_specs(self) -> None:
        """Test H200 SXM profile has correct specifications."""
        profile = get_gpu_profile("H200_SXM")
        assert profile.bf16_tflops == 1979.0
        assert profile.hbm_bandwidth_tb_s == 4.8
        assert profile.hbm_capacity_gb == 141


class TestA100SXMProfile:
    """Tests for A100 SXM profile specifications."""

    def test_a100_sxm_specs(self) -> None:
        """Test A100 SXM profile has correct specifications."""
        profile = get_gpu_profile("A100_SXM")
        assert profile.bf16_tflops == 312.0
        assert profile.hbm_bandwidth_tb_s == 2.0
        assert profile.hbm_capacity_gb == 80


class TestL4Profile:
    """Tests for L4 profile specifications."""

    def test_l4_specs(self) -> None:
        """Test L4 profile has correct specifications."""
        profile = get_gpu_profile("L4")
        assert profile.bf16_tflops == 121.0
        assert profile.hbm_bandwidth_tb_s == 0.3
        assert profile.hbm_capacity_gb == 24


class TestL40SProfile:
    """Tests for L40S profile specifications."""

    def test_l40s_specs(self) -> None:
        """Test L40S profile has correct specifications."""
        profile = get_gpu_profile("L40S")
        assert profile.bf16_tflops == 362.0
        assert profile.hbm_bandwidth_tb_s == 0.864
        assert profile.hbm_capacity_gb == 48
