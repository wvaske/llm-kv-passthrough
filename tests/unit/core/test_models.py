"""Unit tests for kvbench.core.models module."""

from __future__ import annotations

import pytest

from kvbench.core.models import (
    MODEL_PROFILES,
    ModelProfile,
    calculate_kv_cache_requirements,
    get_model_profile,
    get_model_profile_info,
    list_model_profiles,
)


class TestModelProfile:
    """Tests for ModelProfile dataclass."""

    def test_basic_profile_creation(self) -> None:
        """Test basic ModelProfile creation."""
        profile = ModelProfile(
            name="Test Model",
            layers=32,
            hidden=4096,
            kv_heads=8,
            intermediate=14336,
        )
        assert profile.name == "Test Model"
        assert profile.layers == 32
        assert profile.hidden == 4096
        assert profile.kv_heads == 8
        assert profile.intermediate == 14336

    def test_num_heads_default_calculation(self) -> None:
        """Test num_heads defaults based on hidden dimension."""
        profile = ModelProfile(
            name="Test Model",
            layers=32,
            hidden=4096,
            kv_heads=8,
            intermediate=14336,
        )
        # Default: hidden // 128 = 4096 // 128 = 32
        assert profile.num_heads == 32

    def test_num_heads_custom_value(self) -> None:
        """Test custom num_heads value."""
        profile = ModelProfile(
            name="Test Model",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=64,
            intermediate=14336,
        )
        assert profile.num_heads == 64

    def test_head_dim_calculation(self) -> None:
        """Test head_dim property calculation."""
        profile = ModelProfile(
            name="Test Model",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
        )
        # head_dim = hidden // num_heads = 4096 // 32 = 128
        assert profile.head_dim == 128

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        profile = ModelProfile(
            name="Test Model",
            layers=32,
            hidden=4096,
            kv_heads=8,
            intermediate=14336,
        )
        assert profile.vocab_size == 128256
        assert profile.max_seq_len == 131072
        assert profile.dtype == "bf16"

    def test_bytes_per_element_bf16(self) -> None:
        """Test bytes_per_element for bf16."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            intermediate=14336,
            dtype="bf16",
        )
        assert profile.bytes_per_element == 2

    def test_bytes_per_element_fp16(self) -> None:
        """Test bytes_per_element for fp16."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            intermediate=14336,
            dtype="fp16",
        )
        assert profile.bytes_per_element == 2

    def test_bytes_per_element_fp8(self) -> None:
        """Test bytes_per_element for fp8."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            intermediate=14336,
            dtype="fp8",
        )
        assert profile.bytes_per_element == 1

    def test_kv_cache_bytes_per_token(self) -> None:
        """Test KV cache bytes per token per layer calculation."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
            dtype="bf16",
        )
        # KV per token per layer = 2 * kv_heads * head_dim * bytes
        # = 2 * 8 * 128 * 2 = 4096 bytes
        assert profile.kv_cache_bytes_per_token == 4096

    def test_total_kv_cache_bytes_per_token(self) -> None:
        """Test total KV cache bytes per token (all layers)."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
            dtype="bf16",
        )
        # Total = kv_per_layer * layers = 4096 * 32 = 131072 bytes
        assert profile.total_kv_cache_bytes_per_token == 131072

    def test_kv_cache_size_bytes(self) -> None:
        """Test total KV cache size for given tokens."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
            dtype="bf16",
        )
        # For 1000 tokens: 131072 * 1000 = 131072000 bytes
        assert profile.kv_cache_size_bytes(1000) == 131072000

    def test_kv_cache_size_gb(self) -> None:
        """Test KV cache size in GB."""
        profile = ModelProfile(
            name="Test",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
            dtype="bf16",
        )
        size_gb = profile.kv_cache_size_gb(1000)
        expected_gb = 131072000 / (1024**3)
        assert abs(size_gb - expected_gb) < 0.0001

    def test_estimated_params_billions(self) -> None:
        """Test estimated parameters calculation."""
        profile = get_model_profile("llama-3.1-8b")
        # Should be approximately 8 billion
        assert 7.0 < profile.estimated_params_billions < 10.0

    def test_model_size_gb(self) -> None:
        """Test model size estimation in GB."""
        profile = get_model_profile("llama-3.1-8b")
        # 8B params * 2 bytes (bf16) = ~16 GB
        assert 14.0 < profile.model_size_gb < 20.0


class TestModelProfileRegistry:
    """Tests for model profile registry functions."""

    def test_model_profiles_dict_exists(self) -> None:
        """Test MODEL_PROFILES dictionary exists and has entries."""
        assert isinstance(MODEL_PROFILES, dict)
        assert len(MODEL_PROFILES) >= 5  # At least 5 profiles as per spec

    def test_required_profiles_exist(self) -> None:
        """Test all required profiles exist."""
        required_profiles = [
            "llama-3.1-8b",
            "llama-3.1-70b",
            "llama-3.1-405b",
            "qwen-2.5-7b",
            "qwen-2.5-72b",
        ]
        for profile_name in required_profiles:
            assert profile_name in MODEL_PROFILES, f"Missing profile: {profile_name}"

    def test_get_model_profile_valid(self) -> None:
        """Test get_model_profile with valid profile name."""
        profile = get_model_profile("llama-3.1-8b")
        assert isinstance(profile, ModelProfile)
        assert profile.name == "Llama 3.1 8B"

    def test_get_model_profile_invalid(self) -> None:
        """Test get_model_profile with invalid profile name."""
        with pytest.raises(KeyError, match="Unknown model profile"):
            get_model_profile("INVALID_MODEL")

    def test_get_model_profile_error_message(self) -> None:
        """Test error message includes available profiles."""
        with pytest.raises(KeyError) as exc_info:
            get_model_profile("INVALID_MODEL")
        assert "Available profiles:" in str(exc_info.value)

    def test_list_model_profiles(self) -> None:
        """Test list_model_profiles returns sorted list."""
        profiles = list_model_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) >= 5
        assert profiles == sorted(profiles)

    def test_list_model_profiles_contains_required(self) -> None:
        """Test list_model_profiles contains all required profiles."""
        profiles = list_model_profiles()
        assert "llama-3.1-8b" in profiles
        assert "qwen-2.5-72b" in profiles


class TestGetModelProfileInfo:
    """Tests for get_model_profile_info function."""

    def test_get_model_profile_info_returns_dict(self) -> None:
        """Test get_model_profile_info returns a dictionary."""
        info = get_model_profile_info("llama-3.1-8b")
        assert isinstance(info, dict)

    def test_get_model_profile_info_keys(self) -> None:
        """Test get_model_profile_info returns expected keys."""
        info = get_model_profile_info("llama-3.1-8b")
        expected_keys = [
            "name",
            "layers",
            "hidden",
            "kv_heads",
            "num_heads",
            "intermediate",
            "vocab_size",
            "max_seq_len",
            "head_dim",
            "dtype",
            "kv_cache_bytes_per_token",
            "estimated_params_billions",
            "model_size_gb",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_get_model_profile_info_invalid(self) -> None:
        """Test get_model_profile_info with invalid profile."""
        with pytest.raises(KeyError):
            get_model_profile_info("INVALID_MODEL")


class TestCalculateKVCacheRequirements:
    """Tests for calculate_kv_cache_requirements function."""

    def test_calculate_requirements_basic(self) -> None:
        """Test basic KV cache requirements calculation."""
        result = calculate_kv_cache_requirements(
            model_name="llama-3.1-8b",
            max_batch_size=8,
        )
        assert "model" in result
        assert "max_batch_size" in result
        assert "max_seq_len" in result
        assert "per_request_gb" in result
        assert "total_gb" in result
        assert "bytes_per_token" in result

    def test_calculate_requirements_custom_seq_len(self) -> None:
        """Test with custom sequence length."""
        result = calculate_kv_cache_requirements(
            model_name="llama-3.1-8b",
            max_batch_size=1,
            max_seq_len=4096,
        )
        assert result["max_seq_len"] == 4096

    def test_calculate_requirements_default_seq_len(self) -> None:
        """Test default sequence length uses model's max."""
        result = calculate_kv_cache_requirements(
            model_name="llama-3.1-8b",
            max_batch_size=1,
        )
        assert result["max_seq_len"] == 131072  # Llama 3.1 default

    def test_calculate_requirements_invalid_model(self) -> None:
        """Test with invalid model name."""
        with pytest.raises(KeyError):
            calculate_kv_cache_requirements(
                model_name="INVALID_MODEL",
                max_batch_size=1,
            )


class TestLlama31Profiles:
    """Tests for Llama 3.1 family profiles."""

    def test_llama_8b_specs(self) -> None:
        """Test Llama 3.1 8B profile specifications."""
        profile = get_model_profile("llama-3.1-8b")
        assert profile.layers == 32
        assert profile.hidden == 4096
        assert profile.kv_heads == 8
        assert profile.num_heads == 32
        assert profile.intermediate == 14336
        assert profile.vocab_size == 128256

    def test_llama_70b_specs(self) -> None:
        """Test Llama 3.1 70B profile specifications."""
        profile = get_model_profile("llama-3.1-70b")
        assert profile.layers == 80
        assert profile.hidden == 8192
        assert profile.kv_heads == 8
        assert profile.num_heads == 64
        assert profile.intermediate == 28672

    def test_llama_405b_specs(self) -> None:
        """Test Llama 3.1 405B profile specifications."""
        profile = get_model_profile("llama-3.1-405b")
        assert profile.layers == 126
        assert profile.hidden == 16384
        assert profile.kv_heads == 8
        assert profile.num_heads == 128
        assert profile.intermediate == 53248


class TestQwenProfiles:
    """Tests for Qwen 2.5 family profiles."""

    def test_qwen_7b_specs(self) -> None:
        """Test Qwen 2.5 7B profile specifications."""
        profile = get_model_profile("qwen-2.5-7b")
        assert profile.layers == 28
        assert profile.hidden == 3584
        assert profile.kv_heads == 4
        assert profile.intermediate == 18944
        assert profile.vocab_size == 152064

    def test_qwen_72b_specs(self) -> None:
        """Test Qwen 2.5 72B profile specifications."""
        profile = get_model_profile("qwen-2.5-72b")
        assert profile.layers == 80
        assert profile.hidden == 8192
        assert profile.kv_heads == 8
        assert profile.intermediate == 29568


class TestKVCacheCalculations:
    """Tests for KV cache size calculations across models."""

    def test_kv_cache_scales_with_sequence_length(self) -> None:
        """Test that KV cache scales linearly with sequence length."""
        profile = get_model_profile("llama-3.1-8b")
        size_1k = profile.kv_cache_size_bytes(1000)
        size_2k = profile.kv_cache_size_bytes(2000)
        assert size_2k == 2 * size_1k

    def test_kv_cache_scales_with_layers(self) -> None:
        """Test that KV cache scales with number of layers."""
        profile_8b = get_model_profile("llama-3.1-8b")  # 32 layers
        profile_70b = get_model_profile("llama-3.1-70b")  # 80 layers

        bytes_per_token_8b = profile_8b.total_kv_cache_bytes_per_token
        bytes_per_token_70b = profile_70b.total_kv_cache_bytes_per_token

        # 70B should have more per-token bytes due to more layers
        # Note: 70B also has different hidden size, so it's not exactly proportional
        assert bytes_per_token_70b > bytes_per_token_8b

    def test_fp8_reduces_kv_cache_size(self) -> None:
        """Test that FP8 reduces KV cache size by half compared to BF16."""
        profile_bf16 = ModelProfile(
            name="Test BF16",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
            dtype="bf16",
        )
        profile_fp8 = ModelProfile(
            name="Test FP8",
            layers=32,
            hidden=4096,
            kv_heads=8,
            num_heads=32,
            intermediate=14336,
            dtype="fp8",
        )
        assert profile_fp8.kv_cache_size_bytes(1000) == profile_bf16.kv_cache_size_bytes(1000) / 2
