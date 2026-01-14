"""
Model Profile Definitions for KV-Bench.

This module provides LLM model profiles used for KV cache sizing and
latency emulation calculations. Each profile contains the architectural
parameters needed to compute memory requirements and FLOPs.

The profiles include:
- Llama 3.1 family (8B, 70B, 405B)
- Qwen 2.5 family (7B, 72B)
- Mistral family
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelProfile:
    """Hardware profile for an LLM model.

    This dataclass contains the architectural parameters needed for
    KV cache sizing and latency calculations.

    Attributes:
        name: Human-readable name of the model.
        layers: Number of transformer layers.
        hidden: Hidden dimension size.
        kv_heads: Number of KV attention heads (GQA).
        num_heads: Total number of attention heads.
        intermediate: FFN intermediate dimension.
        vocab_size: Vocabulary size.
        max_seq_len: Maximum sequence length.
        head_dim: Dimension per attention head (computed from hidden/num_heads).
        dtype: Default data type for weights and KV cache.
    """

    name: str
    layers: int
    hidden: int
    kv_heads: int
    intermediate: int
    num_heads: int | None = None
    vocab_size: int = 128256  # Common Llama 3 vocab size
    max_seq_len: int = 131072  # 128k context
    dtype: Literal["fp16", "bf16", "fp8"] = "bf16"

    def __post_init__(self) -> None:
        """Set num_heads if not specified."""
        if self.num_heads is None:
            # Default: same as hidden // head_dim where head_dim = 128
            object.__setattr__(self, "num_heads", self.hidden // 128)

    @property
    def head_dim(self) -> int:
        """Compute dimension per attention head."""
        if self.num_heads is None:
            return 128  # Default head dimension
        return self.hidden // self.num_heads

    @property
    def kv_head_dim(self) -> int:
        """Compute KV dimension per head (same as head_dim for standard models)."""
        return self.head_dim

    @property
    def bytes_per_element(self) -> int:
        """Get bytes per element based on dtype."""
        dtype_sizes = {"fp16": 2, "bf16": 2, "fp8": 1}
        return dtype_sizes[self.dtype]

    @property
    def kv_cache_bytes_per_token(self) -> int:
        """Calculate KV cache size in bytes per token per layer.

        For each token, we store K and V tensors:
        - K: [kv_heads, head_dim] = kv_heads * head_dim elements
        - V: [kv_heads, head_dim] = kv_heads * head_dim elements
        - Total: 2 * kv_heads * head_dim * bytes_per_element

        Returns:
            Bytes per token per layer.
        """
        return 2 * self.kv_heads * self.head_dim * self.bytes_per_element

    @property
    def total_kv_cache_bytes_per_token(self) -> int:
        """Calculate total KV cache size in bytes per token (all layers).

        Returns:
            Total bytes per token across all layers.
        """
        return self.kv_cache_bytes_per_token * self.layers

    def kv_cache_size_bytes(self, num_tokens: int) -> int:
        """Calculate total KV cache size for a given number of tokens.

        Args:
            num_tokens: Number of tokens in the sequence.

        Returns:
            Total KV cache size in bytes.
        """
        return self.total_kv_cache_bytes_per_token * num_tokens

    def kv_cache_size_gb(self, num_tokens: int) -> float:
        """Calculate total KV cache size in GB for a given number of tokens.

        Args:
            num_tokens: Number of tokens in the sequence.

        Returns:
            Total KV cache size in GB.
        """
        return self.kv_cache_size_bytes(num_tokens) / (1024**3)

    @property
    def estimated_params_billions(self) -> float:
        """Estimate total model parameters in billions.

        This is an approximation based on architecture dimensions.
        The actual count may vary slightly based on implementation details.

        Returns:
            Estimated parameters in billions.
        """
        # Embedding: vocab_size * hidden
        embedding_params = self.vocab_size * self.hidden

        # Per layer:
        # - QKV projection: hidden * (num_heads + 2 * kv_heads) * head_dim
        # - Output projection: hidden * hidden
        # - FFN: hidden * intermediate * 3 (up, gate, down projections)
        # - Layer norms: hidden * 2
        assert self.num_heads is not None
        qkv_params = self.hidden * (self.num_heads + 2 * self.kv_heads) * self.head_dim
        output_params = self.hidden * self.hidden
        ffn_params = self.hidden * self.intermediate * 3
        norm_params = self.hidden * 2

        per_layer_params = qkv_params + output_params + ffn_params + norm_params
        total_layer_params = per_layer_params * self.layers

        # Final layer norm + LM head
        final_params = self.hidden + self.hidden * self.vocab_size

        total_params = embedding_params + total_layer_params + final_params
        return total_params / 1e9

    @property
    def model_size_gb(self) -> float:
        """Estimate model size in GB based on dtype.

        Returns:
            Estimated model size in GB.
        """
        return self.estimated_params_billions * self.bytes_per_element


# Model Profile Registry
# These profiles are based on official model specifications
MODEL_PROFILES: dict[str, ModelProfile] = {
    # Llama 3.1 family
    "llama-3.1-8b": ModelProfile(
        name="Llama 3.1 8B",
        layers=32,
        hidden=4096,
        kv_heads=8,
        num_heads=32,
        intermediate=14336,
        vocab_size=128256,
        max_seq_len=131072,
    ),
    "llama-3.1-70b": ModelProfile(
        name="Llama 3.1 70B",
        layers=80,
        hidden=8192,
        kv_heads=8,
        num_heads=64,
        intermediate=28672,
        vocab_size=128256,
        max_seq_len=131072,
    ),
    "llama-3.1-405b": ModelProfile(
        name="Llama 3.1 405B",
        layers=126,
        hidden=16384,
        kv_heads=8,
        num_heads=128,
        intermediate=53248,
        vocab_size=128256,
        max_seq_len=131072,
    ),
    # Qwen 2.5 family
    "qwen-2.5-7b": ModelProfile(
        name="Qwen 2.5 7B",
        layers=28,
        hidden=3584,
        kv_heads=4,
        num_heads=28,
        intermediate=18944,
        vocab_size=152064,
        max_seq_len=131072,
    ),
    "qwen-2.5-72b": ModelProfile(
        name="Qwen 2.5 72B",
        layers=80,
        hidden=8192,
        kv_heads=8,
        num_heads=64,
        intermediate=29568,
        vocab_size=152064,
        max_seq_len=131072,
    ),
    # Mistral family
    "mistral-7b": ModelProfile(
        name="Mistral 7B",
        layers=32,
        hidden=4096,
        kv_heads=8,
        num_heads=32,
        intermediate=14336,
        vocab_size=32000,
        max_seq_len=32768,
    ),
    "mixtral-8x7b": ModelProfile(
        name="Mixtral 8x7B",
        layers=32,
        hidden=4096,
        kv_heads=8,
        num_heads=32,
        intermediate=14336,  # Per expert
        vocab_size=32000,
        max_seq_len=32768,
    ),
}


def get_model_profile(name: str) -> ModelProfile:
    """Get a model profile by name.

    Args:
        name: Name of the model profile (e.g., "llama-3.1-8b").

    Returns:
        The ModelProfile for the specified model.

    Raises:
        KeyError: If the model profile is not found.
    """
    if name not in MODEL_PROFILES:
        available = ", ".join(sorted(MODEL_PROFILES.keys()))
        raise KeyError(f"Unknown model profile: {name}. Available profiles: {available}")
    return MODEL_PROFILES[name]


def list_model_profiles() -> list[str]:
    """List all available model profile names.

    Returns:
        List of model profile names.
    """
    return sorted(MODEL_PROFILES.keys())


def get_model_profile_info(name: str) -> dict:
    """Get detailed information about a model profile.

    Args:
        name: Name of the model profile.

    Returns:
        Dictionary with model profile information.
    """
    profile = get_model_profile(name)
    return {
        "name": profile.name,
        "layers": profile.layers,
        "hidden": profile.hidden,
        "kv_heads": profile.kv_heads,
        "num_heads": profile.num_heads,
        "intermediate": profile.intermediate,
        "vocab_size": profile.vocab_size,
        "max_seq_len": profile.max_seq_len,
        "head_dim": profile.head_dim,
        "dtype": profile.dtype,
        "kv_cache_bytes_per_token": profile.total_kv_cache_bytes_per_token,
        "estimated_params_billions": round(profile.estimated_params_billions, 2),
        "model_size_gb": round(profile.model_size_gb, 2),
    }


def calculate_kv_cache_requirements(
    model_name: str,
    max_batch_size: int,
    max_seq_len: int | None = None,
) -> dict:
    """Calculate KV cache memory requirements for a model.

    Args:
        model_name: Name of the model profile.
        max_batch_size: Maximum batch size.
        max_seq_len: Maximum sequence length (uses model default if not specified).

    Returns:
        Dictionary with memory requirements.
    """
    profile = get_model_profile(model_name)
    seq_len = max_seq_len or profile.max_seq_len

    per_request_bytes = profile.kv_cache_size_bytes(seq_len)
    total_bytes = per_request_bytes * max_batch_size

    return {
        "model": model_name,
        "max_batch_size": max_batch_size,
        "max_seq_len": seq_len,
        "per_request_gb": per_request_bytes / (1024**3),
        "total_gb": total_bytes / (1024**3),
        "bytes_per_token": profile.total_kv_cache_bytes_per_token,
    }
