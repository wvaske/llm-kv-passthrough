"""
KV-Bench Core Module.

This module provides the foundational components for KV-Bench:
- Configuration system with environment variable support
- GPU hardware profiles for latency emulation
- Model profiles for KV cache sizing
"""

from kvbench.core.config import (
    ConnectorConfig,
    DistributedConfig,
    GPUEmulationConfig,
    KVBenchConfig,
    MetricsConfig,
    ResourceLimits,
    ServerConfig,
    StorageConfig,
)
from kvbench.core.gpu_profiles import (
    GPU_PROFILES,
    GPUProfile,
    get_gpu_profile,
    get_gpu_profile_info,
    list_gpu_profiles,
)
from kvbench.core.models import (
    MODEL_PROFILES,
    ModelProfile,
    calculate_kv_cache_requirements,
    get_model_profile,
    get_model_profile_info,
    list_model_profiles,
)

__all__ = [
    # Config
    "KVBenchConfig",
    "ResourceLimits",
    "StorageConfig",
    "ConnectorConfig",
    "GPUEmulationConfig",
    "ServerConfig",
    "DistributedConfig",
    "MetricsConfig",
    # GPU Profiles
    "GPUProfile",
    "GPU_PROFILES",
    "get_gpu_profile",
    "get_gpu_profile_info",
    "list_gpu_profiles",
    # Model Profiles
    "ModelProfile",
    "MODEL_PROFILES",
    "get_model_profile",
    "get_model_profile_info",
    "list_model_profiles",
    "calculate_kv_cache_requirements",
]
