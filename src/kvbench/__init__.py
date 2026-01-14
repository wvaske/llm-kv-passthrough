"""
KV-Bench: Distributed KV Cache Benchmarking System.

A distributed mock LLM serving system for benchmarking KV cache management without GPUs.
"""

from kvbench.core.config import KVBenchConfig
from kvbench.core.gpu_profiles import GPUProfile, get_gpu_profile, list_gpu_profiles
from kvbench.core.models import ModelProfile, get_model_profile, list_model_profiles

__version__ = "1.0.0"

__all__ = [
    "KVBenchConfig",
    "GPUProfile",
    "get_gpu_profile",
    "list_gpu_profiles",
    "ModelProfile",
    "get_model_profile",
    "list_model_profiles",
    "__version__",
]
