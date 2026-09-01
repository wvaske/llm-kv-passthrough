"""
KV stack factory.

Creates the KV management stack the servers talk to. There are no mock or
passthrough stacks here: the benchmark only measures real KV management
stacks driving real storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kvbench.kv.lmcache_stack import LMCacheStack

if TYPE_CHECKING:
    from kvbench.core.config import KVBenchConfig
    from kvbench.kv.base import KVStack

# Verified against kvbm 1.2.1 on a CPU-only host (2026-07): KvbmWorker —
# KVBM's entire data plane — panics loading libcuda.so at construction,
# BlockManager requires a KvbmLeader whose init barrier waits on those
# workers, and no DYN_KVBM_* flag provides a CPU or mock device mode.
# Details and upstream requirements: docs/architecture/connectors.md.
_KVBM_UNAVAILABLE = (
    "KV stack 'kvbm' is not yet supported: NVIDIA Dynamo's KV Block Manager "
    "cannot perform storage I/O on a GPU-less host (its data plane, "
    "KvbmWorker, requires the NVIDIA CUDA driver at initialization), and "
    "KV-Bench targets CPU-only benchmark nodes. KVBM support requires "
    "either a CPU device mode in upstream kvbm or a GPU-enabled KV-Bench "
    "build. See docs/architecture/connectors.md#kvbm for the full analysis. "
    "Use stack 'lmcache' instead."
)


def create_kv_stack(config: KVBenchConfig) -> KVStack:
    """Create a KV stack from the KV-Bench configuration.

    Args:
        config: KV-Bench configuration; config.kv selects and configures
            the stack, config.server.model_profile sizes the KV tensors.

    Returns:
        An unstarted KVStack (call await stack.start()).

    Raises:
        ValueError: If the stack is unknown or not yet supported.
    """
    stack = config.kv.stack

    if stack == "lmcache":
        return LMCacheStack(
            model_profile=config.server.model_profile,
            instance_id=config.instance_id,
            config_file=config.kv.lmcache_config_file,
            trace_file=config.kv.trace_file,
            random_fill=config.kv.random_fill,
            random_pool_mb=config.kv.random_pool_mb,
            tp_size=config.gpu.tp_size,
        )

    if stack == "kvbm":
        raise ValueError(_KVBM_UNAVAILABLE)

    raise ValueError(f"Unknown KV stack: {stack!r}. Supported: lmcache")
