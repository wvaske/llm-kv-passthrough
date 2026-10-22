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

    if stack == "hicache":
        from kvbench.kv.hicache_stack import HiCacheStack

        return HiCacheStack(config)

    if stack == "lmcache":
        return LMCacheStack(
            model_profile=config.server.model_profile,
            instance_id=config.instance_id,
            config_file=config.kv.lmcache_config_file,
            trace_file=config.kv.trace_file,
            random_fill=config.kv.random_fill,
            tp_size=config.gpu.tp_size,
            pp_size=config.timing.pp_size,
            model=config.model,
            max_cpu_memory_gb=config.kv.max_cpu_memory_gb,
            max_chunk_mb=config.kv.max_chunk_mb,
            seed=config.kv.seed,
            active_ranks=config.kv.active_ranks,
            payload_generator=config.kv.payload_generator,
            reuse_history_entries=config.kv.lmcache_reuse_history_entries,
        )

    if stack == "lmcache_mp":
        from kvbench.kv.lmcache_mp_stack import LMCacheMPStack

        return LMCacheMPStack(
            model_profile=config.server.model_profile,
            instance_id=config.instance_id,
            config_file=config.kv.lmcache_mp_config_file,
            random_fill=config.kv.random_fill,
            tp_size=config.gpu.tp_size,
            pp_size=config.timing.pp_size,
            model=config.model,
            max_chunk_mb=config.kv.max_chunk_mb,
            seed=config.kv.seed,
            active_ranks=config.kv.active_ranks,
        )

    if stack == "kvbm":
        raise ValueError(_KVBM_UNAVAILABLE)

    raise ValueError(f"Unknown KV stack: {stack!r}. Supported: lmcache, lmcache_mp, hicache")
