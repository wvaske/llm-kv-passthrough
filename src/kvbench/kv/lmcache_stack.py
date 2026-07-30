"""
Real LMCache integration for KV-Bench.

This stack drives the actual LMCache engine (lmcache.v1). LMCache owns the
entire storage control plane: token chunking, prefix hashing, CPU/disk/
remote tiering, eviction, serialization, and every byte of storage I/O.
KV-Bench only supplies token sequences and mock GPU-side KV tensors sized
authentically for the emulated model.

Storage is configured through LMCache's own application configuration:

1. ``lmcache_config_file`` in the KV-Bench config — passed verbatim to
   ``LMCacheEngineConfig.from_file()``. This is the canonical way to
   select and tune backends (local CPU, local disk, remote URL, ...).
2. Otherwise LMCache's ``LMCACHE_*`` environment variables via
   ``LMCacheEngineConfig.from_env()``.

KV-Bench deliberately has no storage settings of its own.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from kvbench.core.models import get_model_profile
from kvbench.kv.base import KVStack, KVStackStats

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_LMCACHE_INSTALL_HINT = (
    "The 'lmcache' package is required to run KV-Bench servers. "
    "Install it with: pip install 'kvbench[lmcache]' (or: pip install lmcache). "
    "LMCache runs CPU-only; no GPU is required."
)


class LMCacheStack(KVStack):
    """KV stack backed by the real LMCache engine.

    Attributes:
        instance_id: LMCache engine instance ID.
        model_profile: Name of the KV-Bench model profile used to size
            KV tensors (layers, KV heads, head dim, dtype).
        stats: Operation statistics.
    """

    def __init__(
        self,
        model_profile: str,
        instance_id: str = "kvbench",
        config_file: Path | str | None = None,
        trace_file: Path | str | None = None,
        random_fill: bool = True,
        random_pool_mb: int = 256,
    ) -> None:
        self.instance_id = instance_id
        self.model_profile = model_profile
        self.config_file = config_file
        self.trace_file = trace_file
        self.random_fill = random_fill
        self.random_pool_mb = random_pool_mb
        self.stats = KVStackStats()
        self._engine: Any = None
        self._connector: Any = None
        self._chunk_size: int | None = None
        self._lmcache_config: Any = None
        self._trace_recorder: Any = None

    @property
    def chunk_size(self) -> int:
        if self._chunk_size is None:
            raise RuntimeError("LMCacheStack not started; call start() first")
        return self._chunk_size

    async def start(self) -> None:
        if self._engine is not None:
            return
        # The engine builder spins up worker threads and allocates the CPU
        # buffer pool; keep it off the event loop.
        await asyncio.to_thread(self._start_engine)

    def _start_engine(self) -> None:
        try:
            import torch
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.metadata import LMCacheMetadata
        except ImportError as e:
            raise RuntimeError(_LMCACHE_INSTALL_HINT) from e

        from kvbench.kv.mock_gpu import MockGPUConnector

        if self.config_file is not None:
            lmcache_config = LMCacheEngineConfig.from_file(str(self.config_file))
            logger.info(f"LMCache configured from file: {self.config_file}")
        else:
            lmcache_config = LMCacheEngineConfig.from_env()
            logger.info("LMCache configured from LMCACHE_* environment / defaults")

        profile = get_model_profile(self.model_profile)
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8": torch.float8_e4m3fn,
        }
        kv_dtype = dtype_map[profile.dtype]
        chunk_size = lmcache_config.chunk_size

        metadata = LMCacheMetadata(
            model_name=self.model_profile,
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=kv_dtype,
            kv_shape=(profile.layers, 2, chunk_size, profile.kv_heads, profile.head_dim),
            chunk_size=chunk_size,
        )
        self._connector = MockGPUConnector(
            num_layers=profile.layers,
            hidden_dim=profile.kv_heads * profile.head_dim,
            dtype=kv_dtype,
            random_fill=self.random_fill,
            random_pool_mb=self.random_pool_mb,
        )
        engine = LMCacheEngineBuilder.get_or_create(
            self.instance_id,
            lmcache_config,
            metadata,
            self._connector,
            broadcast_fn=lambda _tensor, _src: None,
            broadcast_object_fn=lambda obj, _src: obj,
        )
        engine.post_init()
        self._engine = engine
        self._chunk_size = chunk_size
        self._lmcache_config = lmcache_config

        if self.trace_file is not None:
            from kvbench.trace.recorder import TraceRecorder, install_lmcache_trace

            self._trace_recorder = TraceRecorder(self.trace_file)
            wrapped = install_lmcache_trace(engine, self._trace_recorder)
            logger.info(f"KV trace enabled -> {self.trace_file} (backends: {wrapped})")
        logger.info(
            f"LMCache engine started (instance={self.instance_id}, "
            f"model={self.model_profile}, chunk_size={chunk_size}, "
            f"local_cpu={lmcache_config.local_cpu}, "
            f"local_disk={lmcache_config.local_disk}, "
            f"remote={lmcache_config.remote_url})"
        )

    def _require_engine(self) -> Any:
        if self._engine is None:
            raise RuntimeError("LMCacheStack not started; call start() first")
        return self._engine

    async def lookup(self, tokens: list[int]) -> int:
        engine = self._require_engine()
        try:
            hit = await asyncio.to_thread(engine.lookup, tokens)
        except Exception:
            self.stats.errors += 1
            raise
        self.stats.lookups += 1
        self.stats.lookup_tokens += len(tokens)
        self.stats.hit_tokens += hit
        return int(hit)

    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        engine = self._require_engine()
        try:
            await asyncio.to_thread(self._store_sync, engine, tokens, skip_leading)
        except Exception:
            self.stats.errors += 1
            raise
        self.stats.stores += 1
        self.stats.stored_tokens += len(tokens) - skip_leading

    def _store_sync(self, engine: Any, tokens: list[int], skip_leading: int) -> None:
        import torch

        mask = None
        if skip_leading > 0:
            mask = torch.ones(len(tokens), dtype=torch.bool)
            mask[:skip_leading] = False
        kv_source = self._connector.new_kv_tensor(len(tokens))
        engine.store(torch.tensor(tokens, dtype=torch.int64), mask=mask, kv_source=kv_source)

    async def retrieve(self, tokens: list[int]) -> int:
        engine = self._require_engine()
        try:
            retrieved = await asyncio.to_thread(self._retrieve_sync, engine, tokens)
        except Exception:
            self.stats.errors += 1
            raise
        self.stats.retrieves += 1
        self.stats.retrieved_tokens += retrieved
        return retrieved

    def _retrieve_sync(self, engine: Any, tokens: list[int]) -> int:
        import torch

        kv_dest = self._connector.new_kv_tensor(len(tokens))
        ret_mask = engine.retrieve(torch.tensor(tokens, dtype=torch.int64), kv_dest=kv_dest)
        return int(ret_mask.sum())

    def capacity_info(self) -> dict[str, Any]:
        """Report configured tier capacities and KV sizing for this stack.

        Returns:
            Dict with chunk_size, bytes_per_token, chunk_bytes, per-tier
            capacity in bytes, and total_capacity_bytes (0 when the tier is
            disabled). Requires a started stack.
        """
        cfg = self._lmcache_config
        if cfg is None:
            raise RuntimeError("LMCacheStack not started; call start() first")
        profile = get_model_profile(self.model_profile)
        bytes_per_token = profile.total_kv_cache_bytes_per_token
        gb = 1024**3
        local_cpu_bytes = int(cfg.max_local_cpu_size * gb) if cfg.local_cpu else 0
        local_disk_bytes = (
            int(cfg.max_local_disk_size * gb) if getattr(cfg, "local_disk", None) else 0
        )
        return {
            "chunk_size": self.chunk_size,
            "bytes_per_token": bytes_per_token,
            "chunk_bytes": bytes_per_token * self.chunk_size,
            "vocab_size": profile.vocab_size,
            "local_cpu_enabled": bool(cfg.local_cpu),
            "local_cpu_capacity_bytes": local_cpu_bytes,
            "local_disk_path": str(cfg.local_disk) if getattr(cfg, "local_disk", None) else None,
            "local_disk_capacity_bytes": local_disk_bytes,
            "remote_url": cfg.remote_url,
            "total_capacity_bytes": local_cpu_bytes + local_disk_bytes,
        }

    def usage_info(self) -> dict[str, dict[str, Any]]:
        """Report per-backend live usage from the running LMCache engine.

        Returns:
            {backend_class_name: {usage_bytes, keys}} for backends that
            expose them (best-effort; LMCache internals).
        """
        engine = self._require_engine()
        result: dict[str, dict[str, Any]] = {}
        storage_manager = getattr(engine, "storage_manager", None)
        backends = getattr(storage_manager, "storage_backends", None) or {}
        for _name, backend in backends.items():
            info: dict[str, Any] = {}
            usage = getattr(backend, "usage", None)
            if isinstance(usage, (int, float)):
                info["usage_bytes"] = int(usage)
            keys_dict = getattr(backend, "dict", None)
            if keys_dict is not None:
                try:
                    info["keys"] = len(keys_dict)
                except TypeError:
                    pass
            if info:
                result[type(backend).__name__] = info
        return result

    @property
    def trace_recorder(self) -> Any:
        """The active TraceRecorder, or None when tracing is disabled."""
        return self._trace_recorder

    async def close(self) -> None:
        if self._engine is None:
            return
        await asyncio.to_thread(self._close_sync)
        if self._trace_recorder is not None:
            self._trace_recorder.close()
            self._trace_recorder = None

    def _close_sync(self) -> None:
        from lmcache.v1.cache_engine import LMCacheEngineBuilder

        try:
            LMCacheEngineBuilder.destroy(self.instance_id)
        except Exception as e:
            logger.warning(f"Error destroying LMCache engine {self.instance_id}: {e}")
        self._engine = None
        self._connector = None
        self._chunk_size = None
        logger.info(
            f"LMCache engine stopped (instance={self.instance_id}, "
            f"lookups={self.stats.lookups}, stores={self.stats.stores}, "
            f"retrieves={self.stats.retrieves}, "
            f"token_hit_rate={self.stats.token_hit_rate:.1%})"
        )

    def __repr__(self) -> str:
        return (
            f"LMCacheStack(instance_id={self.instance_id!r}, "
            f"model={self.model_profile!r}, "
            f"config_file={str(self.config_file) if self.config_file else None!r})"
        )
