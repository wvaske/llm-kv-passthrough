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
        tp_size: int = 1,
    ) -> None:
        self.instance_id = instance_id
        self.model_profile = model_profile
        self.config_file = config_file
        self.trace_file = trace_file
        self.random_fill = random_fill
        self.random_pool_mb = random_pool_mb
        # Tensor parallelism: emulate one LMCache engine per TP rank, exactly
        # as real vLLM runs one worker-side engine per rank. Each rank stores
        # its own KV shard (kv_heads/tp heads, or 1 replicated head when
        # tp > kv_heads — vLLM replicates KV heads in that regime), keyed
        # with its worker_id, so a logical chunk becomes tp_size files.
        self.tp_size = max(1, int(tp_size))
        self.stats = KVStackStats()
        self._engines: list[Any] = []
        self._connectors: list[Any] = []
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

        tp = self.tp_size
        # vLLM shards KV heads across ranks; when tp > kv_heads each rank
        # holds one replicated head (total stored bytes exceed logical KV).
        heads_per_rank = max(1, profile.kv_heads // tp)
        kv_layers = profile.effective_kv_layers

        if self.trace_file is not None:
            from kvbench.trace.recorder import TraceRecorder

            self._trace_recorder = TraceRecorder(self.trace_file)

        for rank in range(tp):
            metadata = LMCacheMetadata(
                model_name=self.model_profile,
                world_size=tp,
                local_world_size=tp,
                worker_id=rank,
                local_worker_id=rank,
                kv_dtype=kv_dtype,
                kv_shape=(kv_layers, 2, chunk_size, heads_per_rank, profile.head_dim),
                chunk_size=chunk_size,
            )
            connector = MockGPUConnector(
                num_layers=kv_layers,
                hidden_dim=heads_per_rank * profile.head_dim,
                dtype=kv_dtype,
                random_fill=self.random_fill,
                random_pool_mb=self.random_pool_mb,
            )
            rank_instance = (
                self.instance_id if tp == 1 else f"{self.instance_id}-r{rank}"
            )
            engine = LMCacheEngineBuilder.get_or_create(
                rank_instance,
                lmcache_config,
                metadata,
                connector,
                broadcast_fn=lambda _tensor, _src: None,
                broadcast_object_fn=lambda obj, _src: obj,
            )
            engine.post_init()
            self._engines.append(engine)
            self._connectors.append(connector)

            if self._trace_recorder is not None:
                from kvbench.trace.recorder import install_lmcache_trace

                wrapped = install_lmcache_trace(engine, self._trace_recorder)
                logger.info(
                    f"KV trace enabled for rank {rank} -> {self.trace_file} "
                    f"(backends: {wrapped})"
                )

        self._engine = self._engines[0]
        self._connector = self._connectors[0]
        self._chunk_size = chunk_size
        self._lmcache_config = lmcache_config
        logger.info(
            f"LMCache engine(s) started (instance={self.instance_id}, tp={tp}, "
            f"model={self.model_profile}, chunk_size={chunk_size}, "
            f"heads_per_rank={heads_per_rank}, kv_layers={kv_layers}, "
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
        self._require_engine()
        try:
            # All TP ranks store their shard concurrently, as real vLLM
            # workers do.
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        self._store_sync, engine, connector, tokens, skip_leading
                    )
                    for engine, connector in zip(self._engines, self._connectors)
                )
            )
        except Exception:
            self.stats.errors += 1
            raise
        self.stats.stores += 1
        self.stats.stored_tokens += len(tokens) - skip_leading

    def _store_sync(
        self, engine: Any, connector: Any, tokens: list[int], skip_leading: int
    ) -> None:
        import torch

        mask = None
        if skip_leading > 0:
            mask = torch.ones(len(tokens), dtype=torch.bool)
            mask[:skip_leading] = False
        kv_source = connector.new_kv_tensor(len(tokens))
        engine.store(torch.tensor(tokens, dtype=torch.int64), mask=mask, kv_source=kv_source)

    async def retrieve(self, tokens: list[int]) -> int:
        self._require_engine()
        try:
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(self._retrieve_sync, engine, connector, tokens)
                    for engine, connector in zip(self._engines, self._connectors)
                )
            )
            retrieved = results[0] if results else 0
        except Exception:
            self.stats.errors += 1
            raise
        self.stats.retrieves += 1
        self.stats.retrieved_tokens += retrieved
        return retrieved

    def _retrieve_sync(self, engine: Any, connector: Any, tokens: list[int]) -> int:
        import torch

        kv_dest = connector.new_kv_tensor(len(tokens))
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
        # Stored bytes may exceed logical KV when tp > kv_heads (vLLM
        # replicates KV heads across ranks in that regime).
        heads_per_rank = max(1, profile.kv_heads // self.tp_size)
        stored_bytes_per_token = (
            2
            * profile.effective_kv_layers
            * heads_per_rank
            * profile.head_dim
            * profile.bytes_per_element
            * self.tp_size
        )
        return {
            "chunk_size": self.chunk_size,
            "bytes_per_token": bytes_per_token,
            "chunk_bytes": bytes_per_token * self.chunk_size,
            "tp_size": self.tp_size,
            "stored_bytes_per_token": stored_bytes_per_token,
            "files_per_chunk": self.tp_size,
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
