"""Real LMCache classic-path adapter with bounded, serialized rank operations.

Each logical TP/PP rank owns an engine. Public operations finish on all ranks
before returning. Cancellation waits for the blocking operation to finish so
pins/resources cannot outlive the request's cleanup scope. LMCache's backend
workers still perform physical writes asynchronously; submitted bytes are not
reported as confirmed durable bytes.
"""

from __future__ import annotations

import asyncio
import gc
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from kvbench.core.models import ModelProfile, get_model_profile
from kvbench.kv.base import KVStack, KVStackStats
from kvbench.trace.recorder import TraceRecorder

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class LMCacheStack(KVStack):
    def __init__(self, model_profile: str, instance_id: str = "kvbench",
                 config_file: Path | str | None = None, trace_file: Path | str | None = None,
                 random_fill: bool = True, tp_size: int = 1, pp_size: int = 1,
                 model: ModelProfile | None = None, max_cpu_memory_gb: float = 2.0,
                 max_chunk_mb: int = 256, seed: int = 24301):
        self.instance_id = instance_id
        self.model_profile = model_profile
        self.model = model or get_model_profile(model_profile)
        self.config_file = config_file
        self.trace_file = trace_file
        self.random_fill = random_fill
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.layout = self.model.rank_layout(tp_size, pp_size)
        self.max_cpu_memory_gb = max_cpu_memory_gb
        self.max_chunk_mb = max_chunk_mb
        self.seed = seed
        self.stats = KVStackStats()
        self._engines: list[Any] = []
        self._connectors: list[Any] = []
        self._instance_ids: list[str] = []
        self._engine: Any = None
        self._chunk_size: int | None = None
        self._lmcache_config: Any = None
        self._trace_recorder: TraceRecorder | None = None
        self._lock = asyncio.Lock()
        self._closing = False

    @property
    def chunk_size(self) -> int:
        if self._chunk_size is None:
            raise RuntimeError("LMCacheStack not started; call start() first")
        return self._chunk_size

    async def _blocking(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Do not abandon a thread on cancellation (especially lookup(pin=True))."""
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    # Retrieve exceptions rather than leaking an unobserved task.
                    task.result()
                    raise
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def start(self) -> None:
        async with self._lock:
            if self._engine is not None:
                return
            self._closing = False
            try:
                await self._blocking(self._start_sync)
            except BaseException:
                await self._blocking(self._close_sync)
                raise

    def _start_sync(self) -> None:
        try:
            import torch
            from lmcache import torch_device_type
            from lmcache.v1.cache_engine import LMCacheEngineBuilder
            from lmcache.v1.config import LMCacheEngineConfig
            from lmcache.v1.metadata import LMCacheMetadata
        except ImportError as exc:
            raise RuntimeError("Install 'kvbench[lmcache]' to run a server; no inference GPU is required") from exc
        from kvbench.kv import lmcache_compat

        lmcache_compat.apply()  # LMCache 0.5.4 disk tier double-counts re-stored chunks; backport of the 0.5.6 fix
        if torch_device_type != "cpu":
            raise RuntimeError("V1 requires LMCache's CPU platform. Use `kvbench serve`, or hide accelerator devices "
                               "before importing Torch/LMCache in an embedded process (CUDA/HIP/ROCR_VISIBLE_DEVICES='').")
        from kvbench.kv.mock_gpu import MockGPUConnector

        cfg = LMCacheEngineConfig.from_file(str(self.config_file)) if self.config_file else LMCacheEngineConfig.from_env()
        # LMCache defaults to disabling process-wide cyclic GC. Unlike a real
        # worker, this process also owns an async HTTP server and may restart
        # engines. Keep GC enabled; expose the resolved setting in the manifest.
        cfg.py_enable_gc = True
        if not self.config_file and "LMCACHE_MAX_LOCAL_CPU_SIZE" not in os.environ:
            # Safe default when no storage sizing was supplied; explicit LMCache
            # configuration is never rewritten to fit the guard.
            cfg.max_local_cpu_size = 0.25
        unsupported = [name for name in ("use_layerwise", "enable_async_loading", "enable_blending",
                                         "enable_pd", "enable_controller", "enable_hidden_state_cache",
                                         "enable_p2p", "gds_path") if getattr(cfg, name, False)]
        for option in ("enable_nixl_storage", "force_store_wait"):
            if cfg.extra_config and cfg.extra_config.get(option):
                unsupported.append(f"extra_config.{option}")
        if unsupported:
            raise ValueError(f"V1 supports classic LMCache storage only; unsupported modes: {unsupported}")
        ranks = len(self.layout)
        aggregate_gb = cfg.max_local_cpu_size * ranks
        if aggregate_gb <= 0 or aggregate_gb > self.max_cpu_memory_gb:
            raise ValueError(f"LMCache CPU pools request {aggregate_gb:g} GiB across {ranks} ranks; "
                             f"kv.max_cpu_memory_gb guard is {self.max_cpu_memory_gb:g}")
        chunk = cfg.chunk_size
        if chunk < 1:
            raise ValueError("LMCache chunk_size must be positive")
        for rank in self.layout:
            size = rank["bytes_per_token"] * chunk
            aligned = (size + 4095) // 4096 * 4096
            if size > self.max_chunk_mb * 1024**2 or aligned > cfg.max_local_cpu_size * 1024**3:
                raise ValueError(f"Rank chunk ({size} bytes) exceeds payload guard or LMCache CPU pool")
        self._lmcache_config = cfg
        self._chunk_size = chunk
        if self.trace_file:
            from kvbench.trace.recorder import TraceRecorder

            self._trace_recorder = TraceRecorder(self.trace_file)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn}[self.model.effective_kv_dtype]
        fingerprint = hashlib.sha256(json.dumps(asdict(self.model), sort_keys=True).encode()).hexdigest()[:12]
        namespace = f"{self.model_profile}--{fingerprint}--tp{self.tp_size}pp{self.pp_size}"
        for rank in self.layout:
            index = rank["rank"]
            identifier = f"{self.instance_id}-r{index}"
            if LMCacheEngineBuilder.get(identifier) is not None:
                raise ValueError(f"LMCache instance already exists: {identifier}; use a unique instance_id")
            # Register ownership before construction so partial post_init failure
            # still destroys the builder's engine and stats logger.
            self._instance_ids.append(identifier)
            metadata = LMCacheMetadata(model_name=namespace, world_size=ranks, local_world_size=ranks,
                                       worker_id=index, local_worker_id=index, kv_dtype=dtype,
                                       kv_shape=(rank["layers"], 2, chunk, 1, rank["width"]), chunk_size=chunk)
            connector = MockGPUConnector(rank["layers"], rank["width"], dtype,
                                         random_fill=self.random_fill, seed=self.seed, rank=index,
                                         namespace=namespace)
            engine = LMCacheEngineBuilder.get_or_create(identifier, cfg, metadata, connector,
                                                        broadcast_fn=lambda _tensor, _src: None,
                                                        broadcast_object_fn=lambda obj, _src: obj)
            engine.post_init()
            self._engines.append(engine)
            self._connectors.append(connector)
            if self._trace_recorder is not None:
                from kvbench.trace.recorder import install_lmcache_trace

                install_lmcache_trace(engine, self._trace_recorder, rank=index)
        self._engine = self._engines[0]
        logger.info("LMCache started: %s, %d ranks, %d-token chunks, %.3f GiB CPU pools",
                    namespace, ranks, chunk, aggregate_gb)

    def _require_engine(self) -> Any:
        if self._engine is None or self._closing:
            raise RuntimeError("LMCacheStack not started or closing")
        return self._engine

    async def lookup(self, tokens: list[int]) -> int:
        return await self._lookup(tokens, None)

    async def lookup_pinned(self, tokens: list[int], lookup_id: str) -> int:
        return await self._lookup(tokens, lookup_id)

    async def _lookup(self, tokens: list[int], lookup_id: str | None) -> int:
        async with self._lock:
            self._require_engine()
            try:
                hit = await self._blocking(self._lookup_sync, tokens, lookup_id)
            except BaseException:
                if lookup_id is not None:
                    await self._blocking(self._unpin_sync, lookup_id)
                self.stats.errors += 1
                raise
            self.stats.lookups += 1
            self.stats.lookup_tokens += len(tokens)
            self.stats.hit_tokens += hit
            return hit

    def _lookup_sync(self, tokens: list[int], lookup_id: str | None) -> int:
        return min(int(engine.lookup(tokens, lookup_id=lookup_id, pin=lookup_id is not None))
                   for engine in self._engines)

    def _unpin_sync(self, lookup_id: str) -> None:
        failures = []
        for engine in self._engines:
            try:
                engine.lookup_unpin(lookup_id)
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError("Failed to release lookup pins on one or more ranks") from failures[0]

    async def unpin(self, lookup_id: str) -> None:
        async with self._lock:
            await self._blocking(self._unpin_sync, lookup_id)

    def _validate_range(self, tokens: list[int], skip_leading: int) -> None:
        if skip_leading < 0 or skip_leading > len(tokens) or skip_leading % self.chunk_size:
            raise ValueError("skip_leading must be a chunk-aligned prefix of tokens")

    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        async with self._lock:
            self._require_engine()
            self._validate_range(tokens, skip_leading)
            end = len(tokens) // self.chunk_size * self.chunk_size
            if end <= skip_leading:
                return
            try:
                await self._blocking(self._store_sync, tokens[:end], skip_leading)
            except Exception:
                self.stats.errors += 1
                raise
            self.stats.stores += 1
            self.stats.stored_tokens += end - skip_leading
            self.stats.submitted_store_bytes += (end - skip_leading) * sum(r["bytes_per_token"] for r in self.layout)

    def _store_sync(self, tokens: list[int], skip_leading: int) -> None:
        import torch

        ids = torch.tensor(tokens, dtype=torch.int64)
        # One chunk allocation per engine call. LMCache can asynchronously copy
        # to its tiers, but the emulator never allocates a prompt-sized payload.
        for end in range(skip_leading + self.chunk_size, len(tokens) + 1, self.chunk_size):
            mask = torch.zeros(end, dtype=torch.bool)
            mask[end - self.chunk_size:] = True
            for engine in self._engines:
                engine.store(ids[:end], mask=mask, payload_tokens=ids[:end])

    async def retrieve(self, tokens: list[int], skip_leading: int = 0) -> int:
        async with self._lock:
            self._require_engine()
            self._validate_range(tokens, skip_leading)
            try:
                hit = await self._blocking(self._retrieve_sync, tokens, skip_leading)
            except Exception:
                self.stats.errors += 1
                raise
            self.stats.retrieves += 1
            self.stats.retrieved_tokens += hit
            self.stats.retrieved_bytes += hit * sum(r["bytes_per_token"] for r in self.layout)
            return hit

    def _retrieve_sync(self, tokens: list[int], skip_leading: int) -> int:
        import torch

        ids = torch.tensor(tokens, dtype=torch.int64)
        retrieved = 0
        for end in range(skip_leading + self.chunk_size, len(tokens) + 1, self.chunk_size):
            mask = torch.zeros(end, dtype=torch.bool)
            mask[end - self.chunk_size:] = True
            hits = [int(engine.retrieve(ids[:end], mask=mask).sum()) for engine in self._engines]
            if min(hits) != self.chunk_size:
                break
            retrieved += self.chunk_size
        return retrieved

    def capacity_info(self) -> dict[str, Any]:
        cfg = self._lmcache_config
        if cfg is None:
            raise RuntimeError("LMCacheStack not started")
        ranks = len(self.layout)
        cpu = int(cfg.max_local_cpu_size * 1024**3) * ranks if cfg.local_cpu else 0
        disk = int(cfg.max_local_disk_size * 1024**3) * ranks if cfg.local_disk else 0
        stored = sum(rank["bytes_per_token"] for rank in self.layout)
        return {"chunk_size": self.chunk_size, "bytes_per_token": self.model.total_kv_cache_bytes_per_token,
                "chunk_bytes": stored * self.chunk_size, "stored_bytes_per_token": stored,
                "tp_size": self.tp_size, "pp_size": self.pp_size, "files_per_chunk": ranks,
                "ranks": [{**r, "chunk_bytes": r["bytes_per_token"] * self.chunk_size} for r in self.layout],
                "vocab_size": self.model.vocab_size, "local_cpu_enabled": bool(cfg.local_cpu),
                "local_cpu_capacity_bytes": cpu, "local_disk_capacity_bytes": disk,
                "local_disk_path": str(cfg.local_disk) if cfg.local_disk else None,
                "remote_url": cfg.remote_url, "total_capacity_bytes": cpu + disk}

    def usage_info(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for rank, engine in enumerate(self._engines):
            for backend in engine.storage_manager.storage_backends.values():
                info = {"rank": rank}
                usage = getattr(backend, "usage", None)
                if isinstance(usage, (int, float)):
                    info["usage_bytes"] = int(usage)
                mapping = getattr(backend, "dict", None)
                if mapping is not None:
                    info["keys"] = len(mapping)
                if hasattr(backend, "hot_cache"):
                    with backend.cpu_lock:
                        info["keys"] = len(backend.hot_cache)
                    # V1 stores full chunks only; this is resident payload,
                    # excluding transient staging objects and allocator padding.
                    info["usage_bytes"] = info["keys"] * self.layout[rank]["bytes_per_token"] * self.chunk_size
                result[f"{type(backend).__name__}/rank-{rank}"] = info
        return result

    @property
    def trace_recorder(self) -> TraceRecorder | None:
        return self._trace_recorder

    async def close(self) -> None:
        self._closing = True
        async with self._lock:
            await self._blocking(self._close_sync)

    def _close_sync(self) -> None:
        if self._instance_ids:
            from lmcache.v1.cache_engine import LMCacheEngineBuilder

            for identifier in reversed(self._instance_ids):
                LMCacheEngineBuilder.destroy(identifier)
        self._instance_ids.clear()
        self._engines.clear()
        self._connectors.clear()
        self._engine = None
        self._chunk_size = None
        self._lmcache_config = None
        if self._trace_recorder is not None:
            self._trace_recorder.close()
            self._trace_recorder = None
        gc.collect()
