"""CPU workload adapter over SGLang's real HiCacheFile storage implementation.

Host-tier residency and write scheduling are V1 policies, not HiRadixTree.
File format, file admission/eviction, deduplication and I/O are upstream code.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, TypeVar

from kvbench.core.config import KVBenchConfig
from kvbench.core.payload import SFC64_FORMAT, fill_sfc64, scratch_bytes
from kvbench.core.ranks import rank_selection_info
from kvbench.kv.base import KVStack, KVStackStats
from kvbench.kv.hicache_upstream import REVISION, SOURCE_HASHES, VERSION, load_hicache
from kvbench.trace.recorder import TraceRecorder

T = TypeVar("T")


def page_hashes(tokens: list[int], page_size: int) -> list[str]:
    """SGLang regular-token hash: SHA256(previous digest + little-endian u32 page).

    Mirrors hash_binding.cpp without loading/compiling GPU runtime dependencies.
    Partial pages are not eligible for storage.
    """
    prior = b""
    keys = []
    for start in range(0, len(tokens) - page_size + 1, page_size):
        prior = hashlib.sha256(prior + struct.pack(f"<{page_size}I", *tokens[start:start + page_size])).digest()
        keys.append(prior.hex())
    return keys


class HiCacheStack(KVStack):
    def __init__(self, config: KVBenchConfig):
        self.config = config
        self.options = config.kv.hicache
        self.model = config.model
        self.layout = self.model.rank_layout(config.gpu.tp_size, config.timing.pp_size)
        self.selection = rank_selection_info(self.layout, config.kv.active_ranks)
        self.active_layout = [r for r in self.layout if r["rank"] in self.selection["active_rank_ids"]]
        if self.model.cache_components or self.model.kv_latent_dim:
            raise ValueError("HiCache file adapter currently supports homogeneous MHA/GQA profiles only; "
                             "MLA owner-rank deduplication and hybrid sidecar pools require separate adapters")
        identity = {"model": self.model.cache_identity(), "page_size": self.options.page_size,
                    "layout": self.options.mem_layout, "tp": config.gpu.tp_size, "pp": config.timing.pp_size}
        if config.kv.payload_generator == "sfc64":
            identity["payload_generator"] = SFC64_FORMAT
        self.namespace = "kvbench-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        self.stats = KVStackStats()
        self._backends: list[Any] = []
        self._host: list[OrderedDict[str, Any]] = []
        self._free: list[list[Any]] = []
        self._lock = asyncio.Lock()
        self._lease: tuple[str, asyncio.Task[Any] | None] | None = None
        self._started = False
        self.trace_recorder: TraceRecorder | None = None

    @property
    def chunk_size(self) -> int:
        if not self._started:
            raise RuntimeError("HiCacheStack not started")
        return self.options.page_size

    @staticmethod
    async def _blocking(operation: Callable[[], T]) -> T:
        task = asyncio.create_task(asyncio.to_thread(operation))
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                cancelled = True
                if task.done():
                    task.result()
                    raise
        if cancelled:
            raise asyncio.CancelledError
        return result

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        if not self._started:
            raise RuntimeError("HiCacheStack not started")
        if self._lease is not None and self._lease[1] is asyncio.current_task():
            yield
        else:
            async with self._lock:
                if not self._started:
                    raise RuntimeError("HiCacheStack closed")
                yield

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            # Guard before importing Torch or creating any backend/payload.
            host_bytes = int(self.options.host_size_gb * 1024**3)
            peak = host_bytes * len(self.active_layout) + max(r["bytes_per_token"] for r in self.active_layout) * self.options.page_size
            if self.config.kv.random_fill:
                peak += scratch_bytes(self.config.kv.payload_generator)
            if peak > self.config.kv.max_cpu_memory_gb * 1024**3:
                raise ValueError("HiCache host pools plus one staging page exceed kv.max_cpu_memory_gb")
            for rank in self.active_layout:
                page_bytes = rank["bytes_per_token"] * self.options.page_size
                if page_bytes > min(host_bytes, self.config.kv.max_chunk_mb * 1024**2,
                                    self.options.disk_size_gb * 1024**3):
                    raise ValueError("HiCache page exceeds host pool, disk quota or payload guard")
            try:
                await self._blocking(self._start_sync)
                self._started = True
            except BaseException:
                self._backends.clear()
                self._host.clear()
                self._free.clear()
                if self.trace_recorder:
                    self.trace_recorder.close()
                    self.trace_recorder = None
                raise

    def _start_sync(self) -> None:
        upstream = load_hicache()
        # Disallow hidden path overrides: manifests and test-device checks must
        # describe the path that the backend actually opens.
        if os.environ.get("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"):
            raise ValueError("Use kv.hicache.storage_path instead of SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR")
        assert self.options.storage_path is not None
        self.options.storage_path.mkdir(parents=True, exist_ok=True)
        for rank in self.active_layout:
            backend_config = upstream.HiCacheStorageConfig(
                tp_rank=rank["tp_rank"], tp_size=self.config.gpu.tp_size,
                pp_rank=rank["pp_rank"], pp_size=self.config.timing.pp_size,
                attn_cp_rank=0, attn_cp_size=1, is_mla_model=False,
                enable_storage_metrics=False, is_page_first_layout=self.options.mem_layout != "layer_first",
                model_name=self.namespace,
                extra_config={"max_size": int(self.options.disk_size_gb * 1024**3),
                              "min_free_space": self.options.min_free_bytes,
                              "eviction_ratio": self.options.eviction_ratio,
                              "enable_metadata_cache": False},
            )
            self._backends.append(upstream.HiCacheFile(backend_config, str(self.options.storage_path)))
            self._host.append(OrderedDict())
            self._free.append([])
        if self.config.kv.trace_file:
            self.trace_recorder = TraceRecorder(self.config.kv.trace_file)
            self.trace_recorder.record(event="meta", op="rank_selection", backend="HiCacheFile", **self.selection)
            self.trace_recorder.record(event="meta", op="backend_info", backend="HiCacheFile", **self.provenance())
            for index, backend in enumerate(self._backends):
                self._trace_eviction(index, backend)

    def _trace_eviction(self, index: int, backend: Any) -> None:
        original = backend._evictor._evict_one_lru_locked

        def evict() -> Any:
            key = next(iter(backend._evictor._lru), None)
            started = time.perf_counter()
            outcome, freed = original()
            if outcome == "evicted" and self.trace_recorder:
                self.trace_recorder.record(event="logical", op="remove", backend="HiCacheFile",
                                           key=key, size=freed, rank=self.active_layout[index]["rank"],
                                           dur_ms=(time.perf_counter() - started) * 1000, hit=True)
            return outcome, freed

        backend._evictor._evict_one_lru_locked = evict

    def provenance(self) -> dict[str, Any]:
        return {"sglang_version": VERSION, "sglang_revision": REVISION, "source_hashes": SOURCE_HASHES,
                "adapter_scope": "real HiCacheFile L3; V1 modeled host tier, not HiRadixTree",
                "config": self.options.model_dump(mode="json"), "namespace": self.namespace,
                "payload_generator": self.config.kv.payload_generator,
                "payload_generator_format": SFC64_FORMAT if self.config.kv.payload_generator == "sfc64" else "torch-byte-v1",
                "payload_axis_order": "KV,T,L,D" if self.options.mem_layout == "page_first" else "KV,L,T,D",
                "file_format": "raw page, temporary file + rename", "use_odirect": False,
                "fsync_per_page": False}

    def _trace(self, op: str, index: int, key: str, size: int, started: float, hit: bool = True) -> None:
        if self.trace_recorder:
            backend = self._backends[index]
            assert self.options.storage_path is not None
            self.trace_recorder.record(event="io", op=op, backend="HiCacheFile", key=key,
                                       size=size, dur_ms=(time.perf_counter() - started) * 1000, hit=hit,
                                       rank=self.active_layout[index]["rank"],
                                       path=str(self.options.storage_path / f"{backend._get_suffixed_key(key)}.bin"))

    def _write(self, index: int, keys: list[str], values: list[Any]) -> None:
        backend = self._backends[index]
        # Native batch_set itself loops over set; tracing each set observes the
        # actual page granularity instead of misreporting a batch as one write.
        for key, value in zip(keys, values, strict=True):
            exists = backend.exists(key)
            if self.trace_recorder:
                self.trace_recorder.record(event="logical", op="put", backend="HiCacheFile", key=key,
                                           size=value.numel() * value.element_size(), rank=self.active_layout[index]["rank"])
            started = time.perf_counter()
            if not backend.set(key, value):
                raise OSError("HiCacheFile refused/failed a page write; inspect quota and free space")
            if not exists:
                self._trace("write", index, key, value.numel() * value.element_size(), started)

    def _evict_host(self, index: int, count: int) -> None:
        host = self._host[index]
        keys = list(host)[:count]
        if self.options.write_policy == "write_back":
            self._write(index, keys, [host[key] for key in keys])
        for key in keys:
            # Native host pools reuse page slots. Keep our bounded host pages
            # instead of repeatedly mmap/zero-faulting/freeing multi-MiB tensors.
            self._free[index].append(host.pop(key))

    def _payload(self, index: int, key: str) -> Any:
        import torch

        rank = self.active_layout[index]
        # Native MHA page_first is [K/V, token, layer, head, dim]; direct/layer
        # first flatten [K/V, layer, token, head, dim], matching classic LMCache
        # KV_2LTD. Page-first changes axis order, not page bytes or file size.
        shape = (2, rank["layers"], self.options.page_size, rank["width"])
        if self.options.mem_layout == "page_first":
            shape = (2, self.options.page_size, rank["layers"], rank["width"])
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn}[self.model.effective_kv_dtype]
        value = (self._free[index].pop().view(shape) if self._free[index]
                 else torch.empty(shape, dtype=dtype, device="cpu"))
        if self.config.kv.random_fill:
            seed = hashlib.sha256(f"{self.config.kv.seed}:{rank['rank']}:{self.namespace}:{key}".encode()).digest()
            seed_int = int.from_bytes(seed[:8], "little") & (2**63 - 1)
            if self.config.kv.payload_generator == "sfc64":
                import numpy as np

                fill_sfc64(value.view(torch.uint8).numpy(), np.random.SFC64(seed_int))
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed_int)
                value.view(torch.uint8).random_(0, 256, generator=generator)
        else:
            value.zero_()
        return value.flatten()

    def _keys(self, tokens: list[int], skip: int = 0) -> list[str]:
        if skip < 0 or skip > len(tokens) or skip % self.chunk_size:
            raise ValueError("skip_leading must be a chunk-aligned prefix of tokens")
        return page_hashes(tokens, self.chunk_size)[skip // self.chunk_size:]

    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        async with self._operation():
            keys = self._keys(tokens, skip_leading)
            if not keys:
                return
            try:
                await self._blocking(lambda: self._store_sync(keys))
            except Exception:
                self.stats.errors += 1
                raise
            self.stats.stores += 1
            self.stats.stored_tokens += len(keys) * self.chunk_size
            self.stats.submitted_store_bytes += len(keys) * self.chunk_size * self.selection["active_stored_bytes_per_token"]

    def _store_sync(self, keys: list[str]) -> None:
        for index, rank in enumerate(self.active_layout):
            host = self._host[index]
            capacity = int(self.options.host_size_gb * 1024**3) // (rank["bytes_per_token"] * self.chunk_size)
            for key in keys:
                if key in host:
                    host.move_to_end(key)
                    continue
                if len(host) >= capacity:
                    self._evict_host(index, min(self.options.write_batch_pages, len(host)))
                value = self._payload(index, key)
                if self.options.write_policy == "write_through":
                    self._write(index, [key], [value])
                host[key] = value

    def _lookup_sync(self, keys: list[str]) -> int:
        for page, key in enumerate(keys):
            if not all(key in host or backend.exists(key) for host, backend in zip(self._host, self._backends, strict=True)):
                return page * self.chunk_size
            for host in self._host:
                if key in host:
                    host.move_to_end(key)
        return len(keys) * self.chunk_size

    async def lookup(self, tokens: list[int]) -> int:
        async with self._operation():
            hit = await self._blocking(lambda: self._lookup_sync(self._keys(tokens)))
            self.stats.lookups += 1
            self.stats.lookup_tokens += len(tokens)
            self.stats.hit_tokens += hit
            return hit

    async def lookup_pinned(self, tokens: list[int], lookup_id: str) -> int:
        # Lease serializes eviction with lookup -> retrieve without pinning an
        # unbounded prefix in RAM. The server always releases it in finally.
        if self._lease is not None and self._lease[1] is asyncio.current_task():
            raise RuntimeError("Nested HiCache prefix lease")
        await self._lock.acquire()
        self._lease = (lookup_id, asyncio.current_task())
        try:
            return await self.lookup(tokens)
        except BaseException:
            await self.unpin(lookup_id)
            raise

    async def unpin(self, lookup_id: str) -> None:
        if self._lease is not None and self._lease[0] == lookup_id:
            self._lease = None
            self._lock.release()

    async def retrieve(self, tokens: list[int], skip_leading: int = 0) -> int:
        async with self._operation():
            keys = self._keys(tokens, skip_leading)
            hit = await self._blocking(lambda: self._retrieve_sync(keys))
            self.stats.retrieves += 1
            self.stats.retrieved_tokens += hit
            self.stats.retrieved_bytes += hit * self.selection["active_stored_bytes_per_token"]
            return hit

    def _retrieve_sync(self, keys: list[str]) -> int:
        import torch

        for page, key in enumerate(keys):
            for index, (host, backend) in enumerate(zip(self._host, self._backends, strict=True)):
                if key in host:
                    host.move_to_end(key)
                    continue
                size = self.active_layout[index]["bytes_per_token"] * self.chunk_size
                target = torch.empty(size, dtype=torch.uint8, device="cpu")
                started = time.perf_counter()
                value = backend.get(key, target)
                if self.trace_recorder:
                    self.trace_recorder.record(event="logical", op="get", backend="HiCacheFile", key=key,
                                               size=size if value is not None else 0, hit=value is not None,
                                               rank=self.active_layout[index]["rank"])
                if value is None:
                    return page * self.chunk_size
                self._trace("read", index, key, size, started)
                del value, target
        return len(keys) * self.chunk_size

    async def flush(self) -> None:
        """Drain modeled write-back pages through upstream; does not imply fsync."""
        async with self._operation():
            for index, host in enumerate(self._host):
                await self._blocking(partial(self._evict_host, index, len(host)))

    async def close(self) -> None:
        async with self._lock:
            try:
                if self._started:
                    for index, host in enumerate(self._host):
                        await self._blocking(partial(self._evict_host, index, len(host)))
            finally:
                self._started = False
                self._host.clear()
                self._free.clear()
                self._backends.clear()
                if self.trace_recorder:
                    self.trace_recorder.close()
                    self.trace_recorder = None

    def capacity_info(self) -> dict[str, Any]:
        count = len(self.active_layout)
        stored = self.selection["active_stored_bytes_per_token"]
        disk = sum(b._evictor.max_size_bytes for b in self._backends)
        cpu = int(self.options.host_size_gb * 1024**3) * count
        return {**self.selection, "chunk_size": self.chunk_size, "bytes_per_token": self.model.total_kv_cache_bytes_per_token,
                "stored_bytes_per_token": stored, "chunk_bytes": stored * self.chunk_size,
                "modeled_chunk_bytes": self.selection["modeled_stored_bytes_per_token"] * self.chunk_size,
                "files_per_chunk": count, "modeled_files_per_chunk": len(self.layout),
                "tp_size": self.config.gpu.tp_size, "pp_size": self.config.timing.pp_size,
                "ranks": [{**r, "active": r in self.active_layout, "chunk_bytes": r["bytes_per_token"] * self.chunk_size} for r in self.layout],
                "vocab_size": self.model.vocab_size, "local_cpu_enabled": True,
                "local_cpu_capacity_bytes": cpu, "local_disk_capacity_bytes": disk,
                "total_capacity_bytes": cpu + disk, "local_disk_path": str(self.options.storage_path),
                "remote_url": None, "adapter_scope": "HiCacheFile L3 with modeled host tier"}

    def usage_info(self) -> dict[str, dict[str, Any]]:
        result = {}
        for index, (host, backend) in enumerate(zip(self._host, self._backends, strict=True)):
            rank = self.active_layout[index]
            result[f"HiCacheHost/rank-{rank['rank']}"] = {"rank": rank["rank"], "keys": len(host),
                "usage_bytes": len(host) * rank["bytes_per_token"] * self.chunk_size,
                "allocated_bytes": (len(host) + len(self._free[index])) * rank["bytes_per_token"] * self.chunk_size,
                "free_pages": len(self._free[index])}
            result[f"HiCacheFile/rank-{rank['rank']}"] = {"rank": rank["rank"],
                "keys": len(backend._evictor._lru), "usage_bytes": backend._evictor._total_bytes}
        return result
