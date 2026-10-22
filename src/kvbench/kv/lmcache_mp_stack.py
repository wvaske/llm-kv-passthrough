"""LMCache multi-process (MP) storage path: one StorageManager per emulated node.

Reproduces the storage behaviour of an ``lmcache server`` (LMCache MP mode,
as run in production deployments) without vLLM, ZMQ or a GPU:

* one StorageManager per KV-Bench server = one ``lmcache server`` per node,
  shared by all TP ranks: every chunk is stored as one object per kv_rank
  (``kv_world_size = tp``) in the same L1 and L2; pure-MLA models replicate KV
  across TP, so they store one object per chunk (``kv_world_size = 1``);
* L1 = pinned-style CPU pool, L2 = the configured adapters (``fs_native`` on the
  drive under test), write-through from L1 to L2 by LMCache's StoreController;
* L2 eviction = LMCache's own L2EvictionController (watermark + ratio, LRU,
  batch delete), configured in the adapter JSON exactly as on the server;
* lookups use the prefetch PREFIX path (L1 + L2 hit, minimum across ranks),
  retrieves stream L2 -> L1 in windows of ``restore_window_chunks`` chunks
  (a "windowed / deferred L2 restore", as some deployments patch in), so prefixes larger
  than L1 still produce L2 reads.

Configuration: a JSON file (``kv.lmcache_mp_config_file``)::

    {"server_args": ["--l1-size-gb", "2", "--eviction-policy", "LRU", "--chunk-size", "256",
                     "--l2-adapter", "{\\"type\\":\\"fs_native\\", ...}"],
     "restore_window_chunks": 32}

``server_args`` are the ``lmcache server`` storage-manager flags, parsed by
LMCache's own parser (``lmcache.v1.distributed.config``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from kvbench.core.models import ModelProfile, get_model_profile
from kvbench.core.ranks import rank_selection_info, resolve_active_ranks
from kvbench.kv.base import KVStack, KVStackStats

P = ParamSpec("P")
T = TypeVar("T")
logger = logging.getLogger(__name__)

_WAIT_S = 120.0          # max wait for one prefetch (L2 lookup + load) to finish
_RESERVE_RETRY_S = 30.0  # max wait for L1 space on store (L1 is the write buffer; the StoreController frees it)
_L1_RETRIES = 30         # windowed restore: retries of a short window under L1 pressure (patch: up to 30 x 1 s)


class LMCacheMPStack(KVStack):
    def __init__(self, model_profile: str, instance_id: str = "kvbench", config_file: Path | str | None = None,
                 random_fill: bool = True, tp_size: int = 1, pp_size: int = 1, model: ModelProfile | None = None,
                 max_chunk_mb: int = 256, seed: int = 24301, active_ranks: list[int] | None = None, **_ignored: Any):
        if pp_size != 1:
            raise ValueError("lmcache_mp stack models one node: pp_size must be 1")
        if config_file is None:
            raise ValueError("lmcache_mp stack needs kv.lmcache_mp_config_file (server_args JSON)")
        self.model_profile = model_profile
        self.model = model or get_model_profile(model_profile)
        self.instance_id = instance_id
        self.config_file = Path(config_file)
        self.random_fill = random_fill
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.max_chunk_mb = max_chunk_mb
        self.seed = seed
        self.stats = KVStackStats()
        # Byte geometry from the model's researched rank layout. A payload that is the same on every TP rank
        # (MLA latent, replicated components) is stored ONCE per chunk by an LMCache MP server
        # (kv_world_size collapses to 1, every worker reads it); a sharded payload is one object per rank.
        modeled = self.model.rank_layout(tp_size, 1)
        replicated = tp_size > 1 and modeled[0]["bytes_per_token"] == self.model.rank_layout(1, 1)[0]["bytes_per_token"]
        self.replicated = replicated
        self.kv_world_size = 1 if replicated else tp_size
        self.layout = modeled[:1] if replicated else modeled
        # active_ranks (Wes's selected-rank execution): keep the modeled TP geometry, store only these ranks' shards here
        #  (e.g. [0] = one drive per GPU in an 8-GPU node). Replicated payloads have one object, owned by kv_rank 0.
        self.active_rank_ids = resolve_active_ranks(len(self.layout), active_ranks if not replicated else
                                                    ([0] if active_ranks is not None else None))
        self.rank_selection = rank_selection_info(self.layout, self.active_rank_ids)
        self.bytes_per_token_per_rank = self.layout[0]["bytes_per_token"]
        self._sm: Any = None
        self._chunk_size: int | None = None
        self._window = 32
        self._layout_desc: Any = None
        self._attn: Any = None
        self._namespace = ""
        self._lock = asyncio.Lock()
        self._closing = False
        self._fill_lock = threading.Lock()
        self._counts = {"l2_files_written": 0, "l2_bytes_written": 0, "l2_files_deleted": 0, "l2_bytes_deleted": 0,
                        "l2_loads": 0, "l1_reserve_retries": 0, "store_dropped_chunks": 0, "restore_windows": 0, "l1_pressure_retries": 0}
        self._sizes: dict[Any, int] = {}
        self._digest_cache: tuple[Any, list[bytes]] | None = None

    # ------------------------------------------------------------------ helpers
    @property
    def chunk_size(self) -> int:
        if self._chunk_size is None:
            raise RuntimeError("LMCacheMPStack not started")
        return self._chunk_size

    async def _blocking(self, function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
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

    def _digests(self, tokens: list[int], n_chunks: int) -> list[bytes]:
        """Chained per-chunk hashes (like vLLM block hashes: chunk i depends on chunks 0..i), computed once per
        token sequence and reused by every window of the same request (O(n) instead of O(n^2) per request)."""
        cache = self._digest_cache
        if cache is not None and cache[0] is tokens and len(cache[1]) >= n_chunks:
            return cache[1]
        import array

        cs = self.chunk_size
        out: list[bytes] = []
        prev = hashlib.blake2b(self._namespace.encode(), digest_size=16).digest()
        buf = array.array("q", tokens[:n_chunks * cs]).tobytes()
        step = cs * 8
        for c in range(n_chunks):
            prev = hashlib.blake2b(prev + buf[c * step:(c + 1) * step], digest_size=16).digest()
            out.append(prev)
        self._digest_cache = (tokens, out)
        return out

    def _keys(self, tokens: list[int], first_chunk: int, n_chunks: int) -> list[Any]:
        """Object keys in prefix order: chunk-major, kv_rank-minor (LMCache MP AttnWindowDesc stride)."""
        from lmcache.v1.distributed.api import ObjectKey

        digests = self._digests(tokens, first_chunk + n_chunks)
        return [ObjectKey(chunk_hash=digests[c], model_name=self._namespace, kv_rank=r)
                for c in range(first_chunk, first_chunk + n_chunks) for r in self.active_rank_ids]

    def _fill(self, mem_obj: Any, key: Any) -> None:
        """Incompressible, deterministic per-object payload (SFC64 stream seeded by the key, as the classic stack does)."""
        if not self.random_fill:
            return
        t = mem_obj.tensor
        if t is None:
            return
        import torch

        view = t.view(-1).view(torch.uint8)
        seed = int.from_bytes(hashlib.blake2b(key.chunk_hash + key.kv_rank.to_bytes(2, "little"), digest_size=8).digest(), "little")
        try:
            import numpy as np
            from kvbench.core.payload import fill_sfc64

            fill_sfc64(view.numpy(), np.random.SFC64(seed))
        except ImportError:   # older trees without kvbench.core.payload
            g = torch.Generator().manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
            view.random_(0, 256, generator=g)

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        async with self._lock:
            if self._sm is not None:
                return
            self._closing = False
            await self._blocking(self._start_sync)

    def _start_sync(self) -> None:
        import torch
        from lmcache import torch_device_type
        from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
        from lmcache.v1.distributed.config import get_arg_parser, parse_args_to_config
        from lmcache.v1.distributed.internal_api import L2AdapterListener
        from lmcache.v1.distributed.storage_manager import StorageManager

        if torch_device_type != "cpu":
            raise RuntimeError("lmcache_mp stack requires LMCache's CPU platform (hide accelerators)")
        spec = json.loads(self.config_file.read_text())
        args = list(spec.get("server_args", []))
        self._window = int(spec.get("restore_window_chunks", 32))
        # --chunk-size is an `lmcache server` (multiprocess) flag, not a storage-manager flag: take it out here
        chunk = int(spec.get("chunk_size", 256))
        if "--chunk-size" in args:
            i = args.index("--chunk-size"); chunk = int(args[i + 1]); del args[i:i + 2]
        parsed = get_arg_parser().parse_args(args)
        cfg = parse_args_to_config(parsed)
        self._chunk_size = chunk
        size = self.bytes_per_token_per_rank * self._chunk_size
        if size > self.max_chunk_mb * 1024**2:
            raise ValueError(f"object ({size} bytes) exceeds payload guard {self.max_chunk_mb} MiB")
        # storage only sees bytes: bf16/fp16 as 2-byte elements, fp8 and researched "packed" layouts (mixed FP8/FP4/BF16
        # components) as raw bytes; the object size is exactly the rank's bytes_per_token x chunk tokens.
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.model.effective_kv_dtype, torch.uint8)
        elem = torch.tensor([], dtype=dtype).element_size()
        if size % elem:
            raise ValueError("object size not a multiple of the KV element size")
        self._layout_desc = MemoryLayoutDesc(shapes=[torch.Size([size // elem])], dtypes=[dtype])
        self._layout_desc_bytes = size
        self._attn = AttnWindowDesc(num_chunks_in_sw=[-1], world_size=len(self.active_rank_ids))
        fp = hashlib.sha256(json.dumps(asdict(self.model), sort_keys=True).encode()).hexdigest()[:12]
        self._namespace = f"{self.model_profile}-{fp}-tp{self.tp_size}"
        self._sm = StorageManager(cfg)
        stack = self

        class _Counter(L2AdapterListener):
            def on_l2_keys_stored(self, keys, sizes):  # noqa: D401
                for k, s in zip(keys, sizes, strict=False):
                    stack._counts["l2_files_written"] += 1
                    stack._counts["l2_bytes_written"] += int(s)
                    stack._sizes[k] = int(s)

            def on_l2_keys_accessed(self, keys):
                stack._counts["l2_loads"] += len(keys)

            def on_l2_keys_deleted(self, keys):
                for k in keys:
                    stack._counts["l2_files_deleted"] += 1
                    stack._counts["l2_bytes_deleted"] += stack._sizes.pop(k, size)

        self._sm.register_l2_listener(_Counter())
        logger.info("LMCache MP storage started: %s, kv_world_size %d, %d-token chunks, %d-byte objects, window %d",
                    self._namespace, self.kv_world_size, self._chunk_size, size, self._window)

    async def close(self) -> None:
        self._closing = True
        async with self._lock:
            await self._blocking(self._close_sync)

    def _close_sync(self) -> None:
        if self._sm is not None:
            try:
                self._sm.close()
            finally:
                self._sm = None

    # -------------------------------------------------------------- operations
    def _prefetch(self, keys: list[Any]) -> tuple[Any, int]:
        """PREFIX prefetch: lock the L1+L2 prefix hit in L1 (loads L2 hits). Returns (handle, hit_chunks).

        Stock LMCache MP reserves L1 for an L2 load all-or-nothing and returns no L2 hit when L1 has no room.
        Like a windowed-restore patch, a window that comes back short while L1 is full is
        retried after L1 eviction made room (L1 eviction runs every second at the L1 watermark)."""
        from lmcache.v1.distributed.api import PrefetchRequestSpec

        deadline = time.monotonic() + _WAIT_S
        retries = 0
        while True:
            h = self._sm.submit_prefetch_task(PrefetchRequestSpec(keys=keys, group_layout_descs={0: self._layout_desc},
                                                                   attn_desc=self._attn))
            while True:
                if self._sm.wait_prefetch_status(h, 1.0):
                    found = self._sm.query_prefetch_status(h)
                    if found is not None:
                        break
                if time.monotonic() > deadline:
                    raise TimeoutError("lmcache_mp prefetch did not finish")
            hit = min(found.count_leading_ones(), len(keys)) // len(self.active_rank_ids)
            n = len(keys) // len(self.active_rank_ids)
            used, total = self._sm.get_l1_usage()
            l1_pressure = total and (total - used) < self._layout_desc_bytes * len(keys)
            # short window under L1 pressure = maybe an L2 hit that could not be reserved: retry (patch step 3)
            if hit >= n or not l1_pressure or retries >= _L1_RETRIES or time.monotonic() > deadline:
                return h, hit
            self._release(keys[:hit * len(self.active_rank_ids)])
            retries += 1
            self._counts["l1_pressure_retries"] += 1
            time.sleep(1.0)

    def _release(self, keys: list[Any]) -> None:
        if keys:
            self._sm.finish_read_prefetched(keys)

    def _lookup_sync(self, tokens: list[int]) -> int:
        n = len(tokens) // self.chunk_size
        if n == 0:
            return 0
        hit = 0
        # windowed: report the full L1+L2 prefix without holding more than one window in L1
        for start in range(0, n, self._window):
            cnt = min(self._window, n - start)
            keys = self._keys(tokens, start, cnt)
            _, h = self._prefetch(keys)
            self._release(keys[:h * len(self.active_rank_ids)])
            hit += h
            if h < cnt:
                break
        return hit * self.chunk_size

    def _retrieve_sync(self, tokens: list[int], skip_leading: int) -> int:
        cs = self.chunk_size
        first, n = skip_leading // cs, len(tokens) // cs
        got = 0
        for start in range(first, n, self._window):
            cnt = min(self._window, n - start)
            keys = self._keys(tokens, start, cnt)
            _, h = self._prefetch(keys)
            locked = keys[:h * len(self.active_rank_ids)]
            if locked:
                with self._sm.read_prefetched_results(locked) as objs:
                    if objs is None:
                        h = 0
                self._counts["restore_windows"] += 1
            self._release(locked)
            got += h
            if h < cnt:
                break
        return got * cs

    def _store_sync(self, tokens: list[int], skip_leading: int) -> None:
        cs = self.chunk_size
        first, n = skip_leading // cs, len(tokens) // cs
        for c in range(first, n):
            keys = self._keys(tokens, c, 1)
            deadline = time.monotonic() + _RESERVE_RETRY_S
            pending = list(keys)
            while pending:
                got = self._sm.reserve_write(pending, self._layout_desc, "new")
                for k, obj in got.items():
                    self._fill(obj, k)
                if got:
                    self._sm.finish_write(list(got))
                pending = [k for k in pending if k not in got]
                if not pending:
                    break
                # "new" mode skips keys that already exist (L1); only out-of-memory keys need a retry
                pending = self._oom_only(pending)
                if not pending:
                    break
                if time.monotonic() > deadline:
                    self._counts["store_dropped_chunks"] += 1
                    break
                self._counts["l1_reserve_retries"] += 1
                time.sleep(0.005)

    def _oom_only(self, keys: list[Any]) -> list[Any]:
        """Keys whose reservation failed because L1 is full (retry); keys already present/being written are done."""
        from lmcache.v1.distributed.error import L1Error

        res = self._sm._l1_manager.reserve_write(keys=keys, is_temporary=[False] * len(keys),  # noqa: SLF001
                                                  layout_desc=self._layout_desc, mode="new")
        retry, got = [], []
        for k, (err, obj) in res.items():
            if err == L1Error.SUCCESS and obj is not None:
                self._fill(obj, k); got.append(k)
            elif err == L1Error.OUT_OF_MEMORY:
                retry.append(k)
        if got:
            self._sm.finish_write(got)
        return retry

    # ----------------------------------------------------------- async facade
    async def lookup(self, tokens: list[int]) -> int:
        async with self._lock:
            hit = await self._blocking(self._lookup_sync, tokens)
            self.stats.lookups += 1
            self.stats.lookup_tokens += len(tokens)
            self.stats.hit_tokens += hit
            return hit

    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        async with self._lock:
            end = len(tokens) // self.chunk_size * self.chunk_size
            if end <= skip_leading:
                return
            await self._blocking(self._store_sync, tokens[:end], skip_leading)
            self.stats.stores += 1
            self.stats.stored_tokens += end - skip_leading
            self.stats.submitted_store_bytes += (end - skip_leading) * self.bytes_per_token_per_rank * len(self.active_rank_ids)

    async def retrieve(self, tokens: list[int], skip_leading: int = 0) -> int:
        async with self._lock:
            hit = await self._blocking(self._retrieve_sync, tokens, skip_leading)
            self.stats.retrieves += 1
            self.stats.retrieved_tokens += hit
            self.stats.retrieved_bytes += hit * self.bytes_per_token_per_rank * len(self.active_rank_ids)
            return hit

    def capacity_info(self) -> dict[str, Any]:
        l1_used, l1_total = self._sm.get_l1_usage() if self._sm else (0, 0)
        l2 = [(u.total_bytes_used, u.total_capacity_bytes) for us in self._sm.get_l2_usages_by_type().values() for u in us] if self._sm else []
        l2_cap = sum(c for _, c in l2)
        return {"stack": "lmcache_mp", "chunk_size": self.chunk_size, "kv_world_size": self.kv_world_size,
                "bytes_per_token": self.model.total_kv_cache_bytes_per_token,
                "stored_bytes_per_token": self.bytes_per_token_per_rank * len(self.active_rank_ids),
                "replicated_across_tp": self.replicated, **self.rank_selection,
                "ranks": [{**r, "active": r["rank"] in self.active_rank_ids, "chunk_bytes": r["bytes_per_token"] * self.chunk_size}
                          for r in self.layout],
                "object_bytes": self.bytes_per_token_per_rank * self.chunk_size, "tp_size": self.tp_size,
                "files_per_chunk": len(self.active_rank_ids), "vocab_size": self.model.vocab_size,
                "l1_capacity_bytes": l1_total, "l2_capacity_bytes": l2_cap, "restore_window_chunks": self._window,
                # names used by warmup / Prometheus (classic stack): warm up to the L2 (drive) capacity
                "local_cpu_capacity_bytes": l1_total, "local_disk_capacity_bytes": l2_cap, "total_capacity_bytes": l2_cap}

    def usage_info(self) -> dict[str, dict[str, Any]]:
        if self._sm is None:
            return {}
        l1_used, l1_total = self._sm.get_l1_usage()
        out = {"L1": {"usage_bytes": int(l1_used), "capacity_bytes": int(l1_total)}}
        for name, us in self._sm.get_l2_usages_by_type().items():
            for i, u in enumerate(us):
                out[f"L2/{name}/{i}"] = {"usage_bytes": int(u.total_bytes_used), "capacity_bytes": int(u.total_capacity_bytes)}
        out["counters"] = dict(self._counts)
        return out
