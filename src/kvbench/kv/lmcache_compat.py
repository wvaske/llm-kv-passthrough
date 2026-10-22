"""Compatibility fixes for the pinned LMCache release (lmcache==0.5.4).

LocalDiskBackend.submit_put_task in LMCache 0.5.4 skips a put only while the same
key is still *in flight*. A put for a key that is already resident on disk is
admitted again:

1. its size is added to ``current_cache_size`` a second time, while
   ``insert_key`` does not create a second dict entry. The size counter drifts
   above the real usage; once the drift exceeds ``max_local_disk_size`` the
   eviction loop empties the dict and still cannot make room ("No eviction
   candidates found. Disk space under pressure."), so that backend stops
   storing for the rest of the process;
2. the file is rewritten while LRU may evict the same key, so a later eviction
   of that key raises ``FileNotFoundError`` in ``remove()`` and the store fails.

Workloads that store the same chunk more than once trigger it - e.g. agentic
traces (AgentX), where sub-agents re-send the parent's prefix. Measured on a
16-server KV-Bench run: disk tiers stuck on up to 13 of 16 servers, the drive
held ~0.5 TB instead of the ~1.7 TB configured, and WAF read low because the
drive was mostly empty. Synthetic multi-turn chat (unique prefixes) is not
affected.

LMCache 0.5.6 fixes this in ``submit_put_task`` (skip resident keys, atomic
in-flight registration, early give-up when evictable space is insufficient,
always clear the in-flight marker on failure). ``apply()`` installs that logic
on 0.5.4 at runtime; it is a no-op on any other LMCache version and idempotent.
``remove()`` additionally tolerates a file that is already gone.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any

logger = logging.getLogger(__name__)

PATCHED_VERSION = "0.5.4"
_MARK = "_kvbench_disk_restore_fix"


def _lmcache_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("lmcache")
    except Exception:  # pragma: no cover - lmcache not installed
        return None


def _try_insert_put_task(self: Any, key: Any) -> bool:
    """Register a put task unless the key is already active (LMCache 0.5.6)."""
    with self.put_lock:
        if key in self.put_tasks:
            return False
        self.put_tasks.append(key)
        return True


def _submit_put_task(self: Any, key: Any, memory_obj: Any, on_complete_callback: Any = None) -> None:
    """LMCache 0.5.6 ``LocalDiskBackend.submit_put_task`` logic, for 0.5.4."""
    import asyncio

    assert memory_obj.tensor is not None

    # A key already on disk is not written again; refresh its recency instead.
    with self.disk_lock:
        if key in self.dict:
            self.cache_policy.update_on_hit(key, self.dict)
            return None

    required_size = memory_obj.get_physical_size()
    task_registered = False
    admission_succeeded = False
    try:
        with self.disk_lock:
            # Re-check and register in one critical section so a completed write
            # cannot slip in between publishing the key and the in-flight marker.
            if key in self.dict:
                self.cache_policy.update_on_hit(key, self.dict)
                return None
            if required_size > self.max_cache_size:
                logger.warning("Put for %s needs %s bytes, more than the disk cache capacity %s bytes.",
                               key, required_size, self.max_cache_size)
                return None
            if not _try_insert_put_task(self.disk_worker, key):
                return None
            task_registered = True

            required_eviction = self.current_cache_size + required_size - self.max_cache_size
            if required_eviction > 0:
                evictable = sum(m.size for m in self.dict.values() if m.can_evict)
                if evictable < required_eviction:
                    logger.warning("Insufficient evictable disk cache space for %s: need %s bytes, have %s bytes.",
                                   key, required_eviction, evictable)
                    return None

            while self.current_cache_size + required_size > self.max_cache_size:
                evict_keys = self.cache_policy.get_evict_candidates(self.dict, num_candidates=1)
                if not evict_keys:
                    logger.warning("No eviction candidates found. Disk space under pressure.")
                    return None
                for evict_key in evict_keys:
                    self.current_cache_size -= self.dict[evict_key].size
                self.batched_remove(evict_keys, force=False)

            self.current_cache_size += required_size
            self.cache_policy.update_on_put(key)
            admission_succeeded = True
    finally:
        if task_registered and not admission_succeeded:
            self.disk_worker.remove_put_task(key)

    memory_obj.ref_count_up()
    asyncio.run_coroutine_threadsafe(
        self.disk_worker.submit_task("put", self.async_save_bytes_to_disk, key=key, memory_obj=memory_obj,
                                     on_complete_callback=on_complete_callback),
        self.loop,
    )
    return None


def _remove(self: Any, key: Any, force: bool = True) -> bool:
    """LMCache 0.5.4 ``LocalDiskBackend.remove`` that tolerates an already-missing file."""
    lock_context = self.disk_lock if force else nullcontext()
    with lock_context:
        meta = self.dict.pop(key, None)
        if not meta:
            return False
        self.usage -= meta.size
        self.stats_monitor.update_local_storage_usage(self.usage)
        try:
            os.remove(meta.path)
        except FileNotFoundError:
            logger.warning("LMCache disk evict: file already missing: %s", meta.path)
        if force:
            self.cache_policy.update_on_force_evict(key)
    if self.batched_msg_sender is not None:
        from lmcache.v1.cache_controller.message import OpType

        self.batched_msg_sender.add_kv_op(op_type=OpType.EVICT, key=key.chunk_hash)
    return True


def apply() -> bool:
    """Install the disk-backend fix on LMCache 0.5.4. Returns True if the fix is active."""
    if _lmcache_version() != PATCHED_VERSION:
        return False
    from lmcache.v1.storage_backend import local_disk_backend as ldb

    if getattr(ldb.LocalDiskBackend, _MARK, False):
        return True
    if hasattr(ldb.LocalDiskWorker, "try_insert_put_task") and _source_has_resident_check(ldb):
        # Already fixed in this build (e.g. a patched wheel) - leave it alone.
        setattr(ldb.LocalDiskBackend, _MARK, True)
        return True
    ldb.LocalDiskWorker.try_insert_put_task = _try_insert_put_task
    ldb.LocalDiskBackend.submit_put_task = _submit_put_task
    ldb.LocalDiskBackend.remove = _remove
    setattr(ldb.LocalDiskBackend, _MARK, True)
    logger.info("Applied LMCache %s disk-backend re-store fix (backport of 0.5.6 submit_put_task)", PATCHED_VERSION)
    return True


def _source_has_resident_check(ldb: Any) -> bool:
    import inspect

    try:
        src = inspect.getsource(ldb.LocalDiskBackend.submit_put_task)
    except (OSError, TypeError):
        return False
    return "try_insert_put_task" in src or "DCWE439_PATCH" in src


def is_active() -> bool:
    """True when LocalDiskBackend has the re-store fix (applied here or shipped with LMCache)."""
    try:
        from lmcache.v1.storage_backend import local_disk_backend as ldb
    except Exception:  # pragma: no cover
        return False
    return bool(getattr(ldb.LocalDiskBackend, _MARK, False)) or (
        hasattr(ldb.LocalDiskWorker, "try_insert_put_task") and _source_has_resident_check(ldb))
