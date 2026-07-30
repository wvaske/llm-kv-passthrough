"""
KV operation trace recorder.

Records every logical storage operation LMCache performs (put/get/contains/
remove per chunk key) and the physical file I/O those become (whole-file
write/read of chunk files) as one JSON object per line.

Event schema (fields absent when not applicable):
    ts       Wall-clock epoch seconds (float).
    mono_ms  Process-monotonic milliseconds (float) — use for overlap analysis.
    tid      OS thread ident performing the operation.
    thread   Thread name (LMCache worker threads are named).
    event    "logical" (stack-level op), "io" (file-level op), or "meta".
    op       logical: put|get|contains|remove; io: write|read.
    backend  Backend class name (e.g. "LocalDiskBackend").
    key      Chunk cache key (logical events).
    size     Payload bytes.
    dur_ms   Operation duration in milliseconds (measured ops only).
    path     File path (io events).
    hit      contains/get result (logical events).

The recorder also keeps in-memory op/byte counters per (event, op, backend)
for Prometheus export.

Hook installation is per-instance monkeypatching of the live backend
objects, the same seam LMCache's own AuditBackend wraps. It is coupled to
lmcache.v1 internals (pinned in pyproject); install_lmcache_trace() fails
loudly if the expected surface is missing.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TraceRecorder:
    """Thread-safe JSONL trace writer with in-memory counters.

    Attributes:
        path: Output JSONL file path.
        counters: {(event, op, backend): [ops, bytes]} accumulated totals.
    """

    FLUSH_EVERY_RECORDS = 256
    FLUSH_EVERY_SECONDS = 1.0

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: mono_ms is process-relative, so events from a previous
        # run would corrupt overlap analysis if appended to.
        self._fh = open(self.path, "w", buffering=1024 * 1024)  # noqa: SIM115 - lifetime managed by close()
        self._lock = threading.Lock()
        # Anchor wall clock to the monotonic clock once so ts and mono_ms
        # stay mutually consistent for the life of the trace.
        self._mono0 = time.perf_counter()
        self._wall0 = time.time()
        self.counters: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
        self._closed = False
        self._unflushed = 0
        self._last_flush = time.monotonic()

    def record(
        self,
        event: str,
        op: str,
        backend: str,
        key: str | None = None,
        size: int | None = None,
        dur_ms: float | None = None,
        path: str | None = None,
        hit: bool | None = None,
        **extra: Any,
    ) -> None:
        """Record one trace event."""
        if self._closed:
            return
        mono = time.perf_counter() - self._mono0
        rec: dict[str, Any] = {
            "ts": round(self._wall0 + mono, 6),
            "mono_ms": round(mono * 1000, 3),
            "tid": threading.get_ident(),
            "thread": threading.current_thread().name,
            "event": event,
            "op": op,
            "backend": backend,
        }
        if key is not None:
            rec["key"] = key
        if size is not None:
            rec["size"] = size
        if dur_ms is not None:
            rec["dur_ms"] = round(dur_ms, 3)
        if path is not None:
            rec["path"] = path
        if hit is not None:
            rec["hit"] = hit
        rec.update(extra)
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        with self._lock:
            if self._closed:
                return
            self._fh.write(line)
            counter = self.counters[(event, op, backend)]
            counter[0] += 1
            counter[1] += size or 0
            self._unflushed += 1
            now = time.monotonic()
            if (
                self._unflushed >= self.FLUSH_EVERY_RECORDS
                or now - self._last_flush >= self.FLUSH_EVERY_SECONDS
            ):
                self._fh.flush()
                self._unflushed = 0
                self._last_flush = now

    def flush(self) -> None:
        with self._lock:
            if not self._closed:
                self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._fh.flush()
            self._fh.close()


def install_lmcache_trace(engine: Any, recorder: TraceRecorder) -> list[str]:
    """Install trace hooks on a started LMCache engine's storage backends.

    Wraps, per backend instance:
    - batched_submit_put_task: logical "put" per key at submission time.
    - get_blocking / batched_get_non_blocking: logical "get" with duration.
    - contains: logical "contains" with result.
    - remove: logical "remove" (this is the eviction path).
    - LocalDiskBackend.write_file / read_file: physical "io" events with
      duration and thread identity — the operations a disk actually sees.

    LocalCPUBackend is left unwrapped (RAM tier, not storage I/O), matching
    LMCache's own AuditBackend behavior.

    Args:
        engine: A started lmcache.v1 cache engine.
        recorder: The TraceRecorder to write events to.

    Returns:
        Names of the backends that were wrapped.

    Raises:
        RuntimeError: If the engine does not expose the expected
            storage_manager/storage_backends surface.
    """
    storage_manager = getattr(engine, "storage_manager", None)
    backends = getattr(storage_manager, "storage_backends", None)
    if not backends:
        raise RuntimeError(
            "Cannot install KV trace hooks: LMCache engine has no "
            "storage_manager.storage_backends (lmcache internals changed?)"
        )

    wrapped: list[str] = []
    for name, backend in backends.items():
        cls_name = type(backend).__name__
        if cls_name == "LocalCPUBackend":
            continue
        _wrap_logical(backend, cls_name, recorder)
        if hasattr(backend, "write_file") and hasattr(backend, "read_file"):
            _wrap_disk_io(backend, cls_name, recorder)
            recorder.record(
                event="meta",
                op="backend_info",
                backend=cls_name,
                use_odirect=bool(getattr(backend, "use_odirect", False)),
                os_disk_bs=getattr(backend, "os_disk_bs", None),
                name=name,
            )
        wrapped.append(name)
        logger.info(f"KV trace hooks installed on backend {name} ({cls_name})")
    return wrapped


def _obj_size(memory_obj: Any) -> int | None:
    try:
        return len(memory_obj.byte_array)
    except Exception:
        return None


def _wrap_logical(backend: Any, cls_name: str, recorder: TraceRecorder) -> None:
    """Wrap logical (key-level) operations on a backend instance."""
    orig_put = backend.batched_submit_put_task
    orig_get = backend.get_blocking
    orig_contains = backend.contains
    orig_remove = backend.remove

    def traced_batched_submit_put_task(keys, memory_objs, *args, **kwargs):  # type: ignore[no-untyped-def]
        for key, obj in zip(keys, memory_objs, strict=False):
            recorder.record(
                event="logical",
                op="put",
                backend=cls_name,
                key=str(key),
                size=_obj_size(obj),
            )
        return orig_put(keys, memory_objs, *args, **kwargs)

    def traced_get_blocking(key, *args, **kwargs):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = orig_get(key, *args, **kwargs)
        recorder.record(
            event="logical",
            op="get",
            backend=cls_name,
            key=str(key),
            size=_obj_size(result) if result is not None else 0,
            dur_ms=(time.perf_counter() - start) * 1000,
            hit=result is not None,
        )
        return result

    def traced_contains(key, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = orig_contains(key, *args, **kwargs)
        recorder.record(
            event="logical",
            op="contains",
            backend=cls_name,
            key=str(key),
            hit=bool(result),
        )
        return result

    def traced_remove(key, *args, **kwargs):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = orig_remove(key, *args, **kwargs)
        recorder.record(
            event="logical",
            op="remove",
            backend=cls_name,
            key=str(key),
            dur_ms=(time.perf_counter() - start) * 1000,
            hit=bool(result),
        )
        return result

    backend.batched_submit_put_task = traced_batched_submit_put_task
    backend.get_blocking = traced_get_blocking
    backend.contains = traced_contains
    backend.remove = traced_remove

    orig_batched_contains = getattr(backend, "batched_contains", None)
    if orig_batched_contains is not None:

        def traced_batched_contains(keys, *args, **kwargs):  # type: ignore[no-untyped-def]
            result = orig_batched_contains(keys, *args, **kwargs)
            recorder.record(
                event="logical",
                op="batched_contains",
                backend=cls_name,
                size=None,
                hit=bool(result),
                n=len(keys) if hasattr(keys, "__len__") else None,
            )
            return result

        backend.batched_contains = traced_batched_contains

    # Async batched read path (used by the storage manager's prefetcher)
    orig_batched_get = getattr(backend, "batched_get_non_blocking", None)
    if orig_batched_get is not None:

        async def traced_batched_get_non_blocking(*args, **kwargs):  # type: ignore[no-untyped-def]
            start = time.perf_counter()
            result = await orig_batched_get(*args, **kwargs)
            recorder.record(
                event="logical",
                op="batched_get",
                backend=cls_name,
                size=sum(filter(None, (_obj_size(o) for o in result or []))),
                dur_ms=(time.perf_counter() - start) * 1000,
            )
            return result

        backend.batched_get_non_blocking = traced_batched_get_non_blocking


def _wrap_disk_io(backend: Any, cls_name: str, recorder: TraceRecorder) -> None:
    """Wrap physical file I/O on a LocalDiskBackend-shaped instance."""
    orig_write_file = backend.write_file
    orig_read_file = backend.read_file

    def traced_write_file(buffer, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = orig_write_file(buffer, path, *args, **kwargs)
        recorder.record(
            event="io",
            op="write",
            backend=cls_name,
            size=len(buffer),
            dur_ms=(time.perf_counter() - start) * 1000,
            path=str(path),
        )
        return result

    def traced_read_file(key, buffer, path, *args, **kwargs):  # type: ignore[no-untyped-def]
        start = time.perf_counter()
        result = orig_read_file(key, buffer, path, *args, **kwargs)
        recorder.record(
            event="io",
            op="read",
            backend=cls_name,
            size=len(buffer),
            dur_ms=(time.perf_counter() - start) * 1000,
            path=str(path),
        )
        return result

    backend.write_file = traced_write_file
    backend.read_file = traced_read_file
