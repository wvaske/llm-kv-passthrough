"""
Trace analysis: turn a KV I/O trace into workload characteristics.

The summary computed here is the bridge between a recorded LMCache run and
a synthetic workload definition (FIO): operation mix, size distribution,
observed parallelism (concurrent in-flight operations and distinct writer
threads — the property that produces interleaved data on the device), file
churn from eviction, and throughput.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


@dataclass
class OpStats:
    """Statistics for one operation type (e.g. io/write)."""

    count: int = 0
    total_bytes: int = 0
    sizes: Counter = field(default_factory=Counter)
    durations_ms: list[float] = field(default_factory=list)
    threads: set = field(default_factory=set)
    max_concurrency: int = 0
    first_mono_ms: float | None = None
    last_mono_ms: float | None = None

    @property
    def dominant_size(self) -> int | None:
        """Most common payload size (chunk files are near-constant size)."""
        if not self.sizes:
            return None
        return self.sizes.most_common(1)[0][0]

    @property
    def median_dur_ms(self) -> float | None:
        return median(self.durations_ms) if self.durations_ms else None

    @property
    def span_s(self) -> float:
        if self.first_mono_ms is None or self.last_mono_ms is None:
            return 0.0
        return max(0.0, (self.last_mono_ms - self.first_mono_ms) / 1000)

    @property
    def bytes_per_sec(self) -> float:
        span = self.span_s
        return self.total_bytes / span if span > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total_bytes": self.total_bytes,
            "dominant_size": self.dominant_size,
            "size_counts": dict(self.sizes.most_common(10)),
            "median_dur_ms": self.median_dur_ms,
            "threads": len(self.threads),
            "max_concurrency": self.max_concurrency,
            "span_s": round(self.span_s, 3),
            "bytes_per_sec": round(self.bytes_per_sec, 1),
        }


@dataclass
class TraceSummary:
    """Workload characteristics extracted from a trace."""

    backend: str
    io_write: OpStats
    io_read: OpStats
    logical_put: OpStats
    logical_get: OpStats
    logical_remove: OpStats
    unique_write_paths: int
    unique_read_paths: int
    steady_state_files: int
    use_odirect: bool
    events_total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "events_total": self.events_total,
            "use_odirect": self.use_odirect,
            "unique_write_paths": self.unique_write_paths,
            "unique_read_paths": self.unique_read_paths,
            "steady_state_files": self.steady_state_files,
            "io_write": self.io_write.to_dict(),
            "io_read": self.io_read.to_dict(),
            "logical_put": self.logical_put.to_dict(),
            "logical_get": self.logical_get.to_dict(),
            "logical_remove": self.logical_remove.to_dict(),
        }


def load_events(path: Path | str) -> list[dict[str, Any]]:
    """Load trace events from a JSONL file (skips unparseable lines)."""
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _max_concurrency(intervals: list[tuple[float, float]]) -> int:
    """Maximum number of overlapping [start, end] intervals."""
    if not intervals:
        return 0
    points: list[tuple[float, int]] = []
    for start, end in intervals:
        points.append((start, 1))
        points.append((end, -1))
    # Ends sort before starts at the same timestamp: back-to-back ops on
    # one thread must not count as concurrent.
    points.sort(key=lambda p: (p[0], p[1]))
    concurrency = 0
    peak = 0
    for _, delta in points:
        concurrency += delta
        peak = max(peak, concurrency)
    return peak


def _collect(events: list[dict[str, Any]], event: str, op: str, backend: str) -> OpStats:
    stats = OpStats()
    intervals: list[tuple[float, float]] = []
    for e in events:
        if e.get("event") != event or e.get("op") != op or e.get("backend") != backend:
            continue
        stats.count += 1
        size = e.get("size") or 0
        stats.total_bytes += size
        if size:
            stats.sizes[size] += 1
        mono = e.get("mono_ms")
        dur = e.get("dur_ms")
        if dur is not None:
            stats.durations_ms.append(dur)
        if e.get("tid") is not None:
            stats.threads.add(e["tid"])
        if mono is not None:
            # mono_ms is recorded at operation END for measured ops (the
            # recorder logs after the wrapped call returns); reconstruct
            # the start for overlap analysis.
            end = mono
            start = mono - (dur or 0.0)
            intervals.append((start, end))
            stats.first_mono_ms = min(stats.first_mono_ms, start) if stats.first_mono_ms is not None else start
            stats.last_mono_ms = max(stats.last_mono_ms, end) if stats.last_mono_ms is not None else end
    stats.max_concurrency = _max_concurrency(intervals)
    return stats


def summarize(
    events: list[dict[str, Any]],
    backend: str = "LocalDiskBackend",
) -> TraceSummary:
    """Summarize a trace for one backend.

    Args:
        events: Parsed trace events.
        backend: Backend class name to analyze.

    Returns:
        TraceSummary with op mix, sizes, concurrency, and churn.
    """
    io_write = _collect(events, "io", "write", backend)
    io_read = _collect(events, "io", "read", backend)
    logical_put = _collect(events, "logical", "put", backend)
    logical_get = _collect(events, "logical", "get", backend)
    logical_remove = _collect(events, "logical", "remove", backend)

    write_paths = {e["path"] for e in events if e.get("event") == "io" and e.get("op") == "write" and e.get("backend") == backend and e.get("path")}
    read_paths = {e["path"] for e in events if e.get("event") == "io" and e.get("op") == "read" and e.get("backend") == backend and e.get("path")}

    use_odirect = any(
        e.get("event") == "meta" and e.get("backend") == backend and e.get("use_odirect")
        for e in events
    )

    steady_state_files = max(0, io_write.count - logical_remove.count)

    return TraceSummary(
        backend=backend,
        io_write=io_write,
        io_read=io_read,
        logical_put=logical_put,
        logical_get=logical_get,
        logical_remove=logical_remove,
        unique_write_paths=len(write_paths),
        unique_read_paths=len(read_paths),
        steady_state_files=steady_state_files,
        use_odirect=use_odirect,
        events_total=len(events),
    )


def list_backends(events: list[dict[str, Any]]) -> list[str]:
    """List backend names present in a trace."""
    return sorted({e["backend"] for e in events if e.get("backend")})
