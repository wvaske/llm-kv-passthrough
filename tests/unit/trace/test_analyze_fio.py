"""Tests for trace analysis and FIO job generation."""

from __future__ import annotations

import json

import pytest

from kvbench.trace.analyze import (
    _max_concurrency,
    list_backends,
    load_events,
    summarize,
)
from kvbench.trace.fio import generate_fio_job

CHUNK = 32 * 1024 * 1024


def _event(event, op, mono_ms, dur_ms=None, size=CHUNK, tid=1, path=None, **extra):
    e = {
        "ts": 1000.0 + mono_ms / 1000,
        "mono_ms": mono_ms,
        "tid": tid,
        "thread": f"t{tid}",
        "event": event,
        "op": op,
        "backend": "LocalDiskBackend",
        "size": size,
    }
    if dur_ms is not None:
        e["dur_ms"] = dur_ms
    if path is not None:
        e["path"] = path
    e.update(extra)
    return e


@pytest.fixture
def sample_events():
    events = [
        {"event": "meta", "op": "backend_info", "backend": "LocalDiskBackend", "use_odirect": True},
    ]
    # 3 writes on 2 threads; writes 1 and 2 overlap in time
    events.append(_event("io", "write", mono_ms=110, dur_ms=10, tid=1, path="/c/a"))
    events.append(_event("io", "write", mono_ms=112, dur_ms=10, tid=2, path="/c/b"))
    events.append(_event("io", "write", mono_ms=130, dur_ms=10, tid=1, path="/c/c"))
    # 1 read, no overlap
    events.append(_event("io", "read", mono_ms=150, dur_ms=5, tid=3, path="/c/a"))
    # eviction
    events.append(_event("logical", "remove", mono_ms=160, size=None, tid=1, key="k1"))
    return events


class TestMaxConcurrency:
    def test_empty(self):
        assert _max_concurrency([]) == 0

    def test_no_overlap(self):
        assert _max_concurrency([(0, 10), (10, 20), (20, 30)]) == 1

    def test_full_overlap(self):
        assert _max_concurrency([(0, 10), (0, 10), (0, 10)]) == 3

    def test_partial_overlap(self):
        assert _max_concurrency([(0, 10), (5, 15), (12, 20)]) == 2


class TestSummarize:
    def test_summary(self, sample_events):
        summary = summarize(sample_events, backend="LocalDiskBackend")
        assert summary.io_write.count == 3
        assert summary.io_write.total_bytes == 3 * CHUNK
        assert summary.io_write.dominant_size == CHUNK
        assert summary.io_write.max_concurrency == 2
        assert len(summary.io_write.threads) == 2
        assert summary.io_read.count == 1
        assert summary.logical_remove.count == 1
        assert summary.steady_state_files == 2
        assert summary.use_odirect is True
        assert summary.unique_write_paths == 3

    def test_list_backends(self, sample_events):
        assert list_backends(sample_events) == ["LocalDiskBackend"]

    def test_load_events_skips_garbage(self, tmp_path):
        p = tmp_path / "t.jsonl"
        p.write_text('{"event":"io","op":"write","backend":"X"}\nnot json\n\n')
        events = load_events(p)
        assert len(events) == 1


class TestGenerateFioJob:
    def test_generates_valid_job(self, sample_events):
        summary = summarize(sample_events)
        job = generate_fio_job(summary, directory="/data/kv", runtime_s=60)
        assert "[global]" in job
        assert "[kv-chunk-write]" in job
        assert "[kv-chunk-read]" in job
        assert "directory=/data/kv" in job
        assert "bs=32M" in job
        assert "filesize=32M" in job
        assert "direct=1" in job
        assert "numjobs=2" in job  # write concurrency
        assert "runtime=60" in job

    def test_paced_adds_rate(self, sample_events):
        summary = summarize(sample_events)
        job = generate_fio_job(summary, paced=True)
        assert "rate=" in job

    def test_no_reads_omits_read_job(self, sample_events):
        events = [e for e in sample_events if e.get("op") != "read"]
        summary = summarize(events)
        job = generate_fio_job(summary)
        assert "[kv-chunk-read]" not in job

    def test_no_writes_raises(self):
        summary = summarize([])
        with pytest.raises(ValueError, match="no physical write events"):
            generate_fio_job(summary)

    def test_roundtrip_through_file(self, tmp_path, sample_events):
        p = tmp_path / "trace.jsonl"
        with open(p, "w") as f:
            for e in sample_events:
                f.write(json.dumps(e) + "\n")
        summary = summarize(load_events(p))
        job = generate_fio_job(summary)
        assert "bs=32M" in job
