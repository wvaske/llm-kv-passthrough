"""Tests for the KV trace recorder."""

from __future__ import annotations

import json
import threading

import pytest

from kvbench.trace.recorder import TraceRecorder, install_lmcache_trace


class FakeMemoryObj:
    def __init__(self, size: int) -> None:
        self.byte_array = bytearray(size)


class FakeDiskBackend:
    """Duck-typed stand-in for LMCache's LocalDiskBackend."""

    use_odirect = False
    os_disk_bs = 4096

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.store: dict[str, FakeMemoryObj] = {}

    def batched_submit_put_task(self, keys, memory_objs, transfer_spec=None):  # noqa: ARG002
        self.calls.append("put")
        for key, obj in zip(keys, memory_objs, strict=True):
            self.store[str(key)] = obj
            self.write_file(obj.byte_array, f"/fake/{key}")

    def get_blocking(self, key):
        self.calls.append("get")
        obj = self.store.get(str(key))
        if obj is not None:
            self.read_file(key, obj.byte_array, f"/fake/{key}")
        return obj

    def contains(self, key, pin=False):  # noqa: ARG002
        self.calls.append("contains")
        return str(key) in self.store

    def remove(self, key, force=True):  # noqa: ARG002
        self.calls.append("remove")
        return self.store.pop(str(key), None) is not None

    def write_file(self, buffer, path):  # noqa: ARG002
        self.calls.append("write_file")

    def read_file(self, key, buffer, path):  # noqa: ARG002
        self.calls.append("read_file")


class FakeCPUBackend:
    pass


FakeCPUBackend.__name__ = "LocalCPUBackend"


class FakeStorageManager:
    def __init__(self, backends):
        self.storage_backends = backends


class FakeEngine:
    def __init__(self, backends):
        self.storage_manager = FakeStorageManager(backends)


@pytest.fixture
def recorder(tmp_path):
    rec = TraceRecorder(tmp_path / "trace.jsonl")
    yield rec
    rec.close()


class TestTraceRecorder:
    def test_records_jsonl_events(self, recorder):
        recorder.record(event="io", op="write", backend="X", size=100, path="/p")
        recorder.close()
        lines = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        assert len(lines) == 1
        event = lines[0]
        assert event["event"] == "io"
        assert event["op"] == "write"
        assert event["size"] == 100
        assert event["path"] == "/p"
        assert "ts" in event and "mono_ms" in event and "tid" in event

    def test_counters_accumulate(self, recorder):
        recorder.record(event="io", op="write", backend="X", size=100)
        recorder.record(event="io", op="write", backend="X", size=50)
        assert recorder.counters[("io", "write", "X")] == [2, 150]

    def test_truncates_on_open(self, tmp_path):
        path = tmp_path / "trace.jsonl"
        rec1 = TraceRecorder(path)
        rec1.record(event="io", op="write", backend="X", size=1)
        rec1.close()
        rec2 = TraceRecorder(path)
        rec2.record(event="io", op="read", backend="X", size=2)
        rec2.close()
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(lines) == 1
        assert lines[0]["op"] == "read"

    def test_record_after_close_is_noop(self, recorder):
        recorder.close()
        recorder.record(event="io", op="write", backend="X", size=1)

    def test_thread_safety(self, recorder):
        def writer():
            for _ in range(200):
                recorder.record(event="io", op="write", backend="X", size=1)

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        recorder.close()
        lines = recorder.path.read_text().splitlines()
        assert len(lines) == 1600
        for line in lines:
            json.loads(line)


class TestInstallLmcacheTrace:
    def test_wraps_disk_backend_ops(self, recorder):
        disk = FakeDiskBackend()
        engine = FakeEngine({"LocalDiskBackend": disk})
        wrapped = install_lmcache_trace(engine, recorder)
        assert wrapped == ["LocalDiskBackend"]

        key = "chunk-hash-1"
        disk.batched_submit_put_task([key], [FakeMemoryObj(1024)])
        assert disk.contains(key)
        assert disk.get_blocking(key) is not None
        assert disk.remove(key)

        recorder.close()
        events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
        ops = [(e["event"], e["op"]) for e in events]
        assert ("meta", "backend_info") in ops
        assert ("logical", "put") in ops
        assert ("io", "write") in ops
        assert ("logical", "contains") in ops
        assert ("logical", "get") in ops
        assert ("io", "read") in ops
        assert ("logical", "remove") in ops

        put = next(e for e in events if e["op"] == "put")
        assert put["size"] == 1024
        assert put["key"] == key

    def test_skips_cpu_backend(self, recorder):
        cpu = FakeCPUBackend()
        engine = FakeEngine({"LocalCPUBackend": cpu})
        wrapped = install_lmcache_trace(engine, recorder)
        assert wrapped == []

    def test_missing_surface_raises(self, recorder):
        with pytest.raises(RuntimeError, match="storage_backends"):
            install_lmcache_trace(object(), recorder)
