"""Tests for the steady-state warmup controller."""

from __future__ import annotations

import pytest

from kvbench.kv.base import KVStack, KVStackStats
from kvbench.servers.warmup import WarmupController, WarmupRequest


@pytest.fixture(autouse=True)
def fast_eviction_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(WarmupController, "EVICTION_CHECK_ATTEMPTS", 1)
    monkeypatch.setattr(WarmupController, "EVICTION_CHECK_DELAY_S", 0.0)

GB = 1024**3


class FakeKVStack(KVStack):
    """In-memory KV stack with a fixed capacity and FIFO eviction."""

    def __init__(self, capacity_bytes: int = GB, bytes_per_token: int = 131072) -> None:
        self.stats = KVStackStats()
        self.capacity_bytes = capacity_bytes
        self.bytes_per_token = bytes_per_token
        self.sequences: list[tuple[int, ...]] = []
        self.stored_bytes = 0

    @property
    def chunk_size(self) -> int:
        return 256

    async def start(self) -> None:
        pass

    async def lookup(self, tokens: list[int]) -> int:
        for seq in self.sequences:
            if tuple(tokens) == seq:
                return len(tokens)
        return 0

    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:  # noqa: ARG002
        self.sequences.append(tuple(tokens))
        self.stored_bytes += len(tokens) * self.bytes_per_token
        while self.stored_bytes > self.capacity_bytes:
            evicted = self.sequences.pop(0)
            self.stored_bytes -= len(evicted) * self.bytes_per_token

    async def retrieve(self, tokens: list[int]) -> int:
        return await self.lookup(tokens)

    async def close(self) -> None:
        pass

    def capacity_info(self) -> dict:
        return {
            "chunk_size": self.chunk_size,
            "bytes_per_token": self.bytes_per_token,
            "chunk_bytes": self.bytes_per_token * self.chunk_size,
            "vocab_size": 128256,
            "total_capacity_bytes": self.capacity_bytes,
        }


class TestWarmupController:
    @pytest.mark.asyncio
    async def test_fills_to_target_and_detects_eviction(self):
        kv = FakeKVStack(capacity_bytes=GB)
        controller = WarmupController(kv)
        controller.start(WarmupRequest(fill_factor=1.5, seq_tokens=512, concurrency=2))
        await controller._task

        status = controller.status
        assert status.state == "done"
        assert status.stored_bytes >= int(GB * 1.5)
        assert status.evicting is True
        assert kv.stored_bytes <= GB

    @pytest.mark.asyncio
    async def test_explicit_target_gb(self):
        kv = FakeKVStack(capacity_bytes=100 * GB)
        controller = WarmupController(kv)
        controller.start(WarmupRequest(target_gb=0.5, seq_tokens=512, concurrency=1))
        await controller._task

        assert controller.status.state == "done"
        assert controller.status.stored_bytes >= int(0.5 * GB)
        # Cache far from full: earliest sequence still present
        assert controller.status.evicting is False

    @pytest.mark.asyncio
    async def test_rejects_concurrent_runs(self):
        kv = FakeKVStack(capacity_bytes=50 * GB)
        controller = WarmupController(kv)
        controller.start(WarmupRequest(target_gb=10.0, seq_tokens=512, concurrency=1))
        with pytest.raises(RuntimeError, match="already running"):
            controller.start(WarmupRequest(target_gb=1.0))
        await controller.cancel()
        assert controller.status.state == "cancelled"

    @pytest.mark.asyncio
    async def test_no_capacity_and_no_target_raises(self):
        kv = FakeKVStack(capacity_bytes=GB)
        kv.capacity_info = lambda: {  # type: ignore[method-assign]
            "chunk_size": 256,
            "bytes_per_token": 131072,
            "total_capacity_bytes": 0,
        }
        controller = WarmupController(kv)
        with pytest.raises(RuntimeError, match="target_gb"):
            controller.start(WarmupRequest())

    @pytest.mark.asyncio
    async def test_sequences_are_unique(self):
        kv = FakeKVStack(capacity_bytes=100 * GB)
        controller = WarmupController(kv)
        controller.start(WarmupRequest(target_gb=1.0, seq_tokens=256, concurrency=4))
        await controller._task
        assert len(set(kv.sequences)) == len(kv.sequences)

    @pytest.mark.asyncio
    async def test_seq_tokens_chunk_aligned(self):
        kv = FakeKVStack(capacity_bytes=100 * GB)
        controller = WarmupController(kv)
        controller.start(WarmupRequest(target_gb=0.1, seq_tokens=300, concurrency=1))
        await controller._task
        assert all(len(seq) == 256 for seq in kv.sequences)
