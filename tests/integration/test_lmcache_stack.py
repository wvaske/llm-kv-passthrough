"""Integration tests for the real LMCache KV stack.

These tests drive the actual LMCache engine (CPU-only, no GPU required)
and verify that KV-Bench's only storage path is LMCache: data is stored,
looked up, and retrieved through the real engine, and disk artifacts are
written by LMCache itself in its own key format.

Skipped when lmcache is not installed (pip install 'kvbench[lmcache]').
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

lmcache = pytest.importorskip("lmcache")

from kvbench.core.config import KVBenchConfig, KVStackConfig, ServerConfig  # noqa: E402
from kvbench.kv.factory import create_kv_stack  # noqa: E402
from kvbench.kv.lmcache_stack import LMCacheStack  # noqa: E402

pytestmark = pytest.mark.integration


def write_lmcache_config(tmp_path: Path, chunk_size: int = 256) -> Path:
    """Write a minimal LMCache application config with a disk tier."""
    disk_path = tmp_path / "lmcache-disk"
    disk_path.mkdir()
    config_path = tmp_path / "lmcache.yaml"
    config_path.write_text(
        f"""\
chunk_size: {chunk_size}
local_cpu: true
max_local_cpu_size: 1.0
local_disk: "file://{disk_path}/"
max_local_disk_size: 2.0
"""
    )
    return config_path


@pytest.fixture
async def stack(tmp_path: Path):
    """A started LMCacheStack on a small model profile with a disk tier."""
    stack = LMCacheStack(
        model_profile="llama-3.1-8b",
        instance_id=f"kvbench-test-{uuid.uuid4().hex[:8]}",
        config_file=write_lmcache_config(tmp_path),
    )
    await stack.start()
    yield stack
    await stack.close()


class TestLMCacheStack:
    """Round-trip tests through the real LMCache engine."""

    @pytest.mark.asyncio
    async def test_store_lookup_retrieve_roundtrip(self, stack: LMCacheStack) -> None:
        tokens = list(range(512))

        assert await stack.lookup(tokens) == 0

        await stack.store(tokens)
        assert await stack.lookup(tokens) == 512
        assert await stack.retrieve(tokens) == 512

        assert stack.stats.stores == 1
        assert stack.stats.retrieved_tokens == 512

    @pytest.mark.asyncio
    async def test_prefix_lookup(self, stack: LMCacheStack) -> None:
        """A stored sequence's chunk-aligned prefixes are cache hits."""
        tokens = list(range(1024))
        await stack.store(tokens)

        assert await stack.lookup(tokens[:512]) == 512
        assert await stack.lookup(tokens[:256]) == 256

    @pytest.mark.asyncio
    async def test_different_content_does_not_hit(self, stack: LMCacheStack) -> None:
        """Different token content must not collide (LMCache's own hashing)."""
        await stack.store(list(range(512)))
        other = [t + 1_000_000 for t in range(512)]
        assert await stack.lookup(other) == 0

    @pytest.mark.asyncio
    async def test_store_with_skip_leading(self, stack: LMCacheStack) -> None:
        """Storing with a cached-prefix mask completes the sequence."""
        tokens = list(range(512))
        await stack.store(tokens[:256])
        hit = await stack.lookup(tokens)
        assert hit == 256

        await stack.store(tokens, skip_leading=hit)
        assert await stack.lookup(tokens) == 512

    @pytest.mark.asyncio
    async def test_lmcache_owns_the_disk_artifacts(
        self, stack: LMCacheStack, tmp_path: Path
    ) -> None:
        """All disk I/O is performed by LMCache in its own key format."""
        import asyncio

        await stack.store(list(range(512)))
        await asyncio.sleep(1.0)  # LMCache's disk writes are asynchronous

        disk_path = tmp_path / "lmcache-disk"
        files = list(disk_path.rglob("*"))
        artifacts = [f for f in files if f.is_file()]
        assert artifacts, "LMCache wrote no disk artifacts"
        # LMCache key format: <model>@<world_size>@<worker_id>@<hash>...
        assert any("llama-3.1-8b@" in f.name for f in artifacts)

    @pytest.mark.asyncio
    async def test_chunk_size_comes_from_lmcache_config(self, tmp_path: Path) -> None:
        """The stack reports the chunk size from LMCache's own config."""
        stack = LMCacheStack(
            model_profile="llama-3.1-8b",
            instance_id=f"kvbench-test-{uuid.uuid4().hex[:8]}",
            config_file=write_lmcache_config(tmp_path, chunk_size=128),
        )
        await stack.start()
        try:
            assert stack.chunk_size == 128
        finally:
            await stack.close()


class TestFactory:
    """The factory builds a real LMCache stack from KV-Bench config."""

    @pytest.mark.asyncio
    async def test_create_and_run(self, tmp_path: Path) -> None:
        config = KVBenchConfig(
            instance_id=f"kvbench-test-{uuid.uuid4().hex[:8]}",
            kv=KVStackConfig(lmcache_config_file=write_lmcache_config(tmp_path)),
            server=ServerConfig(model_profile="llama-3.1-8b"),
        )
        stack = create_kv_stack(config)
        assert isinstance(stack, LMCacheStack)
        await stack.start()
        try:
            tokens = list(range(256))
            await stack.store(tokens)
            assert await stack.lookup(tokens) == 256
        finally:
            await stack.close()
