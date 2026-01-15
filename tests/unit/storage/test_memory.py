"""Unit tests for kvbench.storage.memory module."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from kvbench.storage.memory import MemoryStorageBackend


class TestMemoryStorageBackend:
    """Tests for MemoryStorageBackend."""

    @pytest.fixture
    def backend(self) -> MemoryStorageBackend:
        """Create a test backend instance."""
        return MemoryStorageBackend(max_size_bytes=1000, name="test")

    @pytest.fixture
    async def async_backend(self) -> MemoryStorageBackend:
        """Create a test backend instance for async tests."""
        backend = MemoryStorageBackend(max_size_bytes=1000, name="test")
        yield backend
        await backend.close()

    def test_init(self, backend: MemoryStorageBackend) -> None:
        """Test backend initialization."""
        assert backend.name == "test"
        assert backend.max_size_bytes == 1000
        assert backend.stats.total_bytes == 1000
        assert backend.stats.used_bytes == 0
        assert backend.stats.item_count == 0

    @pytest.mark.asyncio
    async def test_put_and_get(self, async_backend: MemoryStorageBackend) -> None:
        """Test basic put and get operations."""
        result = await async_backend.put("key1", b"value1")
        assert result is True

        value = await async_backend.get("key1")
        assert value == b"value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, async_backend: MemoryStorageBackend) -> None:
        """Test getting a nonexistent key returns None."""
        value = await async_backend.get("nonexistent")
        assert value is None
        assert async_backend.stats.misses == 1

    @pytest.mark.asyncio
    async def test_put_updates_stats(self, async_backend: MemoryStorageBackend) -> None:
        """Test that put updates statistics correctly."""
        await async_backend.put("key1", b"12345")
        assert async_backend.stats.used_bytes == 5
        assert async_backend.stats.item_count == 1

    @pytest.mark.asyncio
    async def test_put_update_existing(self, async_backend: MemoryStorageBackend) -> None:
        """Test updating an existing key."""
        await async_backend.put("key1", b"12345")
        assert async_backend.stats.used_bytes == 5

        await async_backend.put("key1", b"1234567890")
        assert async_backend.stats.used_bytes == 10
        assert async_backend.stats.item_count == 1  # Still only 1 item

    @pytest.mark.asyncio
    async def test_delete(self, async_backend: MemoryStorageBackend) -> None:
        """Test delete operation."""
        await async_backend.put("key1", b"value1")
        assert async_backend.stats.item_count == 1

        result = await async_backend.delete("key1")
        assert result is True
        assert async_backend.stats.item_count == 0
        assert async_backend.stats.used_bytes == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, async_backend: MemoryStorageBackend) -> None:
        """Test deleting a nonexistent key returns False."""
        result = await async_backend.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, async_backend: MemoryStorageBackend) -> None:
        """Test exists operation."""
        assert await async_backend.exists("key1") is False

        await async_backend.put("key1", b"value1")
        assert await async_backend.exists("key1") is True

    @pytest.mark.asyncio
    async def test_keys(self, async_backend: MemoryStorageBackend) -> None:
        """Test listing keys."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")
        await async_backend.put("other", b"value3")

        all_keys = await async_backend.keys()
        assert set(all_keys) == {"key1", "key2", "other"}

    @pytest.mark.asyncio
    async def test_keys_with_pattern(self, async_backend: MemoryStorageBackend) -> None:
        """Test listing keys with pattern filter."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")
        await async_backend.put("other", b"value3")

        keys = await async_backend.keys("key*")
        assert set(keys) == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_clear(self, async_backend: MemoryStorageBackend) -> None:
        """Test clearing all data."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")

        count = await async_backend.clear()
        assert count == 2
        assert async_backend.stats.item_count == 0
        assert async_backend.stats.used_bytes == 0

    @pytest.mark.asyncio
    async def test_lru_eviction(self) -> None:
        """Test LRU eviction when capacity is exceeded."""
        backend = MemoryStorageBackend(max_size_bytes=20, name="test")
        try:
            # Put items that total 15 bytes
            await backend.put("key1", b"12345")  # 5 bytes
            await backend.put("key2", b"12345")  # 5 bytes
            await backend.put("key3", b"12345")  # 5 bytes

            assert backend.stats.item_count == 3
            assert backend.stats.used_bytes == 15

            # Access key1 to make it recently used
            await backend.get("key1")

            # Add another item that requires eviction
            await backend.put("key4", b"123456789")  # 9 bytes, needs 4 more bytes

            # key2 should be evicted (LRU after key1 was accessed)
            assert await backend.exists("key1") is True
            assert await backend.exists("key2") is False  # Evicted
            assert await backend.exists("key3") is True
            assert await backend.exists("key4") is True
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_put_too_large(self) -> None:
        """Test that values too large for storage are rejected."""
        backend = MemoryStorageBackend(max_size_bytes=10, name="test")
        try:
            result = await backend.put("key1", b"12345678901234567890")  # 20 bytes
            assert result is False
            assert backend.stats.item_count == 0
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_ttl_expiration(self) -> None:
        """Test TTL-based expiration."""
        backend = MemoryStorageBackend(max_size_bytes=1000, name="test")
        try:
            # Create item with expired TTL by manipulating metadata
            await backend.put("key1", b"value1", ttl_seconds=1)

            # Manually expire the item
            meta = backend.get_metadata("key1")
            assert meta is not None
            meta.created_at = datetime.now() - timedelta(seconds=10)
            meta.ttl_seconds = 1

            # Item should be expired
            value = await backend.get("key1")
            assert value is None
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_get_many(self, async_backend: MemoryStorageBackend) -> None:
        """Test batch get operation."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")

        results = await async_backend.get_many(["key1", "key2", "key3"])
        assert results["key1"] == b"value1"
        assert results["key2"] == b"value2"
        assert results["key3"] is None

    @pytest.mark.asyncio
    async def test_put_many(self, async_backend: MemoryStorageBackend) -> None:
        """Test batch put operation."""
        items = {"key1": b"value1", "key2": b"value2"}
        results = await async_backend.put_many(items)

        assert results["key1"] is True
        assert results["key2"] is True
        assert async_backend.stats.item_count == 2

    @pytest.mark.asyncio
    async def test_delete_many(self, async_backend: MemoryStorageBackend) -> None:
        """Test batch delete operation."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")

        results = await async_backend.delete_many(["key1", "key2", "key3"])
        assert results["key1"] is True
        assert results["key2"] is True
        assert results["key3"] is False

    @pytest.mark.asyncio
    async def test_size(self, async_backend: MemoryStorageBackend) -> None:
        """Test size operation."""
        await async_backend.put("key1", b"12345")

        size = await async_backend.size("key1")
        assert size == 5

        size = await async_backend.size("nonexistent")
        assert size is None

    @pytest.mark.asyncio
    async def test_close(self, async_backend: MemoryStorageBackend) -> None:
        """Test closing the backend."""
        await async_backend.put("key1", b"value1")
        await async_backend.close()

        assert async_backend.is_closed is True

    @pytest.mark.asyncio
    async def test_operations_after_close(self) -> None:
        """Test that operations fail after close."""
        backend = MemoryStorageBackend(max_size_bytes=1000, name="test")
        await backend.close()

        with pytest.raises(RuntimeError, match="closed"):
            await backend.get("key1")

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager usage."""
        async with MemoryStorageBackend(max_size_bytes=1000, name="test") as backend:
            await backend.put("key1", b"value1")
            value = await backend.get("key1")
            assert value == b"value1"

        assert backend.is_closed is True

    @pytest.mark.asyncio
    async def test_repr(self, async_backend: MemoryStorageBackend) -> None:
        """Test string representation."""
        repr_str = repr(async_backend)
        assert "MemoryStorageBackend" in repr_str
        assert "test" in repr_str
        assert "1000" in repr_str

    @pytest.mark.asyncio
    async def test_hit_miss_stats(self, async_backend: MemoryStorageBackend) -> None:
        """Test hit/miss statistics."""
        await async_backend.put("key1", b"value1")

        await async_backend.get("key1")  # Hit
        await async_backend.get("key1")  # Hit
        await async_backend.get("key2")  # Miss

        assert async_backend.stats.hits == 2
        assert async_backend.stats.misses == 1
        assert async_backend.stats.hit_rate == pytest.approx(66.67, rel=0.01)

    @pytest.mark.asyncio
    async def test_concurrent_access(self) -> None:
        """Test concurrent access to the backend."""
        backend = MemoryStorageBackend(max_size_bytes=10000, name="test")
        try:
            async def write_task(key: str, value: bytes) -> None:
                await backend.put(key, value)

            async def read_task(key: str) -> bytes | None:
                return await backend.get(key)

            # Concurrent writes
            await asyncio.gather(
                *[write_task(f"key{i}", f"value{i}".encode()) for i in range(100)]
            )

            assert backend.stats.item_count == 100

            # Concurrent reads
            results = await asyncio.gather(
                *[read_task(f"key{i}") for i in range(100)]
            )

            assert all(r is not None for r in results)
        finally:
            await backend.close()
