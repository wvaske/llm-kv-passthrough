"""Unit tests for kvbench.storage.local_disk module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kvbench.storage.local_disk import LocalDiskStorageBackend


class TestLocalDiskStorageBackend:
    """Tests for LocalDiskStorageBackend."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def async_backend(self, temp_dir: Path) -> LocalDiskStorageBackend:
        """Create a test backend instance for async tests."""
        backend = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=10000,
            name="test",
        )
        yield backend
        await backend.close()

    def test_init(self, temp_dir: Path) -> None:
        """Test backend initialization."""
        backend = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=1000,
            name="test",
        )
        assert backend.name == "test"
        assert backend.max_size_bytes == 1000
        assert backend.base_path == temp_dir

    def test_hash_key(self, temp_dir: Path) -> None:
        """Test key hashing produces consistent results."""
        backend = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=1000,
        )
        hash1 = backend._hash_key("test_key")
        hash2 = backend._hash_key("test_key")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_shard_path(self, temp_dir: Path) -> None:
        """Test shard path generation."""
        backend = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=1000,
            shard_depth=2,
            shard_width=2,
        )
        key_hash = "abcdef1234567890"
        shard_path = backend._get_shard_path(key_hash)

        # Should create path like: base/ab/cd/
        assert str(shard_path).endswith("ab/cd")

    @pytest.mark.asyncio
    async def test_put_and_get(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test basic put and get operations."""
        result = await async_backend.put("key1", b"value1")
        assert result is True

        value = await async_backend.get("key1")
        assert value == b"value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test getting a nonexistent key returns None."""
        value = await async_backend.get("nonexistent")
        assert value is None

    @pytest.mark.asyncio
    async def test_put_creates_directories(
        self, async_backend: LocalDiskStorageBackend
    ) -> None:
        """Test that put creates shard directories."""
        await async_backend.put("key1", b"value1")

        file_path = async_backend._get_file_path("key1")
        assert file_path.exists()
        assert file_path.parent.exists()

    @pytest.mark.asyncio
    async def test_delete(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test delete operation."""
        await async_backend.put("key1", b"value1")
        file_path = async_backend._get_file_path("key1")
        assert file_path.exists()

        result = await async_backend.delete("key1")
        assert result is True
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self, async_backend: LocalDiskStorageBackend
    ) -> None:
        """Test deleting a nonexistent key returns False."""
        result = await async_backend.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test exists operation."""
        assert await async_backend.exists("key1") is False

        await async_backend.put("key1", b"value1")
        assert await async_backend.exists("key1") is True

    @pytest.mark.asyncio
    async def test_keys(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test listing keys."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")
        await async_backend.put("other", b"value3")

        all_keys = await async_backend.keys()
        assert set(all_keys) == {"key1", "key2", "other"}

    @pytest.mark.asyncio
    async def test_keys_with_pattern(
        self, async_backend: LocalDiskStorageBackend
    ) -> None:
        """Test listing keys with pattern filter."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")
        await async_backend.put("other", b"value3")

        keys = await async_backend.keys("key*")
        assert set(keys) == {"key1", "key2"}

    @pytest.mark.asyncio
    async def test_clear(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test clearing all data."""
        await async_backend.put("key1", b"value1")
        await async_backend.put("key2", b"value2")

        count = await async_backend.clear()
        assert count == 2
        assert async_backend.stats.item_count == 0

    @pytest.mark.asyncio
    async def test_size(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test size operation."""
        await async_backend.put("key1", b"12345")

        size = await async_backend.size("key1")
        assert size == 5

        size = await async_backend.size("nonexistent")
        assert size is None

    @pytest.mark.asyncio
    async def test_put_update_existing(
        self, async_backend: LocalDiskStorageBackend
    ) -> None:
        """Test updating an existing key."""
        await async_backend.put("key1", b"12345")
        size1 = await async_backend.size("key1")
        assert size1 == 5

        await async_backend.put("key1", b"1234567890")
        size2 = await async_backend.size("key1")
        assert size2 == 10

    @pytest.mark.asyncio
    async def test_capacity_limit(self, temp_dir: Path) -> None:
        """Test that capacity limits are enforced."""
        backend = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=10,
            name="test",
        )
        try:
            result1 = await backend.put("key1", b"12345")  # 5 bytes
            assert result1 is True

            result2 = await backend.put("key2", b"123456")  # 6 bytes, would exceed
            assert result2 is False
        finally:
            await backend.close()

    @pytest.mark.asyncio
    async def test_stats_persistence(self, temp_dir: Path) -> None:
        """Test that statistics are persisted and loaded."""
        # Create backend and add some data
        backend1 = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=10000,
            name="test",
        )
        await backend1.put("key1", b"value1")
        await backend1.put("key2", b"value2")
        await backend1.close()

        # Create new backend with same path
        backend2 = LocalDiskStorageBackend(
            base_path=temp_dir,
            max_size_bytes=10000,
            name="test",
        )
        # Trigger initialization
        await backend2.exists("key1")

        assert backend2.stats.item_count == 2
        await backend2.close()

    @pytest.mark.asyncio
    async def test_close(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test closing the backend."""
        await async_backend.put("key1", b"value1")
        await async_backend.close()

        assert async_backend.is_closed is True

    @pytest.mark.asyncio
    async def test_large_values(self, async_backend: LocalDiskStorageBackend) -> None:
        """Test storing large values."""
        large_value = b"x" * 5000
        result = await async_backend.put("large_key", large_value)
        assert result is True

        retrieved = await async_backend.get("large_key")
        assert retrieved == large_value

    @pytest.mark.asyncio
    async def test_special_characters_in_key(
        self, async_backend: LocalDiskStorageBackend
    ) -> None:
        """Test keys with special characters."""
        key = "user:123/session:456@test"
        await async_backend.put(key, b"value")

        value = await async_backend.get(key)
        assert value == b"value"


class TestLocalDiskSharding:
    """Tests for sharding functionality."""

    def test_shard_depth_1(self) -> None:
        """Test sharding with depth 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalDiskStorageBackend(
                base_path=tmpdir,
                max_size_bytes=1000,
                shard_depth=1,
                shard_width=2,
            )
            key_hash = "abcdef1234567890"
            shard_path = backend._get_shard_path(key_hash)
            assert str(shard_path).endswith("ab")

    def test_shard_depth_3(self) -> None:
        """Test sharding with depth 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalDiskStorageBackend(
                base_path=tmpdir,
                max_size_bytes=1000,
                shard_depth=3,
                shard_width=2,
            )
            key_hash = "abcdef1234567890"
            shard_path = backend._get_shard_path(key_hash)
            assert "ab" in str(shard_path)
            assert "cd" in str(shard_path)
            assert "ef" in str(shard_path)

    def test_shard_width_3(self) -> None:
        """Test sharding with width 3."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = LocalDiskStorageBackend(
                base_path=tmpdir,
                max_size_bytes=1000,
                shard_depth=2,
                shard_width=3,
            )
            key_hash = "abcdef1234567890"
            shard_path = backend._get_shard_path(key_hash)
            assert "abc" in str(shard_path)
            assert "def" in str(shard_path)
