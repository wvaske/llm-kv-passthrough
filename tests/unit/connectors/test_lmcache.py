"""Unit tests for kvbench.connectors.lmcache module."""

from __future__ import annotations

import pytest

from kvbench.connectors.lmcache import LMCacheConfig, LMCacheConnector
from kvbench.storage.memory import MemoryStorageBackend


class TestLMCacheConfig:
    """Tests for LMCacheConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = LMCacheConfig()
        assert config.model_name == "llama-3.1-8b"
        assert config.world_size == 1
        assert config.worker_id == 0
        assert config.chunk_size == 256
        assert config.key_format == "lmcache"
        assert config.key_suffix == "@kv_bytes"
        assert config.bytes_per_token == 512

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = LMCacheConfig(
            model_name="llama-3.1-70b",
            world_size=4,
            worker_id=2,
            chunk_size=512,
        )
        assert config.model_name == "llama-3.1-70b"
        assert config.world_size == 4
        assert config.worker_id == 2
        assert config.chunk_size == 512


class TestLMCacheConnector:
    """Tests for LMCacheConnector."""

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    @pytest.fixture
    async def connector(
        self, storage: MemoryStorageBackend
    ) -> LMCacheConnector:
        """Create a test connector instance."""
        connector = LMCacheConnector(storage=storage, name="test")
        yield connector
        await connector.close()

    def test_make_key(self, storage: MemoryStorageBackend) -> None:
        """Test key generation follows LMCache format."""
        config = LMCacheConfig(
            model_name="llama-3.1-8b",
            world_size=4,
            worker_id=2,
        )
        connector = LMCacheConnector(storage=storage, config=config)

        key = connector.make_key("abc123def456")
        expected = "lmcache@llama-3.1-8b@4@2@abc123def456@kv_bytes"
        assert key == expected

    def test_make_key_override_worker(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test key generation with worker_id override."""
        connector = LMCacheConnector(storage=storage)
        key = connector.make_key("abc123", worker_id=3)
        assert "@3@" in key

    def test_make_keys_for_all_workers(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test generating keys for all workers."""
        config = LMCacheConfig(world_size=4)
        connector = LMCacheConnector(storage=storage, config=config)

        keys = connector.make_keys_for_all_workers("abc123")
        assert len(keys) == 4
        for i, key in enumerate(keys):
            assert f"@{i}@" in key

    def test_compute_chunk_hash(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test chunk hash computation is consistent."""
        connector = LMCacheConnector(storage=storage)

        hash1 = connector.compute_chunk_hash([1, 2, 3, 4, 5])
        hash2 = connector.compute_chunk_hash([1, 2, 3, 4, 5])
        hash3 = connector.compute_chunk_hash([5, 4, 3, 2, 1])

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA-256 hex

    @pytest.mark.asyncio
    async def test_store_and_load(
        self, connector: LMCacheConnector
    ) -> None:
        """Test basic store and load operations."""
        result = await connector.store("chunk1", num_tokens=100)
        assert result is True
        assert connector.stats.stores == 1

        data = await connector.load("chunk1")
        assert data is not None
        assert len(data) == 100 * 512  # bytes_per_token
        assert connector.stats.hits == 1

    @pytest.mark.asyncio
    async def test_load_nonexistent(
        self, connector: LMCacheConnector
    ) -> None:
        """Test loading nonexistent chunk returns None."""
        data = await connector.load("nonexistent")
        assert data is None
        assert connector.stats.misses == 1

    @pytest.mark.asyncio
    async def test_exists(self, connector: LMCacheConnector) -> None:
        """Test exists operation."""
        assert await connector.exists("chunk1") is False

        await connector.store("chunk1", num_tokens=100)
        assert await connector.exists("chunk1") is True

    @pytest.mark.asyncio
    async def test_delete(self, connector: LMCacheConnector) -> None:
        """Test delete operation."""
        await connector.store("chunk1", num_tokens=100)
        assert await connector.exists("chunk1") is True

        result = await connector.delete("chunk1")
        assert result is True
        assert await connector.exists("chunk1") is False

    @pytest.mark.asyncio
    async def test_delete_nonexistent(
        self, connector: LMCacheConnector
    ) -> None:
        """Test deleting nonexistent chunk returns False."""
        result = await connector.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_metadata(
        self, connector: LMCacheConnector
    ) -> None:
        """Test getting chunk metadata."""
        await connector.store("chunk1", num_tokens=100)

        metadata = await connector.get_metadata("chunk1")
        assert metadata is not None
        assert metadata.chunk_hash == "chunk1"
        assert metadata.num_tokens == 100

    @pytest.mark.asyncio
    async def test_store_many(
        self, connector: LMCacheConnector
    ) -> None:
        """Test batch store operation."""
        chunks = [("chunk1", 100), ("chunk2", 200), ("chunk3", 150)]
        results = await connector.store_many(chunks)

        assert results["chunk1"] is True
        assert results["chunk2"] is True
        assert results["chunk3"] is True
        assert connector.stats.stores == 3

    @pytest.mark.asyncio
    async def test_load_many(
        self, connector: LMCacheConnector
    ) -> None:
        """Test batch load operation."""
        await connector.store("chunk1", num_tokens=100)
        await connector.store("chunk2", num_tokens=200)

        results = await connector.load_many(["chunk1", "chunk2", "chunk3"])

        assert results["chunk1"] is not None
        assert results["chunk2"] is not None
        assert results["chunk3"] is None

    @pytest.mark.asyncio
    async def test_exists_many(
        self, connector: LMCacheConnector
    ) -> None:
        """Test batch exists operation."""
        await connector.store("chunk1", num_tokens=100)

        results = await connector.exists_many(["chunk1", "chunk2"])

        assert results["chunk1"] is True
        assert results["chunk2"] is False

    @pytest.mark.asyncio
    async def test_store_with_custom_data(
        self, connector: LMCacheConnector
    ) -> None:
        """Test storing with custom data."""
        custom_data = b"custom kv cache data"
        result = await connector.store("chunk1", num_tokens=100, data=custom_data)
        assert result is True

        loaded = await connector.load("chunk1")
        assert loaded == custom_data

    @pytest.mark.asyncio
    async def test_clear(self, connector: LMCacheConnector) -> None:
        """Test clearing all data."""
        await connector.store("chunk1", num_tokens=100)
        await connector.store("chunk2", num_tokens=200)

        count = await connector.clear()
        assert count == 2
        assert await connector.exists("chunk1") is False

    @pytest.mark.asyncio
    async def test_close(self, storage: MemoryStorageBackend) -> None:
        """Test closing the connector."""
        connector = LMCacheConnector(storage=storage)
        await connector.store("chunk1", num_tokens=100)
        await connector.close()

        assert connector.is_closed is True

    @pytest.mark.asyncio
    async def test_operations_after_close(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test that operations fail after close."""
        connector = LMCacheConnector(storage=storage)
        await connector.close()

        with pytest.raises(RuntimeError, match="closed"):
            await connector.store("chunk1", num_tokens=100)

    @pytest.mark.asyncio
    async def test_context_manager(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test async context manager usage."""
        async with LMCacheConnector(storage=storage) as connector:
            await connector.store("chunk1", num_tokens=100)
            data = await connector.load("chunk1")
            assert data is not None

        assert connector.is_closed is True

    def test_repr(self, storage: MemoryStorageBackend) -> None:
        """Test string representation."""
        config = LMCacheConfig(
            model_name="llama-3.1-70b",
            world_size=4,
            worker_id=2,
            chunk_size=512,
        )
        connector = LMCacheConnector(storage=storage, config=config, name="test")
        repr_str = repr(connector)

        assert "LMCacheConnector" in repr_str
        assert "llama-3.1-70b" in repr_str
        assert "2/4" in repr_str
        assert "512" in repr_str
