"""Unit tests for kvbench.connectors.mooncake module."""

from __future__ import annotations

import pytest

from kvbench.connectors.mooncake import MooncakeConnector
from kvbench.storage.memory import MemoryStorageBackend


class TestMooncakeConnector:
    """Tests for MooncakeConnector stub."""

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    @pytest.fixture
    async def connector(
        self, storage: MemoryStorageBackend
    ) -> MooncakeConnector:
        """Create a test connector instance."""
        connector = MooncakeConnector(storage=storage, name="test")
        yield connector
        await connector.close()

    def test_make_key(self, storage: MemoryStorageBackend) -> None:
        """Test key generation."""
        connector = MooncakeConnector(storage=storage, prefix="mooncake")
        key = connector.make_key("abc123")
        assert key == "mooncake:abc123"

    def test_custom_prefix(self, storage: MemoryStorageBackend) -> None:
        """Test custom prefix."""
        connector = MooncakeConnector(
            storage=storage, prefix="custom_prefix"
        )
        key = connector.make_key("abc123")
        assert key == "custom_prefix:abc123"

    @pytest.mark.asyncio
    async def test_store_and_load(
        self, connector: MooncakeConnector
    ) -> None:
        """Test basic store and load operations."""
        result = await connector.store("chunk1", num_tokens=100)
        assert result is True

        data = await connector.load("chunk1")
        assert data is not None
        assert len(data) == 100 * 512

    @pytest.mark.asyncio
    async def test_exists(self, connector: MooncakeConnector) -> None:
        """Test exists operation."""
        assert await connector.exists("chunk1") is False
        await connector.store("chunk1", num_tokens=100)
        assert await connector.exists("chunk1") is True

    @pytest.mark.asyncio
    async def test_delete(self, connector: MooncakeConnector) -> None:
        """Test delete operation."""
        await connector.store("chunk1", num_tokens=100)
        result = await connector.delete("chunk1")
        assert result is True
        assert await connector.exists("chunk1") is False

    @pytest.mark.asyncio
    async def test_get_metadata(
        self, connector: MooncakeConnector
    ) -> None:
        """Test getting chunk metadata."""
        await connector.store("chunk1", num_tokens=100)
        metadata = await connector.get_metadata("chunk1")
        assert metadata is not None
        assert metadata.chunk_hash == "chunk1"
        assert metadata.num_tokens == 100

    def test_repr(self, storage: MemoryStorageBackend) -> None:
        """Test string representation."""
        connector = MooncakeConnector(
            storage=storage, prefix="mooncake", name="test"
        )
        repr_str = repr(connector)
        assert "MooncakeConnector" in repr_str
        assert "mooncake" in repr_str
