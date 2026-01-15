"""Unit tests for kvbench.connectors.dynamo module."""

from __future__ import annotations

import pytest

from kvbench.connectors.dynamo import DynamoConnector
from kvbench.storage.memory import MemoryStorageBackend


class TestDynamoConnector:
    """Tests for DynamoConnector stub."""

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    @pytest.fixture
    async def connector(
        self, storage: MemoryStorageBackend
    ) -> DynamoConnector:
        """Create a test connector instance."""
        connector = DynamoConnector(storage=storage, name="test")
        yield connector
        await connector.close()

    def test_make_key(self, storage: MemoryStorageBackend) -> None:
        """Test key generation."""
        connector = DynamoConnector(
            storage=storage, prefix="dynamo", namespace="default"
        )
        key = connector.make_key("abc123")
        assert key == "dynamo:default:abc123"

    def test_custom_namespace(self, storage: MemoryStorageBackend) -> None:
        """Test custom namespace."""
        connector = DynamoConnector(
            storage=storage, namespace="custom_ns"
        )
        key = connector.make_key("abc123")
        assert "custom_ns" in key

    @pytest.mark.asyncio
    async def test_store_and_load(
        self, connector: DynamoConnector
    ) -> None:
        """Test basic store and load operations."""
        result = await connector.store("chunk1", num_tokens=100)
        assert result is True

        data = await connector.load("chunk1")
        assert data is not None
        assert len(data) == 100 * 512

    @pytest.mark.asyncio
    async def test_exists(self, connector: DynamoConnector) -> None:
        """Test exists operation."""
        assert await connector.exists("chunk1") is False
        await connector.store("chunk1", num_tokens=100)
        assert await connector.exists("chunk1") is True

    @pytest.mark.asyncio
    async def test_delete(self, connector: DynamoConnector) -> None:
        """Test delete operation."""
        await connector.store("chunk1", num_tokens=100)
        result = await connector.delete("chunk1")
        assert result is True
        assert await connector.exists("chunk1") is False

    @pytest.mark.asyncio
    async def test_get_metadata(
        self, connector: DynamoConnector
    ) -> None:
        """Test getting chunk metadata."""
        await connector.store("chunk1", num_tokens=100)
        metadata = await connector.get_metadata("chunk1")
        assert metadata is not None
        assert metadata.chunk_hash == "chunk1"

    def test_repr(self, storage: MemoryStorageBackend) -> None:
        """Test string representation."""
        connector = DynamoConnector(
            storage=storage, namespace="test_ns", name="test"
        )
        repr_str = repr(connector)
        assert "DynamoConnector" in repr_str
        assert "test_ns" in repr_str
