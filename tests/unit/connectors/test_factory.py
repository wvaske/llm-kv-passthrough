"""Unit tests for kvbench.connectors.factory module."""

from __future__ import annotations

import pytest

from kvbench.connectors.dynamo import DynamoConnector
from kvbench.connectors.factory import create_connector
from kvbench.connectors.lmcache import LMCacheConnector
from kvbench.connectors.mooncake import MooncakeConnector
from kvbench.core.config import ConnectorConfig
from kvbench.storage.memory import MemoryStorageBackend


class TestCreateConnector:
    """Tests for create_connector factory function."""

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    def test_create_lmcache_connector(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test creating an LMCache connector."""
        config = ConnectorConfig(connector_type="lmcache", lmcache_chunk_size=512)
        connector = create_connector(config, storage)

        assert isinstance(connector, LMCacheConnector)
        assert connector.config.chunk_size == 512

    def test_create_mooncake_connector(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test creating a Mooncake connector."""
        config = ConnectorConfig(connector_type="mooncake")
        connector = create_connector(config, storage)

        assert isinstance(connector, MooncakeConnector)

    def test_create_dynamo_connector(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test creating a Dynamo connector."""
        config = ConnectorConfig(connector_type="dynamo")
        connector = create_connector(config, storage)

        assert isinstance(connector, DynamoConnector)

    def test_create_mock_connector(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test creating a mock connector (uses LMCache)."""
        config = ConnectorConfig(connector_type="mock")
        connector = create_connector(config, storage)

        assert isinstance(connector, LMCacheConnector)

    def test_create_unknown_connector(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test that unknown connector type raises error."""
        # Use model_construct to bypass validation
        config = ConnectorConfig.model_construct(connector_type="unknown")

        with pytest.raises(ValueError, match="Unknown connector type"):
            create_connector(config, storage)

    def test_custom_name(self, storage: MemoryStorageBackend) -> None:
        """Test creating a connector with custom name."""
        config = ConnectorConfig(connector_type="lmcache")
        connector = create_connector(config, storage, name="custom_name")

        assert connector.name == "custom_name"

    def test_default_name(self, storage: MemoryStorageBackend) -> None:
        """Test that default name matches connector type."""
        config = ConnectorConfig(connector_type="lmcache")
        connector = create_connector(config, storage)

        assert connector.name == "lmcache"
