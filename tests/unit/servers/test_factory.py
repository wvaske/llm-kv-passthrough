"""Unit tests for kvbench.servers.factory module."""

from __future__ import annotations

import pytest

from kvbench.connectors.lmcache import LMCacheConnector
from kvbench.core.config import KVBenchConfig, ServerConfig
from kvbench.servers.combined import CombinedServer
from kvbench.servers.decode import DecodeServer
from kvbench.servers.factory import create_server
from kvbench.servers.prefill import PrefillServer
from kvbench.servers.proxy import DisaggregatedProxy
from kvbench.storage.memory import MemoryStorageBackend


class TestCreateServer:
    """Tests for create_server factory function."""

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    @pytest.fixture
    def connector(self, storage: MemoryStorageBackend) -> LMCacheConnector:
        """Create a test connector."""
        return LMCacheConnector(storage=storage, name="test")

    def test_create_prefill_server(
        self, connector: LMCacheConnector
    ) -> None:
        """Test creating a prefill server."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="prefill")
        )
        server = create_server(config, connector)

        assert isinstance(server, PrefillServer)

    def test_create_decode_server(
        self, connector: LMCacheConnector
    ) -> None:
        """Test creating a decode server."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="decode")
        )
        server = create_server(config, connector)

        assert isinstance(server, DecodeServer)

    def test_create_combined_server(
        self, connector: LMCacheConnector
    ) -> None:
        """Test creating a combined server."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="combined")
        )
        server = create_server(config, connector)

        assert isinstance(server, CombinedServer)

    def test_create_proxy_server(self) -> None:
        """Test creating a proxy server (no connector needed)."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="proxy")
        )
        server = create_server(config)

        assert isinstance(server, DisaggregatedProxy)

    def test_prefill_missing_connector(self) -> None:
        """Test that prefill server requires connector."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="prefill")
        )

        with pytest.raises(ValueError, match="connector is required"):
            create_server(config)

    def test_decode_missing_connector(self) -> None:
        """Test that decode server requires connector."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="decode")
        )

        with pytest.raises(ValueError, match="connector is required"):
            create_server(config)

    def test_combined_missing_connector(self) -> None:
        """Test that combined server requires connector."""
        config = KVBenchConfig(
            server=ServerConfig(server_type="combined")
        )

        with pytest.raises(ValueError, match="connector is required"):
            create_server(config)

    def test_unknown_server_type(
        self, connector: LMCacheConnector
    ) -> None:
        """Test that unknown server type raises error."""
        # Use model_construct to bypass validation
        config = KVBenchConfig(
            server=ServerConfig.model_construct(server_type="unknown")
        )

        with pytest.raises(ValueError, match="Unknown server type"):
            create_server(config, connector)
