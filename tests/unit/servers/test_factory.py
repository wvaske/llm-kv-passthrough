"""Unit tests for kvbench.servers.factory module."""

from __future__ import annotations

import pytest

from kvbench.core.config import KVBenchConfig, ServerConfig
from kvbench.servers.combined import CombinedServer
from kvbench.servers.decode import DecodeServer
from kvbench.servers.factory import create_server
from kvbench.servers.prefill import PrefillServer
from kvbench.servers.proxy import DisaggregatedProxy
from tests.fakes import FakeKVStack


def make_config(server_type: str) -> KVBenchConfig:
    """Build a configuration for the given server type."""
    return KVBenchConfig(server=ServerConfig(server_type=server_type))  # type: ignore[arg-type]


class TestCreateServer:
    """Tests for create_server."""

    @pytest.fixture
    def kv(self) -> FakeKVStack:
        """Create a fake KV stack."""
        return FakeKVStack()

    def test_create_prefill_server(self, kv: FakeKVStack) -> None:
        """Prefill server is created with a KV stack."""
        server = create_server(make_config("prefill"), kv)
        assert isinstance(server, PrefillServer)
        assert server.kv is kv

    def test_create_decode_server(self, kv: FakeKVStack) -> None:
        """Decode server is created with a KV stack."""
        server = create_server(make_config("decode"), kv)
        assert isinstance(server, DecodeServer)
        assert server.kv is kv

    def test_create_combined_server(self, kv: FakeKVStack) -> None:
        """Combined server is created with a KV stack."""
        server = create_server(make_config("combined"), kv)
        assert isinstance(server, CombinedServer)
        assert server.kv is kv

    def test_create_proxy_server(self) -> None:
        """Proxy server does not need a KV stack."""
        server = create_server(make_config("proxy"))
        assert isinstance(server, DisaggregatedProxy)

    def test_prefill_missing_kv(self) -> None:
        """Prefill without a KV stack fails loudly."""
        with pytest.raises(ValueError, match="KV stack is required"):
            create_server(make_config("prefill"))

    def test_decode_missing_kv(self) -> None:
        """Decode without a KV stack fails loudly."""
        with pytest.raises(ValueError, match="KV stack is required"):
            create_server(make_config("decode"))

    def test_combined_missing_kv(self) -> None:
        """Combined without a KV stack fails loudly."""
        with pytest.raises(ValueError, match="KV stack is required"):
            create_server(make_config("combined"))

    def test_unknown_server_type(self, kv: FakeKVStack) -> None:
        """Unknown server type fails loudly."""
        config = make_config("combined")
        # Bypass pydantic validation to exercise the factory's own check
        object.__setattr__(config.server, "server_type", "warp-drive")
        with pytest.raises(ValueError, match="Unknown server type"):
            create_server(config, kv)
