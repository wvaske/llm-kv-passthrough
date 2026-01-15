"""Unit tests for kvbench.servers.proxy module."""

from __future__ import annotations

import pytest

from kvbench.core.config import DistributedConfig, KVBenchConfig
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from kvbench.servers.proxy import (
    DisaggregatedProxy,
    LoadBalanceStrategy,
    ProxyStats,
    ServerEndpoint,
)


class TestServerEndpoint:
    """Tests for ServerEndpoint dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        endpoint = ServerEndpoint(url="http://localhost:8000")
        assert endpoint.url == "http://localhost:8000"
        assert endpoint.healthy is True
        assert endpoint.active_connections == 0
        assert endpoint.total_requests == 0


class TestProxyStats:
    """Tests for ProxyStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = ProxyStats()
        assert stats.requests_total == 0
        assert stats.prefill_requests == 0
        assert stats.decode_requests == 0

    def test_avg_latency(self) -> None:
        """Test average latency calculation."""
        stats = ProxyStats(requests_total=10, total_latency_ms=500.0)
        assert stats.avg_latency_ms == 50.0


class TestDisaggregatedProxy:
    """Tests for DisaggregatedProxy."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a test configuration."""
        return KVBenchConfig(
            instance_id="test-proxy",
            distributed=DistributedConfig(
                prefill_endpoints=["http://prefill-1:8000", "http://prefill-2:8000"],
                decode_endpoints=["http://decode-1:8000", "http://decode-2:8000"],
            ),
        )

    @pytest.fixture
    async def proxy(self, config: KVBenchConfig) -> DisaggregatedProxy:
        """Create a test proxy instance."""
        proxy = DisaggregatedProxy(config=config)
        yield proxy
        await proxy.stop()

    def test_init(self, config: KVBenchConfig) -> None:
        """Test proxy initialization."""
        proxy = DisaggregatedProxy(config=config)
        assert proxy.instance_id == "test-proxy"
        assert len(proxy.prefill_servers) == 2
        assert len(proxy.decode_servers) == 2

    def test_init_with_custom_endpoints(self) -> None:
        """Test proxy initialization with custom endpoints."""
        config = KVBenchConfig(instance_id="test")
        proxy = DisaggregatedProxy(
            config=config,
            prefill_endpoints=["http://custom-prefill:8000"],
            decode_endpoints=["http://custom-decode:8000"],
        )
        assert len(proxy.prefill_servers) == 1
        assert len(proxy.decode_servers) == 1
        assert proxy.prefill_servers[0].url == "http://custom-prefill:8000"

    def test_select_prefill_round_robin(self, config: KVBenchConfig) -> None:
        """Test round-robin prefill server selection."""
        proxy = DisaggregatedProxy(
            config=config, strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        server1 = proxy._select_prefill_server()
        server2 = proxy._select_prefill_server()

        # Should alternate between servers
        assert server1 is not None
        assert server2 is not None
        assert server1 != server2

    def test_select_decode_round_robin(self, config: KVBenchConfig) -> None:
        """Test round-robin decode server selection."""
        proxy = DisaggregatedProxy(
            config=config, strategy=LoadBalanceStrategy.ROUND_ROBIN
        )

        server1 = proxy._select_decode_server()
        server2 = proxy._select_decode_server()

        assert server1 is not None
        assert server2 is not None
        assert server1 != server2

    def test_select_prefill_least_connections(
        self, config: KVBenchConfig
    ) -> None:
        """Test least-connections prefill server selection."""
        proxy = DisaggregatedProxy(
            config=config, strategy=LoadBalanceStrategy.LEAST_CONNECTIONS
        )

        # Simulate one server with more connections
        proxy.prefill_servers[0].active_connections = 5
        proxy.prefill_servers[1].active_connections = 1

        server = proxy._select_prefill_server()
        assert server is not None
        assert server.active_connections == 1

    def test_select_no_healthy_servers(self) -> None:
        """Test selection when no servers are healthy."""
        config = KVBenchConfig(
            distributed=DistributedConfig(
                prefill_endpoints=["http://prefill-1:8000"],
                decode_endpoints=["http://decode-1:8000"],
            )
        )
        proxy = DisaggregatedProxy(config=config)

        # Mark all servers as unhealthy
        proxy.prefill_servers[0].healthy = False

        server = proxy._select_prefill_server()
        assert server is None

    @pytest.mark.asyncio
    async def test_chat_completions(self, proxy: DisaggregatedProxy) -> None:
        """Test non-streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=10,
        )

        response = await proxy.chat_completions(request)

        assert response.model == "llama-3.1-8b"
        assert proxy.stats.requests_total == 1
        assert proxy.stats.prefill_requests == 1
        assert proxy.stats.decode_requests == 1

    @pytest.mark.asyncio
    async def test_chat_completions_stream(
        self, proxy: DisaggregatedProxy
    ) -> None:
        """Test streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=5,
            stream=True,
        )

        chunks = []
        async for chunk in proxy.chat_completions_stream(request):
            chunks.append(chunk)

        assert len(chunks) >= 5
        assert "data:" in chunks[0]
        assert "[DONE]" in chunks[-1]

    @pytest.mark.asyncio
    async def test_chat_completions_no_prefill_servers(self) -> None:
        """Test error when no prefill servers available."""
        config = KVBenchConfig(
            distributed=DistributedConfig(
                prefill_endpoints=[],
                decode_endpoints=["http://decode-1:8000"],
            )
        )
        proxy = DisaggregatedProxy(config=config)

        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
        )

        response = await proxy.chat_completions(request)
        assert "error" in response.model_dump()

    @pytest.mark.asyncio
    async def test_list_models(self, proxy: DisaggregatedProxy) -> None:
        """Test listing models."""
        models = await proxy.list_models()
        assert len(models.data) == 1
        assert models.data[0].id == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, proxy: DisaggregatedProxy) -> None:
        """Test health check when all servers healthy."""
        health = await proxy.health_check()
        assert health.status == "healthy"
        assert health.server_type == "proxy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, proxy: DisaggregatedProxy) -> None:
        """Test health check when some servers unhealthy."""
        proxy.prefill_servers[0].healthy = False

        health = await proxy.health_check()
        assert health.status == "degraded"

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(
        self, proxy: DisaggregatedProxy
    ) -> None:
        """Test health check when no servers healthy."""
        for server in proxy.prefill_servers:
            server.healthy = False

        health = await proxy.health_check()
        assert health.status == "unhealthy"

    @pytest.mark.asyncio
    async def test_get_metrics(self, proxy: DisaggregatedProxy) -> None:
        """Test getting metrics."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
            max_tokens=5,
        )
        await proxy.chat_completions(request)

        metrics = await proxy.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_success == 1

    def test_get_server_status(self, proxy: DisaggregatedProxy) -> None:
        """Test getting server status."""
        status = proxy.get_server_status()

        assert "prefill_servers" in status
        assert "decode_servers" in status
        assert len(status["prefill_servers"]) == 2
        assert len(status["decode_servers"]) == 2

    @pytest.mark.asyncio
    async def test_start_stop(self, config: KVBenchConfig) -> None:
        """Test proxy start and stop."""
        proxy = DisaggregatedProxy(config=config)

        await proxy.start()
        assert proxy._running is True
        assert proxy._health_check_task is not None

        await proxy.stop()
        assert proxy._running is False

    def test_repr(self, proxy: DisaggregatedProxy) -> None:
        """Test string representation."""
        repr_str = repr(proxy)
        assert "DisaggregatedProxy" in repr_str
        assert "prefill=2" in repr_str
        assert "decode=2" in repr_str
