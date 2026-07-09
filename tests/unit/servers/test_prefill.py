"""Unit tests for kvbench.servers.prefill module."""

from __future__ import annotations

import pytest

from kvbench.core.config import KVBenchConfig
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from kvbench.servers.prefill import PrefillServer, PrefillStats
from tests.fakes import FakeKVStack


class TestPrefillStats:
    """Tests for PrefillStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = PrefillStats()
        assert stats.requests_total == 0
        assert stats.requests_success == 0
        assert stats.requests_failed == 0
        assert stats.tokens_processed == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0

    def test_avg_latency_no_requests(self) -> None:
        """Test avg latency with no requests."""
        stats = PrefillStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self) -> None:
        """Test avg latency calculation."""
        stats = PrefillStats(requests_total=10, total_latency_ms=500.0)
        assert stats.avg_latency_ms == 50.0


class TestPrefillServer:
    """Tests for PrefillServer."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a test configuration."""
        return KVBenchConfig(instance_id="test-prefill")

    @pytest.fixture
    def kv(self) -> FakeKVStack:
        """Create a fake KV stack."""
        return FakeKVStack(chunk_size=16)

    @pytest.fixture
    async def server(self, config: KVBenchConfig, kv: FakeKVStack) -> PrefillServer:
        """Create a test server instance."""
        server = PrefillServer(config=config, kv=kv)
        yield server
        await server.stop()

    def test_init(self, config: KVBenchConfig, kv: FakeKVStack) -> None:
        """Test server initialization."""
        server = PrefillServer(config=config, kv=kv)
        assert server.instance_id == "test-prefill"
        assert server.model_name == "llama-3.1-8b"
        assert server.chunk_size == kv.chunk_size

    def test_simulate_tokenize(self, server: PrefillServer) -> None:
        """Test token simulation."""
        tokens = server._simulate_tokenize("Hello world, how are you?")
        # ~4 chars per token
        assert len(tokens) > 0
        assert len(tokens) <= len("Hello world, how are you?")

    @pytest.mark.asyncio
    async def test_process_prefill(self, server: PrefillServer, kv: FakeKVStack) -> None:
        """Prefill looks up, stores through the KV stack, and reports counts."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content="Hello, this is a test prompt with enough words " * 8,
                )
            ],
        )

        num_tokens, hits, misses, latency = await server.process_prefill(request)

        assert num_tokens > 0
        assert hits == 0  # cold cache
        assert misses > 0
        assert latency >= 0
        assert kv.stats.lookups == 1
        assert kv.stats.stores == 1

    @pytest.mark.asyncio
    async def test_chat_completions(self, server: PrefillServer) -> None:
        """Test chat completion endpoint."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
        )

        response = await server.chat_completions(request)

        assert response.model == "llama-3.1-8b"
        assert server.stats.requests_total == 1
        assert server.stats.requests_success == 1

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_request(
        self, server: PrefillServer, kv: FakeKVStack
    ) -> None:
        """Repeated requests hit the cache and read KV through the stack."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(role=MessageRole.USER, content="Same prompt repeated " * 20)
            ],
        )

        await server.chat_completions(request)
        first_hits = server.stats.cache_hits
        assert first_hits == 0
        assert kv.stats.retrieves == 0

        await server.chat_completions(request)
        assert server.stats.cache_hits > first_hits
        # The cached prefix was read through the stack (the read path)
        assert kv.stats.retrieves == 1
        assert kv.stats.retrieved_tokens > 0

    @pytest.mark.asyncio
    async def test_list_models(self, server: PrefillServer) -> None:
        """Test listing models."""
        models = await server.list_models()
        assert len(models.data) == 1
        assert models.data[0].id == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_health_check(self, server: PrefillServer) -> None:
        """Test health check."""
        health = await server.health_check()
        assert health.status == "healthy"
        assert health.server_type == "prefill"
        assert health.instance_id == "test-prefill"

    @pytest.mark.asyncio
    async def test_get_metrics(self, server: PrefillServer) -> None:
        """Test getting metrics."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
        )
        await server.chat_completions(request)

        metrics = await server.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_success == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, config: KVBenchConfig, kv: FakeKVStack) -> None:
        """Server start/stop does not close the app-owned KV stack."""
        server = PrefillServer(config=config, kv=kv)

        await server.start()
        assert server._running is True

        await server.stop()
        assert server._running is False
        assert kv.closed is False

    def test_repr(self, server: PrefillServer) -> None:
        """Test string representation."""
        repr_str = repr(server)
        assert "PrefillServer" in repr_str
        assert "test-prefill" in repr_str
