"""Unit tests for kvbench.servers.decode module."""

from __future__ import annotations

import pytest

from kvbench.core.config import KVBenchConfig
from kvbench.servers.decode import DecodeServer, DecodeStats
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from tests.fakes import FakeKVStack


class TestDecodeStats:
    """Tests for DecodeStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = DecodeStats()
        assert stats.requests_total == 0
        assert stats.requests_success == 0
        assert stats.requests_failed == 0
        assert stats.tokens_generated == 0

    def test_avg_latency_no_requests(self) -> None:
        """Test avg latency with no requests."""
        stats = DecodeStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self) -> None:
        """Test avg latency calculation."""
        stats = DecodeStats(requests_total=4, total_latency_ms=200.0)
        assert stats.avg_latency_ms == 50.0

    def test_avg_tokens_per_request(self) -> None:
        """Test avg tokens per request."""
        stats = DecodeStats(requests_total=4, tokens_generated=100)
        assert stats.avg_tokens_per_request == 25.0


class TestDecodeServer:
    """Tests for DecodeServer."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a test configuration."""
        return KVBenchConfig(instance_id="test-decode")

    @pytest.fixture
    def kv(self) -> FakeKVStack:
        """Create a fake KV stack."""
        return FakeKVStack(chunk_size=16)

    @pytest.fixture
    async def server(self, config: KVBenchConfig, kv: FakeKVStack) -> DecodeServer:
        """Create a test server instance."""
        server = DecodeServer(config=config, kv=kv)
        yield server
        await server.stop()

    def test_init(self, config: KVBenchConfig, kv: FakeKVStack) -> None:
        """Test server initialization."""
        server = DecodeServer(config=config, kv=kv)
        assert server.instance_id == "test-decode"
        assert server.model_name == "llama-3.1-8b"
        assert server.chunk_size == kv.chunk_size

    def test_generate_mock_token(self, server: DecodeServer) -> None:
        """Test mock token generation."""
        token0 = server._generate_mock_token(0)
        token1 = server._generate_mock_token(1)
        assert token0 is not None
        assert token1 is not None
        assert isinstance(token0, str)

    def test_calculate_decode_latency(self, server: DecodeServer) -> None:
        """Test decode latency calculation."""
        latency_short = server._calculate_decode_latency_ms(100)
        latency_long = server._calculate_decode_latency_ms(10000)
        # Longer context should have higher latency
        assert latency_long > latency_short

    @pytest.mark.asyncio
    async def test_generate_tokens(self, server: DecodeServer) -> None:
        """Test token generation."""
        tokens = []
        async for token, latency in server.generate_tokens(context_length=100, max_tokens=5):
            tokens.append(token)
            assert latency > 0

        assert len(tokens) == 5

    @pytest.mark.asyncio
    async def test_chat_completions(self, server: DecodeServer) -> None:
        """Test chat completion endpoint."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=10,
        )

        response = await server.chat_completions(request, context_length=50)

        assert response.model == "llama-3.1-8b"
        assert response.usage.completion_tokens == 10
        assert server.stats.requests_total == 1
        assert server.stats.tokens_generated == 10

    @pytest.mark.asyncio
    async def test_decode_reads_kv_through_stack(
        self, server: DecodeServer, kv: FakeKVStack
    ) -> None:
        """Decode must look up (and read) the prompt's KV via the KV stack."""
        prompt = "A moderately long prompt for the decode read path " * 8
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content=prompt)],
            max_tokens=2,
        )
        # Pre-populate the cache as a prefill server would, tokenizing the
        # same request text the decode server will see
        tokens = server.processor.simulate_tokenize(request.prompt_text or "")
        await kv.store(tokens)

        await server.chat_completions(request)

        assert kv.stats.lookups == 1
        assert kv.stats.retrieves == 1
        assert kv.stats.retrieved_tokens > 0
        assert server.stats.cache_hits > 0

    @pytest.mark.asyncio
    async def test_chat_completions_stream(self, server: DecodeServer) -> None:
        """Test streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=5,
            stream=True,
        )

        chunks = []
        async for chunk in server.chat_completions_stream(request, context_length=50):
            chunks.append(chunk)

        # Should have: start chunk + 5 content chunks + end chunk + [DONE]
        assert len(chunks) >= 5
        assert "data:" in chunks[0]
        assert "[DONE]" in chunks[-1]
        assert server.stats.requests_total == 1
        assert server.stats.requests_success == 1

    @pytest.mark.asyncio
    async def test_list_models(self, server: DecodeServer) -> None:
        """Test listing models."""
        models = await server.list_models()
        assert len(models.data) == 1
        assert models.data[0].id == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_health_check(self, server: DecodeServer) -> None:
        """Test health check."""
        health = await server.health_check()
        assert health.status == "healthy"
        assert health.server_type == "decode"

    @pytest.mark.asyncio
    async def test_get_metrics(self, server: DecodeServer) -> None:
        """Test getting metrics."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
            max_tokens=5,
        )
        await server.chat_completions(request, context_length=50)

        metrics = await server.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.tokens_processed == 5

    @pytest.mark.asyncio
    async def test_start_stop(self, config: KVBenchConfig, kv: FakeKVStack) -> None:
        """Server start/stop does not close the app-owned KV stack."""
        server = DecodeServer(config=config, kv=kv)

        await server.start()
        assert server._running is True

        await server.stop()
        assert server._running is False
        assert kv.closed is False

    def test_repr(self, server: DecodeServer) -> None:
        """Test string representation."""
        repr_str = repr(server)
        assert "DecodeServer" in repr_str
        assert "test-decode" in repr_str
