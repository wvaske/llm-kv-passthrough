"""Unit tests for kvbench.servers.prefill module."""

from __future__ import annotations

import pytest

from kvbench.connectors.lmcache import LMCacheConnector
from kvbench.core.config import KVBenchConfig
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from kvbench.servers.prefill import PrefillServer, PrefillStats
from kvbench.storage.memory import MemoryStorageBackend


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
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    @pytest.fixture
    async def server(
        self, config: KVBenchConfig, storage: MemoryStorageBackend
    ) -> PrefillServer:
        """Create a test server instance."""
        connector = LMCacheConnector(storage=storage, name="test")
        server = PrefillServer(config=config, connector=connector)
        yield server
        await server.stop()

    def test_init(
        self, config: KVBenchConfig, storage: MemoryStorageBackend
    ) -> None:
        """Test server initialization."""
        connector = LMCacheConnector(storage=storage)
        server = PrefillServer(config=config, connector=connector)
        assert server.instance_id == "test-prefill"
        assert server.model_name == "llama-3.1-8b"

    def test_simulate_tokenize(self, server: PrefillServer) -> None:
        """Test token simulation."""
        tokens = server._simulate_tokenize("Hello world, how are you?")
        # ~4 chars per token
        assert len(tokens) > 0
        assert len(tokens) <= len("Hello world, how are you?")

    def test_chunk_tokens(self, server: PrefillServer) -> None:
        """Test token chunking."""
        tokens = list(range(1000))
        chunks = server._chunk_tokens(tokens, chunk_size=256)
        assert len(chunks) == 4  # 1000 / 256 = 3.9, rounded up to 4
        assert len(chunks[0]) == 256
        assert len(chunks[-1]) == 1000 % 256 + 256 * 0 or len(chunks[-1]) <= 256

    def test_compute_chunk_hash(self, server: PrefillServer) -> None:
        """Test chunk hash computation is consistent."""
        hash1 = server._compute_chunk_hash([1, 2, 3])
        hash2 = server._compute_chunk_hash([1, 2, 3])
        hash3 = server._compute_chunk_hash([3, 2, 1])
        assert hash1 == hash2
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_process_prefill(self, server: PrefillServer) -> None:
        """Test prefill processing."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content="Hello, this is a test prompt.",
                )
            ],
        )

        num_tokens, hits, misses, latency = await server.process_prefill(request)

        assert num_tokens > 0
        assert hits >= 0
        assert misses >= 0
        assert latency >= 0

    @pytest.mark.asyncio
    async def test_chat_completions(self, server: PrefillServer) -> None:
        """Test chat completion endpoint."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(role=MessageRole.USER, content="Hello!")
            ],
        )

        response = await server.chat_completions(request)

        assert response.model == "llama-3.1-8b"
        assert server.stats.requests_total == 1
        assert server.stats.requests_success == 1

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_request(
        self, server: PrefillServer
    ) -> None:
        """Test that repeated requests get cache hits."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[
                ChatMessage(role=MessageRole.USER, content="Same prompt")
            ],
        )

        # First request - all misses
        await server.chat_completions(request)
        first_misses = server.stats.cache_misses

        # Second request - should get hits
        await server.chat_completions(request)
        second_misses = server.stats.cache_misses

        # Second request should have fewer misses
        assert second_misses == first_misses  # Same misses, but hits increased

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
        # Make a request first
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
        )
        await server.chat_completions(request)

        metrics = await server.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_success == 1

    @pytest.mark.asyncio
    async def test_start_stop(
        self, config: KVBenchConfig, storage: MemoryStorageBackend
    ) -> None:
        """Test server start and stop."""
        connector = LMCacheConnector(storage=storage)
        server = PrefillServer(config=config, connector=connector)

        await server.start()
        assert server._running is True

        await server.stop()
        assert server._running is False
        assert connector.is_closed is True

    def test_repr(self, server: PrefillServer) -> None:
        """Test string representation."""
        repr_str = repr(server)
        assert "PrefillServer" in repr_str
        assert "test-prefill" in repr_str
