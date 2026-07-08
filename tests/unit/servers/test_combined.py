"""Unit tests for kvbench.servers.combined module."""

from __future__ import annotations

import pytest

from kvbench.connectors.lmcache import LMCacheConnector
from kvbench.core.config import KVBenchConfig
from kvbench.servers.combined import CombinedServer, CombinedStats
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from kvbench.storage.memory import MemoryStorageBackend


class TestCombinedStats:
    """Tests for CombinedStats dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        stats = CombinedStats()
        assert stats.requests_total == 0
        assert stats.prompt_tokens == 0
        assert stats.completion_tokens == 0

    def test_total_tokens(self) -> None:
        """Test total tokens calculation."""
        stats = CombinedStats(prompt_tokens=100, completion_tokens=50)
        assert stats.total_tokens == 150

    def test_avg_latency(self) -> None:
        """Test average latency calculation."""
        stats = CombinedStats(requests_total=10, total_latency_ms=1000.0)
        assert stats.avg_latency_ms == 100.0


class TestCombinedServer:
    """Tests for CombinedServer."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a test configuration."""
        return KVBenchConfig(instance_id="test-combined")

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a test storage backend."""
        return MemoryStorageBackend(max_size_bytes=10_000_000, name="test")

    @pytest.fixture
    async def server(self, config: KVBenchConfig, storage: MemoryStorageBackend) -> CombinedServer:
        """Create a test server instance."""
        connector = LMCacheConnector(storage=storage, name="test")
        server = CombinedServer(config=config, connector=connector)
        yield server
        await server.stop()

    def test_init(self, config: KVBenchConfig, storage: MemoryStorageBackend) -> None:
        """Test server initialization."""
        connector = LMCacheConnector(storage=storage)
        server = CombinedServer(config=config, connector=connector)
        assert server.instance_id == "test-combined"
        assert server.model_name == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_chat_completions(self, server: CombinedServer) -> None:
        """Test non-streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=10,
        )

        response = await server.chat_completions(request)

        assert response.model == "llama-3.1-8b"
        assert response.usage.completion_tokens == 10
        assert server.stats.requests_total == 1
        assert server.stats.requests_success == 1
        assert server.stats.completion_tokens == 10

    @pytest.mark.asyncio
    async def test_chat_completions_stream(self, server: CombinedServer) -> None:
        """Test streaming chat completion."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
            max_tokens=5,
            stream=True,
        )

        chunks = []
        async for chunk in server.chat_completions_stream(request):
            chunks.append(chunk)

        # Should have multiple chunks
        assert len(chunks) >= 5
        assert "data:" in chunks[0]
        assert "[DONE]" in chunks[-1]
        assert server.stats.requests_success == 1

    @pytest.mark.asyncio
    async def test_cache_behavior(self, server: CombinedServer) -> None:
        """Test cache hit/miss behavior."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Same prompt for caching")],
            max_tokens=5,
        )

        # First request
        await server.chat_completions(request)
        first_hits = server.stats.cache_hits
        first_misses = server.stats.cache_misses

        # Second request with same prompt
        await server.chat_completions(request)
        second_hits = server.stats.cache_hits

        # Should have more hits on second request, and no new misses
        assert second_hits > first_hits
        assert server.stats.cache_misses == first_misses

    @pytest.mark.asyncio
    async def test_list_models(self, server: CombinedServer) -> None:
        """Test listing models."""
        models = await server.list_models()
        assert len(models.data) == 1
        assert models.data[0].id == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_health_check(self, server: CombinedServer) -> None:
        """Test health check."""
        health = await server.health_check()
        assert health.status == "healthy"
        assert health.server_type == "combined"
        assert health.instance_id == "test-combined"

    @pytest.mark.asyncio
    async def test_get_metrics(self, server: CombinedServer) -> None:
        """Test getting metrics."""
        request = ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content="Test")],
            max_tokens=5,
        )
        await server.chat_completions(request)

        metrics = await server.get_metrics()
        assert metrics.requests_total == 1
        assert metrics.requests_success == 1

    @pytest.mark.asyncio
    async def test_start_stop(self, config: KVBenchConfig, storage: MemoryStorageBackend) -> None:
        """Test server start and stop."""
        connector = LMCacheConnector(storage=storage)
        server = CombinedServer(config=config, connector=connector)

        await server.start()
        assert server._running is True

        await server.stop()
        assert server._running is False
        assert connector.is_closed is True

    def test_repr(self, server: CombinedServer) -> None:
        """Test string representation."""
        repr_str = repr(server)
        assert "CombinedServer" in repr_str
        assert "test-combined" in repr_str


class TestCacheKeyCorrectness:
    """Regression tests for cache-key correctness at the server level.

    The original implementation derived cache keys from prompt *length*
    only, so any two same-length prompts shared cache entries and hit-rate
    metrics were meaningless.
    """

    @pytest.fixture
    async def server(self) -> CombinedServer:
        """Create a test server instance."""
        storage = MemoryStorageBackend(max_size_bytes=100_000_000, name="test")
        connector = LMCacheConnector(storage=storage, name="test")
        server = CombinedServer(config=KVBenchConfig(instance_id="test-cache"), connector=connector)
        yield server
        await server.stop()

    @staticmethod
    def _request(content: str) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="llama-3.1-8b",
            messages=[ChatMessage(role=MessageRole.USER, content=content)],
            max_tokens=1,
        )

    @pytest.mark.asyncio
    async def test_same_length_different_content_does_not_share_cache(
        self, server: CombinedServer
    ) -> None:
        """Two same-length, different-content prompts must not hit each
        other's cache entries."""
        await server.chat_completions(self._request("A" * 2000))
        hits_after_first = server.stats.cache_hits
        misses_after_first = server.stats.cache_misses
        assert hits_after_first == 0
        assert misses_after_first > 0

        await server.chat_completions(self._request("B" * 2000))
        # No new hits: different content must miss
        assert server.stats.cache_hits == hits_after_first
        assert server.stats.cache_misses > misses_after_first

    @pytest.mark.asyncio
    async def test_identical_prompt_hits_cache(self, server: CombinedServer) -> None:
        """A repeated identical prompt must hit the cache."""
        await server.chat_completions(self._request("C" * 2000))
        misses_after_first = server.stats.cache_misses

        await server.chat_completions(self._request("C" * 2000))
        assert server.stats.cache_hits > 0
        assert server.stats.cache_misses == misses_after_first

    @pytest.mark.asyncio
    async def test_shared_prefix_hits_only_prefix_chunks(self, server: CombinedServer) -> None:
        """A prompt sharing a long prefix with a previous one must hit the
        prefix chunks and miss its own distinct suffix."""
        prefix = "You are a helpful assistant. " * 200  # ~1450 tokens
        await server.chat_completions(self._request(prefix + "Question one?"))
        hits_first = server.stats.cache_hits

        await server.chat_completions(self._request(prefix + "A different question two?"))
        # Prefix chunks hit, suffix chunk misses
        assert server.stats.cache_hits > hits_first
        assert server.stats.cache_misses > 0
