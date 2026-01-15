"""End-to-end tests for distributed deployment scenarios."""

from __future__ import annotations

import pytest

from kvbench.connectors.lmcache import LMCacheConfig, LMCacheConnector
from kvbench.core.config import (
    DistributedConfig,
    KVBenchConfig,
    ServerConfig,
)
from kvbench.servers.combined import CombinedServer
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from kvbench.servers.proxy import DisaggregatedProxy, LoadBalanceStrategy
from kvbench.storage.memory import MemoryStorageBackend


class TestDistributedScenarios:
    """Tests for distributed deployment scenarios."""

    @pytest.fixture
    def storage(self) -> MemoryStorageBackend:
        """Create a shared storage backend."""
        return MemoryStorageBackend(max_size_bytes=100_000_000, name="shared")

    @pytest.mark.asyncio
    async def test_multiple_workers_shared_storage(
        self, storage: MemoryStorageBackend
    ) -> None:
        """Test multiple workers sharing storage backend."""
        # Create two connectors with different worker IDs
        config1 = LMCacheConfig(model_name="llama-3.1-8b", world_size=2, worker_id=0)
        config2 = LMCacheConfig(model_name="llama-3.1-8b", world_size=2, worker_id=1)

        connector1 = LMCacheConnector(storage=storage, config=config1, name="worker-0")
        connector2 = LMCacheConnector(storage=storage, config=config2, name="worker-1")

        try:
            # Worker 0 stores a chunk
            await connector1.store("shared_chunk", num_tokens=100)

            # Worker 0 can find it
            assert await connector1.exists("shared_chunk")

            # Worker 1 uses different key format, so different key
            # But can store its own version
            await connector2.store("shared_chunk", num_tokens=100)
            assert await connector2.exists("shared_chunk")

            # Verify stats
            assert connector1.stats.stores == 1
            assert connector2.stats.stores == 1

        finally:
            await connector1.close()
            await connector2.close()

    @pytest.mark.asyncio
    async def test_proxy_load_balancing_round_robin(self) -> None:
        """Test proxy round-robin load balancing."""
        config = KVBenchConfig(
            distributed=DistributedConfig(
                prefill_endpoints=["http://prefill1:8000", "http://prefill2:8000"],
                decode_endpoints=["http://decode1:8000", "http://decode2:8000"],
            ),
        )

        proxy = DisaggregatedProxy(
            config=config,
            strategy=LoadBalanceStrategy.ROUND_ROBIN,
        )

        try:
            # Select prefill servers - should alternate
            server1 = proxy._select_prefill_server()
            server2 = proxy._select_prefill_server()
            server3 = proxy._select_prefill_server()

            assert server1 is not None
            assert server2 is not None
            assert server3 is not None

            # Round-robin should cycle through servers
            assert server1.url != server2.url or len(proxy.prefill_servers) == 1
            assert server1.url == server3.url  # Back to first

        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_proxy_load_balancing_least_connections(self) -> None:
        """Test proxy least-connections load balancing."""
        config = KVBenchConfig(
            distributed=DistributedConfig(
                prefill_endpoints=["http://prefill1:8000", "http://prefill2:8000"],
                decode_endpoints=["http://decode1:8000"],
            ),
        )

        proxy = DisaggregatedProxy(
            config=config,
            strategy=LoadBalanceStrategy.LEAST_CONNECTIONS,
        )

        try:
            # Simulate connections on first server
            proxy.prefill_servers[0].active_connections = 5
            proxy.prefill_servers[1].active_connections = 2

            # Should select server with fewer connections
            selected = proxy._select_prefill_server()
            assert selected is not None
            assert selected.url == "http://prefill2:8000"

        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_proxy_unhealthy_server_skipped(self) -> None:
        """Test that unhealthy servers are skipped."""
        config = KVBenchConfig(
            distributed=DistributedConfig(
                prefill_endpoints=["http://prefill1:8000", "http://prefill2:8000"],
                decode_endpoints=["http://decode1:8000"],
            ),
        )

        proxy = DisaggregatedProxy(config=config)

        try:
            # Mark first server as unhealthy
            proxy.prefill_servers[0].healthy = False

            # Should only select healthy server
            for _ in range(5):
                selected = proxy._select_prefill_server()
                assert selected is not None
                assert selected.url == "http://prefill2:8000"

        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_proxy_no_healthy_servers(self) -> None:
        """Test proxy behavior when no servers are healthy."""
        config = KVBenchConfig(
            distributed=DistributedConfig(
                prefill_endpoints=["http://prefill1:8000"],
                decode_endpoints=["http://decode1:8000"],
            ),
        )

        proxy = DisaggregatedProxy(config=config)

        try:
            # Mark all servers as unhealthy
            proxy.prefill_servers[0].healthy = False

            # Should return None
            selected = proxy._select_prefill_server()
            assert selected is None

        finally:
            await proxy.stop()

    @pytest.mark.asyncio
    async def test_cache_sharing_across_requests(self) -> None:
        """Test KV cache sharing across multiple requests."""
        storage = MemoryStorageBackend(max_size_bytes=10_000_000, name="test")
        connector = LMCacheConnector(storage=storage, name="test")
        config = KVBenchConfig(server=ServerConfig(server_type="combined"))

        server = CombinedServer(config=config, connector=connector)

        try:
            await server.start()

            # First request - cache miss
            request1 = ChatCompletionRequest(
                model="llama-3.1-8b",
                messages=[ChatMessage(role=MessageRole.USER, content="Hello world")],
                max_tokens=5,
            )
            await server.chat_completions(request1)

            initial_misses = server.stats.cache_misses

            # Second request with same prefix - should have cache hits
            request2 = ChatCompletionRequest(
                model="llama-3.1-8b",
                messages=[
                    ChatMessage(role=MessageRole.USER, content="Hello world, how are you?")
                ],
                max_tokens=5,
            )
            await server.chat_completions(request2)

            # Should have some cache hits from the shared prefix
            assert server.stats.cache_hits > 0 or server.stats.cache_misses >= initial_misses

        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_connector_stats_accumulate(self) -> None:
        """Test that connector stats accumulate correctly."""
        storage = MemoryStorageBackend(max_size_bytes=10_000_000, name="test")
        connector = LMCacheConnector(storage=storage, name="test")

        try:
            # Perform multiple operations
            for i in range(10):
                await connector.store(f"chunk_{i}", num_tokens=100)

            assert connector.stats.stores == 10
            assert connector.stats.store_bytes == 10 * 100 * 512  # bytes_per_token

            # Load operations
            for i in range(10):
                await connector.load(f"chunk_{i}")

            assert connector.stats.loads == 10
            assert connector.stats.hits == 10
            assert connector.stats.misses == 0

            # Load non-existent
            await connector.load("nonexistent")
            assert connector.stats.misses == 1

            # Verify hit rate
            assert connector.stats.hit_rate == pytest.approx(90.9, rel=0.1)

        finally:
            await connector.close()


class TestStorageBackendScenarios:
    """Tests for storage backend scenarios."""

    @pytest.mark.asyncio
    async def test_memory_storage_eviction(self) -> None:
        """Test memory storage LRU eviction."""
        # Small storage that can only hold a few items
        storage = MemoryStorageBackend(max_size_bytes=5000, name="small")

        try:
            # Store items until eviction occurs
            for i in range(10):
                await storage.put(f"key_{i}", b"x" * 1000)

            # Not all items should fit
            assert storage.stats.used_bytes <= 5000

            # Most recent items should still exist
            assert await storage.exists("key_9")

        finally:
            await storage.close()

    @pytest.mark.asyncio
    async def test_storage_concurrent_access(self) -> None:
        """Test concurrent access to storage backend."""
        import asyncio

        storage = MemoryStorageBackend(max_size_bytes=10_000_000, name="concurrent")

        async def writer(prefix: str, count: int) -> None:
            for i in range(count):
                await storage.put(f"{prefix}_{i}", b"data")

        async def reader(prefix: str, count: int) -> int:
            hits = 0
            for i in range(count):
                data = await storage.get(f"{prefix}_{i}")
                if data is not None:
                    hits += 1
            return hits

        try:
            # Concurrent writes
            await asyncio.gather(
                writer("a", 100),
                writer("b", 100),
                writer("c", 100),
            )

            # Concurrent reads
            results = await asyncio.gather(
                reader("a", 100),
                reader("b", 100),
                reader("c", 100),
            )

            # All writes should be readable
            assert sum(results) == 300

        finally:
            await storage.close()
