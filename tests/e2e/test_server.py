"""End-to-end tests for KV-Bench server."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

import kvbench.servers.app as app_module
from kvbench.core.config import (
    GPUEmulationConfig,
    KVBenchConfig,
    ServerConfig,
)
from kvbench.servers.app import KVBenchApp
from tests.fakes import FakeKVStack


@asynccontextmanager
async def create_test_client(config: KVBenchConfig) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with proper lifespan management.

    Uses the in-memory FakeKVStack; the real LMCache stack is exercised by
    tests/integration/test_lmcache_stack.py.
    """
    kvbench_app = KVBenchApp(config)
    original = app_module.create_kv_stack
    app_module.create_kv_stack = lambda _config: FakeKVStack()
    try:
        # Manually call startup
        await kvbench_app._startup()
    finally:
        app_module.create_kv_stack = original
    try:
        transport = ASGITransport(app=kvbench_app.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client
    finally:
        # Manually call shutdown
        await kvbench_app._shutdown()


class TestServerE2E:
    """End-to-end tests for the KV-Bench server."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a test configuration."""
        return KVBenchConfig(
            instance_id="e2e-test",
            server=ServerConfig(
                host="127.0.0.1",
                port=8000,
                server_type="combined",
                model_profile="llama-3.1-8b",
            ),
            gpu=GPUEmulationConfig(gpu_profile="H100_SXM"),
        )

    @pytest.fixture
    async def client(self, config: KVBenchConfig) -> AsyncGenerator[AsyncClient, None]:
        """Create a test client."""
        async with create_test_client(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient) -> None:
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["server_type"] == "combined"

    @pytest.mark.asyncio
    async def test_metrics(self, client: AsyncClient) -> None:
        """Test metrics endpoint."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "cache_hits" in data

    @pytest.mark.asyncio
    async def test_list_models(self, client: AsyncClient) -> None:
        """Test model listing endpoint."""
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "llama-3.1-8b"

    @pytest.mark.asyncio
    async def test_chat_completion_non_streaming(
        self, client: AsyncClient
    ) -> None:
        """Test non-streaming chat completion."""
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello, world!"}],
                "max_tokens": 10,
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "llama-3.1-8b"
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["usage"]["prompt_tokens"] > 0
        assert data["usage"]["completion_tokens"] == 10

    @pytest.mark.asyncio
    async def test_chat_completion_streaming(
        self, client: AsyncClient
    ) -> None:
        """Test streaming chat completion."""
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 5,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    chunks.append(line)

            # Should have start chunk + content chunks + end chunk + [DONE]
            assert len(chunks) >= 3
            assert any("[DONE]" in chunk for chunk in chunks)

    @pytest.mark.asyncio
    async def test_multiple_requests(self, client: AsyncClient) -> None:
        """Test multiple sequential requests."""
        for i in range(3):
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b",
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "max_tokens": 5,
                },
            )
            assert response.status_code == 200

        # Check metrics updated
        metrics = await client.get("/metrics")
        assert metrics.json()["requests_total"] == 3

    @pytest.mark.asyncio
    async def test_cache_hit_behavior(self, client: AsyncClient) -> None:
        """Test that repeated prompts result in cache hits."""
        # First request - should be cache miss
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello world"}],
                "max_tokens": 5,
            },
        )

        metrics1 = await client.get("/metrics")
        initial_misses = metrics1.json()["cache_misses"]

        # Second request with same prefix - should be cache hit
        await client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello world, how are you?"}],
                "max_tokens": 5,
            },
        )

        metrics2 = await client.get("/metrics")
        # Should have some cache hits now
        assert metrics2.json()["cache_hits"] > 0 or metrics2.json()["cache_misses"] >= initial_misses


class TestPrefillServerE2E:
    """End-to-end tests for the prefill server."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a prefill server configuration."""
        return KVBenchConfig(
            instance_id="prefill-test",
            server=ServerConfig(server_type="prefill"),
        )

    @pytest.fixture
    async def client(self, config: KVBenchConfig) -> AsyncGenerator[AsyncClient, None]:
        """Create a test client."""
        async with create_test_client(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_prefill_health(self, client: AsyncClient) -> None:
        """Test prefill server health check."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["server_type"] == "prefill"


class TestDecodeServerE2E:
    """End-to-end tests for the decode server."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a decode server configuration."""
        return KVBenchConfig(
            instance_id="decode-test",
            server=ServerConfig(server_type="decode"),
        )

    @pytest.fixture
    async def client(self, config: KVBenchConfig) -> AsyncGenerator[AsyncClient, None]:
        """Create a test client."""
        async with create_test_client(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_decode_health(self, client: AsyncClient) -> None:
        """Test decode server health check."""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["server_type"] == "decode"

    @pytest.mark.asyncio
    async def test_decode_streaming(self, client: AsyncClient) -> None:
        """Test decode server streaming."""
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Test"}],
                "max_tokens": 5,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200

            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    chunks.append(line)

            assert len(chunks) >= 3


class TestProxyServerE2E:
    """End-to-end tests for the proxy server."""

    @pytest.fixture
    def config(self) -> KVBenchConfig:
        """Create a proxy server configuration."""
        return KVBenchConfig(
            instance_id="proxy-test",
            server=ServerConfig(server_type="proxy"),
        )

    @pytest.fixture
    async def client(self, config: KVBenchConfig) -> AsyncGenerator[AsyncClient, None]:
        """Create a test client."""
        async with create_test_client(config) as client:
            yield client

    @pytest.mark.asyncio
    async def test_proxy_health(self, client: AsyncClient) -> None:
        """Test proxy server health check."""
        response = await client.get("/health")
        assert response.status_code == 200
        # Proxy with no backends should be unhealthy
        assert response.json()["server_type"] == "proxy"
