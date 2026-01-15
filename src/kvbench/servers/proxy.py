"""
Disaggregated Proxy Implementation.

This module implements a proxy server that routes requests to separate
prefill and decode servers in a disaggregated LLM serving architecture.

The proxy handles:
- Load balancing across multiple prefill/decode servers
- Request routing (prefill → decode handoff)
- Health monitoring of backend servers
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from kvbench.servers.openai_compat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ModelList,
)

if TYPE_CHECKING:
    from kvbench.core.config import KVBenchConfig

logger = logging.getLogger(__name__)


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategy."""

    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"


@dataclass
class ServerEndpoint:
    """Backend server endpoint.

    Attributes:
        url: Server URL.
        healthy: Whether the server is healthy.
        active_connections: Number of active connections.
        total_requests: Total requests processed.
        last_health_check: Time of last health check.
    """

    url: str
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    last_health_check: float = field(default_factory=time.time)


@dataclass
class ProxyStats:
    """Statistics for the proxy server.

    Attributes:
        requests_total: Total number of requests processed.
        requests_success: Number of successful requests.
        requests_failed: Number of failed requests.
        prefill_requests: Requests sent to prefill servers.
        decode_requests: Requests sent to decode servers.
        total_latency_ms: Total latency in milliseconds.
        start_time: Server start time.
    """

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    prefill_requests: int = 0
    decode_requests: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request."""
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total


class DisaggregatedProxy:
    """Proxy for disaggregated prefill/decode architecture.

    Routes requests to separate prefill and decode servers,
    handling the handoff between phases.

    Attributes:
        config: KV-Bench configuration.
        prefill_servers: List of prefill server endpoints.
        decode_servers: List of decode server endpoints.
        stats: Proxy statistics.
    """

    def __init__(
        self,
        config: KVBenchConfig,
        prefill_endpoints: list[str] | None = None,
        decode_endpoints: list[str] | None = None,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
    ) -> None:
        """Initialize the disaggregated proxy.

        Args:
            config: KV-Bench configuration.
            prefill_endpoints: List of prefill server URLs.
            decode_endpoints: List of decode server URLs.
            strategy: Load balancing strategy.
        """
        self.config = config
        self.strategy = strategy
        self.stats = ProxyStats()
        self._running = False

        # Initialize endpoint lists
        prefill_urls = prefill_endpoints or config.distributed.prefill_endpoints
        decode_urls = decode_endpoints or config.distributed.decode_endpoints

        self.prefill_servers = [ServerEndpoint(url=url) for url in prefill_urls]
        self.decode_servers = [ServerEndpoint(url=url) for url in decode_urls]

        # Round-robin counters
        self._prefill_index = 0
        self._decode_index = 0

        # Health check interval
        self._health_check_interval = config.distributed.health_check_interval
        self._health_check_task: asyncio.Task | None = None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.server.model_profile

    @property
    def instance_id(self) -> str:
        """Get the instance ID."""
        return self.config.instance_id

    def _select_prefill_server(self) -> ServerEndpoint | None:
        """Select a prefill server using the configured strategy.

        Returns:
            A healthy prefill server endpoint or None if none available.
        """
        healthy_servers = [s for s in self.prefill_servers if s.healthy]
        if not healthy_servers:
            return None

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            server = healthy_servers[self._prefill_index % len(healthy_servers)]
            self._prefill_index += 1
            return server

        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy_servers, key=lambda s: s.active_connections)

        elif self.strategy == LoadBalanceStrategy.RANDOM:
            import random

            return random.choice(healthy_servers)

        return healthy_servers[0]

    def _select_decode_server(self) -> ServerEndpoint | None:
        """Select a decode server using the configured strategy.

        Returns:
            A healthy decode server endpoint or None if none available.
        """
        healthy_servers = [s for s in self.decode_servers if s.healthy]
        if not healthy_servers:
            return None

        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            server = healthy_servers[self._decode_index % len(healthy_servers)]
            self._decode_index += 1
            return server

        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy_servers, key=lambda s: s.active_connections)

        elif self.strategy == LoadBalanceStrategy.RANDOM:
            import random

            return random.choice(healthy_servers)

        return healthy_servers[0]

    async def _simulate_prefill_request(
        self,
        server: ServerEndpoint,
        request: ChatCompletionRequest,
    ) -> tuple[int, bool]:
        """Simulate sending a prefill request to a backend server.

        In a real implementation, this would make an HTTP request.
        For the mock, we simulate the latency and return mock results.

        Args:
            server: The prefill server endpoint.
            request: The chat completion request.

        Returns:
            Tuple of (prompt_tokens, success).
        """
        server.active_connections += 1
        try:
            # Simulate network + processing latency
            prompt = request.prompt_text or ""
            num_tokens = max(1, len(prompt) // 4)

            # Simulate prefill latency: ~0.1ms per token
            latency_ms = num_tokens * 0.1
            await asyncio.sleep(latency_ms / 1000)

            server.total_requests += 1
            self.stats.prefill_requests += 1

            return num_tokens, True

        finally:
            server.active_connections -= 1

    async def _simulate_decode_request(
        self,
        server: ServerEndpoint,
        request: ChatCompletionRequest,
        context_length: int,  # noqa: ARG002
    ) -> tuple[str, int]:
        """Simulate sending a decode request to a backend server.

        Args:
            server: The decode server endpoint.
            request: The chat completion request.
            context_length: Context length from prefill.

        Returns:
            Tuple of (generated_text, completion_tokens).
        """
        server.active_connections += 1
        try:
            max_tokens = request.max_tokens or 100

            # Simulate decode latency: ~8ms per token
            mock_tokens = [
                " The", " answer", " is", " that", " we", " need",
                " to", " consider", " this", " carefully", ".",
            ]
            generated_text = ""

            for i in range(max_tokens):
                await asyncio.sleep(8 / 1000)  # 8ms per token
                generated_text += mock_tokens[i % len(mock_tokens)]

            server.total_requests += 1
            self.stats.decode_requests += 1

            return generated_text, max_tokens

        finally:
            server.active_connections -= 1

    async def _simulate_decode_stream(
        self,
        server: ServerEndpoint,
        request: ChatCompletionRequest,
        context_length: int,  # noqa: ARG002
    ) -> AsyncIterator[str]:
        """Simulate streaming decode from a backend server.

        Args:
            server: The decode server endpoint.
            request: The chat completion request.
            context_length: Context length from prefill.

        Yields:
            Generated tokens.
        """
        server.active_connections += 1
        try:
            max_tokens = request.max_tokens or 100
            mock_tokens = [
                " The", " answer", " is", " that", " we", " need",
                " to", " consider", " this", " carefully", ".",
            ]

            for i in range(max_tokens):
                await asyncio.sleep(8 / 1000)  # 8ms per token
                yield mock_tokens[i % len(mock_tokens)]

            server.total_requests += 1
            self.stats.decode_requests += 1

        finally:
            server.active_connections -= 1

    async def chat_completions_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Handle a streaming chat completion request.

        Routes the request to prefill server, then streams from decode server.

        Args:
            request: The chat completion request.

        Yields:
            SSE-formatted chunks.
        """
        self.stats.requests_total += 1
        start_time = time.time()

        try:
            # Select prefill server
            prefill_server = self._select_prefill_server()
            if not prefill_server:
                error = ErrorResponse.create(
                    "No healthy prefill servers available",
                    "service_unavailable",
                )
                yield f"data: {error.model_dump_json()}\n\n"
                self.stats.requests_failed += 1
                return

            # Prefill phase
            context_length, success = await self._simulate_prefill_request(
                prefill_server, request
            )
            if not success:
                error = ErrorResponse.create("Prefill failed", "server_error")
                yield f"data: {error.model_dump_json()}\n\n"
                self.stats.requests_failed += 1
                return

            # Select decode server
            decode_server = self._select_decode_server()
            if not decode_server:
                error = ErrorResponse.create(
                    "No healthy decode servers available",
                    "service_unavailable",
                )
                yield f"data: {error.model_dump_json()}\n\n"
                self.stats.requests_failed += 1
                return

            # Stream decode phase
            import uuid

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            # Send initial chunk
            start_chunk = ChatCompletionChunk.create_start(
                model=self.model_name,
                completion_id=completion_id,
            )
            yield start_chunk.to_sse()

            # Stream tokens
            async for token in self._simulate_decode_stream(
                decode_server, request, context_length
            ):
                content_chunk = ChatCompletionChunk.create_content(
                    model=self.model_name,
                    content=token,
                    completion_id=completion_id,
                )
                yield content_chunk.to_sse()

            # Final chunk
            end_chunk = ChatCompletionChunk.create_end(
                model=self.model_name,
                completion_id=completion_id,
                finish_reason="stop",
            )
            yield end_chunk.to_sse()
            yield "data: [DONE]\n\n"

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.total_latency_ms += elapsed_ms
            self.stats.requests_success += 1

        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"Streaming request failed: {e}")
            error = ErrorResponse.create(str(e), "server_error")
            yield f"data: {error.model_dump_json()}\n\n"

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Handle a non-streaming chat completion request.

        Routes the request to prefill server, then to decode server.

        Args:
            request: The chat completion request.

        Returns:
            A ChatCompletionResponse or ErrorResponse.
        """
        self.stats.requests_total += 1
        start_time = time.time()

        try:
            # Select and call prefill server
            prefill_server = self._select_prefill_server()
            if not prefill_server:
                self.stats.requests_failed += 1
                return ErrorResponse.create(
                    "No healthy prefill servers available",
                    "service_unavailable",
                )

            context_length, success = await self._simulate_prefill_request(
                prefill_server, request
            )
            if not success:
                self.stats.requests_failed += 1
                return ErrorResponse.create("Prefill failed", "server_error")

            # Select and call decode server
            decode_server = self._select_decode_server()
            if not decode_server:
                self.stats.requests_failed += 1
                return ErrorResponse.create(
                    "No healthy decode servers available",
                    "service_unavailable",
                )

            generated_text, completion_tokens = await self._simulate_decode_request(
                decode_server, request, context_length
            )

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.total_latency_ms += elapsed_ms
            self.stats.requests_success += 1

            return ChatCompletionResponse.create(
                model=self.model_name,
                content=generated_text,
                prompt_tokens=context_length,
                completion_tokens=completion_tokens,
                finish_reason="stop",
            )

        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"Request failed: {e}")
            return ErrorResponse.create(str(e), "server_error")

    async def _health_check_loop(self) -> None:
        """Periodically check health of backend servers."""
        while self._running:
            try:
                # Check prefill servers
                for server in self.prefill_servers:
                    # In a real implementation, make HTTP health check
                    server.last_health_check = time.time()
                    # For mock, servers are always healthy

                # Check decode servers
                for server in self.decode_servers:
                    server.last_health_check = time.time()

                await asyncio.sleep(self._health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self._health_check_interval)

    async def list_models(self) -> ModelList:
        """List available models."""
        return ModelList(
            data=[
                ModelInfo(
                    id=self.model_name,
                    owned_by="kvbench",
                )
            ]
        )

    async def health_check(self) -> HealthResponse:
        """Check proxy health."""
        uptime = time.time() - self.stats.start_time
        healthy_prefill = sum(1 for s in self.prefill_servers if s.healthy)
        healthy_decode = sum(1 for s in self.decode_servers if s.healthy)

        status = "healthy"
        if healthy_prefill == 0 or healthy_decode == 0:
            status = "unhealthy"
        elif healthy_prefill < len(self.prefill_servers) or healthy_decode < len(
            self.decode_servers
        ):
            status = "degraded"

        return HealthResponse(
            status=status,  # type: ignore[arg-type]
            instance_id=self.instance_id,
            server_type="proxy",
            model=self.model_name,
            uptime_seconds=uptime,
        )

    async def get_metrics(self) -> MetricsResponse:
        """Get proxy metrics."""
        return MetricsResponse(
            requests_total=self.stats.requests_total,
            requests_success=self.stats.requests_success,
            requests_failed=self.stats.requests_failed,
            tokens_processed=0,  # Proxy doesn't track tokens directly
            cache_hits=0,
            cache_misses=0,
            avg_latency_ms=self.stats.avg_latency_ms,
        )

    def get_server_status(self) -> dict:
        """Get status of all backend servers.

        Returns:
            Dictionary with prefill and decode server status.
        """
        return {
            "prefill_servers": [
                {
                    "url": s.url,
                    "healthy": s.healthy,
                    "active_connections": s.active_connections,
                    "total_requests": s.total_requests,
                }
                for s in self.prefill_servers
            ],
            "decode_servers": [
                {
                    "url": s.url,
                    "healthy": s.healthy,
                    "active_connections": s.active_connections,
                    "total_requests": s.total_requests,
                }
                for s in self.decode_servers
            ],
        }

    async def start(self) -> None:
        """Start the proxy server."""
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(
            f"Proxy {self.instance_id} started with "
            f"{len(self.prefill_servers)} prefill, "
            f"{len(self.decode_servers)} decode servers"
        )

    async def stop(self) -> None:
        """Stop the proxy server."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Proxy {self.instance_id} stopped")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DisaggregatedProxy("
            f"instance_id={self.instance_id!r}, "
            f"prefill={len(self.prefill_servers)}, "
            f"decode={len(self.decode_servers)})"
        )
