"""
Decode Server Implementation.

This module implements a mock decode server that emulates the decode phase
of LLM inference. The decode phase generates tokens one at a time and is
typically memory-bound due to KV cache loading.

The server provides an OpenAI-compatible API with streaming support.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
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
    from kvbench.connectors.base import KVConnector
    from kvbench.core.config import KVBenchConfig

logger = logging.getLogger(__name__)


@dataclass
class DecodeStats:
    """Statistics for the decode server.

    Attributes:
        requests_total: Total number of requests processed.
        requests_success: Number of successful requests.
        requests_failed: Number of failed requests.
        tokens_generated: Total tokens generated.
        total_latency_ms: Total latency in milliseconds.
        start_time: Server start time.
    """

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    tokens_generated: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request."""
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total

    @property
    def avg_tokens_per_request(self) -> float:
        """Average tokens generated per request."""
        if self.requests_total == 0:
            return 0.0
        return self.tokens_generated / self.requests_total


class DecodeServer:
    """Mock decode server for KV cache benchmarking.

    The decode server generates tokens one at a time, simulating the
    memory-bound decode phase of LLM inference. It loads KV cache
    entries and generates tokens with appropriate latency.

    Attributes:
        config: KV-Bench configuration.
        connector: KV cache connector.
        stats: Server statistics.
    """

    def __init__(
        self,
        config: KVBenchConfig,
        connector: KVConnector,
    ) -> None:
        """Initialize the decode server.

        Args:
            config: KV-Bench configuration.
            connector: KV cache connector for loading cache entries.
        """
        self.config = config
        self.connector = connector
        self.stats = DecodeStats()
        self._running = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.server.model_profile

    @property
    def instance_id(self) -> str:
        """Get the instance ID."""
        return self.config.instance_id

    def _calculate_decode_latency_ms(self, context_length: int) -> float:
        """Calculate simulated decode latency per token.

        Decode is memory-bound. Latency depends on loading model
        weights and KV cache from memory.

        Args:
            context_length: Current context length (affects KV cache size).

        Returns:
            Estimated decode latency per token in milliseconds.
        """
        efficiency = self.config.gpu.efficiency_factor
        tp_size = self.config.gpu.tp_size

        # Base decode latency: ~5-10ms per token for large models
        # This is a simplified model based on memory bandwidth
        base_latency = 8.0 / efficiency

        # Scale with tensor parallelism
        scaled_latency = base_latency / tp_size

        # Longer contexts have larger KV cache to load
        # Add ~0.01ms per 1000 context tokens
        context_penalty = (context_length / 1000) * 0.01

        return scaled_latency + context_penalty

    def _generate_mock_token(self, position: int) -> str:
        """Generate a mock token for testing.

        Args:
            position: Position in the generation sequence.

        Returns:
            A mock token string.
        """
        # Generate realistic-looking mock tokens
        mock_tokens = [
            " The", " answer", " to", " your", " question", " is",
            " that", " we", " need", " to", " consider", " multiple",
            " factors", " when", " making", " this", " decision", ".",
            " First", ",", " let", " me", " explain", " the",
            " key", " points", " that", " are", " relevant", " here",
            ".", " Based", " on", " the", " information", " provided",
            ",", " I", " would", " recommend", " the", " following",
            " approach", ":", " carefully", " analyze", " the",
            " situation", " and", " then", " proceed", ".",
        ]
        return mock_tokens[position % len(mock_tokens)]

    async def generate_tokens(
        self,
        context_length: int,
        max_tokens: int,
    ) -> AsyncIterator[tuple[str, float]]:
        """Generate tokens with simulated decode latency.

        Args:
            context_length: Initial context length.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Tuple of (token, latency_ms) for each generated token.
        """
        for i in range(max_tokens):
            # Calculate latency for this token
            current_context = context_length + i
            latency_ms = self._calculate_decode_latency_ms(current_context)

            # Simulate decode latency
            await asyncio.sleep(latency_ms / 1000)

            # Generate mock token
            token = self._generate_mock_token(i)

            yield token, latency_ms

    async def chat_completions_stream(
        self,
        request: ChatCompletionRequest,
        context_length: int,
    ) -> AsyncIterator[str]:
        """Handle a streaming chat completion request.

        Args:
            request: The chat completion request.
            context_length: Context length from prefill phase.

        Yields:
            SSE-formatted chunks.
        """
        max_tokens = request.max_tokens or 100
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Send initial chunk with role
        start_chunk = ChatCompletionChunk.create_start(
            model=self.model_name,
            completion_id=completion_id,
        )
        yield start_chunk.to_sse()

        # Generate tokens
        tokens_generated = 0
        total_latency = 0.0

        async for token, latency_ms in self.generate_tokens(context_length, max_tokens):
            tokens_generated += 1
            total_latency += latency_ms

            # Send content chunk
            content_chunk = ChatCompletionChunk.create_content(
                model=self.model_name,
                content=token,
                completion_id=completion_id,
            )
            yield content_chunk.to_sse()

        # Send final chunk
        end_chunk = ChatCompletionChunk.create_end(
            model=self.model_name,
            completion_id=completion_id,
            finish_reason="stop",
        )
        yield end_chunk.to_sse()

        # Send [DONE]
        yield "data: [DONE]\n\n"

        # Update stats
        self.stats.tokens_generated += tokens_generated
        self.stats.total_latency_ms += total_latency

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
        context_length: int = 0,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Handle a non-streaming chat completion request.

        Args:
            request: The chat completion request.
            context_length: Context length from prefill phase.

        Returns:
            A ChatCompletionResponse or ErrorResponse.
        """
        self.stats.requests_total += 1
        start_time = time.time()

        try:
            max_tokens = request.max_tokens or 100

            # Generate all tokens
            generated_text = ""
            tokens_generated = 0

            async for token, _ in self.generate_tokens(context_length, max_tokens):
                generated_text += token
                tokens_generated += 1

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.tokens_generated += tokens_generated
            self.stats.total_latency_ms += elapsed_ms
            self.stats.requests_success += 1

            # Create response
            response = ChatCompletionResponse.create(
                model=self.model_name,
                content=generated_text,
                prompt_tokens=context_length,
                completion_tokens=tokens_generated,
                finish_reason="stop",
            )

            return response

        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"Decode request failed: {e}")
            return ErrorResponse.create(
                message=str(e),
                error_type="server_error",
            )

    async def list_models(self) -> ModelList:
        """List available models.

        Returns:
            ModelList with the configured model.
        """
        return ModelList(
            data=[
                ModelInfo(
                    id=self.model_name,
                    owned_by="kvbench",
                )
            ]
        )

    async def health_check(self) -> HealthResponse:
        """Check server health.

        Returns:
            HealthResponse with current status.
        """
        uptime = time.time() - self.stats.start_time
        return HealthResponse(
            status="healthy",
            instance_id=self.instance_id,
            server_type="decode",
            model=self.model_name,
            uptime_seconds=uptime,
        )

    async def get_metrics(self) -> MetricsResponse:
        """Get server metrics.

        Returns:
            MetricsResponse with current statistics.
        """
        return MetricsResponse(
            requests_total=self.stats.requests_total,
            requests_success=self.stats.requests_success,
            requests_failed=self.stats.requests_failed,
            tokens_processed=self.stats.tokens_generated,
            cache_hits=0,  # Decode doesn't track cache hits directly
            cache_misses=0,
            avg_latency_ms=self.stats.avg_latency_ms,
        )

    async def start(self) -> None:
        """Start the server."""
        self._running = True
        logger.info(f"Decode server {self.instance_id} started")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        await self.connector.close()
        logger.info(f"Decode server {self.instance_id} stopped")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"DecodeServer("
            f"instance_id={self.instance_id!r}, "
            f"model={self.model_name!r}, "
            f"tokens={self.stats.tokens_generated})"
        )
