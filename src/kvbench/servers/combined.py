"""
Combined Server Implementation.

This module implements a combined server that handles both prefill and decode
phases in a single server instance. This is the typical deployment mode for
non-disaggregated LLM serving.

The server provides a complete OpenAI-compatible API with streaming support.
"""

from __future__ import annotations

import asyncio
import hashlib
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
class CombinedStats:
    """Statistics for the combined server.

    Attributes:
        requests_total: Total number of requests processed.
        requests_success: Number of successful requests.
        requests_failed: Number of failed requests.
        prompt_tokens: Total prompt tokens processed.
        completion_tokens: Total completion tokens generated.
        cache_hits: Number of chunk cache hits.
        cache_misses: Number of chunk cache misses.
        total_latency_ms: Total latency in milliseconds.
        start_time: Server start time.
    """

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request."""
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total


class CombinedServer:
    """Combined prefill+decode server for KV cache benchmarking.

    This server handles both phases of LLM inference in a single process,
    which is the typical deployment mode for non-disaggregated serving.

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
        """Initialize the combined server.

        Args:
            config: KV-Bench configuration.
            connector: KV cache connector for cache operations.
        """
        self.config = config
        self.connector = connector
        self.stats = CombinedStats()
        self._running = False

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.server.model_profile

    @property
    def instance_id(self) -> str:
        """Get the instance ID."""
        return self.config.instance_id

    def _simulate_tokenize(self, text: str) -> list[int]:
        """Simulate tokenization of text."""
        num_tokens = max(1, len(text) // 4)
        return list(range(num_tokens))

    def _chunk_tokens(self, tokens: list[int], chunk_size: int) -> list[list[int]]:
        """Split tokens into chunks."""
        chunks = []
        for i in range(0, len(tokens), chunk_size):
            chunks.append(tokens[i : i + chunk_size])
        return chunks

    def _compute_chunk_hash(self, tokens: list[int]) -> str:
        """Compute hash for a token chunk."""
        data = ",".join(str(t) for t in tokens).encode()
        return hashlib.sha256(data).hexdigest()

    def _calculate_prefill_latency_ms(self, num_tokens: int) -> float:
        """Calculate prefill latency."""
        efficiency = self.config.gpu.efficiency_factor
        tp_size = self.config.gpu.tp_size
        base_latency_per_token = 0.1 / efficiency
        scaled_latency = base_latency_per_token / tp_size
        return scaled_latency * num_tokens

    def _calculate_decode_latency_ms(self, context_length: int) -> float:
        """Calculate decode latency per token."""
        efficiency = self.config.gpu.efficiency_factor
        tp_size = self.config.gpu.tp_size
        base_latency = 8.0 / efficiency
        scaled_latency = base_latency / tp_size
        context_penalty = (context_length / 1000) * 0.01
        return scaled_latency + context_penalty

    def _generate_mock_token(self, position: int) -> str:
        """Generate a mock token."""
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

    async def _process_prefill(self, prompt: str) -> tuple[int, int, int]:
        """Process prefill phase.

        Returns:
            Tuple of (num_tokens, cache_hits, cache_misses).
        """
        tokens = self._simulate_tokenize(prompt)
        num_tokens = len(tokens)

        chunk_size = 256
        if hasattr(self.connector, "config"):
            chunk_size = getattr(self.connector.config, "chunk_size", 256)

        chunks = self._chunk_tokens(tokens, chunk_size)

        cache_hits = 0
        cache_misses = 0

        for chunk in chunks:
            chunk_hash = self._compute_chunk_hash(chunk)

            if await self.connector.exists(chunk_hash):
                cache_hits += 1
            else:
                cache_misses += 1
                chunk_latency = self._calculate_prefill_latency_ms(len(chunk))
                await asyncio.sleep(chunk_latency / 1000)
                await self.connector.store(chunk_hash, len(chunk))

        return num_tokens, cache_hits, cache_misses

    async def chat_completions_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[str]:
        """Handle a streaming chat completion request.

        Args:
            request: The chat completion request.

        Yields:
            SSE-formatted chunks.
        """
        self.stats.requests_total += 1
        start_time = time.time()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        try:
            # Prefill phase
            prompt = request.prompt_text or ""
            num_prompt_tokens, hits, misses = await self._process_prefill(prompt)
            self.stats.prompt_tokens += num_prompt_tokens
            self.stats.cache_hits += hits
            self.stats.cache_misses += misses

            # Send initial chunk
            start_chunk = ChatCompletionChunk.create_start(
                model=self.model_name,
                completion_id=completion_id,
            )
            yield start_chunk.to_sse()

            # Decode phase
            max_tokens = request.max_tokens or 100
            context_length = num_prompt_tokens
            tokens_generated = 0

            for i in range(max_tokens):
                latency_ms = self._calculate_decode_latency_ms(context_length + i)
                await asyncio.sleep(latency_ms / 1000)

                token = self._generate_mock_token(i)
                tokens_generated += 1

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
            self.stats.completion_tokens += tokens_generated
            self.stats.total_latency_ms += elapsed_ms
            self.stats.requests_success += 1

        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"Streaming request failed: {e}")
            error_response = ErrorResponse.create(str(e), "server_error")
            yield f"data: {error_response.model_dump_json()}\n\n"

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Handle a non-streaming chat completion request.

        Args:
            request: The chat completion request.

        Returns:
            A ChatCompletionResponse or ErrorResponse.
        """
        self.stats.requests_total += 1
        start_time = time.time()

        try:
            # Prefill phase
            prompt = request.prompt_text or ""
            num_prompt_tokens, hits, misses = await self._process_prefill(prompt)
            self.stats.prompt_tokens += num_prompt_tokens
            self.stats.cache_hits += hits
            self.stats.cache_misses += misses

            # Decode phase
            max_tokens = request.max_tokens or 100
            context_length = num_prompt_tokens
            generated_text = ""
            tokens_generated = 0

            for i in range(max_tokens):
                latency_ms = self._calculate_decode_latency_ms(context_length + i)
                await asyncio.sleep(latency_ms / 1000)

                token = self._generate_mock_token(i)
                generated_text += token
                tokens_generated += 1

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.completion_tokens += tokens_generated
            self.stats.total_latency_ms += elapsed_ms
            self.stats.requests_success += 1

            return ChatCompletionResponse.create(
                model=self.model_name,
                content=generated_text,
                prompt_tokens=num_prompt_tokens,
                completion_tokens=tokens_generated,
                finish_reason="stop",
            )

        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"Request failed: {e}")
            return ErrorResponse.create(str(e), "server_error")

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
        """Check server health."""
        uptime = time.time() - self.stats.start_time
        return HealthResponse(
            status="healthy",
            instance_id=self.instance_id,
            server_type="combined",
            model=self.model_name,
            uptime_seconds=uptime,
        )

    async def get_metrics(self) -> MetricsResponse:
        """Get server metrics."""
        return MetricsResponse(
            requests_total=self.stats.requests_total,
            requests_success=self.stats.requests_success,
            requests_failed=self.stats.requests_failed,
            tokens_processed=self.stats.total_tokens,
            cache_hits=self.stats.cache_hits,
            cache_misses=self.stats.cache_misses,
            avg_latency_ms=self.stats.avg_latency_ms,
        )

    async def start(self) -> None:
        """Start the server."""
        self._running = True
        logger.info(f"Combined server {self.instance_id} started")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        await self.connector.close()
        logger.info(f"Combined server {self.instance_id} stopped")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CombinedServer("
            f"instance_id={self.instance_id!r}, "
            f"model={self.model_name!r}, "
            f"tokens={self.stats.total_tokens})"
        )
