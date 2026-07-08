"""
Prefill Server Implementation.

This module implements a mock prefill server that emulates the prefill phase
of LLM inference. The prefill phase processes all input tokens in parallel
and is typically compute-bound.

The server provides an OpenAI-compatible API and simulates:
- Token chunking and hash computation
- Cache lookups and stores
- Prefill latency based on GPU/model configuration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kvbench.core.latency import LatencyCalculator
from kvbench.core.tokens import TokenProcessor
from kvbench.servers.openai_compat import (
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
class PrefillStats:
    """Statistics for the prefill server.

    Attributes:
        requests_total: Total number of requests processed.
        requests_success: Number of successful requests.
        requests_failed: Number of failed requests.
        tokens_processed: Total tokens processed.
        cache_hits: Number of chunk cache hits.
        cache_misses: Number of chunk cache misses.
        total_latency_ms: Total latency in milliseconds.
        start_time: Server start time.
    """

    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    tokens_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request."""
        if self.requests_total == 0:
            return 0.0
        return self.total_latency_ms / self.requests_total


class PrefillServer:
    """Mock prefill server for KV cache benchmarking.

    The prefill server processes the input prompt, chunks the tokens,
    and stores KV cache entries. It simulates compute-bound prefill
    latency based on the GPU and model configuration.

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
        """Initialize the prefill server.

        Args:
            config: KV-Bench configuration.
            connector: KV cache connector for storing cache entries.
        """
        self.config = config
        self.connector = connector
        self.stats = PrefillStats()
        self._running = False

        chunk_size = (
            getattr(getattr(connector, "config", None), "chunk_size", None)
            or config.connector.lmcache_chunk_size
        )
        self.processor = TokenProcessor(chunk_size=chunk_size)
        self.calculator = LatencyCalculator(
            gpu=config.gpu.gpu_profile,
            model=config.server.model_profile,
            tp_size=config.gpu.tp_size,
            efficiency=config.gpu.efficiency_factor,
        )

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.server.model_profile

    @property
    def instance_id(self) -> str:
        """Get the instance ID."""
        return self.config.instance_id

    def _simulate_tokenize(self, text: str) -> list[int]:
        """Simulate tokenization of text.

        Content-based and prefix-stable: shared text prefixes produce shared
        token prefixes, which is what makes prefix caching measurable.

        Args:
            text: The text to tokenize.

        Returns:
            List of simulated token IDs.
        """
        return self.processor.simulate_tokenize(text)

    def _chunk_tokens(self, tokens: list[int], chunk_size: int) -> list[list[int]]:
        """Split tokens into chunks.

        Args:
            tokens: List of token IDs.
            chunk_size: Number of tokens per chunk.

        Returns:
            List of token chunks.
        """
        return [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

    def _compute_chunk_hash(self, tokens: list[int], prefix_hash: str | None = None) -> str:
        """Compute a hash for a token chunk.

        Args:
            tokens: List of token IDs.
            prefix_hash: Hash of the preceding chunk (prefix chaining).

        Returns:
            SHA-256 hash of the tokens.
        """
        return self.processor.compute_chunk_hash(tokens, prefix_hash=prefix_hash)

    def _calculate_prefill_latency_ms(self, num_tokens: int, context_tokens: int = 0) -> float:
        """Calculate simulated prefill latency using the roofline model.

        Args:
            num_tokens: Number of new tokens to process.
            context_tokens: Already-cached prefix tokens they attend to.

        Returns:
            Estimated prefill latency in milliseconds.
        """
        return self.calculator.prefill_latency(num_tokens, context_tokens=context_tokens).total_ms

    async def process_prefill(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[int, int, int, float]:
        """Process a prefill request.

        Args:
            request: The chat completion request.

        Returns:
            Tuple of (prompt_tokens, cache_hits, cache_misses, latency_ms).
        """
        start_time = time.time()

        # Get prompt text
        prompt = request.prompt_text or ""

        # Simulate tokenization (content-based, prefix-stable)
        tokens = self.processor.simulate_tokenize(prompt)
        num_tokens = len(tokens)

        # Chunk with prefix-chained hashes
        chunks = self.processor.chunk_tokens_with_metadata(tokens)

        cache_hits = 0
        cache_misses = 0
        total_prefill_latency = 0.0

        for chunk in chunks:
            # Check cache
            if await self.connector.exists(chunk.chunk_hash):
                cache_hits += 1
                logger.debug(f"Cache hit for chunk {chunk.chunk_hash[:16]}...")
                # Load the KV data from storage (the passthrough read path)
                await self.connector.load(chunk.chunk_hash)
            else:
                cache_misses += 1
                logger.debug(f"Cache miss for chunk {chunk.chunk_hash[:16]}...")

                # Simulate prefill latency for cache miss; the chunk attends
                # to everything before it in the sequence
                chunk_latency = self._calculate_prefill_latency_ms(
                    chunk.size, context_tokens=chunk.start_position
                )
                total_prefill_latency += chunk_latency
                await asyncio.sleep(chunk_latency / 1000)

                # Store in cache
                await self.connector.store(chunk.chunk_hash, chunk.size)

        elapsed_ms = (time.time() - start_time) * 1000
        return num_tokens, cache_hits, cache_misses, elapsed_ms

    async def chat_completions(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse | ErrorResponse:
        """Handle a chat completion request (prefill only).

        Args:
            request: The chat completion request.

        Returns:
            A ChatCompletionResponse or ErrorResponse.
        """
        self.stats.requests_total += 1

        try:
            # Process prefill
            num_tokens, hits, misses, latency = await self.process_prefill(request)

            # Update stats
            self.stats.tokens_processed += num_tokens
            self.stats.cache_hits += hits
            self.stats.cache_misses += misses
            self.stats.total_latency_ms += latency
            self.stats.requests_success += 1

            # Generate mock response
            # In a real disaggregated setup, this would hand off to decode server
            response = ChatCompletionResponse.create(
                model=self.model_name,
                content="[Prefill complete - handoff to decode server]",
                prompt_tokens=num_tokens,
                completion_tokens=0,
                finish_reason="stop",
            )

            return response

        except Exception as e:
            self.stats.requests_failed += 1
            logger.error(f"Prefill request failed: {e}")
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
            server_type="prefill",
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
            tokens_processed=self.stats.tokens_processed,
            cache_hits=self.stats.cache_hits,
            cache_misses=self.stats.cache_misses,
            avg_latency_ms=self.stats.avg_latency_ms,
        )

    async def start(self) -> None:
        """Start the server."""
        self._running = True
        logger.info(f"Prefill server {self.instance_id} started")

    async def stop(self) -> None:
        """Stop the server."""
        self._running = False
        await self.connector.close()
        logger.info(f"Prefill server {self.instance_id} stopped")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"PrefillServer("
            f"instance_id={self.instance_id!r}, "
            f"model={self.model_name!r}, "
            f"requests={self.stats.requests_total})"
        )
