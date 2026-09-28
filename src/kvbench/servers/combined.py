"""
Combined Server Implementation.

This module implements a combined server that handles both prefill and decode
phases in a single server instance. This is the typical deployment mode for
non-disaggregated LLM serving.

The server provides a complete OpenAI-compatible API with streaming support.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kvbench.core.tokens import TokenProcessor
from kvbench.servers.gpu_engine import build_engine
from kvbench.servers.hotset import HotKVSet
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
from kvbench.timing import create_timing_strategy
from kvbench.timing.fullspeed import emulated_ms

if TYPE_CHECKING:
    from kvbench.core.config import KVBenchConfig
    from kvbench.kv.base import KVStack

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
    # Emulated (model x GPU) time, independent of how long the server actually
    # slept. In full-speed mode this is the only record of the emulated
    # hardware's speed; in normal mode it equals the slept time.
    emulated_prefill_ms: float = 0.0
    emulated_decode_ms: float = 0.0
    emulated_ttft_ms: float = 0.0       # sum over requests of prefill + first decode step
    completed_with_ttft: int = 0
    wall_serving_ms: float = 0.0        # actual wall time spent inside requests
    hot_tokens: int = 0                 # cached-prefix tokens served from the host-resident hot set

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def effective_ttft_ms(self) -> float:
        """Mean emulated time to first token per completed request."""
        if self.completed_with_ttft == 0:
            return 0.0
        return self.emulated_ttft_ms / self.completed_with_ttft

    @property
    def effective_tokens_per_s(self) -> float:
        """Completion tokens per second of emulated (prefill + decode) time."""
        total_ms = self.emulated_prefill_ms + self.emulated_decode_ms
        if total_ms <= 0:
            return 0.0
        return self.completion_tokens / (total_ms / 1000.0)

    @property
    def effective_decode_tokens_per_s(self) -> float:
        """Completion tokens per second of emulated decode time only (1000 / mean TPOT)."""
        if self.emulated_decode_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.emulated_decode_ms / 1000.0)

    @property
    def actual_tokens_per_s(self) -> float:
        """Completion tokens per second of wall time actually spent serving."""
        if self.wall_serving_ms <= 0:
            return 0.0
        return self.completion_tokens / (self.wall_serving_ms / 1000.0)

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
        kv: KV management stack for cache operations.
        stats: Server statistics.
    """

    def __init__(
        self,
        config: KVBenchConfig,
        kv: KVStack,
    ) -> None:
        """Initialize the combined server.

        Args:
            config: KV-Bench configuration.
            kv: Started KV management stack for cache operations.
        """
        self.config = config
        self.kv = kv
        self.stats = CombinedStats()
        self._running = False
        self._request_ttft_ms = 0.0

        self.chunk_size = kv.chunk_size
        self.processor = TokenProcessor(chunk_size=self.chunk_size)
        self.timing = create_timing_strategy(config)
        # Host-resident hot set (see hotset.py): default capacity = the KV
        # stack's local CPU tier, the memory the real serving stack keeps
        # session prefixes in between turns.
        try:
            cap = kv.capacity_info() if hasattr(kv, "capacity_info") else {}
        except Exception:  # stack not started (tests) or no tier info
            cap = {}
        hot_gb = config.timing.hot_kv_gb
        hot_bytes = int(hot_gb * (1 << 30)) if hot_gb else 0
        self.hot = HotKVSet(hot_bytes, self.chunk_size, int(cap.get("chunk_bytes") or 0))
        if self.hot.enabled:
            logger.info("Hot KV set enabled: %.1f GB (%d-token chunks of %d bytes)", hot_bytes / (1 << 30), self.chunk_size, self.hot.chunk_bytes)
        # Continuous-batching engine model (gpu_engine.py); None = per-request timing.
        self.engine = build_engine(config)
        # Optional per-request timing breakdown (JSONL), for validation runs.
        _log = os.environ.get("KVBENCH_REQUEST_LOG")
        self._req_log = open(_log, "a") if _log else None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.config.server.model_profile

    @property
    def instance_id(self) -> str:
        """Get the instance ID."""
        return self.config.instance_id

    def _simulate_tokenize(self, text: str) -> list[int]:
        """Simulate tokenization of text (content-based, prefix-stable)."""
        return self.processor.simulate_tokenize(text)

    def _calculate_prefill_latency_ms(self, num_tokens: int, context_tokens: int = 0) -> float:
        """Prefill latency to sleep (ms); records the emulated time in the stats."""
        result = self.timing.prefill_latency(num_tokens, context_tokens=context_tokens)
        self.stats.emulated_prefill_ms += emulated_ms(result)
        self._request_ttft_ms += emulated_ms(result)
        return result.total_ms

    def _calculate_decode_latency_ms(self, context_length: int) -> float:
        """Decode latency per token to sleep (ms); records the emulated time in the stats."""
        result = self.timing.decode_latency(context_length)
        self.stats.emulated_decode_ms += emulated_ms(result)
        return result.total_ms

    def _generate_mock_token(self, position: int) -> str:
        """Generate a mock token."""
        mock_tokens = [
            " The",
            " answer",
            " to",
            " your",
            " question",
            " is",
            " that",
            " we",
            " need",
            " to",
            " consider",
            " multiple",
            " factors",
            " when",
            " making",
            " this",
            " decision",
            ".",
            " First",
            ",",
            " let",
            " me",
            " explain",
            " the",
            " key",
            " points",
            " that",
            " are",
            " relevant",
            " here",
            ".",
            " Based",
            " on",
            " the",
            " information",
            " provided",
            ",",
            " I",
            " would",
            " recommend",
            " the",
            " following",
            " approach",
            ":",
            " carefully",
            " analyze",
            " the",
            " situation",
            " and",
            " then",
            " proceed",
            ".",
        ]
        return mock_tokens[position % len(mock_tokens)]

    async def _lookup_and_retrieve(self, prompt: str) -> tuple[list[int], int]:
        """Tokenize (off the event loop), look up, and read the cached prefix
        through the KV stack. Returns (tokens, hit_tokens)."""
        # A 100k-token prompt takes ~0.1 s to tokenize; keep it off the loop
        # so other streams' steps are not delayed.
        tokens = await asyncio.to_thread(self.processor.simulate_tokenize, prompt, self.config.timing.chars_per_token)
        hit_tokens = await self.kv.lookup(tokens)
        if hit_tokens:
            await self._retrieve_prefix(tokens, hit_tokens)
        return tokens, hit_tokens

    async def _retrieve_prefix(self, tokens: list[int], hit_tokens: int) -> float:
        """Read the cached prefix through the KV stack; returns elapsed ms."""
        t = time.monotonic()
        await self._retrieve_prefix_inner(tokens, hit_tokens)
        return (time.monotonic() - t) * 1e3

    async def _retrieve_prefix_inner(self, tokens: list[int], hit_tokens: int) -> None:
        hot_tokens = self.hot.resident_prefix_chunks(tokens, hit_tokens // self.chunk_size) * self.chunk_size
        self.stats.hot_tokens += hot_tokens
        if hot_tokens >= hit_tokens:
            return
        if hot_tokens > 0:
            await self.kv.retrieve(tokens[:hit_tokens], skip_leading=hot_tokens)
        else:
            await self.kv.retrieve(tokens[:hit_tokens])

    async def _generate_batched(self, request: ChatCompletionRequest):
        """Serve one request through the batching engine.

        Yields ``(index, token_text)`` per generated token. The front-end
        delay, the engine's queueing, chunked prefill and batched decode
        steps set the pacing; the KV stack I/O is real. The store of the new
        chunks overlaps decoding (vLLM saves layer-wise inside the step).
        """
        assert self.engine is not None
        t0 = time.monotonic()
        tokens = await asyncio.to_thread(self.processor.simulate_tokenize, request.prompt_text or "", self.config.timing.chars_per_token)
        t_tok = time.monotonic()
        lookup_id = uuid.uuid4().hex
        hit_tokens = await self.kv.lookup_pinned(tokens, lookup_id)
        t_look = time.monotonic()
        # The cached prefix is read through the KV stack for real, but the
        # emulator's copy (a CPU memcpy standing in for the engine's load)
        # overlaps the modelled front-end and engine time instead of adding
        # to it: the first token waits for whichever finishes last.
        ret_task = asyncio.create_task(self._retrieve_prefix(tokens, hit_tokens)) if hit_tokens else None
        t_ret = time.monotonic()
        num_tokens = len(tokens)
        miss_tokens = num_tokens - hit_tokens
        self.stats.prompt_tokens += num_tokens
        self.stats.cache_hits += hit_tokens // self.chunk_size
        self.stats.cache_misses += (miss_tokens + self.chunk_size - 1) // self.chunk_size

        target_s = self.config.timing.frontend_us_per_token * num_tokens / 1e6
        spent = time.monotonic() - t0
        if target_s > spent:
            await asyncio.sleep(target_s - spent)

        max_tokens = request.effective_max_tokens
        t_sub = time.monotonic()
        inflight_at_submit = self.engine.in_flight
        seq = self.engine.submit(miss_tokens, hit_tokens, max_tokens, load_tokens=hit_tokens)
        store_task: asyncio.Task | None = None
        rec = None
        try:
            await seq.queue.get()
            retrieve_ms = 0.0
            if ret_task is not None:
                retrieve_ms = await ret_task
                ret_task = None
            await self.kv.unpin(lookup_id)
            lookup_id = None
            t_first = time.monotonic()
            ttft_ms = (t_first - t0) * 1000.0
            if self._req_log is not None:
                rec = {
                    "t0": time.time() - (t_first - t0), "isl": num_tokens, "hit": hit_tokens, "max_tokens": max_tokens,
                    "tok_ms": (t_tok - t0) * 1e3, "lookup_ms": (t_look - t_tok) * 1e3, "retrieve_ms": retrieve_ms,
                    "fe_wait_ms": (t_sub - t_ret) * 1e3, "engine_first_ms": (t_first - t_sub) * 1e3,
                    "inflight_at_submit": inflight_at_submit, "lag_ms_max": self.engine.lag_ms_max,
                }
            self.stats.emulated_prefill_ms += ttft_ms
            self.stats.emulated_ttft_ms += ttft_ms
            self.stats.completed_with_ttft += 1
            store_end = (num_tokens // self.chunk_size) * self.chunk_size
            if miss_tokens and store_end > hit_tokens:
                store_task = asyncio.create_task(self._store_stepwise(tokens, hit_tokens, store_end))
            self.hot.insert(tokens, max(hit_tokens, store_end if miss_tokens else 0) // self.chunk_size)
            yield 0, self._generate_mock_token(0)
            for i in range(1, max_tokens):
                step_ms = await seq.queue.get()
                self.stats.emulated_decode_ms += step_ms
                yield i, self._generate_mock_token(i)
            t_last = time.monotonic()
            if store_task is not None:
                await store_task
                store_task = None
            if rec is not None:
                rec.update(decode_ms=(t_last - t_first) * 1e3, store_tail_ms=(time.monotonic() - t_last) * 1e3)
                self._req_log.write(json.dumps(rec) + "\n")
                self._req_log.flush()
        finally:
            self.engine.cancel(seq)
            if lookup_id is not None:
                # cancelled before the first token: release the pins once any
                # in-flight retrieve has finished with them
                async def _late_unpin(task=ret_task, lid=lookup_id):
                    if task is not None:
                        try:
                            await task
                        except Exception:
                            pass
                    await self.kv.unpin(lid)
                asyncio.create_task(_late_unpin())
            if ret_task is not None and not ret_task.done():
                ret_task.add_done_callback(lambda t: t.exception())
            if store_task is not None and not store_task.done():
                # Let an in-flight store finish in the background; never leak an
                # unawaited failure.
                store_task.add_done_callback(lambda t: t.exception())

    async def _store_stepwise(self, tokens: list[int], start: int, end: int) -> None:
        """Store new chunks in engine-step-sized pieces, as vLLM's connector
        saves the KV computed in each chunked-prefill step (at most the token
        budget per step), instead of one allocation for the whole turn."""
        step = max(self.chunk_size, (self.engine.token_budget // self.chunk_size) * self.chunk_size) if self.engine else end
        pos = start
        while pos < end:
            nxt = min(end, ((pos // self.chunk_size) * self.chunk_size) + step)
            await self.kv.store(tokens[:nxt], skip_leading=pos)
            pos = nxt

    async def _process_prefill(self, prompt: str) -> tuple[int, int, int]:
        """Process prefill phase.

        Returns:
            Tuple of (num_tokens, cache_hits, cache_misses).
        """
        tokens = await asyncio.to_thread(self.processor.simulate_tokenize, prompt, self.config.timing.chars_per_token)
        num_tokens = len(tokens)

        # Ask the KV stack how much of the sequence is already cached
        hit_tokens = await self.kv.lookup(tokens)
        if hit_tokens:
            # The part of the cached prefix still host-resident from this
            # session's previous turns is not re-read; the rest is read
            # through the stack (the storage read path).
            hot_tokens = self.hot.resident_prefix_chunks(tokens, hit_tokens // self.chunk_size) * self.chunk_size
            self.stats.hot_tokens += hot_tokens
            if hot_tokens >= hit_tokens:
                pass
            elif hot_tokens > 0:
                await self.kv.retrieve(tokens[:hit_tokens], skip_leading=hot_tokens)
            else:
                await self.kv.retrieve(tokens[:hit_tokens])

        miss_tokens = num_tokens - hit_tokens
        store_end = (num_tokens // self.chunk_size) * self.chunk_size
        if miss_tokens:
            # Simulate prefill compute for the uncached suffix, which
            # attends to the cached prefix
            prefill_latency = self._calculate_prefill_latency_ms(
                miss_tokens, context_tokens=hit_tokens
            )
            await asyncio.sleep(prefill_latency / 1000)

            # Store new full chunks only, matching the classic
            # LMCacheConnectorV1 path: the trailing partial chunk is never
            # stored (save_unfull_chunk=False / discard_partial_chunks), and
            # when the prompt is fully covered by the hit no store op is
            # emitted at all (vllm_v1_adapter skip rule).
            if store_end > hit_tokens:
                await self.kv.store(tokens[:store_end], skip_leading=hit_tokens)
        # Everything this turn touched (retrieved or stored) is now host-resident
        self.hot.insert(tokens, max(hit_tokens, store_end if miss_tokens else 0) // self.chunk_size)

        cache_hits = hit_tokens // self.chunk_size
        cache_misses = (miss_tokens + self.chunk_size - 1) // self.chunk_size

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
        self._request_ttft_ms = 0.0

        if self.engine is not None:
            try:
                n = 0
                async for i, token in self._generate_batched(request):
                    if i == 0:
                        yield ChatCompletionChunk.create_start(model=self.model_name, completion_id=completion_id).to_sse()
                    yield ChatCompletionChunk.create_content(model=self.model_name, content=token, completion_id=completion_id).to_sse()
                    n += 1
                yield ChatCompletionChunk.create_end(model=self.model_name, completion_id=completion_id, finish_reason="stop").to_sse()
                yield "data: [DONE]\n\n"
                elapsed_ms = (time.time() - start_time) * 1000
                self.stats.completion_tokens += n
                self.stats.total_latency_ms += elapsed_ms
                self.stats.wall_serving_ms += elapsed_ms
                self.stats.requests_success += 1
            except Exception as e:
                self.stats.requests_failed += 1
                logger.error(f"Streaming request failed: {e}")
                yield f"data: {ErrorResponse.create(str(e), 'server_error').model_dump_json()}\n\n"
            return

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
            max_tokens = request.effective_max_tokens
            context_length = num_prompt_tokens
            tokens_generated = 0

            for i in range(max_tokens):
                emu_before = self.stats.emulated_decode_ms
                latency_ms = self._calculate_decode_latency_ms(context_length + i)
                if i == 0:
                    self._request_ttft_ms += self.stats.emulated_decode_ms - emu_before
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
            self.stats.wall_serving_ms += elapsed_ms
            self.stats.emulated_ttft_ms += self._request_ttft_ms
            self.stats.completed_with_ttft += 1
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
        self._request_ttft_ms = 0.0

        if self.engine is not None:
            try:
                text, n = "", 0
                async for _i, token in self._generate_batched(request):
                    text += token
                    n += 1
                elapsed_ms = (time.time() - start_time) * 1000
                self.stats.completion_tokens += n
                self.stats.total_latency_ms += elapsed_ms
                self.stats.wall_serving_ms += elapsed_ms
                self.stats.requests_success += 1
                return ChatCompletionResponse.create(
                    model=self.model_name, content=text, prompt_tokens=0, completion_tokens=n, finish_reason="stop"
                )
            except Exception as e:
                self.stats.requests_failed += 1
                logger.error(f"Request failed: {e}")
                return ErrorResponse.create(str(e), "server_error")

        try:
            # Prefill phase
            prompt = request.prompt_text or ""
            num_prompt_tokens, hits, misses = await self._process_prefill(prompt)
            self.stats.prompt_tokens += num_prompt_tokens
            self.stats.cache_hits += hits
            self.stats.cache_misses += misses

            # Decode phase
            max_tokens = request.effective_max_tokens
            context_length = num_prompt_tokens
            generated_text = ""
            tokens_generated = 0

            for i in range(max_tokens):
                emu_before = self.stats.emulated_decode_ms
                latency_ms = self._calculate_decode_latency_ms(context_length + i)
                if i == 0:
                    self._request_ttft_ms += self.stats.emulated_decode_ms - emu_before
                await asyncio.sleep(latency_ms / 1000)

                token = self._generate_mock_token(i)
                generated_text += token
                tokens_generated += 1

            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.completion_tokens += tokens_generated
            self.stats.total_latency_ms += elapsed_ms
            self.stats.wall_serving_ms += elapsed_ms
            self.stats.emulated_ttft_ms += self._request_ttft_ms
            self.stats.completed_with_ttft += 1
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
            timing_mode="full-speed" if self.config.timing.full_speed else "emulated",
            emulated_model=self.model_name,
            emulated_gpu=f"{self.config.gpu.gpu_profile} x TP{self.config.gpu.tp_size}",
            effective_ttft_ms=self.stats.effective_ttft_ms,
            effective_tokens_per_s=self.stats.effective_tokens_per_s,
            effective_decode_tokens_per_s=self.stats.effective_decode_tokens_per_s,
            actual_tokens_per_s=self.stats.actual_tokens_per_s,
            emulated_prefill_ms=self.stats.emulated_prefill_ms,
            emulated_decode_ms=self.stats.emulated_decode_ms,
        )

    async def start(self) -> None:
        """Start the server."""
        self._running = True
        logger.info(f"Combined server {self.instance_id} started")

    async def stop(self) -> None:
        """Stop the server (the KV stack's lifecycle is owned by the app)."""
        if self.engine is not None:
            logger.info("Batching engine stats: %s", self.engine.info())
            await self.engine.close()
        self._running = False
        st = self.stats
        logger.info(
            "Combined server %s stopped (hot set: %s): %d requests, %d completion tokens; "
            "effective TTFT %.1f ms, effective %.1f tok/s (decode-only %.1f tok/s), "
            "actual %.1f tok/s [%s, %s on %s TP%d]",
            self.instance_id, self.hot.info(), st.requests_success, st.completion_tokens,
            st.effective_ttft_ms, st.effective_tokens_per_s, st.effective_decode_tokens_per_s,
            st.actual_tokens_per_s,
            "full-speed" if self.config.timing.full_speed else "emulated",
            self.model_name, self.config.gpu.gpu_profile, self.config.gpu.tp_size,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CombinedServer("
            f"instance_id={self.instance_id!r}, "
            f"model={self.model_name!r}, "
            f"tokens={self.stats.total_tokens})"
        )
