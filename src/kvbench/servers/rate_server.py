"""V1 serving: workload-sensitive rate scheduling over a real KV manager."""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from kvbench.core.config import KVBenchConfig
from kvbench.core.synthetic import SyntheticTokenizer
from kvbench.kv.base import KVStack
from kvbench.servers.openai_compat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    ModelList,
    PromptTokenDetails,
)
from kvbench.servers.rate_scheduler import OverloadedError, RateScheduler

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class ServingStats:
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_cancelled: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    ttft_ms: float = 0.0
    requests_with_ttft: int = 0
    queue_seconds: float = 0.0
    kv_seconds: float = 0.0
    started: float = field(default_factory=time.monotonic)


@dataclass
class RequestResult:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class RateServer:
    def __init__(self, config: KVBenchConfig, kv: KVStack):
        config.validate_serving()
        self.config = config
        self.kv = kv
        self.chunk_size = kv.chunk_size
        self.stats = ServingStats()
        self.scheduler = RateScheduler(config.scheduler, config.timing.full_speed)
        self.tokenizer = SyntheticTokenizer(config.timing.chars_per_token, config.model.vocab_size)
        self._requests: set[asyncio.Task[Any]] = set()
        # Classic LMCache has one bounded CPU staging pool per rank. Multiple
        # simultaneous pinned prefixes can pin the whole pool while a retrieve
        # waits for eviction. Serialize the complete load lease, not just each
        # individual manager call, to avoid that circular wait.
        self._prefix_load = asyncio.Lock()
        self._running = False
        self._closing = False

    @property
    def model_name(self) -> str:
        return self.config.server.model_profile

    @property
    def instance_id(self) -> str:
        return self.config.instance_id

    def validate_request(self, request: ChatCompletionRequest) -> None:
        if request.model != self.model_name:
            raise ValueError(f"Unknown model {request.model!r}; this engine serves {self.model_name!r}")
        if request.effective_max_tokens > self.config.scheduler.max_output_tokens:
            raise ValueError("Output limit exceeds scheduler.max_output_tokens")
        if request.n not in (None, 1) or request.stop:
            raise ValueError("V1 generates one fixed-length completion; n > 1 and stop are unsupported")
        if request.response_format and request.response_format.type != "text":
            raise ValueError("V1 supports synthetic text output only")
        for message in request.messages:
            if message.tool_calls or message.function_call:
                raise ValueError("V1 does not emulate tool calls; provide their workload as text")
            if isinstance(message.content, list) and any(
                not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str)
                for p in message.content
            ):
                raise ValueError("Only text content parts are supported")

    async def _kv_call(self, operation: Callable[P, Awaitable[T]], *args: P.args, **kwargs: P.kwargs) -> T:
        start = time.monotonic()
        try:
            return await operation(*args, **kwargs)
        finally:
            self.stats.kv_seconds += time.monotonic() - start

    async def _load_prefix(self, tokens: list[int]) -> int:
        async with self._prefix_load:
            lookup_id = uuid.uuid4().hex
            try:
                hit = await self._kv_call(self.kv.lookup_pinned, tokens, lookup_id)
                if hit < 0 or hit > len(tokens) or hit % self.chunk_size:
                    raise RuntimeError("KV manager returned an invalid cached-prefix length")
                if hit:
                    retrieved = await self._kv_call(self.kv.retrieve, tokens[:hit])
                    if retrieved != hit:
                        raise RuntimeError(f"KV retrieval incomplete: looked up {hit}, retrieved {retrieved}")
                return hit
            finally:
                cleanup = asyncio.create_task(self.kv.unpin(lookup_id))
                cancelled = False
                while True:
                    try:
                        await asyncio.shield(cleanup)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
                        if cleanup.done():
                            cleanup.result()
                            raise
                if cancelled:
                    raise asyncio.CancelledError

    async def _generate(self, request: ChatCompletionRequest, result: RequestResult) -> AsyncGenerator[str, None]:
        self.validate_request(request)
        if self._closing:
            raise RuntimeError("Server is shutting down")
        task = asyncio.current_task()
        assert task is not None
        self._requests.add(task)
        self.stats.requests_total += 1
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.config.scheduler.request_timeout_s):
                async with self.scheduler.admit() as queue_time:
                    self.stats.queue_seconds += queue_time
                    # Bound the body before building token metadata. HTTP also has
                    # a pre-parse body limit; this protects direct Python callers.
                    prompt = self.tokenizer.prompt(request.messages)
                    if len(prompt.encode("utf-8")) > self.config.scheduler.max_request_bytes:
                        raise ValueError("Prompt exceeds scheduler.max_request_bytes")
                    tokens = self.tokenizer.encode(prompt)
                    num_prompt = len(tokens)
                    limit = min(self.config.model.max_seq_len, self.config.scheduler.max_context_tokens)
                    if num_prompt + request.effective_max_tokens > limit:
                        raise ValueError(f"Prompt plus output exceeds context limit ({limit} tokens)")
                    result.prompt_tokens = num_prompt
                    self.stats.prompt_tokens += num_prompt
                    hit = await self._load_prefix(tokens)
                    self.stats.cache_hits += hit // self.chunk_size
                    result.cached_tokens = hit
                    self.stats.cache_misses += (num_prompt - hit + self.chunk_size - 1) // self.chunk_size
                    stored = hit

                    async def store_completed(computed: int) -> None:
                        nonlocal stored
                        end = computed // self.chunk_size * self.chunk_size
                        if end > stored:
                            await self._kv_call(self.kv.store, tokens[:end], skip_leading=stored)
                            stored = end

                    computed = hit
                    while computed < num_prompt:
                        count = min(self.config.scheduler.prefill_chunk_tokens, num_prompt - computed)
                        await self.scheduler.run("prefill", count)
                        computed += count
                        await store_completed(computed)
                    if hit == num_prompt:
                        # Even a fully cached prompt needs a model step to sample.
                        await self.scheduler.run("decode", 1)
                    for index in range(request.effective_max_tokens):
                        if index:
                            # The previously sampled token now gains KV. The last
                            # output token remains uncomputed until a later turn.
                            await self.scheduler.run("decode", 1)
                            await store_completed(len(tokens))
                        text, token = self.tokenizer.sample(len(tokens))
                        tokens.append(token)
                        result.completion_tokens += 1
                        self.stats.completion_tokens += 1
                        if index == 0:
                            self.stats.ttft_ms += (time.monotonic() - started) * 1000
                            self.stats.requests_with_ttft += 1
                        yield text
            self.stats.requests_success += 1
        except (asyncio.CancelledError, GeneratorExit):
            self.stats.requests_cancelled += 1
            raise
        except Exception:
            self.stats.requests_failed += 1
            raise
        finally:
            self.stats.total_latency_ms += (time.monotonic() - started) * 1000
            self._requests.discard(task)

    @staticmethod
    def _error(exc: Exception) -> ErrorResponse:
        if isinstance(exc, ValueError):
            return ErrorResponse.create(str(exc), "invalid_request_error", "400")
        if isinstance(exc, OverloadedError):
            return ErrorResponse.create(str(exc), "service_unavailable", "503")
        if isinstance(exc, TimeoutError):
            return ErrorResponse.create("Request timed out", "timeout_error", "504")
        logger.exception("KV serving request failed")
        return ErrorResponse.create(str(exc) or type(exc).__name__, "server_error", "500")

    async def chat_completions(self, request: ChatCompletionRequest) -> ChatCompletionResponse | ErrorResponse:
        result = RequestResult()
        try:
            output = [text async for text in self._generate(request, result)]
            return ChatCompletionResponse.create(self.model_name, "".join(output),
                                                 result.prompt_tokens, result.completion_tokens, "length",
                                                 cached_tokens=result.cached_tokens)
        except Exception as exc:
            return self._error(exc)

    async def chat_completions_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        result = RequestResult()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        generated = self._generate(request, result)
        first = True
        try:
            async for text in generated:
                if first:
                    yield ChatCompletionChunk.create_start(self.model_name, completion_id).to_sse()
                    first = False
                yield ChatCompletionChunk.create_content(self.model_name, text, completion_id).to_sse()
            yield ChatCompletionChunk.create_end(self.model_name, completion_id, "length").to_sse()
            if request.stream_options and request.stream_options.include_usage:
                from kvbench.servers.openai_compat import Usage

                yield ChatCompletionChunk(
                    id=completion_id, model=self.model_name, choices=[],
                    usage=Usage(prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                                total_tokens=result.prompt_tokens + result.completion_tokens,
                                prompt_tokens_details=PromptTokenDetails(cached_tokens=result.cached_tokens)),
                ).to_sse()
        except Exception as exc:
            yield f"data: {self._error(exc).model_dump_json()}\n\n"
        finally:
            await generated.aclose()
        yield "data: [DONE]\n\n"

    async def list_models(self) -> ModelList:
        return ModelList(data=[ModelInfo(id=self.model_name)])

    async def health_check(self) -> HealthResponse:
        return HealthResponse(status="healthy" if self._running and not self._closing else "unhealthy",
                              instance_id=self.instance_id, server_type="combined", model=self.model_name,
                              uptime_seconds=time.monotonic() - self.stats.started)

    async def get_metrics(self) -> MetricsResponse:
        s = self.stats
        completed = s.requests_success + s.requests_failed + s.requests_cancelled
        return MetricsResponse(
            requests_total=s.requests_total, requests_success=s.requests_success, requests_failed=s.requests_failed,
            requests_cancelled=s.requests_cancelled, tokens_processed=s.prompt_tokens + s.completion_tokens,
            cache_hits=s.cache_hits, cache_misses=s.cache_misses,
            avg_latency_ms=s.total_latency_ms / completed if completed else 0,
            timing_mode="rate-v1-full-speed" if self.config.timing.full_speed else "rate-v1",
            emulated_model=self.model_name, emulated_gpu=self.config.gpu.gpu_profile,
            actual_tokens_per_s=s.completion_tokens / max(time.monotonic() - s.started, 1e-9),
            observed_ttft_ms=s.ttft_ms / s.requests_with_ttft if s.requests_with_ttft else 0,
            scheduler=self.scheduler.info(), queue_seconds=s.queue_seconds, kv_seconds=s.kv_seconds,
        )

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._closing = True
        requests = [t for t in self._requests if t is not asyncio.current_task()]
        for task in requests:
            task.cancel()
        await asyncio.gather(*requests, return_exceptions=True)
        await self.scheduler.close()
        self._running = False
