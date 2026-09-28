"""CombinedServer with the continuous-batching engine (--gpu-batching)."""

from __future__ import annotations

import asyncio
import time

import pytest

from kvbench.core.config import KVBenchConfig, TimingConfig
from kvbench.servers.combined import CombinedServer
from kvbench.servers.openai_compat import ChatCompletionRequest, ChatMessage, MessageRole
from tests.fakes import FakeKVStack


def _config() -> KVBenchConfig:
    cfg = KVBenchConfig(instance_id="test-batching")
    return cfg.model_copy(update={"timing": TimingConfig(batching=True, frontend_us_per_token=0.0)})


def _req(text: str, n: int, stream: bool = True) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="llama-3.1-8b", messages=[ChatMessage(role=MessageRole.USER, content=text)], max_tokens=n, stream=stream
    )


async def _consume(server: CombinedServer, req: ChatCompletionRequest) -> tuple[int, float]:
    t0 = time.monotonic()
    n = 0
    async for chunk in server.chat_completions_stream(req):
        if '"content"' in chunk and '"content":""' not in chunk.replace(" ", ""):
            n += 1
    return n, time.monotonic() - t0


@pytest.fixture
async def server():
    s = CombinedServer(config=_config(), kv=FakeKVStack(chunk_size=16))
    assert s.engine is not None
    yield s
    await s.stop()


class TestBatchedServing:
    @pytest.mark.asyncio
    async def test_stream_emits_requested_tokens_and_stores(self, server: CombinedServer) -> None:
        n, _ = await _consume(server, _req("hello world " * 200, 12))
        assert n >= 12
        assert server.stats.requests_success == 1
        assert server.kv.stats.stores == 1
        # a second turn sharing the prefix hits the cache and retrieves it
        await _consume(server, _req("hello world " * 200 + " and more", 4))
        assert server.kv.stats.retrieves == 1
        assert server.stats.cache_hits > 0

    @pytest.mark.asyncio
    async def test_non_streaming_path(self, server: CombinedServer) -> None:
        resp = await server.chat_completions(_req("abc " * 50, 6, stream=False))
        assert resp.usage.completion_tokens == 6

    @pytest.mark.asyncio
    async def test_concurrent_requests_share_the_engine(self, server: CombinedServer) -> None:
        results = await asyncio.gather(*(_consume(server, _req(f"prompt {i} " * 300, 20)) for i in range(3)))
        assert all(n >= 20 for n, _ in results)
        assert server.engine.max_batch >= 2
        assert server.engine.in_flight == 0

    @pytest.mark.asyncio
    async def test_client_disconnect_frees_the_slot(self, server: CombinedServer) -> None:
        gen = server.chat_completions_stream(_req("x " * 100, 10_000))
        got = 0
        async for _chunk in gen:
            got += 1
            if got > 5:
                break
        await gen.aclose()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if server.engine.in_flight == 0:
                break
        assert server.engine.in_flight == 0
