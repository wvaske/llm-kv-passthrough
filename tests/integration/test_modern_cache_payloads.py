"""Actual LMCache file I/O for component/packed layouts; no model weights.

CPU pools are 8 MiB/rank. The reported hundreds of MiB of recurrent inference
state are accounting only and MUST NOT be allocated or added to KV offload.
"""

import asyncio

import pytest

pytest.importorskip("lmcache")

from kvbench.core.config import KVBenchConfig  # noqa: E402
from kvbench.kv.factory import create_kv_stack  # noqa: E402
from kvbench.servers.openai_compat import ChatCompletionRequest, ErrorResponse  # noqa: E402
from kvbench.servers.rate_server import RateServer  # noqa: E402
from kvbench.trace.analyze import load_events  # noqa: E402
from tests.integration.test_v1_storage import make_config, settle  # noqa: E402

pytestmark = pytest.mark.integration


def modern_config(tmp_path, name, tp=1, pp=1, chunk=16):
    data = make_config(tmp_path, tp=tp, pp=pp, chunk=chunk).model_dump(mode="json")
    data["profiles_file"] = None
    data["server"]["model_profile"] = name
    data["kv"]["max_chunk_mb"] = 4
    return KVBenchConfig.model_validate(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("name,tp,pp,expected_bpt", [
    ("deepseek-v4-pro-0813", 1, 1, [4474]),
    ("deepseek-v4-flash-0731", 1, 1, [3125]),
    ("deepseek-v4.1-flash", 1, 1, [890]),
    ("deepseek-v4.1-flash", 2, 1, [890, 890]),
    ("qwen3.8-flash-next", 1, 1, [25344]),
    ("qwen3.8-flash-next", 4, 1, [13056] * 4),
    ("qwen3.8-27b", 1, 1, [65536]),
    ("qwen3.8-2.4t-a95b", 1, 1, [94208]),
    ("glm-5.3", 1, 1, [92628]),
    ("glm-5.3-flash", 1, 1, [11627]),
    ("glm-5.3-flash", 2, 2, [5285, 5285, 6342, 6342]),
    ("kimi-k3", 1, 1, [27648]),
    # Four live ranks keep this under the dev host's 4 GiB process guard.
    # TP replication and simultaneous TP+PP are covered by the GLM case above.
    ("kimi-k3", 1, 4, [6912, 5760, 6912, 8064]),
    ("kimi-linear-48b-a3b", 1, 1, [8064]),
])
async def test_real_packed_payload_bytes_roundtrip(tmp_path, name, tp, pp, expected_bpt):
    chunk = 128 if name.startswith("deepseek-v4-") else 16
    cfg = modern_config(tmp_path, name, tp, pp, chunk=chunk)
    stack = create_kv_stack(cfg)
    await stack.start()
    loops = [engine.storage_manager.loop for engine in stack._engines]
    tokens = list(range(2 * chunk))
    try:
        await stack.store(tokens)
        await settle(stack)
        events = load_events(cfg.kv.trace_file)
        writes = [e for e in events if e["event"] == "io" and e["op"] == "write"]
        assert len(writes) == 2 * tp * pp
        for rank, bpt in enumerate(expected_bpt):
            assert [e["size"] for e in writes if e["rank"] == rank] == [chunk * bpt] * 2
        assert sum(f.stat().st_size for f in (tmp_path / "disk").glob("*.pt")) == 2 * chunk * sum(expected_bpt)
        assert stack.stats.submitted_store_bytes == 2 * chunk * sum(expected_bpt)
        assert stack.capacity_info()["cache_transport_dtype"] == "packed"
        # Force the real file read path, not just a return from the CPU cache.
        for engine in stack._engines:
            await asyncio.to_thread(engine.clear, locations=["LocalCPUBackend"])
        assert await stack.lookup_pinned(tokens, "packed") == 2 * chunk
        assert await stack.retrieve(tokens) == 2 * chunk
        await stack.unpin("packed")
        await settle(stack)
        reads = [e for e in load_events(cfg.kv.trace_file) if e["event"] == "io" and e["op"] == "read"]
        assert sum(e["size"] for e in reads) == 2 * chunk * sum(expected_bpt)
        assert stack.stats.retrieved_bytes == 2 * chunk * sum(expected_bpt)
    finally:
        await stack.close()
    assert all(loop.is_closed() for loop in loops)


@pytest.mark.asyncio
async def test_compression_alignment_rejected_before_allocation(tmp_path):
    cfg = modern_config(tmp_path, "glm-5.3-flash", chunk=6)
    stack = create_kv_stack(cfg)
    with pytest.raises(ValueError, match="multiple of 4"):
        await stack.start()
    assert not stack._engines


@pytest.mark.asyncio
async def test_deepseek_output_growth_has_published_byte_slope(tmp_path):
    cfg = modern_config(tmp_path, "deepseek-v4.1-flash")
    stack = create_kv_stack(cfg)
    await stack.start()
    server = RateServer(cfg, stack)
    try:
        for text, output in (("a" * 128, 1), ("b" * 128, 65)):
            before = stack.stats.submitted_store_bytes
            response = await server.chat_completions(ChatCompletionRequest(
                model="deepseek-v4.1-flash", messages=[{"role": "user", "content": text}],
                max_completion_tokens=output,
            ))
            assert not isinstance(response, ErrorResponse)
            expected = (response.usage.prompt_tokens + output - 1) // 16 * 16 * 890
            assert stack.stats.submitted_store_bytes - before == expected
        await settle(stack)
        writes = [e for e in load_events(cfg.kv.trace_file) if e["event"] == "io" and e["op"] == "write"]
        assert sum(e["size"] for e in writes) == stack.stats.submitted_store_bytes
    finally:
        await server.stop()
        await stack.close()
