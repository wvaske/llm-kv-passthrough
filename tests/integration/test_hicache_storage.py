"""CPU-only tests driving the pinned, unmodified upstream file I/O methods."""

import asyncio
import hashlib
import struct
from importlib.metadata import PackageNotFoundError, version

import pytest

from kvbench.core.config import KVBenchConfig
from kvbench.kv.factory import create_kv_stack
from kvbench.kv.hicache_stack import page_hashes
from kvbench.kv.hicache_upstream import load_hicache
from kvbench.servers.app import KVBenchApp
from kvbench.trace.analyze import load_events

try:
    available = version("sglang") == "0.5.21"
except PackageNotFoundError:
    available = False

pytestmark = [pytest.mark.integration, pytest.mark.skipif(not available, reason="pinned SGLang source wheel required")]


def config(tmp_path, *, policy="write_through", ranks=(7,), layout="page_first_direct", disk_pages=8):
    # Llama 8B TP=8: 16 KiB/token/rank, 32 KiB/page. Just 64 KiB host/rank.
    return KVBenchConfig.model_validate({
        "gpu": {"tp_size": 8}, "timing": {"full_speed": True},
        "kv": {"stack": "hicache", "active_ranks": list(ranks), "max_cpu_memory_gb": 1 / 1024,
               "trace_file": str(tmp_path / "trace.jsonl"),
               "hicache": {"storage_path": str(tmp_path / "disk"), "page_size": 2,
                           "host_size_gb": 65536 / 1024**3, "disk_size_gb": disk_pages * 32768 / 1024**3,
                           "write_policy": policy, "mem_layout": layout, "eviction_ratio": 1.0,
                           "min_free_bytes": 0, "write_batch_pages": 2}},
    })


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct", "layer_first"])
@pytest.mark.parametrize("ranks", [(7,), (0, 7)])
async def test_real_file_write_read_dedup_and_rank_identity(tmp_path, layout, ranks):
    stack = create_kv_stack(config(tmp_path, ranks=ranks, layout=layout))
    await stack.start()
    tokens = list(range(8))
    try:
        assert type(stack._backends[0]) is load_hicache().HiCacheFile
        await stack.store(tokens)
        await stack.flush()  # release host entries to force actual file reads
        assert await stack.lookup_pinned(tokens, "lease") == 8
        assert await stack.retrieve(tokens) == 8
        await stack.unpin("lease")
        files = list((tmp_path / "disk").glob("*.bin"))
        assert len(files) == len(ranks) * 4
        assert all(f.stat().st_size == 32768 for f in files)
        assert {f.stem.rsplit("_", 2)[1] for f in files} == {str(r) for r in ranks}
        stack.trace_recorder.flush()
        events = load_events(tmp_path / "trace.jsonl")
        assert sum(e["size"] for e in events if e["op"] == "write") == len(ranks) * 131072
        assert sum(e["size"] for e in events if e["op"] == "read") == len(ranks) * 131072
        await stack.store(tokens)
        stack.trace_recorder.flush()
        assert len([e for e in load_events(tmp_path / "trace.jsonl") if e["op"] == "write"]) == len(files)
        assert stack.capacity_info()["modeled_rank_count"] == 8
        assert stack.stats.retrieved_bytes == len(ranks) * 131072
    finally:
        await stack.unpin("lease")
        await stack.close()
    assert stack.usage_info() == {}


async def test_write_back_defers_then_evicts_in_page_batch(tmp_path):
    stack = create_kv_stack(config(tmp_path, policy="write_back"))
    await stack.start()
    try:
        await stack.store([1, 2, 3, 4])
        assert not list((tmp_path / "disk").glob("*.bin"))
        assert await stack.lookup([1, 2, 3, 4]) == 4
        await stack.store([1, 2, 3, 4, 5, 6], skip_leading=4)
        assert len(list((tmp_path / "disk").glob("*.bin"))) == 2
        assert await stack.retrieve([1, 2, 3, 4, 5, 6]) == 6
        await stack.flush()
        assert len(list((tmp_path / "disk").glob("*.bin"))) == 3
    finally:
        await stack.close()


async def test_native_lru_enforces_disk_quota_and_restart_discovers_files(tmp_path):
    cfg = config(tmp_path, disk_pages=3)
    stack = create_kv_stack(cfg)
    await stack.start()
    await stack.store(list(range(16)))
    await stack.close()
    assert len([e for e in load_events(tmp_path / "trace.jsonl") if e["op"] == "remove"]) == 5
    assert len(list((tmp_path / "disk").glob("*.bin"))) == 3
    restarted = create_kv_stack(cfg)
    await restarted.start()
    try:
        assert restarted.usage_info()["HiCacheFile/rank-7"]["usage_bytes"] == 3 * 32768
        assert await restarted.lookup(list(range(16))) == 0  # old prefix was evicted
        assert await restarted.retrieve(list(range(16)), skip_leading=10) == 6
    finally:
        await restarted.close()


async def test_prefix_lease_blocks_eviction_until_unpin(tmp_path):
    stack = create_kv_stack(config(tmp_path))
    await stack.start()
    try:
        await stack.store([1, 2, 3, 4])
        await stack.lookup_pinned([1, 2, 3, 4], "lease")
        writer = asyncio.create_task(stack.store([8, 9]))
        await asyncio.sleep(0.02)
        assert not writer.done()
        assert await stack.retrieve([1, 2, 3, 4]) == 4
        await stack.unpin("lease")
        await writer
    finally:
        await stack.unpin("lease")
        await stack.close()


async def test_http_manifest_and_serving(tmp_path):
    import httpx

    cfg = config(tmp_path, disk_pages=512)
    app = KVBenchApp(cfg)
    async with app.app.router.lifespan_context(app.app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app.app), base_url="http://test"
    ) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": cfg.server.model_profile, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5,
            })
            assert response.status_code == 200, response.text
            manifest = (await client.get("/kvbench/manifest")).json()
            assert manifest["hicache"]["sglang_version"] == "0.5.21"
            assert manifest["hicache"]["file_format"] == "raw page, temporary file + rename"
            state = (await client.get("/kvbench/state")).json()
            assert state["kv_stats"]["submitted_store_bytes"] > 0
            assert 'kvbench_kv_rank_count{scope="active"} 1.0' in (await client.get("/metrics")).text


def test_native_hash_cpp_regular_token_parity():
    a = hashlib.sha256(struct.pack("<2I", 42, 77)).digest()
    b = hashlib.sha256(a + struct.pack("<2I", 1, 4294967295)).hexdigest()
    assert page_hashes([42, 77, 1, 4294967295, 12], 2) == [a.hex(), b]
    with pytest.raises(struct.error):
        page_hashes([-1, 0], 2)


async def test_guards_and_write_failures_are_not_success(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    stack = create_kv_stack(cfg)
    await stack.start()
    monkeypatch.setattr(stack._backends[0], "set", lambda *_args: False)
    try:
        with pytest.raises(OSError, match="refused/failed"):
            await stack.store([1, 2])
        assert stack.stats.submitted_store_bytes == 0
        assert stack.stats.errors == 1
    finally:
        await stack.close()
    data = cfg.model_dump(mode="json")
    data["kv"]["max_cpu_memory_gb"] = 32768 / 1024**3
    guarded = create_kv_stack(KVBenchConfig.model_validate(data))
    with pytest.raises(ValueError, match="staging page"):
        await guarded.start()


def test_unsupported_hybrid_layout_is_explicit(tmp_path):
    data = config(tmp_path).model_dump(mode="json")
    data["server"]["model_profile"] = "kimi-k3"
    with pytest.raises(ValueError, match="MHA/GQA"):
        create_kv_stack(KVBenchConfig.model_validate(data))


async def test_pp_uses_native_tp_pp_suffix_and_global_trace_rank(tmp_path):
    data = config(tmp_path).model_dump(mode="json")
    data["gpu"]["tp_size"] = 2
    data["timing"]["pp_size"] = 4
    stack = create_kv_stack(KVBenchConfig.model_validate(data))
    await stack.start()
    try:
        await stack.store([1, 2])
        files = list((tmp_path / "disk").glob("*.bin"))
        assert len(files) == 1 and files[0].name.endswith("_1_2_4_3.bin")
        stack.trace_recorder.flush()
        assert {e["rank"] for e in load_events(tmp_path / "trace.jsonl") if e["event"] == "io"} == {7}
    finally:
        await stack.close()


@pytest.mark.parametrize("policy", ["write_through", "write_back"])
async def test_host_page_buffers_are_reused_within_the_pool_bound(tmp_path, monkeypatch, policy):
    import torch

    cfg = config(tmp_path, policy=policy, disk_pages=32)
    cfg.kv.payload_generator = "sfc64"
    stack = create_kv_stack(cfg)
    await stack.start()
    original = torch.empty
    allocations = []

    def allocate(*args, **kwargs):
        allocations.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", allocate)
    try:
        await stack.store(list(range(40)))  # twenty pages through a two-page pool
        assert len(allocations) == 2
        usage = stack.usage_info()["HiCacheHost/rank-7"]
        assert usage["allocated_bytes"] == 65536
        assert usage["keys"] + usage["free_pages"] == 2
        await stack.flush()
        payloads = [p.read_bytes() for p in (tmp_path / "disk").glob("*.bin")]
        assert len(payloads) == len(set(payloads)) == 20  # no overwritten/shared or repeated pages
    finally:
        await stack.close()
    assert not stack._free
