"""One/two real LMCache engines representing an eight-rank system.

The oracle is each rank's actual full-topology geometry and original ID, not
the active engine count. Pools are only 8 MiB per active rank.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from lmcache.v1.cache_engine import LMCacheEngineBuilder  # noqa: E402

from kvbench.core.config import KVBenchConfig  # noqa: E402
from kvbench.kv.factory import create_kv_stack  # noqa: E402
from kvbench.metrics.prometheus import MetricsExporter  # noqa: E402
from kvbench.servers.openai_compat import ChatCompletionRequest, ErrorResponse  # noqa: E402
from kvbench.servers.rate_server import RateServer  # noqa: E402
from kvbench.servers.warmup import WarmupController, WarmupRequest  # noqa: E402
from kvbench.trace.analyze import load_events  # noqa: E402
from tests.integration.test_v1_storage import make_config, settle  # noqa: E402

pytestmark = pytest.mark.integration


def selected_config(tmp_path, active, *, tp=8, pp=1, model="llama-3.1-8b"):
    data = make_config(tmp_path).model_dump(mode="json")
    data["profiles_file"] = None
    data["server"]["model_profile"] = model
    data["gpu"]["tp_size"] = tp
    data["timing"]["pp_size"] = pp
    data["kv"]["active_ranks"] = active
    # Would reject all eight pools: this budget fits exactly the selected ones.
    data["kv"]["max_cpu_memory_gb"] = 0.0078125 * len(active)
    return KVBenchConfig.model_validate(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("active", [[0], [7], [0, 7], [2, 4, 7]])
async def test_tp8_only_selected_engines_do_real_io(tmp_path, active):
    cfg = selected_config(tmp_path, active)
    stack = create_kv_stack(cfg)
    await stack.start()
    chunk, count, bpt = 16, len(active), 16384
    tokens = list(range(chunk * 2))
    try:
        assert len(stack._engines) == count
        assert len(stack._connectors) == count
        assert stack._instance_ids == [f"{cfg.instance_id}-r{i}" for i in active]
        for rank in range(8):
            assert (LMCacheEngineBuilder.get(f"{cfg.instance_id}-r{rank}") is not None) == (rank in active)
        for rank, engine in zip(active, stack._engines, strict=True):
            assert engine.metadata.world_size == engine.metadata.local_world_size == 8
            assert engine.metadata.worker_id == engine.metadata.local_worker_id == rank
            assert "--tp8pp1" in engine.metadata.model_name

        await stack.store(tokens)
        await settle(stack)
        events = load_events(cfg.kv.trace_file)
        selection = next(e for e in events if e["op"] == "rank_selection")
        assert selection["active_rank_ids"] == active
        assert selection["modeled_rank_count"] == 8
        writes = [e for e in events if e["event"] == "io" and e["op"] == "write"]
        assert len(writes) == 2 * count
        assert {e["rank"] for e in writes} == set(active)
        assert all(e["size"] == chunk * bpt for e in writes)
        files = list((tmp_path / "disk").glob("*.pt"))
        assert len(files) == 2 * count
        assert {int(f.name.split("@")[2]) for f in files} == set(active)
        assert {f.name.split("@")[1] for f in files} == {"8"}
        assert sum(f.stat().st_size for f in files) == 32 * bpt * count
        assert stack.stats.stored_tokens == 32  # no request/token downsampling
        assert stack.stats.submitted_store_bytes == 32 * bpt * count

        cap = stack.capacity_info()
        assert cap["tp_size"] == cap["modeled_rank_count"] == 8
        assert cap["active_rank_count"] == cap["files_per_chunk"] == count
        assert cap["modeled_files_per_chunk"] == 8
        assert cap["local_cpu_capacity_bytes"] == count * 8 * 1024**2
        assert cap["local_disk_capacity_bytes"] == count * 32 * 1024**2
        assert cap["stored_bytes_per_token"] == count * bpt
        assert cap["modeled_stored_bytes_per_token"] == 8 * bpt
        assert [r["rank"] for r in cap["ranks"] if r["active"]] == active
        usage = stack.usage_info()
        assert {info["rank"] for info in usage.values()} == set(active)
        for rank in active:
            assert usage[f"LocalCPUBackend/rank-{rank}"]["usage_bytes"] == 32 * bpt

        # Force disk readback and prove there is no attempted load from absent ranks.
        for engine in stack._engines:
            await asyncio.to_thread(engine.clear, locations=["LocalCPUBackend"])
        assert await stack.lookup_pinned(tokens, "subset") == 32
        assert await stack.retrieve(tokens) == 32
        await stack.unpin("subset")
        await settle(stack)
        reads = [e for e in load_events(cfg.kv.trace_file) if e["event"] == "io" and e["op"] == "read"]
        assert len(reads) == 2 * count
        assert {e["rank"] for e in reads} == set(active)
        assert stack.stats.retrieved_bytes == 32 * bpt * count
        assert all(not engine.lookup_pins for engine in stack._engines)

        metrics = MetricsExporter(SimpleNamespace(_kv=stack)).render().decode()
        assert 'kvbench_kv_rank_count{scope="modeled"} 8.0' in metrics
        assert f'kvbench_kv_rank_count{{scope="active"}} {count}.0' in metrics
        for rank in range(8):
            assert f'kvbench_kv_rank_active{{rank="{rank}"}} {int(rank in active)}.0' in metrics
    finally:
        await stack.close()
    assert stack.usage_info() == {}
    assert all(LMCacheEngineBuilder.get(f"{cfg.instance_id}-r{rank}") is None for rank in range(8))


@pytest.mark.asyncio
async def test_subset_does_not_change_rank_namespace_or_payload(tmp_path):
    snapshots = []
    for index, active in enumerate(([7], [0, 7])):
        directory = tmp_path / str(index)
        directory.mkdir()
        cfg = selected_config(directory, active)
        stack = create_kv_stack(cfg)
        await stack.start()
        try:
            await stack.store(list(range(32)))
            await settle(stack)
            snapshots.append({f.name: f.read_bytes() for f in (directory / "disk").glob("*@7@*.pt")})
        finally:
            await stack.close()
    assert len(snapshots[0]) == 2
    assert snapshots[0] == snapshots[1]


@pytest.mark.asyncio
async def test_warmup_sizes_only_the_active_drive(tmp_path):
    cfg = selected_config(tmp_path, [7])
    stack = create_kv_stack(cfg)
    await stack.start()
    warmup = WarmupController(stack)
    try:
        status = warmup.start(WarmupRequest(fill_factor=1.25))
        assert status.target_bytes == 50 * 1024**2  # 1.25 * (8 MiB CPU + 32 MiB disk)
        assert status.params["bytes_per_token"] == 16384
        await warmup.cancel()  # Cancel before the filler executes; no large test writes.
        assert status.stored_bytes == 0
    finally:
        await warmup.cancel()
        await stack.close()


@pytest.mark.asyncio
async def test_all_rank_mode_still_checks_all_cpu_pools(tmp_path):
    cfg = selected_config(tmp_path, [7])
    data = cfg.model_dump(mode="json")
    data["kv"]["active_ranks"] = None
    stack = create_kv_stack(KVBenchConfig.model_validate(data))
    with pytest.raises(ValueError, match="8 active ranks"):
        await stack.start()
    assert not stack._engines


@pytest.mark.asyncio
async def test_nonuniform_pp_rank_keeps_global_identity_and_size(tmp_path):
    cfg = selected_config(tmp_path, [7], tp=2, pp=4, model="kimi-k3")
    stack = create_kv_stack(cfg)
    await stack.start()
    try:
        assert stack.active_layout[0]["pp_rank"] == 3
        assert stack.active_layout[0]["tp_rank"] == 1
        await stack.store(list(range(16)))
        await settle(stack)
        assert stack.stats.submitted_store_bytes == 16 * 8064
        assert stack.usage_info()["LocalCPUBackend/rank-7"]["usage_bytes"] == 16 * 8064
        assert stack.capacity_info()["modeled_chunk_bytes"] == 16 * 55296
    finally:
        await stack.close()


@pytest.mark.asyncio
async def test_partial_selected_startup_cleans_only_owned_rank_ids(tmp_path, monkeypatch):
    cfg = selected_config(tmp_path, [3, 7])
    created = []
    original = LMCacheEngineBuilder.get_or_create

    def fail_last(instance_id, *args, **kwargs):
        created.append(instance_id)
        if instance_id.endswith("-r7"):
            raise RuntimeError("selected rank failed")
        return original(instance_id, *args, **kwargs)

    monkeypatch.setattr(LMCacheEngineBuilder, "get_or_create", fail_last)
    stack = create_kv_stack(cfg)
    with pytest.raises(RuntimeError, match="selected rank failed"):
        await stack.start()
    assert created == [f"{cfg.instance_id}-r3", f"{cfg.instance_id}-r7"]
    assert all(LMCacheEngineBuilder.get(f"{cfg.instance_id}-r{i}") is None for i in range(8))
    assert not stack._engines


@pytest.mark.asyncio
async def test_serving_one_rank_receives_the_whole_request_workload(tmp_path):
    cfg = selected_config(tmp_path, [7])
    stack = create_kv_stack(cfg)
    await stack.start()
    server = RateServer(cfg, stack)
    try:
        for content in ("a" * 128, "b" * 128):
            result = await server.chat_completions(ChatCompletionRequest(
                model="llama-3.1-8b", messages=[{"role": "user", "content": content}], max_tokens=33,
            ))
            assert not isinstance(result, ErrorResponse)
            expected_tokens = (result.usage.prompt_tokens + 32) // 16 * 16
            await settle(stack)
        assert server.stats.requests_success == 2
        assert stack.stats.stored_tokens == 2 * expected_tokens
        assert stack.stats.submitted_store_bytes == 2 * expected_tokens * 16384
        assert server.scheduler.config == cfg.scheduler
        writes = [e for e in load_events(cfg.kv.trace_file) if e["event"] == "io" and e["op"] == "write"]
        assert {e["rank"] for e in writes} == {7}
        assert sum(e["size"] for e in writes) == stack.stats.submitted_store_bytes
    finally:
        await server.stop()
        await stack.close()
