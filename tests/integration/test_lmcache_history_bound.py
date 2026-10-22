"""Bound upstream diagnostic metadata without changing native LRU behavior."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

pytest.importorskip("lmcache")

from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy  # noqa: E402

from kvbench.core.config import KVBenchConfig  # noqa: E402
from kvbench.kv.factory import create_kv_stack  # noqa: E402
from kvbench.kv.lmcache_stack import LMCacheStack  # noqa: E402
from tests.integration.test_v1_storage import make_config  # noqa: E402

pytestmark = pytest.mark.integration


def test_reuse_history_limit_does_not_change_eviction_order():
    reference, bounded = LRUCachePolicy(), LRUCachePolicy()
    stack = LMCacheStack("llama-3.1-8b", reuse_history_entries=8)
    engine = SimpleNamespace(storage_manager=SimpleNamespace(storage_backends={
        "disk": SimpleNamespace(cache_policy=bounded),
    }))
    stack._bound_reuse_history(engine)
    caches = [OrderedDict(), OrderedDict()]
    for key in range(10000):
        for policy, cache in zip((reference, bounded), caches, strict=True):
            policy.update_on_put(key)
            cache[key] = SimpleNamespace(can_evict=True)
            if len(cache) > 16:
                victim = policy.get_evict_candidates(cache, 1)[0]
                del cache[victim]
                policy.update_on_force_evict(victim)
            policy.update_on_hit(key - 3 if key >= 3 else key, cache)
        assert list(caches[0]) == list(caches[1])
        assert len(bounded.chunk_hash_to_init_timestamp) <= 8
    assert len(reference.chunk_hash_to_init_timestamp) == 10000


async def test_real_manager_churn_keeps_all_policy_histories_bounded(tmp_path):
    cfg = make_config(tmp_path, extra="max_local_disk_size: 0.000244140625\n")  # 8 small chunks
    data = cfg.model_dump(mode="json")
    data["kv"]["lmcache_reuse_history_entries"] = 8
    stack = create_kv_stack(KVBenchConfig.model_validate(data))
    await stack.start()
    try:
        for index in range(256):
            await stack.store([index, *range(1, 16)])
            await stack.flush()
        assert stack.stats.submitted_store_bytes == 256 * 16 * 2048
        usage = stack.usage_info()
        histories = [item for item in usage.values() if "reuse_history_entries" in item]
        assert len(histories) == 2
        assert all(item["reuse_history_entries"] <= item["reuse_history_limit"] == 8 for item in histories)
        assert len(list((tmp_path / "disk").glob("*.pt"))) <= 8
        assert stack.capacity_info()["lmcache_reuse_history_entries"] == 8
    finally:
        await stack.close()
