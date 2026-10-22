"""Selected rank execution is independent of the emulated TP/PP world."""

import os

import pytest

from kvbench.core.config import KVBenchConfig, KVStackConfig
from kvbench.core.ranks import parse_active_ranks
from kvbench.kv.factory import create_kv_stack
from kvbench.kv.lmcache_stack import LMCacheStack


def configured(selection=None, tp=8, pp=1, model="llama-3.1-8b"):
    return KVBenchConfig.model_validate({
        "gpu": {"tp_size": tp}, "timing": {"pp_size": pp},
        "server": {"model_profile": model}, "kv": {"active_ranks": selection},
    })


@pytest.mark.parametrize("value,expected", [
    (None, None), ("all", None), (" ALL ", None), ([7, 0], [0, 7]),
    ("7, 0", [0, 7]), ("[7, 0]", [0, 7]), ("0", [0]),
])
def test_selection_input_forms(value, expected):
    assert KVStackConfig(active_ranks=value).active_ranks == expected


@pytest.mark.parametrize("value", [
    [], "[]", "", " ", "0,", "0,,1", [0, 0], "0,0", [-1], "-1",
    [True], [False], [1.0], '[1.0]', ["1"], '["1"]', True, 1, {"rank": 0},
])
def test_bad_selection_fails_instead_of_loading_unexpected_ranks(value):
    with pytest.raises(ValueError):
        parse_active_ranks(value)


def test_rank_ids_are_checked_against_the_full_topology():
    with pytest.raises(ValueError, match="0..7"):
        configured([8])
    with pytest.raises(ValueError, match="0..7"):
        LMCacheStack("llama-3.1-8b", tp_size=8, active_ranks=[8])
    cfg = configured([15], pp=2)
    selection = cfg.manifest()["kv_rank_selection"]
    assert selection["modeled_rank_count"] == 16
    assert selection["active_rank_ids"] == [15]


def test_default_still_instantiates_all_ranks():
    cfg = configured()
    stack = create_kv_stack(cfg)
    assert stack.active_rank_ids == list(range(8))
    assert stack.active_layout == stack.layout
    assert cfg.manifest()["kv_rank_selection"]["is_subset"] is False


def test_subset_keeps_full_sharding_and_scheduler_rates():
    full, partial = configured(), configured([7])
    full_stack, partial_stack = create_kv_stack(full), create_kv_stack(partial)
    assert full.scheduler == partial.scheduler
    assert partial.gpu.tp_size == 8
    assert partial_stack.layout == full_stack.layout
    assert partial_stack.active_layout == [full_stack.layout[7]]
    assert partial_stack.active_layout[0]["bytes_per_token"] == 16384  # NOT TP=1's 131072
    selection = partial.manifest()["kv_rank_selection"]
    assert selection["active_rank_ids"] == [7]
    assert selection["active_stored_bytes_per_token"] == 16384
    assert selection["modeled_stored_bytes_per_token"] == 131072
    assert selection["cache_result_scope"] == "active_ranks"


def test_replicated_mla_is_not_divided_by_active_or_modeled_rank_count():
    cfg = configured([7], model="deepseek-v4.1-flash")
    selection = cfg.manifest()["kv_rank_selection"]
    assert selection["active_stored_bytes_per_token"] == 890
    assert selection["modeled_stored_bytes_per_token"] == 8 * 890


def test_nonuniform_pp_selection_uses_original_global_rank():
    cfg = configured([7], tp=2, pp=4, model="kimi-k3")
    stack = create_kv_stack(cfg)
    assert stack.active_layout[0]["pp_rank"] == 3
    assert stack.active_layout[0]["tp_rank"] == 1
    selection = cfg.manifest()["kv_rank_selection"]
    assert selection["active_stored_bytes_per_token"] == 8064
    assert selection["modeled_stored_bytes_per_token"] == 55296  # NOT 8 * 8064


def test_env_and_yaml_roundtrip(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("KVBENCH_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("KVBENCH_GPU__TP_SIZE", "8")
    monkeypatch.setenv("KVBENCH_KV__ACTIVE_RANKS", "7,0")
    cfg = KVBenchConfig.from_env()
    assert cfg.kv.active_ranks == [0, 7]
    path = tmp_path / "run.yaml"
    cfg.to_yaml(path)
    loaded = KVBenchConfig.from_yaml(path)
    assert loaded.manifest()["kv_rank_selection"] == cfg.manifest()["kv_rank_selection"]
    assert loaded.gpu.tp_size == 8
