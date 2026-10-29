"""Independent published/config-derived KV byte oracles (no network/weights).

Sources and pinned revisions are in docs/architecture/model-catalog.md and the
catalog entries. These values deliberately distinguish K/V heads from MLA
latents, index owners from query indexers, and finite state from token growth.
"""

from dataclasses import replace

import pytest

from kvbench.core.cache_layout import CacheComponent
from kvbench.core.models import get_model_profile


@pytest.mark.parametrize("name,owners,bpt,alignment", [
    ("deepseek-v4-pro-0813", 61, 4474, 128),
    ("deepseek-v4-flash-0731", 41, 3125, 128),
    ("deepseek-v4.1-flash", 4, 890, 2),
    ("qwen3.8-flash-next", 12, 25344, 4),
    ("qwen3.8-27b", 16, 65536, 1),
    ("qwen3.8-2.4t-a95b", 23, 94208, 1),
    ("glm-5.3", 78, 92628, 1),
    ("glm-5.3-flash", 11, 11627, 4),
    ("kimi-k3", 24, 27648, 1),
    ("kimi-linear-48b-a3b", 7, 8064, 1),
])
def test_published_layout_byte_oracles(name, owners, bpt, alignment):
    profile = get_model_profile(name)
    assert profile.effective_kv_layers == owners
    assert profile.total_kv_cache_bytes_per_token == bpt
    assert profile.cache_token_alignment == alignment
    assert profile.kv_cache_size_bytes(256) == 256 * bpt
    assert profile.rank_layout()[0]["bytes_per_token"] == bpt
    assert profile.source_repo and len(profile.source_revision) == 40
    assert any(profile.source_revision in url for url in profile.source_urls)
    assert profile.verified_on == "2026-09-29"


def test_deepseek_41_counts_kv_owners_and_quant_scales_not_query_indexers():
    p = get_model_profile("deepseek-v4.1-flash")
    # 512 FP4 values + one byte scale / 16; 128 FP4 index values + scale / 32.
    main = 512 // 2 + 512 // 16
    index = 128 // 2 + 128 // 32
    assert main == 288 and index == 68
    for tokens in (0, 1, 2, 3, 127, 128, 1001, 1048576):
        assert p.kv_cache_size_bytes(tokens) == (3 * (tokens // 2) + tokens) * (main + index)
    growing = [c for c in p.cache_components if c.export]
    assert {i for c in growing for i in c.layer_ids} == {2, 8, 14, 20}
    # The sliding-window/cache compressor state is not multiplied by 1M tokens.
    short, long = p.cache_footprint(128), p.cache_footprint(1048576)
    assert short["window_bytes"] == long["window_bytes"] == 40 * 128 * (512 + 16)
    assert long["fixed_bytes"] == 3 * 2 * 2 * 512 * 4
    with pytest.raises(ValueError, match="PP=1"):
        p.rank_layout(pp_size=2)


def test_deepseek_v4_csa_hca_ratios_have_discrete_growth():
    pro = get_model_profile("deepseek-v4-pro-0813")
    flash = get_model_profile("deepseek-v4-flash-0731")
    for n in (1, 4, 127, 128, 256, 1024):
        assert pro.kv_cache_size_bytes(n) == 30 * (n // 4) * (512 + 68) + 31 * (n // 128) * 512
        assert flash.kv_cache_size_bytes(n) == 21 * (n // 4) * (512 + 68) + 20 * (n // 128) * 512


def test_qsa_compresses_index_keys_not_main_attention_kv():
    p = get_model_profile("qwen3.8-flash-next")
    assert p.total_kv_cache_bytes_per_token == 12 * (2 * 2 * 256 * 2 + 128 * 2 // 4)
    # Top-k budget 2048 must not cap the stored context at 2048.
    assert p.kv_cache_size_bytes(8192) == 4 * p.kv_cache_size_bytes(2048)
    # Only GQA heads shard/replicate; each rank retains the shared index key.
    assert p.rank_layout(tp_size=2)[0]["bytes_per_token"] == 12288 + 768


def test_glm_indexshare_does_not_share_main_mla_cache():
    p = get_model_profile("glm-5.3")
    assert p.total_kv_cache_bytes_per_token == 78 * (512 + 64) * 2 + 21 * (128 + 4)
    assert all(r["bytes_per_token"] == 92628 for r in p.rank_layout(tp_size=4))
    flash = get_model_profile("glm-5.3-flash")
    assert flash.total_kv_cache_bytes_per_token == 11 * (512 * 2 + (128 + 4) // 4)
    assert flash.total_kv_cache_bytes_per_token % 2 == 1  # must not pad or truncate odd byte strides


def test_kimi_nope_retains_shared_64_dim_projection_and_layer_indexing():
    p = get_model_profile("kimi-k3")
    assert p.total_kv_cache_bytes_per_token == 24 * (512 + 64) * 2
    mla = next(c for c in p.cache_components if c.name == "mla_kv")
    assert mla.layer_ids[-2:] == (91, 92)
    # V1 balances contiguous physical layers (remainder to earlier stages),
    # rather than evenly redistributing only the 24 cache-bearing layers.
    assert [r["layers"] for r in p.rank_layout(pp_size=4)] == [6, 5, 6, 7]
    assert sum(r["bytes_per_token"] for r in p.rank_layout(pp_size=4)) == 27648


@pytest.mark.parametrize("name,matrices,heads,conv_bytes", [
    ("kimi-k3", 69, 96, 69 * 96 * 128 * 3 * 3 * 2),
    ("kimi-linear-48b-a3b", 20, 32, 20 * 32 * 128 * 3 * 3 * 2),
    ("qwen3.8-27b", 48, 48, 48 * (2 * 16 + 48) * 128 * 3 * 2),
    ("qwen3.8-2.4t-a95b", 69, 128, 69 * (2 * 16 + 128) * 128 * 3 * 2),
])
def test_recurrent_state_is_fixed_not_zero_or_per_token(name, matrices, heads, conv_bytes):
    p = get_model_profile(name)
    expected = matrices * heads * 128 * 128 * 4 + conv_bytes
    assert p.cache_footprint(16)["fixed_bytes"] == expected
    assert p.cache_footprint(131072)["fixed_bytes"] == expected
    assert p.cache_footprint(0)["accounted_resident_bytes"] == 0
    assert p.cache_footprint(16)["accounted_resident_bytes"] == p.kv_cache_size_bytes(16) + expected


def test_cache_precision_does_not_follow_fp4_or_fp8_weights():
    p = get_model_profile("kimi-k3")
    assert "MXFP4" in p.weight_format
    assert p.total_kv_cache_bytes_per_token == 27648
    assert replace(p, dtype="fp8").total_kv_cache_bytes_per_token == 27648
    with pytest.raises(ValueError, match="specify precision"):
        replace(p, kv_dtype="fp8")


@pytest.mark.parametrize("change", [
    {"bits": 4.0}, {"bits": 3}, {"heads": 0}, {"layer_ids": [0, 0]},
    {"scale_bytes": 4}, {"quant_group_size": 16}, {"scope": "fixed"},
    {"scope": "window", "export": False}, {"export": "false"},
])
def test_invalid_cache_component_fails(change):
    with pytest.raises(ValueError):
        CacheComponent(**{"name": "test", "layer_ids": [0], "elements_per_head": 128, **change})


def test_unsupported_fractional_stride_is_not_silently_rounded():
    c = CacheComponent("fractional", (0,), 128, tokens_per_entry=3)
    assert c.size_bytes(3) == 256
    with pytest.raises(ValueError, match="no rounding"):
        c.growth_bytes_per_token()


def test_comment_updates_do_not_change_cache_identity():
    p = get_model_profile("deepseek-v4.1-flash")
    components = tuple(replace(c, notes="updated prose") for c in p.cache_components)
    revised = replace(p, cache_notes="updated prose", cache_components=components,
                      verified_on="2026-09-30", source_urls=("https://example.com",))
    assert revised.cache_identity() == p.cache_identity()
    assert replace(p, source_revision="a" * 40).cache_identity() != p.cache_identity()


def test_muse_spark_exports_full_attention_layers_only():
    p = get_model_profile("muse-spark")
    # 16 NoPE full-attention owners at 1, 5, ..., 61; 16 KV heads x 64 dims x K,V in BF16.
    assert p.effective_kv_layers == 16
    assert p.total_kv_cache_bytes_per_token == 16 * 2 * 16 * 64 * 2 == 65536
    growing = next(c for c in p.cache_components if c.export)
    assert growing.layer_ids == tuple(range(1, 62, 4))
    # Reproduces the published ~64 GiB / 1M tokens and ~8.4 GiB / 128K (with the 46 x 2048 window).
    assert p.kv_cache_size_bytes(1048576) == 64 * 2**30
    foot = p.cache_footprint(131072)
    assert foot["window_bytes"] == 46 * 2048 * 2 * 16 * 64 * 2
    assert abs((p.kv_cache_size_bytes(131072) + foot["window_bytes"]) / 2**30 - 8.36) < 0.01
    assert p.cache_footprint(1048576)["window_bytes"] == foot["window_bytes"]
    # TP=8 shards the 16 KV heads: rank 0 carries 8 KiB/token.
    assert p.rank_layout(tp_size=8)[0]["bytes_per_token"] == 8192
    assert p.source_repo == "meta-models/Muse-Spark-open-816B-A42B"


def test_legacy_profiles_keep_the_old_identity_fields():
    p = get_model_profile("llama-3.1-8b")
    assert "cache_components" not in p.cache_identity()
    assert "source_repo" not in p.cache_identity()
    assert p.total_kv_cache_bytes_per_token == 131072
