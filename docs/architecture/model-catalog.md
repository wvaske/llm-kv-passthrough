# Researched model catalog — 2026-09-29

These entries were researched from current **official publisher repositories**,
checkpoint configurations, and serving implementations. Names and dimensions
were not inferred from parameter count or remembered releases. Updated entries
record immutable HF revisions, the verification date, format notes, and source
URLs in `src/kvbench/core/profiles/catalog.yaml`.

## Release selection

The official owner feeds were queried sorted by creation date:

- [DeepSeek](https://huggingface.co/api/models?author=deepseek-ai&sort=createdAt&direction=-1&limit=15):
  newest main release **DeepSeek-V4.1-Flash**; also retain the latest V4 Pro
  **0813** checkpoint and preceding V4 Flash **0731** for comparisons.
- [Qwen](https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1&limit=20):
  newest general text backbone **Qwen3.8-Flash-Next**, flagship open
  **Qwen3.8-2.4T-A95B**, and existing **Qwen3.8-27B** reverified. Newer image
  generation/drive checkpoints are not substitutes for these text backbones.
  API names “Qwen3.8-Max/Flash” are not additional weight checkpoints.
- [Z.ai](https://huggingface.co/api/models?author=zai-org&sort=createdAt&direction=-1&limit=15):
  **GLM-5.3** and **GLM-5.3-Flash**. Weight precision does not set KV precision.
- [Moonshot](https://huggingface.co/api/models?author=moonshotai&sort=createdAt&direction=-1&limit=15):
  **Kimi K3** is newest. The separate **Kimi Linear 48B-A3B** research line is
  included for its KDA/MLA layout; it is older than K3.

Creation timestamps are not publication dates. This is a dated snapshot, not
an automatically moving `latest` alias. Multimodal entries describe text backbones.

## Growing payload bytes

Figures are **bytes per source token, TP=1**, including the specified index-key
payload/scales. They describe explicit formats, not universal backend footprints.

| Profile ID | Selected cache format | Growing B/token | Chunk multiple |
|---|---|---:|---:|
| `deepseek-v4.1-flash` | Shared CSA2 FP4 main + FP4 index | **890** | 2 |
| `deepseek-v4-pro-0813` | Uniform FP8 main + FP4 index; 30 CSA / 31 HCA | **4,474** | 128 |
| `deepseek-v4-flash-0731` | Uniform FP8 main + FP4 index; 21 CSA / 20 HCA | **3,125** | 128 |
| `qwen3.8-flash-next` | 12 BF16 GQA layers + BF16 pooled QSA index | **25,344** | 4 |
| `qwen3.8-27b` | 16 BF16 GQA layers | **65,536** | 1 |
| `qwen3.8-2.4t-a95b` | 23 BF16 GQA layers | **94,208** | 1 |
| `glm-5.3` | 78 BF16 MLA layers + 21 FP8/FP32-scale index owners | **92,628** | 1 |
| `glm-5.3-flash` | 11 BF16 NoPE MLA layers + 4:1 FP8 index | **11,627** | 4 |
| `kimi-k3` | 24 BF16 absorbed gated MLA layers | **27,648** | 1 |
| `kimi-linear-48b-a3b` | 7 BF16 absorbed MLA layers | **8,064** | 1 |

### DeepSeek: sharing, temporal compression and quantization scales

[V4.1 config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/config.json),
[card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/README.md),
and [reference code](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py):

```text
main KV row = 512 × 4/8 + 512/16 × 1-byte scale = 288 bytes
index-K row = 128 × 4/8 + 128/32 × 1-byte scale = 68 bytes
global growth = (3 encoder owners / 2 + 1 decoder owner) × (288 + 68)
              = 890 bytes/source token
```

Only layers **2, 8, 14, 20** own buffers. Eight index-source layer IDs do not
mean eight distinct K caches: other layers reindex/reuse. At odd lengths the
exact exported size is `(3*floor(N/2) + N) * 356`, not `890*N`.

SWA is bounded to 128 tokens per layer. The published **SWA bounded replay**
policy avoids persisting SWA to SSD; V1 does not implement its replay compute.
The declared V4.1 window uses FP8 plus per-32 one-byte scales: 528 B/row.

V4 Pro/Flash have 4:1 CSA and 128:1 HCA, not V4.1 sharing. Their profiles select
SGLang's **TRT-LLM uniform 512-byte FP8 row** plus 68-byte FP4 index row.
The **FlashMLA 584-byte mixed FP8/BF16 row**, with additional page padding, is
a different format and is **not** represented by these profiles. See the
[pinned pool implementation](https://github.com/sgl-project/sglang/blob/bf80732b926eabadbd6f54f702ffa74916a9121b/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py).

```text
Pro:   30 × 512/4 + 31 × 512/128 + 30 × 68/4 = 4,474 B/token
Flash: 21 × 512/4 + 20 × 512/128 + 21 × 68/4 = 3,125 B/token
```

Their SWA checkpoint traffic and backend-dependent compression accumulators
are not in V1 offload. Growing-KV figures are not full resumable-state sizes.

### Qwen: QSA compresses index keys, not main KV

[Flash-Next config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)
has 36 GDN and 12 full/QSA layers. Main KV per full layer is
`2(K,V) * 2 heads * 256 * 2 bytes = 2048 B/token`.
The optimized [SGLang QSA pool](https://github.com/sgl-project/sglang/blob/bf80732b926eabadbd6f54f702ffa74916a9121b/python/sglang/srt/mem_cache/qsa_kv_pool.py)
stores one BF16 128-element index key per four source tokens: 64 B/token/layer.
Total is `12 * (2048 + 64) = 25,344 B/token`. Top-k=2048 reduces compute, not
retained main KV. HF's reference caches raw index keys and pools on demand;
it does not have the same memory layout as this optimized profile.

[27B](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
and [2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json)
have 16 and 23 GQA layers respectively; remaining layers are GDN. The open
checkpoints' native context is **262,144**, not the API's 1M configuration.

### GLM: IndexShare is not main-KV sharing

[GLM-5.3 config](https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/config.json)
retains `512+64` MLA in all 78 layers. IndexShare reuses top-k selections,
**not main MLA KV**. Its 21 `full` indexers own K caches. The selected BF16 MLA /
FP8-index reference totals `78*576*2 + 21*(128+4) = 92,628 B/token`.

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json)
has **34 KDA + 11 DSA/MLA** layers, explicitly zero RoPE key dimensions, and
4:1 index k-pooling: `11 * (512*2 + (128+4)/4) = 11,627 B/token`.
The FP8 keys, FP32 scales, skipped index owners and BF16 pending rings are in
[DSATokenToKVPool](https://github.com/sgl-project/sglang/blob/bf80732b926eabadbd6f54f702ffa74916a9121b/python/sglang/srt/mem_cache/memory_pool.py)
and [IndexKeyCache](https://github.com/sgl-project/sglang/blob/bf80732b926eabadbd6f54f702ffa74916a9121b/python/sglang/srt/mem_cache/index_key_cache.py).

### Kimi: NoPE does not discard the shared projection

[K3 config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json),
[publisher code](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py),
and [SGLang code](https://github.com/sgl-project/sglang/blob/bf80732b926eabadbd6f54f702ffa74916a9121b/python/sglang/srt/models/kimi_k3.py)
retain `512+64` latent dimensions with `skip_rope=True`. K3 has **69 KDA + 24
gated MLA** layers: `24*576*2 = 27,648 B/token`. MXFP4 weights do not mean FP4 KV.

The [separate Linear checkpoint](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json)
has 20 KDA and **seven** MLA layers, not rounded `27/4`. Both configs' layer
lists are **one-based**; the catalog converts them to zero-based owners. K3's
last two layers are both MLA, relevant to PP. This is absorbed-MLA sizing;
naive expanded HF K/V tensors cost more.

## Fixed state and exported traffic

[SGLang's shape/dtype implementation](https://github.com/sgl-project/sglang/blob/bf80732b926eabadbd6f54f702ffa74916a9121b/python/sglang/srt/configs/mamba_utils.py)
uses FP32 recurrent matrices and default BF16 convolution history with
`kernel_size - 1` slots. Other backends/state overrides can differ.

| Model | Accounted fixed bytes per active sequence, TP=1 |
|---|---:|
| Qwen3.8-27B | 146.8125 MiB |
| Qwen3.8-Flash-Next | ~110.1212 MiB including QSA pending state |
| Qwen3.8-2.4T-A95B | ~560.0859 MiB |
| GLM-5.3-Flash | ~140.8027 MiB including index pending state |
| Kimi K3 | ~428.5547 MiB |
| Kimi Linear 48B-A3B | 41.40625 MiB |

For example K3's matrices alone cost `69*96*128*128*4 = 414 MiB`, plus ~14.55
MiB convolution history. A hybrid can use **more** memory at short contexts
despite much lower long-context growth.

`cache_footprint(N)` separates growing, bounded-window, fixed, exported, and
summed **accounted** resident payload. It is not an HBM/allocator reservation
estimate. Weights, workspace, page padding, draft/vision state, PLE/Engram
auxiliary state and unlisted backend buffers are excluded.

**V1 exports only `export: true` growing KV/indexer components.** It does not
offload recurrent/window checkpoints or implement numerical resume, replay,
sparse kernels or native connector layouts. Those policies need separate
fidelity work. Fixed state is reported, not fabricated as per-token I/O.

## Schema and transport

`cache_components` replaces `kv_layers/kv_latent_dim/kv_dtype` for heterogeneous
models. Entries declare buffer owners, head counts, elements, bit width,
quantization group/scales, compression ratio, lifetime and TP partition mode.

The classic adapter carries exact aggregate growing bytes in a one-plane
**uint8 envelope** per rank, preserving logical token/chunk counts. It does not
pretend an odd 11,627-byte stride is BF16 K/V. No silent rounding is allowed.
Chunks must align to compression groups; default chunk size 256 works for all
bundled profiles.

Component PP balances contiguous **physical backbone layers** (remainder to
early stages) and then counts owners. This is V1 placement, not a claim about
framework defaults. V4.1 is restricted to PP=1 until shared-cache cross-stage
ownership is modeled. Fractional byte strides and state-only stages fail
explicitly rather than acquiring invented heads/padding.

`kvbench info --model kimi-k3` exposes provenance, growth and maximum-context
footprint. Tests verify independently derived formulas and real LMCache file
bytes with small CPU pools. Model weights are neither downloaded nor executed.

## Validation

The memory-bounded suite passes **604 tests** (one expected optional-dependency
hint test skips when LMCache is installed). The new matrix checks actual
LMCache file bytes/readback, including odd byte strides, compression alignment,
TP replication, physical-layer PP placement and discrete compression-group
growth. It also checks fixed state independently from per-token payload.

The expanded matrix exposed LMCache 0.5.4 leaving stopped storage-loop default
executors alive. The adapter now drains/closes its owned stopped loops after
engine destruction. Tests disable background usage-report HTTP requests and
retain the existing 4 GiB virtual-memory limit instead of relaxing it.
