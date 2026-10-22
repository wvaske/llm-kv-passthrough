# Profiles and fidelity levels

For current DeepSeek, Qwen, GLM and Kimi releases, see the
[researched catalog and byte formulas](model-catalog.md). Component layouts
separate compressed/shared KV, sparse index keys and fixed recurrent state.

V1 performance is defined by explicit aggregate prefill/decode rates. GPU peak
specifications do **not** change those rates automatically. Hardware-dependent
coupled costs, real framework schedulers, HBM admission and preemption belong to
fidelity level 2. Chakra replay belongs to level 3.

## File-backed catalogs

Built-ins are in `src/kvbench/core/profiles/catalog.yaml`. Custom catalogs use
the same `version: 1`, `models`, and `gpus` structure:

```yaml
version: 1
models:
  my-gqa-model:
    name: My GQA model
    layers: 32
    hidden: 4096
    num_heads: 32
    kv_heads: 8
    intermediate: 14336
    dtype: bf16
    kv_dtype: fp8
    max_seq_len: 131072
gpus:
  my-hardware:
    name: My hardware description
    bf16_tflops: 100
    hbm_bandwidth_tb_s: 0.25
    hbm_capacity_gb: 64
```

Select with `profiles_file`, `server.model_profile`, and `gpu.gpu_profile` in
the run config. Overrides are per configuration instance. Unknown fields fail;
there is no Python registration step.

## KV byte geometry

For ordinary GQA/MQA, rank-chunk bytes are:

```text
2 × rank_KV_layers × rank_KV_heads × head_dim × KV_dtype_bytes × chunk_tokens
```

`head_dim_override` handles nonstandard head dimensions. `kv_layers` specifies
only cache-bearing layers in hybrid models. `kv_latent_dim` represents MLA as
a single latent per layer, replicated across TP ranks. The classic adapter
represents its bytes as two halves; native MLA layout interoperability is not
claimed. Constant-size recurrent state is not part of V1 storage traffic.

TP divides KV heads, or replicates heads when TP is a multiple of the number of
heads. Homogeneous legacy profiles distribute cache-bearing layers across PP;
component profiles partition physical backbone layers and count their actual
buffer owners. The resulting rank geometry is included in the manifest. The manager's
cache namespace includes model-definition and rank-layout identity.

`dtype` and `kv_dtype` are independent. `params_billions_override` is an active
compute estimate, while `total_params_billions` describes total weight capacity.
Neither drives V1 rates. Older legacy profile estimates were inherited from
earlier experiments; the researched current-family entries carry pinned source
revisions and explicit cache formats. Inspect/override geometry for the actual
deployment under test.

## Historical analytical utilities

`core/latency.py`, `timing/`, and `servers/gpu_engine.py` contain experimental
roofline/calibration/batching calculations. Their anchors mix measured, fitted,
borrowed and projected operating points. They are not V1 validation evidence,
and their timing flags are rejected at the V1 serving boundary.
