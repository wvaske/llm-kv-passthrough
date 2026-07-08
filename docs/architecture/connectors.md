# KV Management Stacks

KV-Bench integrates with real KV cache management stacks through the
`KVStack` interface (`src/kvbench/kv/`). Servers speak in token sequences;
the stack owns chunking, hashing, tiering, and all storage I/O. There are
no mock or passthrough stacks — the benchmark measures real KV management
code paths driving real storage.

## Interface

```python
class KVStack:
    async def lookup(tokens: list[int]) -> int      # cached prefix length
    async def store(tokens, skip_leading=0) -> None  # write path
    async def retrieve(tokens: list[int]) -> int     # read path
```

The servers use these three operations the way vLLM uses LMCache:

1. **Prefill**: `lookup` the prompt's tokens → `retrieve` the cached prefix
   (storage read) → simulate compute for the uncached suffix → `store` the
   new KV (storage write, cached prefix masked out).
2. **Decode**: `lookup` + `retrieve` the prompt's KV before generating, so
   storage read latency appears in TTFT — the disaggregated read path.

## LMCache (supported)

`LMCacheStack` drives the real `lmcache` engine (v1 API). It runs CPU-only:
LMCache's platform layer detects the absence of CUDA and uses its CPU
device stub, and KV-Bench supplies mock GPU-side tensors through LMCache's
`GPUConnectorInterface`. Everything downstream — memory objects, tiering,
disk and remote I/O — is unmodified LMCache code.

Configuration is pass-through: point `kv.lmcache_config_file` (or
`--lmcache-config`) at LMCache's own config file, or use `LMCACHE_*`
environment variables. KV-Bench adds no storage settings of its own.

```yaml
# kvbench config
kv:
  stack: lmcache
  lmcache_config_file: lmcache.yaml
```

Install with `pip install "kvbench[lmcache]"`.

## KVBM (planned)

NVIDIA Dynamo's KV Block Manager is the second stack on the roadmap. The
factory rejects `stack: kvbm` with a clear error until the integration is
real — there is deliberately no stub implementation.
