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

## KVBM

NVIDIA Dynamo's [KV Block Manager](https://docs.nvidia.com/dynamo/latest/architecture/kvbm_components.html)
(KVBM) is the second stack on the roadmap. `stack: kvbm` is accepted in
configuration but rejected at startup with the reason below — there is
deliberately no stub implementation.

### Why KVBM is not yet supported

Verified empirically against `kvbm` 1.2.1 from PyPI on a CPU-only host
(July 2026):

1. **The data plane requires the CUDA driver.** `KvbmWorker` — the
   component that registers KV memory and executes all tier transfers
   (device → host → disk → remote, via NIXL) — panics at construction
   attempting to dynamically load `libcuda.so`, even when handed CPU
   tensors. Without a worker, no bytes move.
2. **The control plane cannot stand alone.** `BlockManager` requires a
   `KvbmLeader`, and the leader's initialization barrier waits for
   workers to register — which circles back to requirement 1.
3. **No escape hatch.** The `DYN_KVBM_*` configuration surface (extracted
   from the shipped binary) tunes cache sizes, disk paths, and transfer
   batching, but offers no CPU-device or mock-transfer mode. NVIDIA's
   documentation describes the CPU (G2) and disk (G3) tiers only as
   offload targets fed from GPU memory (G1).

By contrast, LMCache ships a CPU platform stub and a pluggable
GPU-connector interface, which is exactly the seam KV-Bench mocks.

### What would unblock it

- **Upstream**: a CPU device layout / mock transfer backend in KVBM's
  worker, letting host memory stand in for G1 the way KV-Bench's
  `MockGPUConnector` does for LMCache. KVBM's design (per-tier block
  pools behind one lifecycle API) is compatible with this; the current
  wheel just hard-binds the transfer engine to CUDA.
- **Alternatively**: a GPU-enabled KV-Bench deployment mode, where KVBM
  runs its real data plane on a GPU node while KV-Bench still simulates
  inference timing. This trades away KV-Bench's GPU-less premise and is
  only worth building against a testable environment.

KVBM's storage-relevant configuration is, like LMCache's, its own:
`DYN_KVBM_CPU_CACHE_GB`, `DYN_KVBM_DISK_CACHE_GB`,
`DYN_KVBM_DISK_CACHE_DIR`, etc. — when the integration lands, KV-Bench
will pass that surface through unmodified, mirroring the LMCache
approach.
