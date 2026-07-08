# Storage

**KV-Bench never performs storage I/O itself.** The storage under test sits
under the KV management stack — exactly where it sits in a real vLLM +
LMCache deployment. All KV cache operations (lookup, store, retrieve) go
through the real LMCache engine, and LMCache owns the entire storage
control plane: token chunking, prefix hashing, tier placement, spilling,
eviction, serialization, and every byte written to or read from storage.

```
Mock inference servers (prefill / decode / combined)
        │  token sequences + mock GPU-side KV tensors
        ▼
LMCache engine (real library, CPU-only capable)
        │  chunking · hashing · tiering · eviction
        ▼
Storage under test
  CPU RAM tier → local disk tier → remote backend (Redis, Mooncake, ...)
```

## Why this design

Benchmarking storage for LLM inference is only meaningful if the I/O
pattern matches what the KV management stack actually produces: LMCache's
chunk sizes, its key format, its tiering and spill decisions, its
serialization. Reimplementing storage backends inside the benchmark would
measure the benchmark's own I/O, not LMCache's. So KV-Bench has no storage
backends of its own — anything LMCache supports, KV-Bench benchmarks.

## Configuring storage

Storage is configured **through LMCache's own application configuration**,
not through KV-Bench settings:

```yaml
# lmcache.yaml — passed to `kvbench serve --lmcache-config lmcache.yaml`
chunk_size: 256

# Tier 1: CPU memory
local_cpu: true
max_local_cpu_size: 4.0        # GB

# Tier 2: local disk (NVMe under test)
local_disk: "file:///var/lib/lmcache/"
max_local_disk_size: 100.0     # GB

# Tier 3: remote backend (shared across prefill/decode fleets)
# remote_url: "redis://cache-host:6379"
# remote_serde: "naive"
```

Alternatively, set LMCache's `LMCACHE_*` environment variables and omit
the file. See the [LMCache documentation](https://docs.lmcache.ai/) for
the full option set — every option applies unmodified.

## Sizing and authenticity

KV tensor sizes are derived from the emulated model profile (layers, KV
heads, head dimension, dtype), so the number of bytes flowing into LMCache
per token matches the real model. The GPU side is mocked with CPU tensors
via LMCache's `GPUConnectorInterface`; everything downstream of that
interface is unmodified LMCache code, including the on-disk key format
(`<model>@<world_size>@<worker_id>@<hash>`).
