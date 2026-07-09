# LMCache Integration

KV-Bench integrates the **real [LMCache](https://github.com/LMCache/LMCache)
library** — not an emulation. Every KV cache operation runs through the
actual LMCache engine: its chunking, its prefix hashing, its multi-tier
placement and eviction, its serialization, and its storage I/O. The GPU is
the only mocked component (CPU tensors stand in for GPU KV memory via
LMCache's `GPUConnectorInterface`), which is what lets KV-Bench run on
GPU-less benchmark nodes.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     KV-Bench Server                             │
│                  (OpenAI-compatible API)                        │
└─────────────────────────────────────────────────────────────────┘
                              │  token sequences + mock KV tensors
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                LMCache Engine (real library)                    │
│        chunking · prefix hashing · tiering · eviction           │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌───────────┐   ┌─────────────┐
        │ CPU RAM  │   │ Local disk│   │ Remote      │
        │ (hot)    │   │ (warm)    │   │ (Redis, ...)│
        └──────────┘   └───────────┘   └─────────────┘
                    Storage under test
```

## Installation

```bash
pip install "kvbench[lmcache]"
```

LMCache runs CPU-only in this configuration; no GPU or CUDA is required.

## Configuration

Storage is configured through **LMCache's own application configuration**.
KV-Bench passes the config file through verbatim:

```yaml
# lmcache.yaml
chunk_size: 256
local_cpu: true
max_local_cpu_size: 4.0            # GB, hot tier
local_disk: "file:///var/lib/lmcache/"
max_local_disk_size: 100.0         # GB, warm tier (NVMe under test)
# remote_url: "redis://cache-host:6379"   # cold/shared tier
```

```bash
kvbench serve --model llama-3.1-8b --gpu H100_SXM --lmcache-config lmcache.yaml
```

Or use LMCache's `LMCACHE_*` environment variables and omit the file:

```bash
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_DISK="file:///var/lib/lmcache/"
export LMCACHE_MAX_LOCAL_DISK_SIZE=100
kvbench serve
```

Every option in the [LMCache configuration
reference](https://docs.lmcache.ai/) applies unmodified.

## What the benchmark measures

- **Prefill**: the server looks up the prompt's tokens in LMCache,
  retrieves the cached prefix (storage read), simulates GPU compute for
  the uncached suffix using the roofline model, and stores the new KV
  (storage write).
- **Decode**: the server retrieves the prompt's KV through LMCache before
  generating, so storage read latency appears in TTFT — the disaggregated
  read path.

Because tensor sizes come from the emulated model profile, bytes-per-token
match the real model, and because LMCache performs the I/O, chunk sizes,
key formats, and tier behavior match a real deployment.

## Performance tuning

Tune LMCache, not KV-Bench — chunk size, tier capacities, and remote
serialization are all LMCache settings:

```bash
for size in 64 128 256 512; do
  sed "s/^chunk_size:.*/chunk_size: $size/" lmcache.yaml > lmcache-$size.yaml
  kvbench serve --lmcache-config lmcache-$size.yaml &
  sleep 5
  python benchmark.py --output results_chunk_${size}.json
  kill %1
done
```

## Troubleshooting

**Low hit rate** — prompts must share exact prefixes for prefix caching to
apply; check chunk alignment (hits are chunk-granular) and chunk size
against your workload's shared-prefix length.

**`lmcache` import errors** — install the extra: `pip install
"kvbench[lmcache]"`. The engine is required to run servers; KV-Bench fails
fast with an install hint when it's missing.
