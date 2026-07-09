# LMCache Deployment Guide

LMCache is KV-Bench's KV management stack: it owns every byte of KV cache
storage. Deploying KV-Bench is therefore two steps — deploy the storage
infrastructure you want to test, then point LMCache's configuration at it.
KV-Bench itself has no storage settings.

## Installation

```bash
pip install "kvbench[lmcache]"
```

LMCache runs CPU-only in this configuration. No GPU, CUDA, or special
drivers are required on the benchmark nodes.

## Configuration file

Write LMCache's own config file and hand it to `kvbench serve`:

```yaml
# /etc/kvbench/lmcache.yaml
chunk_size: 256                    # tokens per KV chunk

# Tier 1: CPU memory (hot)
local_cpu: true
max_local_cpu_size: 4.0            # GB

# Tier 2: local disk (warm) — the NVMe under test
local_disk: "file:///var/lib/lmcache/"
max_local_disk_size: 100.0         # GB

# Tier 3: remote backend (cold / shared across nodes)
remote_url: "redis://cache-host:6379"
remote_serde: "naive"
```

```bash
kvbench serve --model llama-3.1-8b --gpu H100_SXM \
    --lmcache-config /etc/kvbench/lmcache.yaml
```

Every option in the [LMCache configuration
reference](https://docs.lmcache.ai/) applies unmodified; KV-Bench passes
the file through verbatim.

## Environment variables

For container deployments, LMCache's `LMCACHE_*` variables replace the
file (KV-Bench uses them automatically when no config file is set):

```bash
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_MAX_LOCAL_CPU_SIZE=4
export LMCACHE_LOCAL_DISK="file:///var/lib/lmcache/"
export LMCACHE_MAX_LOCAL_DISK_SIZE=100
export LMCACHE_REMOTE_URL="redis://cache-host:6379"
kvbench serve
```

## Multi-node shared cache (disaggregated)

In a disaggregated deployment, prefill servers write KV cache and decode
servers read it back. Give every server the same `remote_url` so the
shared tier flows through the storage you want to characterize:

```
prefill nodes ──► LMCache ──► remote tier (Redis/Valkey/S3/fs/...) ◄── LMCache ◄── decode nodes
```

Point the proxy at the fleets with `KVBENCH_DISTRIBUTED__*` endpoints (see
the [Docker guide](docker.md) for a complete compose topology).

## Verifying that LMCache owns the I/O

After a few requests, the disk tier contains artifacts written by LMCache
in its own key format — evidence that the benchmark exercises the real
storage path:

```bash
$ ls /var/lib/lmcache/
llama-3.1-8b@1@0@-a4e4a87b8c11903@bfloat16.pt
```

The `/metrics` endpoint reports cache hits/misses; hit-path latency
includes the real storage reads LMCache performs.

## Sizing guidance

Per-token KV bytes are authentic for the emulated model — for
`llama-3.1-8b` (32 layers, 8 KV heads, head dim 128, bf16) that is 128 KB
per token, so a 4096-token prompt moves ~512 MB through the configured
tiers. Size `max_local_cpu_size` and prompt lengths accordingly, and
expect benchmark clients with long prompts to generate substantial I/O
quickly.
