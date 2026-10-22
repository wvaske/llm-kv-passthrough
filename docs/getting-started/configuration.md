# V1 configuration

The runnable reference is `examples/rate-v1.yaml`. Schema implementation:
`src/kvbench/core/config.py`.

## Resolution

- CLI flags override a selected `--config` YAML file.
- Without a file, `KVBENCH_*` environment variables are used; `__` separates
  nested fields, e.g. `KVBENCH_SCHEDULER__DECODE_TOKENS_PER_SECOND=500`.
- File and environment bases are alternatives, not merged layers.
- Paths inside a KV-Bench file are relative to that file. LMCache's own path
  semantics apply inside its configuration.
- Unknown fields, nonfinite values, invalid profile names and invalid KV head
  partitioning fail. One process runs one combined engine (`workers: 1`).

## Rate scheduler

| Field under `scheduler` | Default | Meaning |
|---|---:|---|
| `prefill_tokens_per_second` | 10000 | Aggregate engine prefill rate |
| `decode_tokens_per_second` | 1000 | Aggregate engine decode rate |
| `prefill_chunk_tokens` | 256 | Maximum prefill work per scheduler round/request |
| `max_active_requests` | 16 | Active request slots, including I/O-blocked requests |
| `max_pending_requests` | 64 | Additional queued requests |
| `max_context_tokens` | 131072 | Prompt + output bound, also limited by the model |
| `max_output_tokens` | 4096 | Maximum requested completion length |
| `max_request_bytes` | 4194304 | HTTP chat-body limit before JSON parsing |
| `request_timeout_s` | 600 | Request deadline after request handling begins |

Cancellation/deadline cleanup drains any already-running blocking LMCache call
before releasing its pins. This can outlast the request deadline; the underlying
manager/backend controls its own I/O timeouts. Prefix-load leases are serialized
to avoid circular waits for a fully pinned CPU staging pool.

`timing.full_speed` skips compute sleeps without changing scheduling/KV semantics.
`timing.chars_per_token` defaults to 4.0 and means **UTF-8 bytes per synthetic
token**; fractional spans work. `timing.pp_size` and `gpu.tp_size` specify the
logical KV rank layout. Rates already describe the whole rank group and are
not multiplied by either rank count.

GPU peak specifications are descriptive/offline in V1. Set rates explicitly
for each intended hardware/model operating point.

## Profiles

`profiles_file` extends/overrides the bundled YAML catalog for this run.
See [profile schema](../architecture/gpu-emulation.md). Resolved definitions
appear in `/kvbench/manifest` with rates, ranks and package versions.

## KV adapter

| Field under `kv` | Default | Meaning |
|---|---|---|
| `stack` | `lmcache` | `lmcache` or `hicache` (SGLang HiCacheFile adapter) |
| `lmcache_config_file` | unset | LMCache YAML; otherwise `LMCACHE_*` environment |
| `active_ranks` | `null` (all) | Global TP/PP rank IDs to instantiate, e.g. `[0]` or `[0, 7]` |
| `trace_file` | unset | JSONL trace; use a unique path for each process/run |
| `max_cpu_memory_gb` | 2 | Aggregate active-rank CPU-pool allocation guard, GiB |
| `max_chunk_mb` | 256 | Largest permitted rank-chunk payload, MiB |
| `lmcache_reuse_history_entries` | 65536 | Maximum entries in each LMCache LRU diagnostic reuse-history table |
| `random_fill` | true | High-entropy deterministic payloads; false means zeros |
| `payload_generator` | `torch` | Legacy byte RNG, or `sfc64` for fast word-oriented high-entropy generation |
| `seed` | 24301 | Payload seed, mixed with rank and token prefix |

LMCache sizes are **per active rank**. The guard checks their sum before creating
any engine. With no explicit CPU sizing, the adapter uses 0.25 GiB per active rank. It keeps
process-wide cyclic GC enabled. It does not allocate model weights or simulated
HBM, and it does not change LMCache chunk size or explicit tier sizes.

For HiCache, use `kv.stack: hicache` with `kv.hicache.storage_path`, `page_size`,
`host_size_gb`, `disk_size_gb`, `mem_layout` and `write_policy`. Its quota fields
are also GiB per active rank. See the [HiCache contract](../architecture/hicache.md)
for installation and the distinction between native L3 storage and V1 host policy.
Do not pass an LMCache config file to a HiCache run.

`payload_generator: sfc64` uses fresh deterministic SFC64 words, filling existing
CPU tensors with at most 256 KiB of scratch memory. It preserves payload sizes
and axis order and uses a separate versioned cache namespace. Both comparison
arms must use the same generator. This is a statistical PRNG, not cryptographic
randomness; it does not reuse a repeated random-data pool. NumPy version and
the selected generator are recorded in manifests. The scratch allocation is
included in the CPU-memory guard.

LMCache 0.5.4 separately remembers first-seen hashes for reuse statistics, even
after payload eviction. Its default permits 12.5 million entries per policy.
The adapter limits that diagnostic table using the upstream reset threshold;
it does not alter the native cache's OrderedDict, quota or eviction order.
The effective history limit/count appears in traces, capacity and per-tier usage.
Reuse-statistics history is therefore windowed, while physical I/O/WAF counters
remain actual observed bytes. This prevents cold endurance workloads from
exhausting the process's address-space guard merely by accumulating statistics.

### Selected-rank execution

Use `examples/tp8-single-drive.yaml` to measure a single rank's storage traffic
while emulating a larger engine:

```yaml
gpu:
  tp_size: 8
timing:
  pp_size: 1
kv:
  active_ranks: [0]  # [0, 7] for two ranks, null for every rank
```

Equivalent CLI selection (add `--lmcache-config` for disk storage):

```bash
kvbench serve --model llama-3.1-8b --tp-size 8 --kv-ranks 0 \
  --prefill-tps 10000 --decode-tps 1000
```

Without `--config`, environment equivalents are:

```bash
KVBENCH_GPU__TP_SIZE=8 KVBENCH_KV__ACTIVE_RANKS='0,7' kvbench serve
```

Environment/CLI selection accepts comma-separated IDs or a JSON list such as
`'[0, 7]'`. `--kv-ranks all` explicitly clears a file/environment subset.
YAML `null`, `all`, or an omitted field selects every rank. Lists are sorted;
empty, duplicate, negative, noninteger and out-of-range IDs fail validation.

Global rank is `pp_rank * tp_size + tp_rank`, ranging from 0 to `TP * PP - 1`.
For TP=2/PP=4, rank 7 is TP rank 1 on PP stage 3. IDs are never renumbered when
rank 0 is omitted. Selection preserves full-world LMCache metadata, cache keys,
payload seeds, TP sharding/replication and PP layer ownership.

**Every request is processed by each active rank.** Selection changes neither
the request stream nor the configured aggregate compute rates. Inactive ranks
allocate no engine, connector or CPU pool and issue no storage operations.
Cache lookup/retrieve completion is the common prefix across active ranks only;
no cache, eviction or storage delay is simulated for omitted ranks.

#### Reading subset results

- `/kvbench/manifest`: `ranks` retains the full layout; `kv_rank_selection`
  reports active/inactive IDs and modeled versus active bytes/token.
- `/kvbench/state` → `capacity`: `stored_bytes_per_token`, `chunk_bytes`,
  `files_per_chunk` and tier capacities describe active ranks.
  `modeled_stored_bytes_per_token`, `modeled_chunk_bytes` and
  `modeled_files_per_chunk` describe the full topology's geometry. Each entry
  in `ranks` has an `active` flag.
- Usage labels and physical trace events retain global IDs such as `rank-7`.
  A trace metadata event (`op: rank_selection`) records the selection.
- Prometheus exposes `kvbench_kv_rank_count{scope="modeled"|"active"}` and
  `kvbench_kv_rank_active{rank="…"}`. KV byte/capacity counters cover active ranks;
  logical token counters count each sequence once, not once per rank.
- Warmup's target and submitted-byte accounting use active-rank capacities and
  payload geometry. A one-rank run does not fill eight ranks' worth of data.

No measured bytes or IOPS are extrapolated to the whole engine. Symmetric GQA
shards can have equal sizes; replicated MLA or heterogeneous PP stages do not
necessarily scale as `TP * PP` times the selected rank. Actual whole-system
storage behavior also depends on omitted ranks' cache state and contention.

## Metrics and migration

`metrics.enabled` and `metrics.include_histograms` control the exporter. Metrics
are on the same HTTP port as serving. The historical `prometheus_port` setting
does not start a separate listener; nondefault values are rejected for V1.

Legacy per-request/roofline/batching/calibration timing flags are offline
experiments, not selectable V1 serving modes. Replace them with explicit rates.
Legacy `storage`, `connector`, and `random_pool_mb` settings are rejected. The
LMCache file owns storage. Standalone prefill/decode/proxy serving is deferred;
run independent combined replicas instead.
