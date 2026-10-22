# Storage ownership and measurements

KV payload I/O goes through **real LMCache** or **SGLang's HiCacheFile backend**.
The [HiCache adapter](hicache.md) uses a modeled host tier over native L3 file
operations; the sections below describe LMCache unless noted. LMCache configuration controls chunk size,
CPU/local-disk capacities and optional remote backends.

Each active logical TP/PP rank owns a CPU pool and backend instance. Configured tier
sizes are per active rank; `/kvbench/state` reports their aggregate and per-rank chunk
geometry. A shared local-disk directory alone does not provide cross-process
discovery: LMCache's local-disk index is process-local. Use an appropriate
LMCache remote backend for cross-process sharing, and validate that deployment.

Use a fresh directory for cold-run comparisons. A restarted local-disk engine
does not account for old files in its new in-memory index; its per-run capacity
limit is not a cleanup policy for artifacts from earlier runs.

KV `stored_tokens` counts logical tokens **submitted** to LMCache once per
sequence; `submitted_store_bytes` sums payloads across active ranks. Neither
confirms durable writes. LMCache can deduplicate or decline cache admission under
pressure, and disk writes happen on its workers. Trace `io/write` and `io/read`
events measure completed file operations. These are application-level file
operations, not block-device tracing; remote physical I/O is not instrumented.

`kv.active_ranks` selects measured ranks without altering modeled TP/PP geometry.
Omitted ranks allocate nothing and issue no I/O. A single selected BF16 Llama 3.1
8B rank at TP=8 submits 16 KiB/token (4 MiB per 256-token chunk), versus 128
KiB/token across all eight ranks. Multiple selected ranks share the configured
disk path but retain distinct rank keys and per-rank quotas; a shared directory
does not give them separate physical drives.

Byte counters, tier capacities, usage and trace I/O cover only active ranks.
Full-topology byte/chunk geometry is reported separately in `modeled_*` capacity
fields. There is no automatic whole-system I/O extrapolation. Cache-hit decisions
also cover only active ranks; omitted-rank eviction and latency are not modeled.
See [selected-rank configuration](../getting-started/configuration.md#selected-rank-execution).

Warmup reports submitted bytes and an earliest-prefix eviction probe. That is
evidence of eviction, not a proof that every tier is in steady state. Warmup
and measured serving must not overlap. Keep separate traces/run windows when
deriving workload characteristics. Both warmup capacity targets and byte/token
conversion use the active ranks' actual geometry and configured quotas.
