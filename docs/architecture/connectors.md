# KV manager adapter

`LMCacheStack` in `src/kvbench/kv/` drives the real LMCache 0.5.4 classic engine
API. `HiCacheStack` drives pinned SGLang 0.5.21 HiCacheFile L3 storage with a
modeled V1 host tier; see its [integration contract](hicache.md).
`KVStack` is the shared boundary for managers.

Servers operate on token sequences using lookup/pin, retrieve, store and unpin.
The adapter validates rank geometry, checks CPU-pool/chunk allocation guards,
and serializes blocking calls. Cancellation waits for an already-running call
before cleanup. Every created rank engine is destroyed on close or partial
startup failure.

By default every modeled TP/PP rank is instantiated. `kv.active_ranks` can select
any nonempty subset, including a single nonzero rank. The adapter keeps the full
rank layout for sizing and namespace identity, then creates resources only for
selected IDs. LMCache metadata retains the modeled world size and original
global worker IDs. Cache keys and deterministic payloads for a rank therefore
do not change when peers are omitted. Store/retrieve/lookup/unpin operations
visit only active engines; prefix completion is checked across those engines.
Scheduler rates remain aggregate full-engine inputs.

LMCache itself owns prefix hashing, chunk identity, backend allocation,
tiering, eviction and physical storage I/O. KV-Bench fills manager-allocated
payloads with deterministic high-entropy bytes; no inference tensors or weights
are computed. Final device transfer is omitted at fidelity level 1.

Only whole-chunk synchronous operations are supported. Async loading,
layerwise/blending, PD, controller, hidden-state and device-specific paths are
rejected rather than silently mapped onto the wrong protocol. Remote-backend
compatibility depends on LMCache and its dependencies; CPU/disk paths are the
integration-tested matrix in this repository.

KVBM remains unimplemented and is rejected. Earlier notes about its CUDA-only
worker are historical research, not a current upstream compatibility guarantee.
