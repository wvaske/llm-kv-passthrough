# HiCache integration and comparison contract

## Upstream and scope

HiCache belongs to **SGLang**, not vLLM. This integration targets SGLang
**v0.5.21**, commit `e00930c5489053f26d86b179cee0d087f846acbb`.
The first CPU-only adapter exercises its real **HiCacheFile L3 backend**,
including raw page serialization, deduplication and LRU file eviction. V1
supplies synthetic host pages and an explicitly modeled bounded host tier;
it does not run SGLang's GPU scheduler, HiRadixTree or GPU transfer kernels.
Results must identify that boundary rather than claim native end-to-end
SGLang/vLLM performance.

Install the source wheel without SGLang's GPU-serving dependencies:

```bash
python -m pip install -e '.[hicache]'
python -m pip install --no-deps 'sglang==0.5.21'
kvbench serve --config examples/hicache-tp8-single-drive.yaml
```

The CPU loader verifies the installed version and SHA256 hashes of four upstream
source files before execution. It loads the real environment definitions, size
parser, LRU evictor and file backend in a private namespace, redirecting only
their SGLang-internal imports. Storage method bodies are unchanged. No network
fetch, GPU import, source modification or fake `sglang` modules occur at serving
time. The manifest records source hashes and adapter scope.

`kv.stack: hicache` selects the adapter. `kv.hicache` configures `storage_path`,
`page_size`, `host_size_gb`, `disk_size_gb` (GiB per active rank), `min_free_bytes`,
`eviction_ratio`, `mem_layout` and `write_policy`. The memory guard includes the
host tiers plus one temporary page. The initial supported profile family is
homogeneous MHA/GQA; MLA storage-owner deduplication and hybrid sidecar pools are
rejected explicitly rather than represented as native-compatible layouts.

Supported memory layouts are `page_first`, `page_first_direct` and `layer_first`.
MHA page-first bytes have axis order `[K/V, token, layer, head, dim]`; direct and
layer-first pages flatten `[K/V, layer, token, head, dim]`. Classic LMCache's
`KV_2LTD` connector has the **same flattened axis order as page-first-direct**;
do not confuse its metadata shape with its actual connector tensor shape.
Payloads are deterministic high-entropy
synthetic bytes, not floating-point inference outputs.

The modeled host `write_through` policy submits each completed page immediately;
`write_back` submits least-recently-used host pages on eviction, in groups up to
`write_batch_pages` (default 32), and drains remaining pages at shutdown. That is
a V1 scheduling policy, **not** SGLang's native L1→L2 write-back trigger. It must
be labeled separately in comparisons. Native HiCacheFile owns L3 quota/eviction.
Successful file operations are buffered writes, not per-page fsync durability.
Prefix-load leases prevent this adapter's eviction between lookup and retrieval.

Sources:

- [HiCache design](https://docs.sglang.io/advanced_features/hicache_design)
- [HiCacheFile](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/hicache_storage.py)
- [File eviction](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/storage/file/lru_file_evictor.py)
- [Chained page hashing](https://github.com/sgl-project/sglang/blob/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp)

## Hypothesis, not a baked-in result

HiCache supports write-through, selective write-through and write-back. Its
file backend writes one raw `.bin` object per page using a temporary file and
rename; `batch_set` loops over pages. A batch is **not** one contiguous file
write, and page-contiguous host memory does not prove contiguous device LBAs.
LMCache classic local-disk workers write chunk objects concurrently. Neither
solution's name alone establishes its actual stream ordering or NAND WAF.

Compare equal page/chunk tokens, model/KV format, selected ranks, aggregate
rates, request workload, host capacity, storage quota, filesystem and device.
Run a same-policy comparison first, then a separately labeled write-back
experiment. Preserve per-file size/timing/rank traces and independently
capture block-device operations when evaluating LBA sequentiality.

## Device discovery (2026-10-09)

Dev's `/mnt/work` is XFS on a single `/dev/nvme6n1p1`, backed by a
Dell/Micron 6500 RI FIPS 30.7 TB drive, serial `2337448A0113`. It currently
contains existing data and was approximately 14% full at discovery. The
development workspace itself is RAID10 and is not a valid single-drive target.

The drive exposes both standard SMART `data_units_written` (512,000 bytes per
unit) and OCP `Physical media units written` (128-bit byte counter). Counter
support is confirmed; a valid endurance comparison has **not** yet run.
Use interval deltas, never lifetime NAND/host ratios, for test WAF.

The long-run harness must verify a single direct NVMe backing device, stable
serial/firmware, high occupancy, real allocated filler (not sparse/fallocate-only),
and at least three namespace-capacity equivalents of observed host writes per
measured phase. Keep setup, preconditioning, measurement and drain/idle intervals
separate. Missing NAND counters, resets and telemetry gaps invalidate WAF rather
than becoming zero. Record raw counters and publish interval/run WAF to Grafana.
