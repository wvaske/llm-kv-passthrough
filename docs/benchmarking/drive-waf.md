# Single-NVMe cache-layout and write-amplification comparison

The [active campaign launch record](active-campaign.md) identifies the supervised
run, frozen code revision, artifact locations and stop command.

## What is measured

Three distinct layers are retained:

1. **KV/file operations:** page/chunk bytes, files, API latency, writer threads,
   overlap and eviction traces. These are not device LBAs.
2. **Block issue order:** bounded `blktrace` captures, parsed into request sizes,
   adjacent-write transitions, maximum contiguous-LBA runs, unique written-LBA
   coverage and revisited-write bytes/fraction. This is device
   issue order, not NAND placement or reconstructed per-file streams.
3. **Device counters:** standard SMART host bytes and OCP NAND bytes, sampled
   independently every five seconds and saved with raw logs and timestamps.

`WAF = delta NAND bytes / delta host bytes`. Host units are **512,000 bytes**;
the supported OCP physical-media counter is a **128-bit byte count**. Deltas
are calculated with exact integers before dashboard float conversion. Host
counter quantization and independently updated NAND counters can distort tiny
intervals; use longer windows and cumulative phase totals for interpretation.

Zero host writes produce no interval WAF, not zero. Idle NAND activity is kept.
Missing counters, resets, power-cycle/identity changes and sampling gaps invalidate
cumulative WAF. All counters are **controller-wide**, including other processes,
filesystem metadata and controller housekeeping. An idle-write-rate check catches
existing activity; it cannot prove unrelated writers will remain inactive later.

## Qualification and protocol

The runner requires direct ext4/XFS on one NVMe namespace, rejects system root,
RAID/LVM/ZFS/device-mapper backing, and checks controller/serial and exactly one
allocated namespace. The filesystem must span at least 90% of the namespace;
the namespace must span nearly the controller's reported capacity.

The example requires:

- **80–95% occupied**, with preparation targeting **85%**. Existing real data
  contributes; dedicated filler supplies the rest without deleting it.
- One workload preconditioning drive fill for each arm, separately recorded.
- **Three full drive capacities of observed host writes** per measured arm,
  plus a minimum duration. KV submissions or NAND writes cannot satisfy this.
- Matched model, KV precision, selected ranks, chunk/page bytes, host/disk quotas,
  scheduler settings, seed and request distribution, checked before any workload
  and again against live capacities. Each worker uses the same deterministic
  prompt sequence across arms; faster arms may process more requests before the
  observed-device-write stopping criterion is satisfied.
- ABBA order: LMCache, HiCache, HiCache, LMCache, to expose order/history effects.
- 60-second idle baseline, 5-second sampling, and a 300-second drain/idle tail.
- 60-second block-issue sample during measurement. Dropped trace events invalidate
  the LBA sample. A missing block tracer is an error unless explicitly disabled.
- Separate process per arm, one active rank of TP=8, 0.25 GiB host cache and
  64 GiB disk quota. Server virtual memory is capped at 4 GiB. Full-speed compute
  mode is intentional; actual payload generation and storage remain real work.
- The accelerated profile labels the system as 8×B300, with illustrative aggregate
  rates of 200,000 prefill / 20,000 decode tokens/s, 32 active slots, 16 clients,
  8,192-token prefill slices and 512 output tokens per request. Full-speed mode
  already skips compute sleeps: throughput comes from actual generation/I/O,
  not from multiplying the configured rates. Both arms use `payload_generator:
  sfc64`, with 256 KiB bounded scratch and fresh high-entropy bytes.

### Pattern-focused full-speed workload

Both cache arms use **`timing.full_speed: true` with 16 producer streams**.
The scheduler skips emulated GPU compute sleeps and feeds the real KV manager
as fast as the host and storage path allow. Configured GPU token rates impose
no pacing in this mode. Admission stays at 32 active / 64 pending slots.

Acceleration does not multiply concurrent streams, per-request prompt/output
sizes, prefill batches, chunk/page bytes, cache quotas or payload bytes. Real
manager admission, backpressure, eviction and storage determine actual I/O.
There is no synthetic bandwidth or counter multiplier. The objective is the
resulting write/eviction/address-reuse pattern, not matching GPU throughput.

The proposed 10× stream multiplier was removed before deployment: increasing
stream count would alter the concurrency pattern. The running frozen campaign
already uses full-speed mode with 16 streams, so this correction requires no
interruption of preparation.

The pattern comparisons emphasize:

- File/page size distribution and per-request storage order.
- Concurrent writer streams, file creation and native cache eviction.
- Contiguous issued-LBA runs and address revisits under cache churn.
- Device NAND/host WAF over observed drive fills.

`revisited_write_bytes` is sampled issued-write bytes minus the union of sampled
written LBA ranges, with partial overlap counted exactly. It includes filesystem
metadata and device reissues, and only detects repeats **within that sample**.
It is not a NAND-write counter or evidence that an application overwrote the same
file. Native cache churn may unlink/create immutable page objects and let the
filesystem reuse their LBAs. A disk-backed ordered interval set beside the trace
keeps coverage analysis bounded in RAM; original traces remain available.
The currently running frozen campaign predates these new LBA-reuse summary
fields. Its retained `block-issues.txt` files can be analyzed with the updated
`summarize_issues` function after capture, without rerunning storage I/O.

The large quota gives buffered data time to reach writeback before eviction;
tiny fast-turnover caches can unlink dirty files before the device sees them.
The observed-host-write goal catches that case instead of counting fake drive
fills. Both adapters retain their real buffered-I/O behavior. The runner drains
manager writes and calls `sync -f` on the test filesystem at phase boundaries,
then keeps collecting NAND activity. It does not issue per-page fsync, trim,
format, sanitize, or a global drop-caches operation.

Trace/telemetry/control files must live off the tested filesystem. Benchmark
artifact storage retains a 16 GiB free-space reserve; the run stops if it is
reached. Full per-file traces can consume tens of gigabytes in a drive-fill test.
Benchmark
directories are new and unique per arm. Existing cache files and filler are
preserved; directory/file creation cannot silently overwrite previous runs.
A controller lease prevents overlapping fill/comparison jobs from this harness.

Three fills on the discovered 30.7 TB drive are **92.16 TB of host writes per
measured arm**, in addition to 30.72 TB preconditioning. ABBA therefore writes
roughly **491.5 TB**, plus initial filler and drain. Runtime is determined by
observed throughput and can be many days. The maximum duration is a failure
deadline, never permission to call an incomplete run qualified.

## Commands

Install `nvme-cli`, `blktrace` (including `blkparse`), and the two Python adapters.
The user needs read-only NVMe admin access and permission to attach block tracing.
See [HiCache installation](../architecture/hicache.md) and
`deployment/monitoring/README.md` for the existing lab collector/dashboard.

```bash
# Read-only: establish device mapping, counter support and occupancy.
python -m kvbench.benchmark.drive preflight --target /mnt/work/kvbench \
  --controller /dev/nvme6 --expected-serial 2337448A0113 --sudo

# Create REAL, rate-limited random data in a NEW directory; never sparse/fallocate-only.
# This example can write ~22 TB on the discovered drive. Files remain after it ends.
python -m kvbench.benchmark.drive fill --target /mnt/work/kvbench/waf-filler \
  --controller /dev/nvme6 --expected-serial 2337448A0113 --sudo \
  --occupancy 0.85 --rate-mib 128

# --max-bytes bounds filler work for a functional check; an unfinished fill is
# recorded as "incomplete", never accepted as preparation.

# Comparison after qualification:
python -m kvbench.benchmark.drive compare --config examples/drive-compare.yaml

# Or unattended preparation + comparison, with durable preparation telemetry:
python -m kvbench.benchmark.drive campaign --config examples/drive-compare.yaml \
  --rate-mib 2048 --filler-generator sfc64 --fsync-mib 256

# Small functional check only, ALWAYS qualified:false. Request count is bounded.
python -m kvbench.benchmark.drive compare --config examples/drive-compare.yaml --mode smoke
```

The lab launcher `bash scripts/launch-drive-campaign.sh` requires a clean commit,
freezes source/configs in a detached worktree, and starts the full campaign under
systemd as the invoking user. Its whole process group has a 3 GiB memory-high
threshold, 4 GiB hard memory limit, no swap, eight CPU-equivalents, low I/O priority,
and a 3 GB/s read/write cap on the test namespace. Filler has a 2,048 MiB/s rate
ceiling and uses SFC64 rather than a cryptographic system RNG, fsyncing each
256 MiB. These are ceilings, not guaranteed throughput. These controls also
affect page-cache writeback and must be retained
for both arms when comparing results. The source revision is recorded.

The unit is `kvbench-waf-<eight-character-commit>.service`; inspect with
`systemctl status` / `journalctl -u`, and stop it with `sudo systemctl stop`.
It does not auto-restart an interrupted experiment or delete incomplete artifacts.

Filler uses an 8 MiB buffer (plus at most 256 KiB scratch for SFC64), defaults to
fsync every 64 MiB unless overridden, and creates files no larger than
8 GiB, and stops at a 5% free-space floor. It preserves existing data and the
new filler. The campaign samples during preparation, then fails if preparation
or telemetry is incomplete. Do not restart an interrupted fill over its old
directory; its manifest records partial progress and existing data remains.

Existing filler contributes to filesystem occupancy after a campaign restart.
The new campaign creates a fresh owned directory and only supplies the remaining
space; it does not erase or rewrite the previous preparation files.

On dev, the bounded 4 MiB-page CPU microbenchmark measured about 81 MiB/s with
the legacy Torch-byte RNG and 2,041 MiB/s with SFC64 (about 25×). Reproduce with
`scripts/benchmark-payloads.py` under its documented process/thread limits.
This measures only payload generation; real device throughput is measured
separately by the NVMe collector. Page size, dtype, selected ranks and byte
geometry are unchanged, and fast payloads use a separate cache namespace.

## Artifacts and Grafana

Each run creates `experiment.json`, resolved `server.yaml`, manager config,
`manifest.json`, server log, file-I/O JSONL, raw/precomputed NVMe JSONL per phase,
phase summaries, final KV state, raw per-CPU block traces, `block-issues.txt` and
`block-summary.json`. Failed runs retain `qualified:false` and their error.
Small smoke runs are never qualified, even if counter goals are met.

Set `influx_file` to `/var/lib/kvbench-nvme/experiment.influx`. The collector also
appends every sample to `experiment.stream`; Telegraf tails it, avoiding lost
samples at polling/phase boundaries. The existing sampler continuously records
the same drive independently. Select the experiment's run ID in the
[Grafana dashboard](http://172.16.2.20:3000/d/kvbench-nvme-waf). Window WAF divides
summed byte deltas, not averaged WAF ratios. Phase-drain series retain that
phase's cumulative baseline so idle GC contributes to its final WAF.

## Bounded real-drive validation (2026-10-09)

Artifacts: `/tmp/opencode/kvbench-drive-smoke/20261009T153658-8dc19c8d`.
Four cold requests per arm, two concurrent clients, TP=8/rank 0, BF16 Llama 8B,
256-token pages/chunks, 4 MiB per file, equal 0.25/0.5 GiB host/disk quotas.
Both produced **16 files / 64 MiB**; both file traces showed one concurrent
writer in this small rate-limited run. The three-second block samples recorded:

| Observation | LMCache | HiCacheFile |
|---|---:|---:|
| Write issues | 524 | 513 |
| Dominant block write size | 128 KiB | 128 KiB |
| Adjacent write transitions | 508 / 523 | 511 / 512 |
| Longest contiguous issued-LBA run | 24 MiB | 64 MiB |

Both traces reported zero dropped events. Host and NAND deltas were nonzero,
saved with raw counters, and exposed in Grafana. A prior tracer invocation found
that this `blktrace` requires `-D directory -o basename`, not an absolute `-o`
path, and may still exit zero on output errors. The runner now validates its log.

These observations validate instrumentation and motivate the sustained test;
they establish **no comparative WAF or steady-state performance claim**. The
drive was only **13.4% occupied**, and each smoke arm wrote approximately
0.00000218 drive fills. The normal endurance preflight correctly rejected it.

See [the HiCache contract](../architecture/hicache.md) for the distinction
between upstream HiCacheFile and the modeled host write-back policy. In
particular, native HiRadixTree write-back behavior is not claimed by these runs.
