# AgentX storage matrix: one LMCache instance of an 8x B300 node

`scripts/agentx/run_agentx_matrix.sh` runs one KV-storage test point per model
profile on one block device and keeps a flat JSON feed a dashboard can follow.
It was written for the PR 6 validation of Qwen3.8 Flash Next, GLM-5.3 and
Meta Muse Spark Open with the AgentX (SemiAnalysis Claude Code trace) workload.

## What one test point does

1. `kvbench serve --model M --gpu B300_SXM --tp-size 8 --kv-ranks 0 --full-speed`
   starts one real LMCache engine holding rank 0's shard of a TP=8 engine.
   The modeled node has eight LMCache instances (one per GPU); only one is
   instantiated, so the device sees exactly one instance's traffic:

   | Profile | Node B/token | Rank 0 B/token | 256-token chunk on disk |
   |---|---:|---:|---:|
   | `qwen3.8-flash-next` | 25,344 (104,448 stored across 8 ranks: the QSA index key is replicated per rank) | 13,056 | 3.19 MiB |
   | `glm-5.3` | 92,628 (MLA latent, replicated on every rank) | 92,628 | 22.6 MiB |
   | `muse-spark` | 65,536 | 8,192 | 2.0 MiB |

2. A fresh LMCache disk tier on the device (`--tier-pct` of the filesystem,
   default 90%) plus a small CPU tier; `kvbench warmup --fill-factor 1.05`
   fills past capacity so the load runs in the eviction-active steady state.
3. AIPerf drives the OpenAI endpoint with AgentX traces
   (`--custom-dataset-type weka_trace --input-file <dir>` or
   `--public-dataset semianalysis-cc-traces-weka`), closed loop at `--users`
   concurrency for `--duration` seconds, trace think-times stripped unless
   `--trace-delays 1`.
4. `/proc/diskstats` deltas for the resolved device, LMCache's own counters
   (`/kvbench/state`), request stats (`/stats`), Prometheus text and the
   AIPerf export are written to `--out/<run_id>/`, and
   `--out/dashboard/{runs,events}.json` is updated at every phase boundary
   (`scripts/agentx/dashboard_feed.py`).

The runner needs no lab-specific helper: no nvme-cli, tmux, bpftrace or
LMCache patch scripts. The LMCache 0.5.4 disk-tier fix from PR 6 is applied at
runtime by `kvbench.kv.lmcache_compat`, so an unpatched 0.5.4 install is fine.

## Running on the dev host against the zvol

```bash
sudo AIPERF=/root/aiperf-venv/bin/aiperf scripts/agentx/run_agentx_matrix.sh \
  --dev /dev/zvol/databases/testdev --mnt /mnt/testdev --format 1 \
  --models qwen3.8-flash-next,glm-5.3,muse-spark --gpu B300_SXM --tp 8 --ranks 0 \
  --workload agentx --traces /root/agentx/weka_all --max-isl 131072 \
  --users 16 --duration 900 --cpu-gb 2 \
  --out /mnt/development/$USER/agentx-runs --tag zvol-testdev
```

- `--format 1` runs `mkfs.xfs -f` on the resolved zvol (`/dev/zdN`) and mounts
  it at `--mnt`; omit it to reuse an existing mount, which must be from that
  device. The cache directory `<mnt>/kvbench_agentx` is recreated per model.
- With no local trace copy, `--public-dataset semianalysis-cc-traces-weka`
  (or the `-with-subagents` variants) lets AIPerf download the dataset from
  Hugging Face.
- `--tokenizer` names the AIPerf tokenizer: the lab default `gpt2` needs a
  populated HF cache, `builtin` fetches tiktoken's o200k_base once, and a local
  directory holding `tokenizer.json` + `tokenizer_config.json` works with no
  network at all (run AIPerf without `HF_HUB_OFFLINE` for a local directory).
- A zvol reports block I/O through `/proc/diskstats` like any disk, so the
  read/write deltas are valid; NAND-level write amplification is a property of
  the physical pool, not the zvol, and is not collected here.
- One run writes roughly warmup (1.05 x tier) plus the load's store volume to
  the device; size `--duration` and `--tier-pct` for the zvol's capacity.

## Following the matrix live

`--out/dashboard/runs.json` holds one row per test point with phase, status,
rank-0 bytes/token, warmup GiB, request counts, LMCache token hit rate, KV
bytes submitted and retrieved, device read/write GiB and the write
amplification above the cache. `events.json` is the phase log. Both are plain
JSON arrays: import them into any dashboard, or hand them to the session that
maintains the claude.ai dashboard for this matrix.
