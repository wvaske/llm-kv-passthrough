# AgentX matrix on dev: handoff

Branch `claude/great-hawking-t7l3y5` = Murat's PR 6 head (`d3e27af`) plus:

- `muse-spark` model profile (Meta Muse Spark Open 816B-A42B), tests, docs
- `scripts/agentx/run_agentx_matrix.sh` + `scripts/agentx/dashboard_feed.py`
- `docs/benchmarking/agentx-matrix.md`

## What was verified in the cloud sandbox (no lab access)

| Check | Result |
|---|---|
| `pytest tests/unit` on PR head | 648 passed, 1 skipped |
| `pytest tests/integration -m integration` (real LMCache 0.5.4) | 53 passed, 18 skipped |
| `scripts/drive-test/check_lmcache_mp_stack.py` | ALL PASS |
| `ruff check` | 4 style nits in `lmcache_mp_stack.py` (as the PR notes) |
| Serve + steady-state warmup, B300_SXM, TP=8, rank 0, each of the 3 models | OK |
| AIPerf 0.13.0 closed-loop load against each server (synthetic long-session stand-in) | see dashboard |

AgentX traces could not be used in the sandbox (Hugging Face is blocked there).

## Run on dev (needs root for mkfs/mount on the zvol)

```bash
cd /path/to/llm-kv-passthrough && git fetch origin && git checkout claude/great-hawking-t7l3y5
uv sync --extra lmcache --extra dev        # or: pip install -e ".[lmcache]"
sudo AIPERF=/root/aiperf-venv/bin/aiperf scripts/agentx/run_agentx_matrix.sh \
  --dev /dev/zvol/databases/testdev --mnt /mnt/testdev --format 1 \
  --models qwen3.8-flash-next,glm-5.3,muse-spark --gpu B300_SXM --tp 8 --ranks 0 \
  --workload agentx --traces /root/agentx/weka_all --max-isl 131072 \
  --users 16 --duration 900 --cpu-gb 2 \
  --out /mnt/development/$USER/agentx-runs --tag zvol-testdev
```

`--format 1` wipes the zvol. Drop it to reuse an existing mount of that device.
Without a local trace copy use `--public-dataset semianalysis-cc-traces-weka`.

Results: `<out>/<run_id>/` per model and `<out>/dashboard/{runs,events}.json`;
send the two dashboard files back (or paste them) and the claude.ai dashboard
is refreshed from them.
