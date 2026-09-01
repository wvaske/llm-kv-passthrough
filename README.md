# KV-Bench

Distributed mock LLM serving system for benchmarking KV cache management without GPUs.

## Features

- **Real LMCache integration** — all KV cache operations go through the
  actual LMCache engine (CPU-only, no GPU required); LMCache owns chunking,
  hashing, tiering, eviction, and every byte of storage I/O
- **Storage configured through LMCache's own application config** — CPU RAM,
  local disk, and remote backends (Redis, Mooncake, ...) via LMCache's
  config file or `LMCACHE_*` environment variables
- **Steady-state warmup** — one command fills every cache tier past capacity
  so benchmarks measure the eviction-active steady state, not an empty cache
- **Prometheus metrics** at `/metrics` — KV operations, token hit rate, live
  tier usage, warmup progress, request latency and TTFT histograms
- **KV I/O tracing → FIO** — record every logical and physical storage
  operation LMCache performs, then `kvbench trace2fio` derives an FIO job
  file reproducing the workload (chunk sizes, read/write mix, writer
  parallelism, eviction churn)
- **Incompressible KV data** — mock KV tensors are filled with random bytes
  so storage systems with compression/dedup see realistic entropy
- **Disaggregated prefill/decode** architecture emulation
- **OpenAI-compatible API** — benchmark with NVIDIA AIPerf/GenAI-Perf or any
  OpenAI-style load generator
- **GPU timing emulation** based on real hardware specs (roofline model)

Planned future work: NVIDIA Dynamo **KVBM** as a second KV stack (blocked on
a CPU device mode in upstream kvbm — its data plane requires the CUDA driver
at initialization; see `docs/architecture/connectors.md`).

## Installation

```bash
# With the LMCache KV management stack (required to run servers)
pip install "kvbench[lmcache]"

# With development dependencies
pip install "kvbench[dev]"

# With all optional dependencies
pip install "kvbench[all]"
```

## Quick Start

### Start the Server

```bash
# Basic server (LMCache with default CPU-memory tier)
kvbench serve --model llama-3.1-8b --gpu H100_SXM

# With an LMCache config file selecting storage tiers (disk, Redis, ...)
kvbench serve --model llama-3.1-8b --lmcache-config lmcache.yaml

# List available profiles
kvbench list-profiles

# Show profile information
kvbench info --gpu H100_SXM --model llama-3.1-8b
```

### Send Requests

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Benchmark Workflow (steady state → measure → derive FIO)

```bash
# 1. Serve with an LMCache config sized for the storage under test,
#    recording every KV storage operation to a trace file
kvbench serve --model llama-3.1-8b \
  --lmcache-config examples/lmcache-local-disk.yaml \
  --trace-file /tmp/kv-trace.jsonl

# 2. Fill all cache tiers past capacity so every further store evicts
#    (verifies steady state by confirming the earliest data was evicted)
kvbench warmup --url http://localhost:8000 --fill-factor 1.25

# 3. Run your benchmark against the OpenAI endpoint (AIPerf, GenAI-Perf, ...)
#    and watch KV activity live at http://localhost:8000/metrics

# 4. Derive an FIO job file that reproduces the disk workload LMCache
#    generated: chunk file size, read/write mix, concurrent writers,
#    eviction churn — reviewable, commented, hand-tunable
kvbench trace2fio /tmp/kv-trace.jsonl -o kv_workload.fio --directory /mnt/nvme/kvtest
fio kv_workload.fio
```

Key endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST/GET/DELETE /kvbench/warmup` | Start / poll / cancel steady-state warmup |
| `GET /kvbench/state` | Tier capacities, live usage, KV op stats |
| `GET /metrics` | Prometheus exposition format |
| `GET /stats` | JSON metrics summary |

### Docker Deployment

```bash
# Single server
docker run -p 8000:8000 kvbench:latest

# Distributed deployment
cd deployment/docker
docker-compose -f docker-compose.distributed.yml up -d
```

## Configuration

KV-Bench can be configured via environment variables, YAML files, or CLI arguments.

### Environment Variables

```bash
export KVBENCH_SERVER__HOST=0.0.0.0
export KVBENCH_SERVER__PORT=8000
export KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b
export KVBENCH_GPU__GPU_PROFILE=H100_SXM

# Storage is configured through LMCache's own environment variables
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_DISK="file:///var/lib/lmcache/"
export LMCACHE_MAX_LOCAL_DISK_SIZE=100
```

### YAML Configuration

```yaml
# config.yaml
instance_id: kvbench-0

server:
  port: 8000
  model_profile: llama-3.1-8b
  server_type: combined

gpu:
  gpu_profile: H100_SXM
  efficiency_factor: 0.7

kv:
  stack: lmcache
  lmcache_config_file: lmcache.yaml   # LMCache's own config controls storage
```

```yaml
# lmcache.yaml — LMCache application configuration
chunk_size: 256
local_cpu: true
max_local_cpu_size: 4.0
local_disk: "file:///var/lib/lmcache/"
max_local_disk_size: 100.0
# remote_url: "redis://cache-host:6379"
```

```bash
kvbench serve --config config.yaml
```

## Supported GPU Profiles

TFLOPS values are dense (non-sparsity) tensor-core numbers for consistent cross-GPU comparisons.

| Profile | BF16 TFLOPS | HBM Bandwidth | HBM Capacity |
|---------|-------------|---------------|--------------|
| H100_SXM | 989.5 | 3.35 TB/s | 80 GB |
| H100_PCIe | 756.5 | 2.0 TB/s | 80 GB |
| H200_SXM | 989.5 | 4.8 TB/s | 141 GB |
| A100_SXM | 312 | 2.0 TB/s | 80 GB |
| L4 | 60.5 | 0.3 TB/s | 24 GB |
| L40S | 181.05 | 0.864 TB/s | 48 GB |

## Supported Model Profiles

| Model | Layers | Hidden | KV Heads | Parameters |
|-------|--------|--------|----------|------------|
| llama-3.1-8b | 32 | 4096 | 8 | ~8B |
| llama-3.1-70b | 80 | 8192 | 8 | ~70B |
| llama-3.1-405b | 126 | 16384 | 8 | ~405B |
| qwen-2.5-7b | 28 | 3584 | 4 | ~7B |
| qwen-2.5-72b | 80 | 8192 | 8 | ~72B |
| mistral-7b | 32 | 4096 | 8 | ~7B |
| mixtral-8x7b | 32 | 4096 | 8 | ~47B |

## Storage

KV-Bench never performs storage I/O itself. All KV cache operations go
through the **real LMCache engine**, and storage — CPU-memory tier, local
disk tier, remote backends — is selected and tuned entirely in LMCache's
own application configuration (`--lmcache-config` file or `LMCACHE_*`
environment variables). Anything LMCache supports as a backend, KV-Bench
benchmarks; the storage under test sits under the KV management stack,
exactly as it does in a real vLLM + LMCache deployment.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Requests                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Disaggregated Proxy                         │
└─────────────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│     Prefill Servers       │   │      Decode Servers       │
└───────────────────────────┘   └───────────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              LMCache (real engine — KV management)              │
│        chunking · hashing · tiering · eviction · all I/O        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│     Storage under test: CPU RAM → local disk → remote (...)     │
│           configured via LMCache's own application config       │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone the repository
git clone https://github.com/wvaske/llm-kv-passthrough.git
cd llm-kv-passthrough

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=kvbench --cov-report=html

# Format code
ruff format .

# Run linting
ruff check .

# Type checking
mypy src/kvbench
```

## Project Structure

```
kvbench/
├── src/kvbench/
│   ├── core/           # Configuration, GPU/model profiles
│   ├── kv/             # KV management stack integration (real LMCache)
│   ├── servers/        # HTTP servers, OpenAI API
│   └── cli/            # Command-line interface
├── tests/
│   ├── unit/           # Unit tests
│   ├── integration/    # Real-LMCache integration tests
│   └── e2e/            # End-to-end tests
├── docs/               # MkDocs documentation
├── deployment/
│   ├── docker/         # Docker Compose files
│   └── ansible/        # Ansible playbooks
└── scripts/            # Utility scripts
```

## Documentation

Full documentation available at https://github.com/wvaske/llm-kv-passthrough (docs/)

Build locally:

```bash
cd docs
pip install mkdocs mkdocs-material
mkdocs serve
```

## Benchmarking

### GenAI-Perf

```bash
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --concurrency 10 \
  --num-requests 100
```

### LMCache Integration

```bash
./scripts/lmcache_test.sh
```

## License

BSD 3-Clause License — see [LICENSE](LICENSE).
