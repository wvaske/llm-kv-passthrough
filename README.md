# KV-Bench

Distributed mock LLM serving system for benchmarking KV cache management without GPUs.

## Features

- **Multi-host deployment** with shared storage (Redis, NFS, Weka, S3/MinIO, Mooncake)
- **Disaggregated prefill/decode** architecture emulation
- **Pluggable KV backends** (LMCache, Mooncake connectors)
- **OpenAI-compatible API** for easy integration
- **GPU timing emulation** based on real hardware specs
- **GenAI-Perf compatible** for standardized benchmarking

## Installation

```bash
# Basic installation
pip install kvbench

# With development dependencies
pip install kvbench[dev]

# With all optional dependencies
pip install kvbench[all]
```

## Quick Start

### Start the Server

```bash
# Basic server (in-memory storage)
kvbench serve --model llama-3.1-8b --gpu H100_SXM

# With Redis storage
kvbench serve --model llama-3.1-8b --storage redis

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
export KVBENCH_STORAGE__BACKEND_TYPE=redis
export KVBENCH_STORAGE__REDIS_URL=redis://localhost:6379
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

storage:
  backend_type: redis
  redis_url: redis://localhost:6379
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

## Storage Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `memory` | In-memory with LRU | Testing, development |
| `local_disk` | Local NVMe storage | Single-node production |
| `redis` | Redis/Cluster | Multi-node shared cache |
| `s3` | S3/MinIO | Cloud deployments |
| `nfs` | NFS filesystem | On-premise clusters |
| `weka` | Weka storage | HPC environments |
| `mooncake` | Mooncake engine | Disaggregated serving |

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
│                      KV Cache Connector                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Backend                            │
└─────────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone the repository
git clone https://github.com/your-org/kvbench.git
cd kvbench

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
│   ├── connectors/     # LMCache, Mooncake connectors
│   ├── storage/        # Storage backends (7 implementations)
│   ├── servers/        # HTTP servers, OpenAI API
│   └── cli/            # Command-line interface
├── tests/
│   ├── unit/           # Unit tests
│   └── e2e/            # End-to-end tests
├── docs/               # MkDocs documentation
├── deployment/
│   ├── docker/         # Docker Compose files
│   └── ansible/        # Ansible playbooks
└── scripts/            # Utility scripts
```

## Documentation

Full documentation available at https://your-org.github.io/kvbench/

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

MIT License
