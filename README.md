# KV-Bench

Distributed mock LLM serving system for benchmarking KV cache management without GPUs.

## Features

- **Multi-host deployment** with shared storage (Redis, NFS, Ceph, Weka, MinIO)
- **Disaggregated prefill/decode** architecture emulation
- **Pluggable KV backends** (LMCache, Mooncake, Dynamo)
- **Configurable resources** (CPU memory, NVMe, external storage)
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

```bash
# List available profiles
kvbench list-profiles

# Show profile information
kvbench info --gpu H100_SXM --model llama-3.1-8b

# Start the server (implementation coming in Phase 4)
kvbench serve --model llama-3.1-8b --gpu H100_SXM
```

## Configuration

KV-Bench can be configured via:

1. **Environment variables** (prefix: `KVBENCH_`)
2. **YAML configuration file**
3. **Programmatic configuration**

### Environment Variables

```bash
export KVBENCH_INSTANCE_ID=kvbench-0
export KVBENCH_RESOURCES__CPU_MEMORY_GB=64.0
export KVBENCH_STORAGE__BACKEND_TYPE=redis
export KVBENCH_STORAGE__REDIS_URL=redis://localhost:6379
export KVBENCH_GPU__GPU_PROFILE=H100_SXM
export KVBENCH_SERVER__PORT=8000
```

### YAML Configuration

```yaml
# config.yaml
instance_id: kvbench-0
resources:
  cpu_memory_gb: 64.0
  nvme_storage_gb: 500.0
storage:
  backend_type: redis
  redis_url: redis://localhost:6379
gpu:
  gpu_profile: H100_SXM
  efficiency_factor: 0.7
server:
  port: 8000
  model_profile: llama-3.1-8b
```

## Supported GPU Profiles

| Profile | BF16 TFLOPS | HBM Bandwidth | HBM Capacity |
|---------|-------------|---------------|--------------|
| H100_SXM | 1979 | 3.35 TB/s | 80 GB |
| H100_PCIe | 1513 | 2.0 TB/s | 80 GB |
| H200_SXM | 1979 | 4.8 TB/s | 141 GB |
| A100_SXM | 312 | 2.0 TB/s | 80 GB |
| L4 | 121 | 0.3 TB/s | 24 GB |
| L40S | 362 | 0.864 TB/s | 48 GB |

## Supported Model Profiles

| Model | Layers | Hidden | KV Heads | Parameters |
|-------|--------|--------|----------|------------|
| llama-3.1-8b | 32 | 4096 | 8 | ~8B |
| llama-3.1-70b | 80 | 8192 | 8 | ~70B |
| llama-3.1-405b | 126 | 16384 | 8 | ~405B |
| qwen-2.5-7b | 28 | 3584 | 4 | ~7B |
| qwen-2.5-72b | 80 | 8192 | 8 | ~72B |

## Development

```bash
# Clone the repository
git clone https://github.com/kvbench/kvbench.git
cd kvbench

# Install development dependencies
make install-dev

# Run tests
make test

# Run tests with coverage
make coverage

# Format code
make format

# Run type checks
make type-check
```

## Project Structure

```
kv-bench/
├── src/kvbench/
│   ├── core/           # Config, GPU/model profiles
│   ├── kv/             # KV cache management
│   ├── connectors/     # LMCache, Mooncake, Dynamo
│   ├── storage/        # Storage backends
│   ├── servers/        # HTTP servers
│   ├── distributed/    # Distributed coordination
│   ├── metrics/        # Prometheus metrics
│   └── cli/            # Command-line interface
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
├── deployment/
│   ├── docker/
│   └── ansible/
└── benchmarks/
```

## License

MIT License
