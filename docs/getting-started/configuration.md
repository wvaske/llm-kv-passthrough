# Configuration

KV-Bench can be configured via command-line arguments, environment variables, or YAML configuration files.

## Environment Variables

All configuration options support environment variables with the prefix `KVBENCH_` and nested delimiter `__`.

```bash
# Server configuration
export KVBENCH_SERVER__HOST=0.0.0.0
export KVBENCH_SERVER__PORT=8000
export KVBENCH_SERVER__SERVER_TYPE=combined
export KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b

# GPU emulation
export KVBENCH_GPU__GPU_PROFILE=H100_SXM
export KVBENCH_GPU__EFFICIENCY_FACTOR=0.7
export KVBENCH_GPU__TP_SIZE=1

# Storage
export KVBENCH_STORAGE__BACKEND_TYPE=memory
export KVBENCH_STORAGE__REDIS_URL=redis://localhost:6379

# Resources
export KVBENCH_RESOURCES__CPU_MEMORY_GB=8.0
export KVBENCH_RESOURCES__NVME_STORAGE_GB=100.0
```

## YAML Configuration

Create a `config.yaml` file:

```yaml
instance_id: kvbench-prod

server:
  host: 0.0.0.0
  port: 8000
  server_type: combined
  model_profile: llama-3.1-8b
  workers: 4
  log_level: info

gpu:
  gpu_profile: H100_SXM
  efficiency_factor: 0.7
  tp_size: 1

storage:
  backend_type: redis
  redis_url: redis://localhost:6379
  redis_cluster: false

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256

resources:
  cpu_memory_gb: 16.0
  nvme_storage_gb: 500.0
  nvme_path: /var/lib/kvbench/nvme
  memory_allocation: lazy

distributed:
  prefill_endpoints:
    - http://prefill-1:8000
    - http://prefill-2:8000
  decode_endpoints:
    - http://decode-1:8000
    - http://decode-2:8000
  health_check_interval: 10.0

metrics:
  enabled: true
  prometheus_port: 9090
  include_histograms: true
```

Load with:

```bash
kvbench serve --config config.yaml
```

## Configuration Reference

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `host` | `0.0.0.0` | Host address to bind to |
| `port` | `8000` | Port number |
| `server_type` | `combined` | Server type: combined, prefill, decode, proxy |
| `model_profile` | `llama-3.1-8b` | Model to emulate |
| `workers` | `1` | Number of worker processes |
| `log_level` | `info` | Logging level |

### GPU Emulation Options

| Option | Default | Description |
|--------|---------|-------------|
| `gpu_profile` | `H100_SXM` | GPU profile to emulate |
| `efficiency_factor` | `0.7` | GPU efficiency (0.1-1.0) |
| `tp_size` | `1` | Tensor parallelism size |

### Storage Options

| Option | Default | Description |
|--------|---------|-------------|
| `backend_type` | `memory` | Storage backend |
| `redis_url` | `None` | Redis connection URL |
| `redis_cluster` | `false` | Use Redis cluster mode |
| `filesystem_path` | `None` | Path for NFS/Weka backends |
| `s3_endpoint` | `None` | S3/MinIO endpoint URL |
| `s3_bucket` | `None` | S3 bucket name |

### Available GPU Profiles

| Profile | BF16 TFLOPS | HBM BW (TB/s) | HBM (GB) |
|---------|-------------|---------------|----------|
| `H100_SXM` | 1979 | 3.35 | 80 |
| `H100_PCIe` | 1513 | 2.0 | 80 |
| `H200_SXM` | 1979 | 4.8 | 141 |
| `A100_SXM` | 312 | 2.0 | 80 |
| `A100_PCIe` | 312 | 2.0 | 80 |
| `L4` | 121 | 0.3 | 24 |
| `L40S` | 362 | 0.864 | 48 |

### Available Model Profiles

| Profile | Layers | Hidden | KV Heads | Est. Params |
|---------|--------|--------|----------|-------------|
| `llama-3.1-8b` | 32 | 4096 | 8 | ~8B |
| `llama-3.1-70b` | 80 | 8192 | 8 | ~70B |
| `llama-3.1-405b` | 126 | 16384 | 8 | ~405B |
| `qwen-2.5-7b` | 28 | 3584 | 4 | ~7B |
| `qwen-2.5-72b` | 80 | 8192 | 8 | ~72B |
| `mistral-7b` | 32 | 4096 | 8 | ~7B |
| `mixtral-8x7b` | 32 | 4096 | 8 | ~47B |
