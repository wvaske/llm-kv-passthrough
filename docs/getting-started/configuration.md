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

# KV management stack (storage is configured through LMCache itself)
export KVBENCH_KV__STACK=lmcache
export KVBENCH_KV__LMCACHE_CONFIG_FILE=/etc/kvbench/lmcache.yaml

# Or configure LMCache via its own environment variables instead of a file
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOCAL_DISK="file:///var/lib/lmcache/"
export LMCACHE_MAX_LOCAL_DISK_SIZE=100
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

timing:
  simple_mode: false            # true = fixed ms/token instead of roofline
  prefill_ms_per_token: 0.1     # simple mode only
  decode_ms_per_token: 1.0      # simple mode only
  include_tp_communication: true   # AllReduce timing when tp_size > 1
  include_pp_communication: true   # send/recv timing when pp_size > 1
  pp_size: 1
  # nvlink_bandwidth_gb_s: 900.0   # defaults to the GPU profile's NVLink

kv:
  stack: lmcache
  # LMCache's own config file controls all storage (tiers, backends, sizes)
  lmcache_config_file: /etc/kvbench/lmcache.yaml

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

### Timing Options

| Option | Default | Description |
|--------|---------|-------------|
| `simple_mode` | `false` | Fixed ms/token timing instead of the roofline model |
| `prefill_ms_per_token` | `0.1` | Prefill latency per token in ms (simple mode) |
| `decode_ms_per_token` | `1.0` | Decode latency per token in ms (simple mode) |
| `include_tp_communication` | `true` | Add AllReduce timing when `tp_size > 1` (roofline mode) |
| `include_pp_communication` | `true` | Add pipeline send/recv timing when `pp_size > 1` (roofline mode) |
| `pp_size` | `1` | Pipeline parallelism size |
| `nvlink_bandwidth_gb_s` | GPU profile | Interconnect bandwidth for communication timing |

CLI shortcuts: `--simple-timing/--roofline-timing`, `--prefill-ms-per-token`,
`--decode-ms-per-token`, `--tp-size`, and `--pp-size`.

### KV Stack Options

| Option | Default | Description |
|--------|---------|-------------|
| `stack` | `lmcache` | KV management stack (`kvbm` planned) |
| `lmcache_config_file` | `None` | LMCache's own config file; `LMCACHE_*` env vars are used when unset |

Storage backends, tier sizes, and eviction are configured in LMCache's own
application configuration — see [Storage](../architecture/storage.md).

### Available GPU Profiles

| Profile | BF16 TFLOPS | HBM BW (TB/s) | HBM (GB) |
|---------|-------------|---------------|----------|
| `H100_SXM` | 989.5 | 3.35 | 80 |
| `H100_PCIe` | 756.5 | 2.0 | 80 |
| `H200_SXM` | 989.5 | 4.8 | 141 |
| `A100_SXM` | 312 | 2.0 | 80 |
| `A100_PCIe` | 312 | 2.0 | 80 |
| `L4` | 60.5 | 0.3 | 24 |
| `L40S` | 181.05 | 0.864 | 48 |

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
