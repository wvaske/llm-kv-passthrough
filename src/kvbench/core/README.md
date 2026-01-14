# KV-Bench Core Module

This module provides the foundational components for KV-Bench.

## Components

### Configuration System (`config.py`)

The configuration system uses Pydantic v2 for validation and supports:

- Environment variables (prefix: `KVBENCH_`, nested delimiter: `__`)
- YAML configuration files
- Programmatic configuration

```python
from kvbench.core import KVBenchConfig

# Load from environment variables
config = KVBenchConfig.from_env()

# Load from YAML file
config = KVBenchConfig.from_yaml("config.yaml")

# Programmatic configuration
config = KVBenchConfig(
    instance_id="kvbench-0",
    resources=ResourceLimits(cpu_memory_gb=64.0),
    storage=StorageConfig(backend_type="redis", redis_url="redis://localhost:6379"),
)
```

#### Configuration Classes

| Class | Description |
|-------|-------------|
| `KVBenchConfig` | Main configuration container |
| `ResourceLimits` | CPU memory and NVMe storage limits |
| `StorageConfig` | Storage backend configuration |
| `ConnectorConfig` | KV connector (LMCache, etc.) settings |
| `GPUEmulationConfig` | GPU profile and efficiency settings |
| `ServerConfig` | HTTP server settings |
| `DistributedConfig` | Multi-node deployment settings |
| `MetricsConfig` | Prometheus metrics settings |

### GPU Profiles (`gpu_profiles.py`)

GPU hardware profiles for latency emulation using the roofline model.

```python
from kvbench.core import get_gpu_profile, list_gpu_profiles

# List all profiles
profiles = list_gpu_profiles()
# ['A100_PCIe', 'A100_SXM', 'H100_PCIe', 'H100_SXM', 'H200_SXM', 'L4', 'L40S']

# Get a specific profile
h100 = get_gpu_profile("H100_SXM")
print(h100.bf16_tflops)        # 1979.0
print(h100.hbm_bandwidth_tb_s) # 3.35
print(h100.hbm_capacity_gb)    # 80
```

#### Available GPU Profiles

| Profile | BF16 TFLOPS | HBM BW (TB/s) | HBM (GB) | TDP (W) |
|---------|-------------|---------------|----------|---------|
| H100_SXM | 1979 | 3.35 | 80 | 700 |
| H100_PCIe | 1513 | 2.0 | 80 | 350 |
| H200_SXM | 1979 | 4.8 | 141 | 700 |
| A100_SXM | 312 | 2.0 | 80 | 400 |
| A100_PCIe | 312 | 2.0 | 80 | 300 |
| L4 | 121 | 0.3 | 24 | 72 |
| L40S | 362 | 0.864 | 48 | 350 |

### Model Profiles (`models.py`)

LLM model profiles for KV cache sizing and latency calculations.

```python
from kvbench.core import get_model_profile, calculate_kv_cache_requirements

# Get a model profile
llama = get_model_profile("llama-3.1-8b")
print(llama.layers)                          # 32
print(llama.hidden)                          # 4096
print(llama.kv_heads)                        # 8
print(llama.total_kv_cache_bytes_per_token)  # 131072

# Calculate KV cache requirements
reqs = calculate_kv_cache_requirements(
    model_name="llama-3.1-8b",
    max_batch_size=8,
    max_seq_len=4096,
)
print(reqs["total_gb"])  # Total KV cache in GB
```

#### Available Model Profiles

| Model | Layers | Hidden | KV Heads | Params (B) |
|-------|--------|--------|----------|------------|
| llama-3.1-8b | 32 | 4096 | 8 | ~8 |
| llama-3.1-70b | 80 | 8192 | 8 | ~70 |
| llama-3.1-405b | 126 | 16384 | 8 | ~405 |
| qwen-2.5-7b | 28 | 3584 | 4 | ~7 |
| qwen-2.5-72b | 80 | 8192 | 8 | ~72 |
| mistral-7b | 32 | 4096 | 8 | ~7 |
| mixtral-8x7b | 32 | 4096 | 8 | ~46 |

## Usage Examples

### Calculate KV Cache Size

```python
from kvbench.core import get_model_profile

model = get_model_profile("llama-3.1-70b")

# KV cache for 8k context
kv_size_gb = model.kv_cache_size_gb(8192)
print(f"KV cache for 8k tokens: {kv_size_gb:.2f} GB")

# Per-token KV cache size
bytes_per_token = model.total_kv_cache_bytes_per_token
print(f"Bytes per token: {bytes_per_token}")
```

### Check GPU Memory Capacity

```python
from kvbench.core import get_gpu_profile, get_model_profile

gpu = get_gpu_profile("H100_SXM")
model = get_model_profile("llama-3.1-70b")

# How many tokens fit in HBM (after model weights)?
model_size_gb = model.model_size_gb
available_gb = gpu.hbm_capacity_gb - model_size_gb
max_tokens = int(available_gb * 1024**3 / model.total_kv_cache_bytes_per_token)
print(f"Max tokens in KV cache: {max_tokens:,}")
```
