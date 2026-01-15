# GPU Emulation

KV-Bench emulates GPU behavior to provide realistic timing without actual hardware. This enables benchmarking KV cache strategies on commodity hardware.

## How It Works

The GPU emulation layer calculates operation timing based on:

1. **GPU specifications** (compute, memory bandwidth, capacity)
2. **Model architecture** (layers, dimensions, heads)
3. **Operation type** (prefill, decode, cache transfer)
4. **Efficiency factor** (configurable overhead simulation)

## GPU Profiles

### Available Profiles

| Profile | BF16 TFLOPS | HBM BW (TB/s) | HBM (GB) | TDP (W) |
|---------|-------------|---------------|----------|---------|
| `H100_SXM` | 1979 | 3.35 | 80 | 700 |
| `H100_PCIe` | 1513 | 2.0 | 80 | 350 |
| `H200_SXM` | 1979 | 4.8 | 141 | 700 |
| `A100_SXM` | 312 | 2.0 | 80 | 400 |
| `A100_PCIe` | 312 | 2.0 | 80 | 300 |
| `L4` | 121 | 0.3 | 24 | 72 |
| `L40S` | 362 | 0.864 | 48 | 350 |

### Profile Selection

Choose a GPU profile based on your target deployment:

```bash
# High-end datacenter GPU
kvbench serve --gpu H100_SXM

# Cost-optimized inference
kvbench serve --gpu L4

# Legacy datacenter
kvbench serve --gpu A100_SXM
```

## Timing Calculations

### Prefill Phase

During prefill, the model processes all input tokens in parallel. Time is dominated by compute:

```python
# FLOPs for attention computation
attention_flops = 2 * num_tokens * num_tokens * hidden_dim * num_layers

# FLOPs for FFN computation
ffn_flops = 8 * num_tokens * hidden_dim * hidden_dim * num_layers

# Total compute time
prefill_time = (attention_flops + ffn_flops) / (gpu_tflops * 1e12 * efficiency)
```

### Decode Phase

During decode, tokens are generated one at a time. Time is dominated by memory bandwidth:

```python
# KV cache size per token
kv_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * 2  # 2 for K and V, 2 for bf16

# Memory access per decode step
memory_bytes = kv_bytes_per_token * sequence_length

# Decode time per token
decode_time = memory_bytes / (hbm_bandwidth * 1e12 * efficiency)
```

### Cache Transfer

When loading or storing KV cache:

```python
# Transfer time based on backend bandwidth
transfer_time = cache_size_bytes / backend_bandwidth
```

## Efficiency Factor

The efficiency factor (0.1 to 1.0) simulates real-world overhead:

| Efficiency | Scenario |
|------------|----------|
| 0.9-1.0 | Ideal, optimized deployment |
| 0.7-0.8 | Typical production |
| 0.5-0.6 | Suboptimal configuration |
| 0.3-0.4 | Heavy contention/overhead |

```bash
# High efficiency (optimized)
kvbench serve --gpu H100_SXM -e KVBENCH_GPU__EFFICIENCY_FACTOR=0.85

# Conservative estimate
kvbench serve --gpu H100_SXM -e KVBENCH_GPU__EFFICIENCY_FACTOR=0.65
```

## Tensor Parallelism

For multi-GPU configurations, specify tensor parallelism:

```bash
# 4-way tensor parallelism
export KVBENCH_GPU__TP_SIZE=4
kvbench serve --gpu H100_SXM
```

Timing is adjusted:
- Compute scales linearly with TP size
- Memory bandwidth scales linearly with TP size
- Communication overhead is added

## Custom GPU Profiles

Create custom profiles by extending the base class:

```python
from kvbench.core.profiles import GPUProfile, register_gpu_profile

custom_gpu = GPUProfile(
    name="Custom_GPU",
    bf16_tflops=500.0,
    hbm_bandwidth_tb_s=2.5,
    hbm_capacity_gb=96,
    tdp_watts=450,
)

register_gpu_profile("Custom_GPU", custom_gpu)
```

## Validation

Compare emulated timing against real hardware:

```bash
# Run benchmark and compare
kvbench benchmark --gpu H100_SXM --validate

# Output timing comparison report
kvbench benchmark --gpu H100_SXM --report timing_comparison.json
```

## Best Practices

1. **Match target hardware**: Use the GPU profile matching your production deployment
2. **Conservative efficiency**: Start with 0.7 efficiency and adjust based on validation
3. **Include TP overhead**: For multi-GPU, account for communication costs
4. **Validate periodically**: Compare emulated vs. real timing when possible
