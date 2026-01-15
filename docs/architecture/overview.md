# Architecture Overview

KV-Bench is a distributed mock LLM serving system designed to benchmark KV cache management strategies without requiring actual GPUs.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Requests                          │
│                    (OpenAI-compatible API)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Disaggregated Proxy                         │
│              (Load Balancing, Health Checks)                    │
└─────────────────────────────────────────────────────────────────┘
                    │                       │
                    ▼                       ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│     Prefill Servers       │   │      Decode Servers       │
│  (Prompt Processing)      │   │   (Token Generation)      │
└───────────────────────────┘   └───────────────────────────┘
                    │                       │
                    ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      KV Cache Connector                         │
│              (LMCache, Mooncake, Custom)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Backend                            │
│    (Memory, Redis, S3, NFS, Weka, Local Disk, Mooncake)        │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Servers

KV-Bench supports multiple server modes for different deployment scenarios:

| Mode | Description | Use Case |
|------|-------------|----------|
| `combined` | Single server handling both prefill and decode | Development, testing |
| `prefill` | Dedicated prefill server | Disaggregated production |
| `decode` | Dedicated decode server | Disaggregated production |
| `proxy` | Load balancer for prefill/decode servers | Production routing |

### 2. GPU Emulation

The GPU emulation layer simulates realistic timing based on:

- **Compute throughput**: BF16 TFLOPS for attention computation
- **Memory bandwidth**: HBM bandwidth for KV cache transfers
- **Memory capacity**: HBM size for cache eviction decisions

Supported GPU profiles: H100_SXM, H100_PCIe, H200_SXM, A100_SXM, A100_PCIe, L4, L40S

### 3. Model Profiles

Model profiles define the architecture parameters used for timing calculations:

- Number of layers
- Hidden dimension
- Number of KV heads
- Head dimension
- Vocabulary size

Supported models: LLaMA 3.1 (8B, 70B, 405B), Qwen 2.5 (7B, 72B), Mistral 7B, Mixtral 8x7B

### 4. KV Cache Connectors

Connectors manage KV cache storage and retrieval:

| Connector | Description |
|-----------|-------------|
| `lmcache` | LMCache-compatible connector with chunking |
| `mooncake` | Mooncake transfer engine connector |
| `custom` | Base class for custom implementations |

### 5. Storage Backends

Storage backends provide the actual data persistence:

| Backend | Description | Best For |
|---------|-------------|----------|
| `memory` | In-memory with LRU eviction | Testing, small deployments |
| `local_disk` | Local NVMe storage | Single-node deployments |
| `redis` | Redis/Redis Cluster | Multi-node shared cache |
| `s3` | S3/MinIO object storage | Cloud deployments |
| `nfs` | NFS shared filesystem | On-premise clusters |
| `weka` | Weka distributed storage | High-performance clusters |
| `mooncake` | Mooncake transfer engine | Disaggregated inference |

## Request Flow

### Combined Mode

1. Client sends chat completion request
2. Server tokenizes input
3. Server checks KV cache for prefix matches
4. Cache hit: Load cached KV states
5. Cache miss: Simulate prefill computation
6. Store new KV cache entries
7. Simulate decode token generation
8. Return response (streaming or complete)

### Disaggregated Mode

1. Client sends request to proxy
2. Proxy routes to available prefill server
3. Prefill server processes prompt, stores KV cache
4. Proxy routes decode phase to decode server
5. Decode server loads KV cache, generates tokens
6. Response streamed back through proxy

## Performance Modeling

KV-Bench calculates realistic timing based on:

### Prefill Timing
```
prefill_time = (num_tokens * model_flops) / (gpu_tflops * efficiency)
```

### Decode Timing
```
decode_time = kv_cache_size / (hbm_bandwidth * efficiency)
```

### Cache Transfer Timing
```
transfer_time = data_size / backend_bandwidth
```

## Configuration Hierarchy

Configuration follows this precedence (highest to lowest):

1. Command-line arguments
2. Environment variables (`KVBENCH_*`)
3. YAML configuration file
4. Default values

## Metrics and Monitoring

KV-Bench exposes metrics via:

- `/metrics` endpoint (JSON format)
- Prometheus-compatible metrics (optional)
- Per-request timing headers

Key metrics:
- `requests_total`: Total requests processed
- `cache_hits`: Number of cache hits
- `cache_misses`: Number of cache misses
- `prefill_time_seconds`: Prefill latency histogram
- `decode_time_seconds`: Decode latency histogram
- `ttft_seconds`: Time to first token histogram
