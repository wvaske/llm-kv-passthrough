# LMCache Integration

KV-Bench provides compatibility with [LMCache](https://github.com/LMCache/LMCache) for KV cache management benchmarking.

> **Deployment Guide**: For detailed deployment instructions including multi-tier storage setup, see the [LMCache Deployment Guide](../deployment/lmcache.md).

## Overview

LMCache is a KV cache management system for LLM serving that provides:

- **Prefix caching**: Reuse KV cache for shared prompt prefixes
- **Distributed caching**: Share cache across multiple servers
- **Chunked storage**: Efficient memory management with fixed-size chunks
- **Multi-tier storage**: CPU memory → Local NVMe → Remote shared storage

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     KV-Bench Server                             │
│                  (OpenAI-compatible API)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LMCache Connector                            │
│              (Chunking, Key Management)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    Local Storage        │     │   LMCache Server        │
│  (Memory/Disk/Redis)    │     │  (Optional Remote)      │
└─────────────────────────┘     └─────────────────────────┘
```

## Configuration

### Basic Setup

```yaml
connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

```bash
export KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache
export KVBENCH_CONNECTOR__LMCACHE_CHUNK_SIZE=256
kvbench serve
```

### With Remote LMCache Server

```bash
# Start LMCache server
lmcache_server localhost:8080

# Configure KV-Bench to use remote server
export KVBENCH_CONNECTOR__LMCACHE_REMOTE_URL=lm://localhost:8080
kvbench serve
```

### Multi-Worker Setup

```yaml
connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
  lmcache_world_size: 4
  lmcache_worker_id: 0  # Set per-worker
```

## Chunking Strategy

LMCache divides KV cache into fixed-size chunks:

```
Sequence: [token_0, token_1, ..., token_999]

Chunk Size: 256 tokens

Chunks:
  - Chunk 0: tokens 0-255
  - Chunk 1: tokens 256-511
  - Chunk 2: tokens 512-767
  - Chunk 3: tokens 768-999 (partial)
```

### Chunk Size Selection

| Chunk Size | Memory Overhead | Cache Granularity | Best For |
|------------|-----------------|-------------------|----------|
| 64 | High | Fine | Short prompts |
| 256 | Medium | Medium | General use |
| 512 | Low | Coarse | Long documents |

## Prefix Caching

LMCache enables prefix caching for shared prompt patterns:

### Example: System Prompt Caching

```python
# Request 1: Cache the system prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What is Python?"}
]
# Caches: system_prompt_chunk_0, system_prompt_chunk_1, ...

# Request 2: Reuse system prompt cache
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "What is JavaScript?"}
]
# Hits: system_prompt_chunk_0, system_prompt_chunk_1, ...
# Only processes new user content
```

### Example: Few-Shot Learning

```python
# All requests share the same examples
examples = """
Q: What is 2+2?
A: 4

Q: What is 3+3?
A: 6
"""

# Request 1
prompt1 = examples + "\nQ: What is 4+4?\nA:"
# Caches example chunks

# Request 2
prompt2 = examples + "\nQ: What is 5+5?\nA:"
# Hits example chunks, only processes new question
```

## Benchmarking Cache Performance

### Cache Hit Rate Test

```bash
# Run the LMCache test script
scripts/lmcache_test.sh
```

### Manual Test

```python
import httpx
import asyncio

async def test_cache_hits():
    async with httpx.AsyncClient() as client:
        # Request 1 - cache miss
        r1 = await client.post(
            'http://localhost:8000/v1/chat/completions',
            json={
                'model': 'llama-3.1-8b',
                'messages': [{'role': 'user', 'content': 'Hello world'}],
                'max_tokens': 10
            }
        )

        # Request 2 - cache hit (same prefix)
        r2 = await client.post(
            'http://localhost:8000/v1/chat/completions',
            json={
                'model': 'llama-3.1-8b',
                'messages': [{'role': 'user', 'content': 'Hello world, how are you?'}],
                'max_tokens': 10
            }
        )

        # Check metrics
        metrics = await client.get('http://localhost:8000/metrics')
        print(metrics.json())

asyncio.run(test_cache_hits())
```

### Expected Output

```json
{
  "cache_hits": 1,
  "cache_misses": 1,
  "hit_rate": 50.0,
  "connector": {
    "stores": 2,
    "loads": 2,
    "hits": 1,
    "misses": 1
  }
}
```

## Performance Tuning

### Optimize Chunk Size

```bash
# Test different chunk sizes
for size in 64 128 256 512; do
  export KVBENCH_CONNECTOR__LMCACHE_CHUNK_SIZE=$size
  kvbench serve &
  sleep 2

  # Run benchmark
  python benchmark.py --output results_chunk_${size}.json

  kill %1
done
```

### Memory Management

```yaml
storage:
  backend_type: memory
  max_size_bytes: 10737418240  # 10 GB

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

### Distributed Caching

```yaml
storage:
  backend_type: redis
  redis_url: redis://redis-cluster:6379
  redis_cluster: true

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

## Metrics

LMCache connector exposes detailed metrics:

| Metric | Description |
|--------|-------------|
| `connector.stores` | Number of cache store operations |
| `connector.loads` | Number of cache load operations |
| `connector.hits` | Number of cache hits |
| `connector.misses` | Number of cache misses |
| `connector.hit_rate` | Cache hit rate percentage |
| `connector.store_bytes` | Total bytes stored |
| `connector.load_bytes` | Total bytes loaded |

## Comparison with Real LMCache

KV-Bench emulates LMCache behavior for benchmarking:

| Feature | Real LMCache | KV-Bench |
|---------|--------------|----------|
| Chunking | Yes | Yes |
| Prefix caching | Yes | Yes |
| Remote server | Yes | Optional |
| GPU memory | Required | Emulated |
| Actual inference | Yes | Simulated |

## Troubleshooting

### Low Hit Rate

```bash
# Check chunk alignment
# Ensure prompts share common prefixes

# Verify chunk size matches workload
export KVBENCH_CONNECTOR__LMCACHE_CHUNK_SIZE=128
```

### Memory Pressure

```bash
# Increase storage capacity
export KVBENCH_STORAGE__MAX_SIZE_BYTES=21474836480  # 20 GB

# Or use disk storage
export KVBENCH_STORAGE__BACKEND_TYPE=local_disk
```

### Connection Issues

```bash
# Test LMCache server connectivity
curl http://localhost:8080/health

# Check logs
kvbench serve --log-level debug
```
