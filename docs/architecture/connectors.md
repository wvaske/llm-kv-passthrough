# KV Cache Connectors

Connectors bridge the gap between the inference server and storage backends, providing cache key management, chunking, and protocol compatibility.

## Connector Architecture

```
┌─────────────────────────────────────────┐
│            Inference Server             │
│         (Prefill / Decode)              │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│              Connector                  │
│  - Key generation                       │
│  - Chunking / Dechunking                │
│  - Protocol translation                 │
│  - Statistics collection                │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│           Storage Backend               │
└─────────────────────────────────────────┘
```

## LMCache Connector

The LMCache connector provides compatibility with [LMCache](https://github.com/LMCache/LMCache) protocol.

### Configuration

```yaml
connector:
  connector_type: lmcache
  lmcache_chunk_size: 256  # Tokens per chunk
  lmcache_remote_url: null  # Optional remote LMCache server
```

```bash
export KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache
export KVBENCH_CONNECTOR__LMCACHE_CHUNK_SIZE=256
kvbench serve
```

### Key Format

LMCache uses a hierarchical key format:

```
{model_name}/{world_size}/{worker_id}/{chunk_hash}
```

Example:
```
llama-3.1-8b/1/0/a1b2c3d4e5f6
```

### Chunking

KV cache is stored in fixed-size chunks:

```python
# 256-token chunks
chunk_size = 256

# 1000 token sequence = 4 chunks
# Chunk 0: tokens 0-255
# Chunk 1: tokens 256-511
# Chunk 2: tokens 512-767
# Chunk 3: tokens 768-999 (partial)
```

### Prefix Caching

LMCache supports prefix caching for shared prompts:

```
Request 1: "What is the capital of France?"
  → Stores chunks for full prompt

Request 2: "What is the capital of Germany?"
  → Hits cache for "What is the capital of "
  → Only processes " Germany?" tokens
```

### Integration with Real LMCache

Connect to a real LMCache server:

```bash
# Start LMCache server
lmcache_server localhost:8080

# Connect KV-Bench
export KVBENCH_CONNECTOR__LMCACHE_REMOTE_URL=lm://localhost:8080
kvbench serve
```

## Mooncake Connector

The Mooncake connector integrates with the Mooncake transfer engine for disaggregated inference.

### Configuration

```yaml
connector:
  connector_type: mooncake
  mooncake_local_hostname: node-1
  mooncake_metadata_server: etcd://etcd:2379
```

```bash
export KVBENCH_CONNECTOR__CONNECTOR_TYPE=mooncake
export KVBENCH_CONNECTOR__MOONCAKE_LOCAL_HOSTNAME=$(hostname)
kvbench serve
```

### Transfer Modes

Mooncake supports multiple transfer protocols:

| Protocol | Latency | Use Case |
|----------|---------|----------|
| RDMA | ~10μs | Same datacenter |
| TCP | ~100μs | Cross-datacenter |

### Zero-Copy Transfers

Mooncake enables zero-copy KV cache transfers:

```
Prefill Server                    Decode Server
     │                                 │
     │  1. Compute KV cache            │
     │                                 │
     │  2. Register with Mooncake      │
     │         ─────────────────────►  │
     │                                 │
     │  3. RDMA read                   │
     │         ◄─────────────────────  │
     │                                 │
     │                          4. Decode
```

## Custom Connectors

Create custom connectors by implementing the base interface:

```python
from kvbench.connectors.base import BaseConnector, ConnectorStats

class CustomConnector(BaseConnector):
    """Custom KV cache connector."""

    def __init__(self, storage: StorageBackend, name: str = "custom"):
        super().__init__(storage=storage, name=name)
        self._stats = ConnectorStats()

    @property
    def stats(self) -> ConnectorStats:
        return self._stats

    async def store(self, key: str, num_tokens: int) -> None:
        """Store KV cache data."""
        # Generate cache data
        data = self._generate_kv_data(num_tokens)

        # Store in backend
        await self.storage.put(key, data)

        # Update stats
        self._stats.stores += 1
        self._stats.store_bytes += len(data)

    async def load(self, key: str) -> bytes | None:
        """Load KV cache data."""
        data = await self.storage.get(key)

        self._stats.loads += 1
        if data is not None:
            self._stats.hits += 1
            self._stats.load_bytes += len(data)
        else:
            self._stats.misses += 1

        return data

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.storage.exists(key)

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        return await self.storage.delete(key)

    async def close(self) -> None:
        """Cleanup resources."""
        pass
```

Register the custom connector:

```python
from kvbench.connectors import register_connector

register_connector("custom", CustomConnector)
```

## Connector Statistics

All connectors track operational statistics:

```python
@dataclass
class ConnectorStats:
    stores: int = 0          # Number of store operations
    loads: int = 0           # Number of load operations
    hits: int = 0            # Cache hits
    misses: int = 0          # Cache misses
    store_bytes: int = 0     # Total bytes stored
    load_bytes: int = 0      # Total bytes loaded

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
```

Access statistics via API:

```bash
curl http://localhost:8000/metrics
```

```json
{
  "connector": {
    "stores": 1000,
    "loads": 5000,
    "hits": 4500,
    "misses": 500,
    "hit_rate": 90.0,
    "store_bytes": 512000000,
    "load_bytes": 2304000000
  }
}
```

## Best Practices

1. **Chunk Size**: Use 256-512 tokens for optimal prefix sharing
2. **Key Design**: Include model name and version in keys
3. **Statistics**: Monitor hit rate to tune cache size
4. **Cleanup**: Implement TTL or LRU eviction for long-running deployments
