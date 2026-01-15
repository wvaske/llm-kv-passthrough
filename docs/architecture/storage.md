# Storage Backends

KV-Bench supports multiple storage backends for KV cache persistence. Choose based on your deployment requirements.

## Backend Comparison

| Backend | Latency | Throughput | Scalability | Persistence | Best For |
|---------|---------|------------|-------------|-------------|----------|
| Memory | ~1μs | Very High | Single Node | No | Testing |
| Local Disk | ~100μs | High | Single Node | Yes | Single-node |
| Redis | ~1ms | High | Multi-node | Optional | Shared cache |
| S3 | ~50ms | Medium | Unlimited | Yes | Cloud |
| NFS | ~5ms | Medium | Multi-node | Yes | On-premise |
| Weka | ~1ms | Very High | Multi-node | Yes | HPC |
| Mooncake | ~100μs | Very High | Multi-node | No | Disaggregated |

## Memory Backend

In-memory storage with LRU eviction. Fastest option but no persistence.

```yaml
storage:
  backend_type: memory
  # Maximum memory usage
  max_size_bytes: 10737418240  # 10 GB
```

```bash
kvbench serve --storage memory
```

**Features:**
- LRU eviction when capacity exceeded
- Thread-safe operations
- Zero serialization overhead

## Local Disk Backend

NVMe-optimized local storage with memory-mapped files.

```yaml
storage:
  backend_type: local_disk
  local_disk_path: /var/lib/kvbench/cache
  local_disk_max_size_gb: 100.0
```

```bash
export KVBENCH_STORAGE__LOCAL_DISK_PATH=/nvme/kvbench
kvbench serve --storage local_disk
```

**Features:**
- Memory-mapped I/O for performance
- Automatic cleanup on capacity limits
- Persistent across restarts

## Redis Backend

Distributed caching with Redis or Redis Cluster.

```yaml
storage:
  backend_type: redis
  redis_url: redis://localhost:6379
  redis_cluster: false
  redis_prefix: kvbench
  redis_ttl: 3600  # 1 hour
```

```bash
# Single Redis
export KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
kvbench serve --storage redis

# Redis Cluster
export KVBENCH_STORAGE__REDIS_URL=redis://redis-1:6379,redis-2:6379,redis-3:6379
export KVBENCH_STORAGE__REDIS_CLUSTER=true
kvbench serve --storage redis
```

**Features:**
- Shared cache across multiple servers
- Optional TTL for automatic expiration
- Cluster mode for horizontal scaling

## S3 Backend

Object storage compatible with S3, MinIO, and other S3-compatible services.

```yaml
storage:
  backend_type: s3
  s3_endpoint: https://s3.amazonaws.com
  s3_bucket: kvbench-cache
  s3_prefix: kv-cache/
  s3_region: us-east-1
```

```bash
# AWS S3
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export KVBENCH_STORAGE__S3_BUCKET=kvbench-cache
kvbench serve --storage s3

# MinIO
export KVBENCH_STORAGE__S3_ENDPOINT=http://minio:9000
export KVBENCH_STORAGE__S3_BUCKET=kvbench
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
kvbench serve --storage s3
```

**Features:**
- Unlimited capacity
- Multi-region support
- Cost-effective for large caches

## NFS Backend

Network filesystem for shared storage across nodes.

```yaml
storage:
  backend_type: nfs
  filesystem_path: /mnt/nfs/kvbench
```

```bash
# Mount NFS first
mount -t nfs nfs-server:/exports/kvbench /mnt/nfs/kvbench

export KVBENCH_STORAGE__FILESYSTEM_PATH=/mnt/nfs/kvbench
kvbench serve --storage nfs
```

**Features:**
- Shared across all nodes
- POSIX filesystem semantics
- Works with existing NFS infrastructure

## Weka Backend

High-performance distributed filesystem optimized for AI workloads.

```yaml
storage:
  backend_type: weka
  filesystem_path: /mnt/weka/kvbench
  weka_mount_point: /mnt/weka
```

```bash
export KVBENCH_STORAGE__FILESYSTEM_PATH=/mnt/weka/kvbench
kvbench serve --storage weka
```

**Features:**
- Sub-millisecond latency
- Parallel I/O support
- Optimized for GPU workloads

## Mooncake Backend

Integration with Mooncake transfer engine for disaggregated inference.

```yaml
storage:
  backend_type: mooncake
  mooncake_local_hostname: node-1
  mooncake_metadata_server: etcd://etcd:2379
  mooncake_protocol: rdma  # or tcp
```

```bash
export KVBENCH_STORAGE__MOONCAKE_LOCAL_HOSTNAME=$(hostname)
export KVBENCH_STORAGE__MOONCAKE_METADATA_SERVER=etcd://etcd:2379
kvbench serve --storage mooncake
```

**Features:**
- Zero-copy transfers
- RDMA support for ultra-low latency
- Designed for disaggregated serving

## Tiered Storage

Combine multiple backends for tiered caching:

```yaml
storage:
  backend_type: tiered
  tiered_backends:
    - type: memory
      max_size_bytes: 1073741824  # 1 GB hot tier
    - type: local_disk
      path: /nvme/cache  # 100 GB warm tier
    - type: s3
      bucket: kvbench-cold  # Unlimited cold tier
```

## Performance Tuning

### Memory Backend
- Increase `max_size_bytes` for better hit rates
- Monitor eviction rate via metrics

### Redis Backend
- Use Redis Cluster for >100GB caches
- Enable persistence only if needed
- Tune `maxmemory-policy` to `allkeys-lru`

### S3 Backend
- Use local SSD cache for frequently accessed keys
- Enable multipart uploads for large objects
- Use regional endpoints for lower latency

### Filesystem Backends
- Use NVMe SSDs for local disk
- Mount NFS/Weka with `noatime` option
- Ensure sufficient I/O threads
