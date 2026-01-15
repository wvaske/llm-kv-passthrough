# LMCache Server Deployment

This guide covers deploying KV-Bench with LMCache connector support, including configuration for CPU memory, local storage, and remote shared storage.

## Overview

LMCache provides a hierarchical storage system for KV cache management:

```
┌─────────────────────────────────────────────────────────────────┐
│                     LMCache Storage Tiers                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  CPU Memory  │───▶│ Local NVMe   │───▶│ Remote Store │       │
│  │  (Hot Cache) │    │ (Warm Cache) │    │ (Cold Cache) │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│       ~1ms              ~10ms               ~100ms               │
│     High BW            Medium BW           Lower BW              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Storage Tier Configuration

### Tier 1: CPU Memory (Hot Cache)

CPU memory provides the fastest access for frequently-used KV cache chunks.

```yaml
# config.yaml
resources:
  cpu_memory_gb: 64.0          # Allocate 64 GB for KV cache
  memory_allocation: eager      # Pre-allocate memory at startup

storage:
  backend_type: memory
  max_size_bytes: 68719476736   # 64 GB limit

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**Environment Variables:**

```bash
export KVBENCH_RESOURCES__CPU_MEMORY_GB=64.0
export KVBENCH_RESOURCES__MEMORY_ALLOCATION=eager
export KVBENCH_STORAGE__BACKEND_TYPE=memory
export KVBENCH_STORAGE__MAX_SIZE_BYTES=68719476736
export KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache
```

**Sizing Guidelines:**

| Model Size | Recommended CPU Memory | Max Sequences |
|------------|----------------------|---------------|
| 7-8B | 16-32 GB | ~500 |
| 70B | 64-128 GB | ~200 |
| 405B | 256-512 GB | ~50 |

### Tier 2: Local NVMe Storage (Warm Cache)

Local NVMe provides secondary storage for overflow from CPU memory.

```yaml
# config.yaml
resources:
  cpu_memory_gb: 32.0
  nvme_storage_gb: 500.0
  nvme_path: /mnt/nvme/kvbench

storage:
  backend_type: local_disk
  local_disk_path: /mnt/nvme/kvbench/cache
  local_disk_max_size_gb: 500.0
  local_disk_shard_depth: 2
  local_disk_shard_width: 256

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**Environment Variables:**

```bash
export KVBENCH_RESOURCES__NVME_STORAGE_GB=500.0
export KVBENCH_RESOURCES__NVME_PATH=/mnt/nvme/kvbench
export KVBENCH_STORAGE__BACKEND_TYPE=local_disk
export KVBENCH_STORAGE__LOCAL_DISK_PATH=/mnt/nvme/kvbench/cache
export KVBENCH_STORAGE__LOCAL_DISK_MAX_SIZE_GB=500.0
```

**NVMe Setup:**

```bash
# Format and mount NVMe drive
sudo mkfs.ext4 /dev/nvme0n1
sudo mkdir -p /mnt/nvme
sudo mount /dev/nvme0n1 /mnt/nvme

# Add to /etc/fstab for persistence
echo '/dev/nvme0n1 /mnt/nvme ext4 defaults,noatime 0 2' | sudo tee -a /etc/fstab

# Set permissions
sudo mkdir -p /mnt/nvme/kvbench
sudo chown -R kvbench:kvbench /mnt/nvme/kvbench
```

### Tier 3: Remote Shared Storage (Cold Cache)

Remote storage enables cache sharing across multiple servers.

#### Option A: Redis (Recommended for Multi-Node)

```yaml
# config.yaml
storage:
  backend_type: redis
  redis_url: redis://redis-cluster:6379
  redis_cluster: true
  redis_prefix: lmcache
  redis_ttl: 3600            # 1 hour TTL
  redis_max_connections: 100

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**Redis Cluster Setup:**

```bash
# Deploy Redis Cluster with 3 masters + 3 replicas
docker run -d --name redis-node-1 -p 7001:7001 redis:7 \
  redis-server --port 7001 --cluster-enabled yes

# Repeat for nodes 2-6...

# Create cluster
redis-cli --cluster create \
  192.168.1.1:7001 192.168.1.2:7002 192.168.1.3:7003 \
  192.168.1.4:7004 192.168.1.5:7005 192.168.1.6:7006 \
  --cluster-replicas 1
```

#### Option B: NFS (For On-Premise)

```yaml
# config.yaml
storage:
  backend_type: nfs
  filesystem_path: /mnt/nfs/kvbench

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**NFS Mount:**

```bash
# Mount NFS share
sudo mount -t nfs nfs-server:/exports/kvbench /mnt/nfs/kvbench

# Add to /etc/fstab
echo 'nfs-server:/exports/kvbench /mnt/nfs/kvbench nfs defaults,_netdev 0 0' | sudo tee -a /etc/fstab
```

#### Option C: S3/MinIO (For Cloud)

```yaml
# config.yaml
storage:
  backend_type: s3
  s3_endpoint: https://s3.amazonaws.com  # Or MinIO endpoint
  s3_bucket: kvbench-cache
  s3_prefix: lmcache/
  s3_region: us-east-1

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

#### Option D: Weka (For HPC)

```yaml
# config.yaml
storage:
  backend_type: weka
  filesystem_path: /mnt/weka/kvbench
  weka_client_path: /opt/weka

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

## Multi-Tier Configuration

For production deployments, configure multiple storage tiers:

```yaml
# config.yaml - Full multi-tier setup
instance_id: lmcache-server-01

server:
  host: 0.0.0.0
  port: 8000
  server_type: combined
  model_profile: llama-3.1-70b
  workers: 4

gpu:
  gpu_profile: H100_SXM
  efficiency_factor: 0.7
  tp_size: 4

# Tier 1: CPU Memory (Hot)
resources:
  cpu_memory_gb: 128.0
  memory_allocation: eager
  nvme_storage_gb: 1000.0
  nvme_path: /mnt/nvme/kvbench

# Tier 2 & 3: Local + Remote Storage
storage:
  backend_type: redis            # Remote tier
  redis_url: redis://redis:6379
  redis_cluster: false

  # Local disk cache settings
  local_disk_path: /mnt/nvme/kvbench/cache
  local_disk_max_size_gb: 500.0

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
  lmcache_model_name: llama-3.1-70b
  lmcache_world_size: 4
  lmcache_worker_id: 0           # Set per-worker

metrics:
  enabled: true
  prometheus_port: 9090
```

## Distributed LMCache Deployment

### Architecture

```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    │   (HAProxy/Nginx)   │
                    └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌────────────┐  ┌────────────┐  ┌────────────┐
       │ LMCache    │  │ LMCache    │  │ LMCache    │
       │ Server 1   │  │ Server 2   │  │ Server 3   │
       │ Worker 0   │  │ Worker 1   │  │ Worker 2   │
       └────────────┘  └────────────┘  └────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                    ┌─────────────────────┐
                    │   Redis Cluster     │
                    │  (Shared KV Cache)  │
                    └─────────────────────┘
```

### Server Configuration

**Server 1 (Worker 0):**

```yaml
# server-1.yaml
instance_id: lmcache-01
connector:
  connector_type: lmcache
  lmcache_world_size: 3
  lmcache_worker_id: 0
```

**Server 2 (Worker 1):**

```yaml
# server-2.yaml
instance_id: lmcache-02
connector:
  connector_type: lmcache
  lmcache_world_size: 3
  lmcache_worker_id: 1
```

**Server 3 (Worker 2):**

```yaml
# server-3.yaml
instance_id: lmcache-03
connector:
  connector_type: lmcache
  lmcache_world_size: 3
  lmcache_worker_id: 2
```

## Docker Deployment

### Single Server

```bash
docker run -d \
  --name lmcache-server \
  -p 8000:8000 \
  -v /mnt/nvme:/data/nvme \
  -e KVBENCH_RESOURCES__CPU_MEMORY_GB=64 \
  -e KVBENCH_STORAGE__BACKEND_TYPE=local_disk \
  -e KVBENCH_STORAGE__LOCAL_DISK_PATH=/data/nvme/cache \
  -e KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache \
  kvbench:latest
```

### Docker Compose (Multi-Tier)

```yaml
# docker-compose.lmcache.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  lmcache-server:
    image: kvbench:latest
    ports:
      - "8000:8000"
      - "9090:9090"
    volumes:
      - nvme_cache:/data/nvme
    environment:
      KVBENCH_INSTANCE_ID: lmcache-01
      KVBENCH_RESOURCES__CPU_MEMORY_GB: "64"
      KVBENCH_RESOURCES__NVME_STORAGE_GB: "500"
      KVBENCH_STORAGE__BACKEND_TYPE: redis
      KVBENCH_STORAGE__REDIS_URL: redis://redis:6379
      KVBENCH_STORAGE__LOCAL_DISK_PATH: /data/nvme/cache
      KVBENCH_CONNECTOR__CONNECTOR_TYPE: lmcache
      KVBENCH_CONNECTOR__LMCACHE_CHUNK_SIZE: "256"
    depends_on:
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  redis_data:
  nvme_cache:
```

## Ansible Deployment

See the [Ansible LMCache Playbook](../ansible/playbooks/deploy-lmcache.yml) for automated deployment.

### Quick Start

```bash
# Deploy LMCache servers with Redis backend
ansible-playbook -i inventory/lmcache.yml playbooks/deploy-lmcache.yml

# Deploy with custom storage settings
ansible-playbook -i inventory/lmcache.yml playbooks/deploy-lmcache.yml \
  -e "kvbench_cpu_memory_gb=128" \
  -e "kvbench_nvme_storage_gb=1000"
```

## Performance Tuning

### Chunk Size Optimization

| Workload | Recommended Chunk Size | Rationale |
|----------|----------------------|-----------|
| Short prompts (<512 tokens) | 64-128 | Fine-grained caching |
| General chat | 256 | Balanced |
| Long documents | 512-1024 | Reduce overhead |
| RAG applications | 256 | Match retrieval chunks |

### Memory Allocation Strategy

| Strategy | Use Case | Startup Time | Memory Efficiency |
|----------|----------|--------------|-------------------|
| `lazy` | Development | Fast | Lower |
| `eager` | Production | Slower | Higher |

### Connection Pool Sizing

```yaml
# For Redis backend
storage:
  redis_max_connections: 100    # Adjust based on workers * concurrency
  redis_socket_timeout: 5.0
  redis_socket_connect_timeout: 2.0
```

## Monitoring

### Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| `cache_hit_rate` | Chunk cache hit percentage | >80% |
| `storage_used_bytes` | Current storage utilization | <90% capacity |
| `avg_store_latency_ms` | Average write latency | <10ms (memory), <50ms (disk) |
| `avg_load_latency_ms` | Average read latency | <5ms (memory), <20ms (disk) |

### Prometheus Queries

```promql
# Cache hit rate
rate(kvbench_cache_hits_total[5m]) / rate(kvbench_cache_requests_total[5m])

# Storage utilization
kvbench_storage_used_bytes / kvbench_storage_capacity_bytes

# Request latency P99
histogram_quantile(0.99, rate(kvbench_request_duration_seconds_bucket[5m]))
```

## Troubleshooting

### High Cache Miss Rate

1. Check chunk size alignment with prompt patterns
2. Verify sufficient memory allocation
3. Ensure Redis connectivity for distributed deployments

### Slow Storage Performance

1. Check NVMe health: `nvme smart-log /dev/nvme0n1`
2. Verify Redis latency: `redis-cli --latency`
3. Check network bandwidth for remote storage

### Memory Pressure

```bash
# Monitor memory usage
watch -n 1 'free -h && curl -s localhost:8000/metrics | grep memory'

# Adjust allocation
export KVBENCH_RESOURCES__CPU_MEMORY_GB=96
```
