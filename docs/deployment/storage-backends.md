# Storage Backends for KV Cache

KV-Bench supports multiple storage backends for characterizing storage workload patterns in LLM serving systems. This guide covers deployment and configuration for each backend.

## Overview

The primary purpose of KV-Bench is to characterize storage I/O patterns when using shared KV caches for LLM inference. Understanding these patterns helps:

- Size storage systems appropriately
- Choose the right storage technology
- Optimize storage configurations
- Predict performance under load

### Storage Backend Comparison

| Backend | Use Case | Latency | Throughput | Shared Access |
|---------|----------|---------|------------|---------------|
| **NFS** | On-premise shared storage | Medium | Medium-High | Yes |
| **Ceph** | Distributed object storage | Medium | High | Yes |
| **MinIO/S3** | Cloud-native object storage | Medium-High | High | Yes |
| **Redis** | In-memory caching | Low | Medium | Yes |
| **Weka** | High-performance parallel FS | Low | Very High | Yes |
| **Local Disk** | Single-node caching | Very Low | High | No |

---

## NFS Storage

NFS (Network File System) is ideal for on-premise deployments where you need shared storage with POSIX semantics.

### NFS Server Setup

```bash
# Install NFS server (Ubuntu/Debian)
sudo apt-get install nfs-kernel-server

# Create export directory
sudo mkdir -p /exports/kvbench
sudo chown nobody:nogroup /exports/kvbench
sudo chmod 755 /exports/kvbench

# Configure exports
echo '/exports/kvbench *(rw,sync,no_subtree_check,no_root_squash)' | sudo tee -a /etc/exports

# Apply and start
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

### NFS Client Configuration

```yaml
# config.yaml
storage:
  backend_type: nfs
  filesystem_path: /mnt/nfs/kvbench

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**Mount Options for Performance:**

```bash
# High-performance mount
sudo mount -t nfs -o rw,sync,hard,intr,rsize=1048576,wsize=1048576 \
  nfs-server:/exports/kvbench /mnt/nfs/kvbench

# Add to /etc/fstab
echo 'nfs-server:/exports/kvbench /mnt/nfs/kvbench nfs rw,sync,hard,intr,rsize=1048576,wsize=1048576,_netdev 0 0' \
  | sudo tee -a /etc/fstab
```

### Ansible Inventory for NFS

```yaml
# inventory/nfs-deployment.yml
all:
  vars:
    kvbench_storage: nfs
    kvbench_filesystem_path: /mnt/nfs/kvbench
    lmcache_nfs_server: nfs-server.example.com
    lmcache_nfs_export: /exports/kvbench
    lmcache_nfs_options: "rw,sync,hard,intr,rsize=1048576,wsize=1048576"

  children:
    nfs_servers:
      hosts:
        nfs-server:
          ansible_host: 10.0.0.5
          nfs_export_path: /exports/kvbench
          nfs_export_options: "*(rw,sync,no_subtree_check)"

    lmcache_servers:
      hosts:
        worker-1:
          ansible_host: 10.0.0.11
```

### Docker Compose with NFS

```yaml
version: '3.8'

services:
  lmcache-server:
    image: kvbench:latest
    volumes:
      - type: volume
        source: nfs_cache
        target: /data/cache
        volume:
          nocopy: true
    environment:
      KVBENCH_STORAGE__BACKEND_TYPE: nfs
      KVBENCH_STORAGE__FILESYSTEM_PATH: /data/cache

volumes:
  nfs_cache:
    driver: local
    driver_opts:
      type: nfs
      o: addr=nfs-server.example.com,rw,sync
      device: ":/exports/kvbench"
```

### NFS Performance Tuning

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `rsize` | 1048576 | Read buffer size (1MB) |
| `wsize` | 1048576 | Write buffer size (1MB) |
| `hard` | Yes | Retry indefinitely on failure |
| `intr` | Yes | Allow interrupt of hung operations |
| `sync` | Yes | Synchronous writes for consistency |
| `noatime` | Yes | Don't update access times |

---

## Ceph Storage

Ceph provides distributed object storage with high availability and scalability.

### Ceph Cluster Prerequisites

```bash
# Install Ceph client
sudo apt-get install ceph-common

# Copy Ceph configuration from cluster
sudo scp admin@ceph-mon:/etc/ceph/ceph.conf /etc/ceph/
sudo scp admin@ceph-mon:/etc/ceph/ceph.client.admin.keyring /etc/ceph/
```

### Ceph RADOS Configuration

```yaml
# config.yaml
storage:
  backend_type: ceph
  ceph_pool: kvbench-pool
  ceph_conf_path: /etc/ceph/ceph.conf
  ceph_keyring_path: /etc/ceph/ceph.client.admin.keyring
  ceph_client_name: client.admin
  ceph_namespace: lmcache

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**Environment Variables:**

```bash
export KVBENCH_STORAGE__BACKEND_TYPE=ceph
export KVBENCH_STORAGE__CEPH_POOL=kvbench-pool
export KVBENCH_STORAGE__CEPH_CONF_PATH=/etc/ceph/ceph.conf
export KVBENCH_STORAGE__CEPH_NAMESPACE=lmcache
```

### Ceph Pool Setup

```bash
# Create pool for KV cache (on Ceph admin node)
ceph osd pool create kvbench-pool 128 128

# Set pool application
ceph osd pool application enable kvbench-pool rbd

# Optional: Set replication factor
ceph osd pool set kvbench-pool size 3

# Optional: Enable compression
ceph osd pool set kvbench-pool compression_mode aggressive
ceph osd pool set kvbench-pool compression_algorithm snappy
```

### Ansible Inventory for Ceph

```yaml
# inventory/ceph-deployment.yml
all:
  vars:
    kvbench_storage: ceph
    kvbench_ceph_pool: kvbench-pool
    kvbench_ceph_conf_path: /etc/ceph/ceph.conf
    kvbench_ceph_keyring_path: /etc/ceph/ceph.client.kvbench.keyring
    kvbench_ceph_client_name: client.kvbench
    kvbench_ceph_namespace: lmcache

  children:
    ceph_monitors:
      hosts:
        ceph-mon-1:
          ansible_host: 10.0.0.50

    lmcache_servers:
      hosts:
        worker-1:
          ansible_host: 10.0.0.11
          kvbench_lmcache_worker_id: 0
        worker-2:
          ansible_host: 10.0.0.12
          kvbench_lmcache_worker_id: 1
```

### Docker Compose with Ceph

```yaml
version: '3.8'

services:
  lmcache-server:
    image: kvbench:latest
    volumes:
      - /etc/ceph:/etc/ceph:ro
    environment:
      KVBENCH_STORAGE__BACKEND_TYPE: ceph
      KVBENCH_STORAGE__CEPH_POOL: kvbench-pool
      KVBENCH_STORAGE__CEPH_CONF_PATH: /etc/ceph/ceph.conf
      KVBENCH_STORAGE__CEPH_NAMESPACE: lmcache
      KVBENCH_CONNECTOR__CONNECTOR_TYPE: lmcache
```

### Ceph Performance Tuning

```bash
# Increase client cache
ceph config set client rbd_cache_size 134217728  # 128MB

# Enable read-ahead
ceph config set client rbd_readahead_max_bytes 4194304  # 4MB

# Optimize for small objects (KV cache chunks)
ceph config set osd bluestore_min_alloc_size_ssd 4096
```

---

## MinIO / S3 Storage

MinIO provides S3-compatible object storage for cloud-native deployments.

### MinIO Server Setup

```bash
# Single-node MinIO
docker run -d \
  --name minio \
  -p 9000:9000 \
  -p 9001:9001 \
  -v minio_data:/data \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"

# Create bucket
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/kvbench-cache
```

### MinIO Distributed Setup

```yaml
# docker-compose.minio-cluster.yml
version: '3.8'

services:
  minio1:
    image: minio/minio
    command: server http://minio{1...4}/data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio1_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  minio2:
    image: minio/minio
    command: server http://minio{1...4}/data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio2_data:/data

  minio3:
    image: minio/minio
    command: server http://minio{1...4}/data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio3_data:/data

  minio4:
    image: minio/minio
    command: server http://minio{1...4}/data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio4_data:/data

volumes:
  minio1_data:
  minio2_data:
  minio3_data:
  minio4_data:
```

### MinIO/S3 Configuration

```yaml
# config.yaml
storage:
  backend_type: s3
  s3_endpoint: http://minio:9000
  s3_bucket: kvbench-cache
  s3_prefix: lmcache/
  s3_access_key: minioadmin
  s3_secret_key: minioadmin
  s3_region: us-east-1
  s3_use_ssl: false

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

**Environment Variables:**

```bash
export KVBENCH_STORAGE__BACKEND_TYPE=s3
export KVBENCH_STORAGE__S3_ENDPOINT=http://minio:9000
export KVBENCH_STORAGE__S3_BUCKET=kvbench-cache
export KVBENCH_STORAGE__S3_PREFIX=lmcache/
export KVBENCH_STORAGE__S3_ACCESS_KEY=minioadmin
export KVBENCH_STORAGE__S3_SECRET_KEY=minioadmin
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

### AWS S3 Configuration

```yaml
# config.yaml for AWS S3
storage:
  backend_type: s3
  s3_endpoint: https://s3.amazonaws.com
  s3_bucket: my-kvbench-cache
  s3_prefix: production/lmcache/
  s3_region: us-west-2
  s3_use_ssl: true
  # Credentials via IAM role or environment variables

connector:
  connector_type: lmcache
  lmcache_chunk_size: 256
```

### Ansible Inventory for MinIO

```yaml
# inventory/minio-deployment.yml
all:
  vars:
    kvbench_storage: s3
    kvbench_s3_endpoint: http://minio.example.com:9000
    kvbench_s3_bucket: kvbench-cache
    kvbench_s3_prefix: lmcache/
    kvbench_s3_access_key: "{{ vault_minio_access_key }}"
    kvbench_s3_secret_key: "{{ vault_minio_secret_key }}"
    kvbench_s3_region: us-east-1

  children:
    minio_servers:
      hosts:
        minio-1:
          ansible_host: 10.0.0.40
        minio-2:
          ansible_host: 10.0.0.41
        minio-3:
          ansible_host: 10.0.0.42
        minio-4:
          ansible_host: 10.0.0.43

    lmcache_servers:
      hosts:
        worker-1:
          ansible_host: 10.0.0.11
```

### Docker Compose with MinIO

```yaml
version: '3.8'

services:
  minio:
    image: minio/minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio-setup:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minioadmin minioadmin;
      mc mb local/kvbench-cache --ignore-existing;
      exit 0;
      "

  lmcache-server:
    image: kvbench:latest
    ports:
      - "8000:8000"
    environment:
      KVBENCH_STORAGE__BACKEND_TYPE: s3
      KVBENCH_STORAGE__S3_ENDPOINT: http://minio:9000
      KVBENCH_STORAGE__S3_BUCKET: kvbench-cache
      KVBENCH_STORAGE__S3_PREFIX: lmcache/
      KVBENCH_STORAGE__S3_ACCESS_KEY: minioadmin
      KVBENCH_STORAGE__S3_SECRET_KEY: minioadmin
      KVBENCH_CONNECTOR__CONNECTOR_TYPE: lmcache
    depends_on:
      minio-setup:
        condition: service_completed_successfully

volumes:
  minio_data:
```

### MinIO Performance Tuning

```bash
# Enable batch operations
mc admin config set local api requests_max=1000
mc admin config set local api requests_deadline=10s

# Optimize for small objects
mc admin config set local storage_class standard=EC:2

# Enable compression (for text-heavy workloads)
mc admin config set local compression enable=on
mc admin config set local compression extensions=.bin,.dat
```

---

## Storage Workload Characterization

### Metrics to Collect

| Metric | Description | Prometheus Query |
|--------|-------------|------------------|
| Write IOPS | Writes per second | `rate(kvbench_storage_writes_total[5m])` |
| Read IOPS | Reads per second | `rate(kvbench_storage_reads_total[5m])` |
| Write Throughput | MB/s written | `rate(kvbench_storage_write_bytes[5m])` |
| Read Throughput | MB/s read | `rate(kvbench_storage_read_bytes[5m])` |
| Latency P99 | 99th percentile latency | `histogram_quantile(0.99, rate(kvbench_storage_latency_bucket[5m]))` |
| Object Size | Average object size | `kvbench_storage_write_bytes / kvbench_storage_writes_total` |

### Benchmark Script

```bash
#!/bin/bash
# benchmark-storage.sh - Compare storage backend performance

BACKENDS=("nfs" "ceph" "s3")
RESULTS_DIR="./benchmark-results"
DURATION=300  # 5 minutes per test

mkdir -p $RESULTS_DIR

for backend in "${BACKENDS[@]}"; do
    echo "Testing $backend..."

    # Start server with backend
    export KVBENCH_STORAGE__BACKEND_TYPE=$backend
    kvbench serve &
    SERVER_PID=$!
    sleep 5

    # Run benchmark
    genai-perf \
        --endpoint http://localhost:8000/v1/chat/completions \
        --model llama-3.1-8b \
        --concurrency 10 \
        --duration $DURATION \
        --output $RESULTS_DIR/${backend}_results.json

    # Collect metrics
    curl -s http://localhost:8000/metrics > $RESULTS_DIR/${backend}_metrics.txt

    kill $SERVER_PID
    sleep 2
done

echo "Results saved to $RESULTS_DIR"
```

### Expected I/O Patterns

| Workload | Read/Write Ratio | Object Size | Access Pattern |
|----------|------------------|-------------|----------------|
| Prefill (cold) | 10:90 | 64KB-1MB | Sequential |
| Prefill (warm) | 80:20 | 64KB-1MB | Random |
| Decode | 95:5 | 64KB-256KB | Random |
| Mixed | 60:40 | 64KB-1MB | Mixed |

---

## Choosing a Storage Backend

### Decision Matrix

| Requirement | Recommended Backend |
|-------------|---------------------|
| Lowest latency | Redis or Local Disk |
| Highest throughput | Weka or NFS (tuned) |
| Cloud deployment | MinIO / S3 |
| On-premise distributed | Ceph |
| Simplest setup | NFS |
| Cost-effective | NFS or MinIO |
| Enterprise support | Weka or Ceph |

### Sizing Guidelines

| Model Size | Cache Size (per request) | Recommended Storage |
|------------|-------------------------|---------------------|
| 7-8B | ~2GB | 500GB - 1TB |
| 70B | ~20GB | 2TB - 5TB |
| 405B | ~100GB | 10TB - 20TB |

For multi-tenant deployments, multiply by expected concurrent users and cache hit rate.
