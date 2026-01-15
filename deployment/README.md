# KV-Bench Deployment

This directory contains deployment configurations for KV-Bench, a system designed to characterize storage workload patterns for shared KV caches in LLM serving.

## Deployment Options

| Method | Use Case | Directory |
|--------|----------|-----------|
| Docker | Containerized deployment | `docker/` |
| Ansible | Bare-metal automation | `ansible/` |

## Storage Backend Options

KV-Bench supports multiple storage backends for characterizing I/O patterns:

| Backend | Docker Compose File | Ansible Inventory | Use Case |
|---------|--------------------|--------------------|----------|
| **NFS** | `docker-compose.nfs.yml` | `inventory/example.yml` | On-premise shared storage |
| **Ceph** | - | `inventory/ceph.yml` | Distributed object storage |
| **MinIO/S3** | `docker-compose.minio.yml` | `inventory/minio.yml` | Cloud-native object storage |
| **Redis** | `docker-compose.lmcache.yml` | `inventory/lmcache.yml` | In-memory caching |

## Quick Start

### Docker - NFS Storage (Recommended for Workload Characterization)

```bash
cd docker
# Edit docker-compose.nfs.yml to set your NFS server address
docker-compose -f docker-compose.nfs.yml up -d
```

### Docker - MinIO Storage

```bash
cd docker
docker-compose -f docker-compose.minio.yml up -d

# With monitoring
docker-compose -f docker-compose.minio.yml --profile monitoring up -d
```

### Docker - LMCache with Redis

```bash
cd docker
docker-compose -f docker-compose.lmcache.yml up -d
```

### Ansible - NFS Deployment

```bash
cd ansible
cp inventory/example.yml inventory/production.yml
# Edit inventory/production.yml with your NFS server details
ansible-playbook -i inventory/production.yml playbooks/deploy-lmcache.yml
```

### Ansible - Ceph Deployment

```bash
cd ansible
cp inventory/ceph.yml inventory/production.yml
# Edit inventory/production.yml
ansible-playbook -i inventory/production.yml playbooks/deploy-lmcache.yml
```

### Ansible - MinIO Deployment

```bash
cd ansible
cp inventory/minio.yml inventory/production.yml
# Edit inventory/production.yml
ansible-playbook -i inventory/production.yml playbooks/deploy-lmcache.yml
```

## Deployment Architectures

### Single Server (Development)
- 1 KV-Bench server (combined mode)
- In-memory or local disk storage

### Single Server with Shared Storage
- 1 KV-Bench server
- NFS/Ceph/MinIO for storage workload characterization

### Distributed (Production)
- 2+ Prefill servers
- 2+ Decode servers
- 1 Proxy/Load balancer
- Shared storage (NFS/Ceph/MinIO) for KV cache

## Storage Workload Characterization

The primary purpose of KV-Bench is to characterize storage I/O patterns. Key metrics:

| Metric | Description |
|--------|-------------|
| Write IOPS | Cache store operations per second |
| Read IOPS | Cache load operations per second |
| Throughput | MB/s for reads and writes |
| Latency | P50, P95, P99 operation latencies |
| Object Size | Distribution of KV cache chunk sizes |

### Running Benchmarks

```bash
# Start server
docker-compose -f docker-compose.nfs.yml up -d

# Run workload
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --concurrency 10 \
  --duration 300

# Collect metrics
curl http://localhost:8000/metrics > metrics.txt
```

## Configuration

See the documentation for detailed configuration:

- [Configuration Guide](../docs/getting-started/configuration.md)
- [Storage Backends](../docs/deployment/storage-backends.md)
- [LMCache Deployment](../docs/deployment/lmcache.md)

## Monitoring

KV-Bench exposes metrics at `/metrics` endpoint. Integrate with:
- Prometheus (configurations included)
- Grafana
- Custom monitoring solutions

Start monitoring stack:

```bash
docker-compose -f docker-compose.minio.yml --profile monitoring up -d
```
