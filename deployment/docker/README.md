# KV-Bench Docker Deployment

Docker Compose configurations for deploying KV-Bench.

## Files

| File | Description |
|------|-------------|
| `docker-compose.yml` | Simple single-server deployment |
| `docker-compose.distributed.yml` | Distributed deployment with prefill/decode separation |

## Quick Start

### Simple Deployment

```bash
docker-compose up -d
```

### Distributed Deployment

```bash
docker-compose -f docker-compose.distributed.yml up -d
```

## Architecture

### Simple Mode
- Single KV-Bench container
- Optional Redis for storage

### Distributed Mode
- 2 Prefill servers
- 2 Decode servers
- 1 Proxy/Load balancer
- Redis for shared KV cache

## Configuration

All configuration via environment variables:

```bash
# Server
KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b
KVBENCH_SERVER__SERVER_TYPE=combined

# GPU emulation
KVBENCH_GPU__GPU_PROFILE=H100_SXM

# Storage
KVBENCH_STORAGE__BACKEND_TYPE=redis
KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
```

## Scaling

```bash
# Scale prefill servers
docker-compose -f docker-compose.distributed.yml up -d --scale prefill=4

# Scale decode servers
docker-compose -f docker-compose.distributed.yml up -d --scale decode=8
```

## Health Checks

```bash
# Check health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics
```
