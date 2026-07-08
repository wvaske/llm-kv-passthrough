# Docker Deployment

> **⚠️ Outdated**: This guide predates the real-LMCache integration and
> references removed KV-Bench storage settings (`storage:`, `connector:`,
> `KVBENCH_STORAGE_*`). Storage is now configured entirely through
> LMCache's own application configuration — see
> [Storage](../architecture/storage.md) and
> [LMCache Integration](../benchmarking/lmcache.md). The infrastructure
> steps below (installing Redis, mounting NFS, etc.) remain useful;
> point LMCache's `remote_url`/`local_disk` at the result.

KV-Bench provides Docker images for easy deployment in containerized environments.

## Quick Start

### Single Container

```bash
# Pull the image
docker pull kvbench:latest

# Run with default settings
docker run -p 8000:8000 kvbench:latest

# Run with custom configuration
docker run -p 8000:8000 \
  -e KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-70b \
  -e KVBENCH_GPU__GPU_PROFILE=H100_SXM \
  kvbench:latest
```

### Build from Source

```bash
cd /path/to/kvbench
docker build -t kvbench:local .
```

## Docker Compose

### Simple Deployment

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  kvbench:
    image: kvbench:latest
    ports:
      - "8000:8000"
    environment:
      - KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b
      - KVBENCH_GPU__GPU_PROFILE=H100_SXM
      - KVBENCH_STORAGE__BACKEND_TYPE=memory
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
```

### With Redis Storage

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes

  kvbench:
    image: kvbench:latest
    ports:
      - "8000:8000"
    environment:
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    depends_on:
      - redis

volumes:
  redis-data:
```

## Distributed Deployment

For production deployments with prefill/decode separation:

```yaml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  prefill-1:
    image: kvbench:latest
    environment:
      - KVBENCH_INSTANCE_ID=prefill-1
      - KVBENCH_SERVER__SERVER_TYPE=prefill
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  prefill-2:
    image: kvbench:latest
    environment:
      - KVBENCH_INSTANCE_ID=prefill-2
      - KVBENCH_SERVER__SERVER_TYPE=prefill
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  decode-1:
    image: kvbench:latest
    environment:
      - KVBENCH_INSTANCE_ID=decode-1
      - KVBENCH_SERVER__SERVER_TYPE=decode
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  decode-2:
    image: kvbench:latest
    environment:
      - KVBENCH_INSTANCE_ID=decode-2
      - KVBENCH_SERVER__SERVER_TYPE=decode
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    depends_on:
      redis:
        condition: service_healthy

  proxy:
    image: kvbench:latest
    ports:
      - "8000:8000"
    environment:
      - KVBENCH_INSTANCE_ID=proxy
      - KVBENCH_SERVER__SERVER_TYPE=proxy
      - KVBENCH_DISTRIBUTED__PREFILL_ENDPOINTS=["http://prefill-1:8000","http://prefill-2:8000"]
      - KVBENCH_DISTRIBUTED__DECODE_ENDPOINTS=["http://decode-1:8000","http://decode-2:8000"]
    depends_on:
      - prefill-1
      - prefill-2
      - decode-1
      - decode-2

volumes:
  redis-data:

networks:
  default:
    name: kvbench-distributed
```

```bash
docker-compose -f docker-compose.distributed.yml up -d
```

## Configuration

### Environment Variables

All configuration options can be set via environment variables:

```bash
# Server configuration
KVBENCH_SERVER__HOST=0.0.0.0
KVBENCH_SERVER__PORT=8000
KVBENCH_SERVER__SERVER_TYPE=combined
KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b

# GPU emulation
KVBENCH_GPU__GPU_PROFILE=H100_SXM
KVBENCH_GPU__EFFICIENCY_FACTOR=0.7

# Storage
KVBENCH_STORAGE__BACKEND_TYPE=redis
KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
```

### Volume Mounts

For persistent storage:

```yaml
services:
  kvbench:
    volumes:
      # Config file
      - ./config.yaml:/app/config.yaml:ro
      # Local disk cache
      - kvbench-cache:/var/lib/kvbench/cache
      # Logs
      - ./logs:/var/log/kvbench
```

### Resource Limits

Set appropriate resource limits:

```yaml
services:
  kvbench:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## Health Checks

KV-Bench provides health endpoints:

```bash
# Liveness probe
curl http://localhost:8000/health

# Readiness probe (checks storage connection)
curl http://localhost:8000/ready
```

Kubernetes-style probes:

```yaml
services:
  kvbench:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
```

## Scaling

### Horizontal Scaling

Scale prefill/decode servers independently:

```bash
# Scale prefill servers
docker-compose up -d --scale prefill=4

# Scale decode servers
docker-compose up -d --scale decode=8
```

### Load Balancing

Use the built-in proxy or external load balancers:

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - kvbench-1
      - kvbench-2
```

## Monitoring

### Prometheus Integration

```yaml
services:
  kvbench:
    environment:
      - KVBENCH_METRICS__ENABLED=true
      - KVBENCH_METRICS__PROMETHEUS_PORT=9090

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
```

### Logging

Configure logging output:

```yaml
services:
  kvbench:
    environment:
      - KVBENCH_SERVER__LOG_LEVEL=info
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs kvbench

# Check health
docker inspect --format='{{.State.Health.Status}}' kvbench
```

### Connection Issues

```bash
# Test network connectivity
docker exec kvbench curl http://redis:6379

# Check DNS resolution
docker exec kvbench nslookup redis
```

### Performance Issues

```bash
# Check resource usage
docker stats kvbench

# Check metrics
curl http://localhost:8000/metrics
```
