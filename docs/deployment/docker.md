# Docker Deployment

## Build the image

The image includes the LMCache stack (CPU-only; the torch dependency makes
it a large image, ~2.5 GB):

```bash
docker build -t kvbench:latest .
```

## Single node

`deployment/docker/docker-compose.yml` runs a combined server with
LMCache's CPU + disk tiers. The disk tier lives on a named volume — bind
it to a host NVMe mount to put a real device under test:

```bash
cd deployment/docker
docker compose up -d
curl http://localhost:8000/health
```

Storage is configured through LMCache's `LMCACHE_*` environment variables
in the compose file (or mount an LMCache config file and set
`KVBENCH_KV__LMCACHE_CONFIG_FILE`):

```yaml
environment:
  LMCACHE_CHUNK_SIZE: "256"
  LMCACHE_MAX_LOCAL_CPU_SIZE: "4"
  LMCACHE_LOCAL_DISK: "file:///var/lib/lmcache/"
  LMCACHE_MAX_LOCAL_DISK_SIZE: "20"
volumes:
  - /mnt/nvme-under-test:/var/lib/lmcache
```

## Disaggregated topology

`deployment/docker/docker-compose.distributed.yml` runs the full
disaggregated shape: a proxy in front of two prefill and two decode
servers, all sharing a Redis remote tier through LMCache. Prefill writes
KV to Redis via LMCache; decode reads it back — the shared-storage path
under test:

```bash
cd deployment/docker
docker compose -f docker-compose.distributed.yml up -d

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 10}'
```

Swap the `redis` service and `LMCACHE_REMOTE_URL` for any storage you
want to characterize — `valkey://`, `s3://` (MinIO), `fs://` on a shared
mount — see [Storage Under Test](storage-backends.md).

## Verifying

```bash
# LMCache-written artifacts on the disk tier
docker compose exec kvbench ls /var/lib/lmcache

# Cache hit/miss counters
curl -s http://localhost:8000/metrics
```
