# Docker Deployment

See [docs/deployment/docker.md](../../docs/deployment/docker.md) for the
full guide.

- `docker-compose.yml` — single combined server; LMCache CPU + disk tiers
  (bind `/var/lib/lmcache` to a host NVMe path to test a real device)
- `docker-compose.distributed.yml` — proxy + 2 prefill + 2 decode servers
  sharing a Redis remote tier through LMCache

```bash
docker compose up -d                                       # single node
docker compose -f docker-compose.distributed.yml up -d     # disaggregated
curl http://localhost:8000/health
```

Storage is configured via `LMCACHE_*` environment variables in the
compose files — swap `LMCACHE_REMOTE_URL` to point at the storage you
want to characterize.
