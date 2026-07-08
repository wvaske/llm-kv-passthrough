# KV-Bench Deployment

Deployment configurations for KV-Bench, a system for characterizing
storage workload patterns for shared KV caches in LLM serving.

KV-Bench performs no storage I/O of its own — all KV cache operations go
through the real LMCache engine, and **storage is selected and tuned in
LMCache's own configuration** (config file or `LMCACHE_*` environment
variables). Deploying a benchmark therefore means: deploy the storage you
want to test, then point LMCache at it.

## Layout

| Path | Contents |
|------|----------|
| `docker/docker-compose.yml` | Single combined server, LMCache CPU + disk tiers |
| `docker/docker-compose.distributed.yml` | Proxy + prefill/decode fleets sharing a Redis remote tier |
| `ansible/` | Fleet deployment: venv, `kvbench[lmcache]`, rendered configs, systemd |

## Quick start

```bash
# Single node
cd docker && docker compose up -d

# Disaggregated with shared Redis tier
cd docker && docker compose -f docker-compose.distributed.yml up -d

# Fleet via Ansible
cd ansible && ansible-playbook -i inventory/example.yml playbooks/deploy.yml
```

## Choosing the storage under test

Map your storage to an LMCache tier — local NVMe and shared filesystems
via the disk tier (`local_disk`), network stores via the remote tier
(`remote_url`: `redis://`, `valkey://`, `s3://` for MinIO/S3, `fs://` for
mounted filesystems, `lm://` for LMCache's cache server, `mooncakestore://`,
`infinistore://`). See
[docs/deployment/storage-backends.md](../docs/deployment/storage-backends.md)
for per-system instructions.

## What the benchmark measures

Serving-level metrics (TTFT, ITL, throughput, cache hit rate) as a
function of the storage underneath: GPU compute is simulated from
hardware profiles and held constant; KV cache I/O is real and performed
by LMCache. Swap the storage tier, rerun the same workload, and the delta
is attributable to storage.
