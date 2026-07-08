# Storage Under Test

KV-Bench characterizes storage by putting it **under LMCache** — the same
position it occupies in a production vLLM + LMCache deployment. This page
maps common storage systems to the LMCache configuration that exercises
them. Deploy the infrastructure, then point LMCache at it; KV-Bench needs
no storage-specific configuration.

## Local NVMe

Mount the device, point LMCache's disk tier at it:

```bash
mkfs.ext4 /dev/nvme1n1 && mount /dev/nvme1n1 /mnt/nvme-under-test
```

```yaml
# lmcache.yaml
local_disk: "file:///mnt/nvme-under-test/lmcache/"
max_local_disk_size: 500.0
```

Keep `max_local_cpu_size` small relative to the working set so traffic
actually spills to disk instead of being absorbed by the CPU tier.

## NFS / parallel filesystems (Weka, Lustre, ...)

Mount the filesystem on every node and use it either as each node's disk
tier (shared namespace → cross-node cache reuse) or via LMCache's `fs://`
remote connector:

```bash
mount -t nfs storage-server:/export/kvcache /mnt/shared-kv
```

```yaml
# Option A: disk tier on the shared mount
local_disk: "file:///mnt/shared-kv/lmcache/"
max_local_disk_size: 1000.0

# Option B: remote tier through the filesystem connector
# remote_url: "fs:///mnt/shared-kv/lmcache/"
```

## Redis / Valkey

A network KV store as the shared remote tier:

```bash
docker run -d -p 6379:6379 redis:7-alpine \
    redis-server --maxmemory 16gb --maxmemory-policy allkeys-lru
```

```yaml
remote_url: "redis://cache-host:6379"
remote_serde: "naive"
```

Sentinel topologies use `redis-sentinel://host1:26379,host2:26379/mymaster`;
Valkey uses `valkey://host:6379`.

## S3 / MinIO

Object storage as the remote tier via LMCache's `s3://` connector:

```bash
docker run -d -p 9000:9000 minio/minio server /data
```

```yaml
remote_url: "s3://kv-cache-bucket"
```

Credentials and endpoint follow the AWS SDK conventions
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`).

## LMCache cache server

LMCache ships its own remote cache server for a dedicated cache node:

```bash
python -m lmcache.v1.server localhost 65432
```

```yaml
remote_url: "lm://cache-node:65432"
```

## Disaggregated stores (Mooncake, InfiniStore)

```yaml
remote_url: "mooncakestore://metadata-server:2379"
# or
remote_url: "infinistore://host:port"
```

These require their respective services deployed; consult the LMCache
documentation for connector-specific options.

## Baseline / no-op

`blackhole://` discards all remote writes — useful for isolating the
CPU-tier and compute-simulation baseline from storage effects.

## What changed from earlier KV-Bench versions

Earlier versions implemented seven storage backends inside KV-Bench and a
mock "LMCache-compatible" key formatter. Those measured KV-Bench's own
I/O, not a real KV management stack's, and were removed. If you have an
old config with a `storage:` section, delete it and express the same
intent in `lmcache.yaml` using the mappings above.
