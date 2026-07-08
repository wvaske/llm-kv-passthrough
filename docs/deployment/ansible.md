# Ansible Deployment

`deployment/ansible/` deploys KV-Bench across a fleet: a Python venv per
host, the `kvbench[lmcache]` package, two rendered config files (KV-Bench
and LMCache), and a systemd unit.

> These playbooks are provided as a starting point and are not exercised
> in CI — review the rendered configs (`/etc/kvbench/*.yaml`) on a staging
> host before a full rollout.

## Inventory

```yaml
# inventory/example.yml
all:
  children:
    kvbench_servers:
      children:
        prefill:
          hosts:
            prefill-1.example.com:
            prefill-2.example.com:
        decode:
          hosts:
            decode-1.example.com:
            decode-2.example.com:
        proxy:
          hosts:
            proxy-1.example.com:
```

## Variables

KV-Bench settings (`config.yaml.j2`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `kvbench_server_type` | `combined` | prefill / decode / combined / proxy |
| `kvbench_model` | `llama-3.1-8b` | model profile |
| `kvbench_gpu` | `H100_SXM` | GPU profile |
| `kvbench_pip_spec` | git+https of this repo | pip requirement to install |

LMCache settings (`lmcache.yaml.j2`) — this is where storage lives:

| Variable | Default | Purpose |
|----------|---------|---------|
| `lmcache_chunk_size` | `256` | tokens per KV chunk |
| `lmcache_max_local_cpu_size_gb` | `4.0` | CPU tier size |
| `lmcache_local_disk_path` | unset | disk tier path (NVMe under test) |
| `lmcache_max_local_disk_size_gb` | `100.0` | disk tier size |
| `lmcache_remote_url` | unset | shared remote tier (`redis://...`, `s3://...`, `fs://...`) |

Example group vars for a disaggregated benchmark with shared Redis:

```yaml
# group_vars/kvbench_servers.yml
lmcache_chunk_size: 256
lmcache_max_local_cpu_size_gb: 2.0
lmcache_remote_url: "redis://cache-host:6379"
```

## Run

```bash
cd deployment/ansible
ansible-playbook -i inventory/example.yml playbooks/deploy.yml
ansible-playbook -i inventory/example.yml playbooks/benchmark.yml   # drive load
ansible-playbook -i inventory/example.yml playbooks/teardown.yml
```

The deploy playbook health-checks `http://localhost:8000/health` on each
host after starting the service.
