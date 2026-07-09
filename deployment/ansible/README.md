# KV-Bench Ansible Deployment

Ansible playbooks and templates for deploying KV-Bench across multiple servers.

## Directory Structure

```
deployment/ansible/
├── inventory/
│   └── example.yml      # Example inventory file
├── playbooks/
│   ├── deploy.yml       # Deploy KV-Bench
│   ├── configure.yml    # Update configuration
│   ├── benchmark.yml    # Run benchmarks
│   └── teardown.yml     # Remove KV-Bench
├── templates/
│   ├── config.yaml.j2   # Configuration template
│   └── kvbench.service.j2  # Systemd service template
└── README.md
```

## Quick Start

### 1. Create Inventory

```bash
cp inventory/example.yml inventory/production.yml
# Edit inventory/production.yml with your server details
```

### 2. Deploy

```bash
ansible-playbook -i inventory/production.yml playbooks/deploy.yml
```

### 3. Run Benchmarks

```bash
ansible-playbook -i inventory/production.yml playbooks/benchmark.yml
```

## Playbooks

| Playbook | Description |
|----------|-------------|
| `deploy.yml` | Full deployment including package installation |
| `configure.yml` | Update configuration and restart services |
| `benchmark.yml` | Run performance benchmarks |
| `teardown.yml` | Remove KV-Bench from all servers |

## Configuration Variables

See `inventory/example.yml` for all available variables.

### Common Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `kvbench_pip_spec` | git+https of this repo | pip requirement to install |
| `kvbench_model` | `llama-3.1-8b` | Model profile |
| `kvbench_gpu` | `H100_SXM` | GPU profile |
| `lmcache_chunk_size` | `256` | LMCache tokens per KV chunk |
| `lmcache_max_local_cpu_size_gb` | `4.0` | LMCache CPU tier size |
| `lmcache_local_disk_path` | unset | LMCache disk tier path (NVMe under test) |
| `lmcache_remote_url` | unset | LMCache shared remote tier (`redis://...`, `s3://...`) |

Storage lives entirely in the LMCache variables (rendered into
`/etc/kvbench/lmcache.yaml`); KV-Bench itself has no storage settings.

## Usage Examples

### Deploy to specific group

```bash
ansible-playbook -i inventory/production.yml playbooks/deploy.yml -l prefill
```

### Update configuration only

```bash
ansible-playbook -i inventory/production.yml playbooks/configure.yml
```

### Teardown with data preservation

```bash
ansible-playbook -i inventory/production.yml playbooks/teardown.yml -e preserve_data=true
```

## Requirements

- Ansible 2.9+
- SSH access to target servers
- Python 3.10+ on target servers
