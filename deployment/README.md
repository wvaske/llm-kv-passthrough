# KV-Bench Deployment

This directory contains deployment configurations for KV-Bench.

## Deployment Options

| Method | Use Case | Directory |
|--------|----------|-----------|
| Docker | Containerized deployment | `docker/` |
| Ansible | Bare-metal automation | `ansible/` |

## Quick Start

### Docker (Recommended)

```bash
cd docker
docker-compose up -d
```

### Ansible

```bash
cd ansible
cp inventory/example.yml inventory/production.yml
# Edit inventory/production.yml
ansible-playbook -i inventory/production.yml playbooks/deploy.yml
```

## Deployment Architectures

### Single Server (Development)
- 1 KV-Bench server (combined mode)
- In-memory storage

### Single Server with Redis
- 1 KV-Bench server
- 1 Redis server for persistent cache

### Distributed (Production)
- 2+ Prefill servers
- 2+ Decode servers
- 1 Proxy/Load balancer
- Redis Cluster for shared cache

## Configuration

See [Configuration Guide](../docs/getting-started/configuration.md) for all options.

## Monitoring

KV-Bench exposes metrics at `/metrics` endpoint. Integrate with:
- Prometheus
- Grafana
- Custom monitoring solutions
