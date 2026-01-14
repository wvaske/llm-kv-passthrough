# KV-Bench: Distributed KV Cache Benchmarking System
## Comprehensive Development Plan

**Project:** KV-Bench | **Version:** 1.0.0 | **Duration:** 12-16 weeks

---

## Executive Summary

KV-Bench is a distributed mock LLM serving system for benchmarking KV cache management without GPUs:

- **Multi-host deployment** with shared storage (Redis, NFS, Ceph, Weka, MinIO)
- **Disaggregated prefill/decode** architecture emulation
- **Pluggable KV backends** (LMCache, Mooncake, Dynamo)
- **Configurable resources** (CPU memory, NVMe, external storage)
- **GenAI-Perf compatible** for standardized benchmarking

---

## Development Phases

| Phase | Name | Weeks | Deliverables | Coverage |
|-------|------|-------|--------------|----------|
| 1 | Foundation | 1-2 | Config, GPU/Model profiles | ≥95% |
| 2 | Core Engine | 3-5 | Latency calc, KV manager | ≥90% |
| 3 | Storage | 6-8 | 7 storage backends | ≥90% |
| 4 | Distributed | 9-10 | Servers, proxy, connectors | ≥85% |
| 5 | Integration | 11-12 | E2E tests, GenAI-Perf | ≥90% |
| 6 | Deployment | 13-14 | Docs, Ansible, Docker | ≥90% |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      KV-Bench System                         │
├─────────────────────────────────────────────────────────────┤
│  CLI Tools │ Metrics Exporter │ GenAI-Perf Adapter │ Docs  │
├─────────────────────────────────────────────────────────────┤
│                    API Gateway / Proxy                       │
├─────────────────────────────────────────────────────────────┤
│           Prefill Server    │    Decode Server              │
│           (OpenAI API)      │    (OpenAI API)               │
├─────────────────────────────────────────────────────────────┤
│     Latency Calculator │ KV Manager │ Token Processor       │
├─────────────────────────────────────────────────────────────┤
│  LMCache Connector │ Mooncake Connector │ Dynamo Connector  │
├─────────────────────────────────────────────────────────────┤
│ Memory│Disk│Redis│NFS│Ceph│Weka│MinIO  (Storage Backends)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
kv-bench/
├── README.md
├── pyproject.toml
├── Makefile
│
├── docs/                        # MkDocs documentation
│   ├── mkdocs.yml
│   ├── index.md
│   ├── getting-started/
│   │   └── README.md, installation.md, quickstart.md, configuration.md
│   ├── architecture/
│   │   └── README.md, overview.md, gpu-emulation.md, storage.md
│   ├── deployment/
│   │   └── README.md, docker.md, ansible.md
│   └── benchmarking/
│       └── README.md, genai-perf.md, lmcache.md
│
├── src/kvbench/
│   ├── core/                    # README.md + config, latency, models, gpu_profiles, tokens
│   ├── kv/                      # README.md + manager, chunk, metadata
│   ├── connectors/              # README.md + base, lmcache/, mooncake/, dynamo/
│   ├── storage/                 # README.md + base, memory, local_disk, redis, nfs, ceph, weka, minio
│   ├── servers/                 # README.md + prefill, decode, proxy, combined, openai_compat
│   ├── distributed/             # README.md + registry, coordinator, health
│   ├── metrics/                 # README.md + prometheus, collectors
│   └── cli/                     # README.md + main, serve, benchmark
│
├── tests/
│   ├── unit/                    # core/, kv/, connectors/, storage/, servers/
│   ├── integration/             # prefill_decode, storage_backends, connectors
│   └── e2e/                     # genai_perf, lmcache, distributed
│
├── scripts/
│   └── run_tests.sh, coverage_report.sh, genai_perf_test.sh, lmcache_test.sh
│
├── examples/
│   └── single_node/, multi_node/, disaggregated/, benchmarks/
│
├── deployment/
│   ├── docker/                  # Dockerfile, docker-compose.yml, docker-compose.distributed.yml
│   └── ansible/                 # inventory/, playbooks/, roles/, templates/
│
└── benchmarks/
    └── configs/, results/, analysis/
```

---

## Phase 1: Foundation (Weeks 1-2)

### 1.1 Configuration System

```python
# src/kvbench/core/config.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class ResourceLimits(BaseModel):
    cpu_memory_gb: float = Field(default=8.0, ge=0.1, le=1024.0)
    nvme_storage_gb: float = Field(default=100.0, ge=0.0, le=10000.0)
    nvme_path: Path = Path("/var/lib/kvbench/nvme")
    memory_allocation: Literal["greedy", "lazy", "pool"] = "lazy"

class StorageConfig(BaseModel):
    backend_type: Literal["memory","local_disk","redis","nfs","ceph","weka","minio"] = "memory"
    redis_url: Optional[str] = None
    redis_cluster: bool = False
    filesystem_path: Optional[Path] = None
    s3_endpoint: Optional[str] = None
    s3_bucket: Optional[str] = None

class ConnectorConfig(BaseModel):
    connector_type: Literal["lmcache", "mooncake", "dynamo", "mock"] = "lmcache"
    lmcache_chunk_size: int = Field(default=256, ge=16, le=4096)
    lmcache_remote_url: Optional[str] = None

class GPUEmulationConfig(BaseModel):
    gpu_profile: str = "H100_SXM"
    efficiency_factor: float = Field(default=0.7, ge=0.1, le=1.0)
    tp_size: int = Field(default=1, ge=1, le=16)

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1024, le=65535)
    server_type: Literal["prefill", "decode", "combined", "proxy"] = "combined"
    model_profile: str = "llama-3.1-8b"

class KVBenchConfig(BaseSettings):
    instance_id: str = "kvbench-0"
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    connector: ConnectorConfig = Field(default_factory=ConnectorConfig)
    gpu: GPUEmulationConfig = Field(default_factory=GPUEmulationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    class Config:
        env_prefix = "KVBENCH_"
        env_nested_delimiter = "__"
```

### 1.2 GPU Profiles

```python
# src/kvbench/core/gpu_profiles.py
GPU_PROFILES = {
    "H100_SXM":  GPUProfile(bf16_tflops=1979, hbm_bandwidth_tb_s=3.35, hbm_capacity_gb=80),
    "H100_PCIe": GPUProfile(bf16_tflops=1513, hbm_bandwidth_tb_s=2.0,  hbm_capacity_gb=80),
    "H200_SXM":  GPUProfile(bf16_tflops=1979, hbm_bandwidth_tb_s=4.8,  hbm_capacity_gb=141),
    "A100_SXM":  GPUProfile(bf16_tflops=312,  hbm_bandwidth_tb_s=2.0,  hbm_capacity_gb=80),
    "L4":        GPUProfile(bf16_tflops=121,  hbm_bandwidth_tb_s=0.3,  hbm_capacity_gb=24),
    "L40S":      GPUProfile(bf16_tflops=362,  hbm_bandwidth_tb_s=0.864,hbm_capacity_gb=48),
}
```

### 1.3 Model Profiles

```python
# src/kvbench/core/models.py
MODEL_PROFILES = {
    "llama-3.1-8b":   ModelProfile(layers=32, hidden=4096,  kv_heads=8,  intermediate=14336),
    "llama-3.1-70b":  ModelProfile(layers=80, hidden=8192,  kv_heads=8,  intermediate=28672),
    "llama-3.1-405b": ModelProfile(layers=126,hidden=16384, kv_heads=8,  intermediate=53248),
    "qwen-2.5-7b":    ModelProfile(layers=28, hidden=3584,  kv_heads=4,  intermediate=18944),
    "qwen-2.5-72b":   ModelProfile(layers=80, hidden=8192,  kv_heads=8,  intermediate=29568),
}
```

### Phase 1 Checklist
- [ ] Project structure, pyproject.toml, Makefile
- [ ] Pre-commit hooks (ruff, black, mypy)
- [ ] Configuration with env var support
- [ ] 6 GPU profiles, 5+ model profiles
- [ ] Unit tests ≥95% coverage
- [ ] README.md at root and core/

---

## Phase 2: Core Engine (Weeks 3-5)

### 2.1 Latency Calculator (Roofline Model)

```python
# src/kvbench/core/latency.py
@dataclass
class LatencyBreakdown:
    compute_ms: float
    memory_ms: float
    total_ms: float
    is_compute_bound: bool
    
    @property
    def bottleneck(self) -> str:
        return "compute" if self.is_compute_bound else "memory"

class LatencyCalculator:
    def __init__(self, gpu: str, model: str, tp_size: int = 1, efficiency: float = 0.7):
        self.gpu = get_gpu_profile(gpu)
        self.model = get_model_profile(model)
        self.tp_size = tp_size
        self.efficiency = efficiency
    
    def prefill_latency(self, num_tokens: int, batch_size: int = 1) -> LatencyBreakdown:
        """Compute-bound: processes all tokens in parallel."""
        # Attention FLOPs (quadratic in seq_len) + FFN FLOPs (linear)
        # time = max(compute_time, memory_time)
        
    def decode_latency(self, context_length: int, batch_size: int = 1) -> LatencyBreakdown:
        """Memory-bound: loads model + KV cache per token."""
        # bytes = model_params + kv_cache_for_context
        # time dominated by memory bandwidth
    
    def kv_transfer_latency(self, num_tokens: int, bandwidth_gb_s: float = 10.0) -> float:
        """KV transfer time for disaggregated prefill."""
    
    def estimate_ttft(self, input_tokens: int, cache_hit_tokens: int = 0) -> float:
        """Time to first token with cache hits."""
```

### 2.2 Token Processing

```python
# src/kvbench/core/tokens.py
class TokenProcessor:
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
    
    def simulate_tokenize(self, text: str) -> List[int]
    def chunk_tokens(self, tokens: List[int]) -> List[List[int]]
    def compute_chunk_hash(self, tokens: List[int]) -> str  # SHA-256

class CacheKeyGenerator:
    """LMCache-compatible: format@model@world_size@worker_id@hash@suffix"""
    def make_key(self, chunk_hash: str, worker_id: int, suffix: str = "@kv_bytes") -> str
    def make_keys_for_all_workers(self, chunk_hash: str) -> List[str]
```

### Phase 2 Checklist
- [ ] LatencyCalculator with roofline model
- [ ] Prefill (compute-bound) and decode (memory-bound) emulation
- [ ] TokenProcessor with chunking
- [ ] CacheKeyGenerator (LMCache format)
- [ ] Unit tests ≥90% coverage

---

## Phase 3: Storage Backends (Weeks 6-8)

### 3.1 Abstract Interface

```python
# src/kvbench/storage/base.py
class StorageBackend(ABC):
    def __init__(self, max_size_bytes: int, name: str = "unknown"):
        self.max_size_bytes = max_size_bytes
        self._stats = StorageStats(total_bytes=max_size_bytes)
    
    @abstractmethod
    async def get(self, key: str) -> Optional[bytes]: pass
    
    @abstractmethod
    async def put(self, key: str, value: bytes, ttl_seconds: Optional[int] = None) -> bool: pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool: pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool: pass
    
    @abstractmethod
    async def keys(self, pattern: Optional[str] = None) -> List[str]: pass
    
    @abstractmethod
    async def clear(self) -> int: pass
    
    @abstractmethod
    async def close(self) -> None: pass
```

### 3.2 Implementations

| Backend | File | Resource Config | Features |
|---------|------|-----------------|----------|
| Memory | memory.py | `cpu_memory_gb` | LRU eviction, asyncio locks |
| Local Disk | local_disk.py | `nvme_storage_gb`, `nvme_path` | Sharded directories, async I/O |
| Redis | redis_backend.py | `redis_url`, `redis_cluster` | Standalone/Cluster/Sentinel |
| NFS | nfs.py | `filesystem_path` | Async file operations |
| Ceph | ceph.py | `ceph_pool` | librados bindings |
| Weka | weka.py | `filesystem_path` | High-performance parallel FS |
| MinIO | minio.py | `s3_endpoint`, `s3_bucket` | aioboto3, S3-compatible |

### 3.3 Factory

```python
# src/kvbench/storage/factory.py
def create_storage_backend(config: StorageConfig, resources: ResourceLimits) -> StorageBackend:
    if config.backend_type == "memory":
        return MemoryStorageBackend(max_size_bytes=int(resources.cpu_memory_gb * 1024**3))
    elif config.backend_type == "local_disk":
        return LocalDiskStorageBackend(base_path=resources.nvme_path, ...)
    elif config.backend_type == "redis":
        return RedisStorageBackend(redis_url=config.redis_url, cluster_mode=config.redis_cluster)
    # ... etc
```

### Phase 3 Checklist
- [ ] Abstract StorageBackend interface
- [ ] MemoryStorageBackend with LRU
- [ ] LocalDiskStorageBackend with sharding
- [ ] RedisStorageBackend (all modes)
- [ ] NFSStorageBackend
- [ ] CephStorageBackend
- [ ] WekaStorageBackend
- [ ] MinIOStorageBackend
- [ ] Storage factory
- [ ] Unit + integration tests ≥90%

---

## Phase 4: Distributed System (Weeks 9-10)

### 4.1 Mock Servers

#### Prefill Server
```python
# src/kvbench/servers/prefill.py
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    tokens = processor.simulate_tokenize(prompt)
    chunks = processor.chunk_tokens(tokens)
    
    for chunk in chunks:
        chunk_hash = processor.compute_chunk_hash(chunk)
        if not await connector.exists(chunk_hash):
            # Simulate prefill latency
            await asyncio.sleep(calc.prefill_latency(len(chunk)).total_ms / 1000)
            await connector.store(chunk_hash, len(chunk))
    
    # Return first token + metadata for decode handoff
```

#### Decode Server
```python
# src/kvbench/servers/decode.py
async def generate_tokens(context_length: int, max_tokens: int):
    for i in range(max_tokens):
        latency = calc.decode_latency(context_length + i).total_ms
        await asyncio.sleep(latency / 1000)
        yield f"token_{i}"
```

#### Proxy
```python
# src/kvbench/servers/proxy.py
class DisaggregatedProxy:
    def __init__(self, prefill_endpoints: List[str], decode_endpoints: List[str]):
        self.prefill_endpoints = prefill_endpoints
        self.decode_endpoints = decode_endpoints
    
    def select_prefill(self) -> str  # Round-robin
    def select_decode(self) -> str   # Round-robin
```

### 4.2 KV Connectors (Pluggable)

```python
# src/kvbench/connectors/base.py
class KVConnector(ABC):
    @abstractmethod
    async def store(self, chunk_hash: str, num_tokens: int) -> bool: pass
    
    @abstractmethod
    async def load(self, chunk_hash: str) -> Optional[bytes]: pass
    
    @abstractmethod
    async def exists(self, chunk_hash: str) -> bool: pass

# src/kvbench/connectors/lmcache/connector.py
class LMCacheConnector(KVConnector):
    """Full implementation with LMCache-compatible key format."""

# src/kvbench/connectors/mooncake/connector.py
class MooncakeConnector(KVConnector):
    """Stub for future Mooncake support."""

# src/kvbench/connectors/dynamo/connector.py
class DynamoConnector(KVConnector):
    """Stub for future NVIDIA Dynamo support."""
```

### Phase 4 Checklist
- [ ] PrefillServer (OpenAI API)
- [ ] DecodeServer (streaming)
- [ ] CombinedServer
- [ ] DisaggregatedProxy
- [ ] KVConnector base class
- [ ] LMCacheConnector (full)
- [ ] MooncakeConnector (stub)
- [ ] DynamoConnector (stub)
- [ ] Service discovery
- [ ] Unit tests ≥85%

---

## Phase 5: Integration & Testing (Weeks 11-12)

### 5.1 GenAI-Perf Test Script

```bash
#!/bin/bash
# scripts/genai_perf_test.sh
set -e

ENDPOINT="${KVBENCH_ENDPOINT:-http://localhost:8000}"
MODEL="${KVBENCH_MODEL:-llama-3.1-8b}"

# Start server
kvbench serve --model $MODEL &
sleep 5

# Run benchmark
genai-perf profile \
    -m $MODEL \
    --service-kind openai \
    --endpoint-type chat \
    --url $ENDPOINT \
    --streaming \
    --synthetic-input-tokens-mean 1000 \
    --output-tokens-mean 100 \
    --concurrency 8 \
    --request-count 100 \
    --generate-plots

echo "✅ GenAI-Perf test complete"
```

### 5.2 LMCache Test Script

```bash
#!/bin/bash
# scripts/lmcache_test.sh
set -e

# Start LMCache server
lmcache_server localhost:8080 &
sleep 3

# Start KV-Bench with LMCache
KVBENCH_CONNECTOR__CONNECTOR_TYPE=lmcache \
KVBENCH_CONNECTOR__LMCACHE_REMOTE_URL=lm://localhost:8080 \
kvbench serve &
sleep 3

# Test cache hit/miss
python -c "
import httpx, asyncio

async def test():
    async with httpx.AsyncClient() as c:
        # First request - cache miss
        r1 = await c.post('http://localhost:8000/v1/chat/completions', 
            json={'model':'llama-3.1-8b','messages':[{'role':'user','content':'Hello world'}]})
        print(f'Request 1: {r1.status_code}')
        
        # Same prefix - cache hit
        r2 = await c.post('http://localhost:8000/v1/chat/completions',
            json={'model':'llama-3.1-8b','messages':[{'role':'user','content':'Hello world, how are you?'}]})
        print(f'Request 2: {r2.status_code}')

asyncio.run(test())
"

echo "✅ LMCache test complete"
```

### 5.3 Coverage Report Script

```bash
#!/bin/bash
# scripts/coverage_report.sh
set -e

PHASE=${1:-"final"}
MIN_COVERAGE=${2:-90}

echo "══════════════════════════════════════════════"
echo "KV-Bench Coverage Report: Phase ${PHASE}"
echo "══════════════════════════════════════════════"

pytest tests/ \
    --cov=src/kvbench \
    --cov-report=term-missing \
    --cov-report=html:coverage_html \
    --cov-report=xml:coverage.xml \
    --cov-branch \
    -v

COVERAGE=$(coverage report | grep TOTAL | awk '{print $4}' | tr -d '%')

echo ""
echo "Coverage: ${COVERAGE}% (minimum: ${MIN_COVERAGE}%)"

if (( $(echo "$COVERAGE >= $MIN_COVERAGE" | bc -l) )); then
    echo "✅ PASSED"
    exit 0
else
    echo "❌ FAILED"
    exit 1
fi
```

### Phase 5 Checklist
- [ ] genai_perf_test.sh
- [ ] lmcache_test.sh
- [ ] coverage_report.sh
- [ ] E2E test suite
- [ ] Distributed deployment tests
- [ ] Coverage ≥90%

---

## Phase 6: Documentation & Deployment (Weeks 13-14)

### 6.1 MkDocs Configuration

```yaml
# docs/mkdocs.yml
site_name: KV-Bench Documentation
theme:
  name: material
  features:
    - navigation.tabs
    - content.code.copy

nav:
  - Home: index.md
  - Getting Started:
    - Installation: getting-started/installation.md
    - Quick Start: getting-started/quickstart.md
    - Configuration: getting-started/configuration.md
  - Architecture:
    - Overview: architecture/overview.md
    - GPU Emulation: architecture/gpu-emulation.md
    - Storage Backends: architecture/storage.md
    - KV Connectors: architecture/connectors.md
  - Deployment:
    - Docker: deployment/docker.md
    - Ansible: deployment/ansible.md
  - Benchmarking:
    - GenAI-Perf: benchmarking/genai-perf.md
    - LMCache: benchmarking/lmcache.md
  - API Reference: api-reference/

plugins:
  - search
  - mkdocstrings
```

### 6.2 Docker Compose (Distributed)

```yaml
# deployment/docker/docker-compose.distributed.yml
version: "3.8"

services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  prefill-1:
    build: .
    environment:
      - KVBENCH_INSTANCE_ID=prefill-1
      - KVBENCH_SERVER__SERVER_TYPE=prefill
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    ports: ["8001:8000"]

  prefill-2:
    build: .
    environment:
      - KVBENCH_INSTANCE_ID=prefill-2
      - KVBENCH_SERVER__SERVER_TYPE=prefill
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    ports: ["8002:8000"]

  decode-1:
    build: .
    environment:
      - KVBENCH_INSTANCE_ID=decode-1
      - KVBENCH_SERVER__SERVER_TYPE=decode
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    ports: ["8101:8000"]

  decode-2:
    build: .
    environment:
      - KVBENCH_INSTANCE_ID=decode-2
      - KVBENCH_SERVER__SERVER_TYPE=decode
      - KVBENCH_STORAGE__BACKEND_TYPE=redis
      - KVBENCH_STORAGE__REDIS_URL=redis://redis:6379
    ports: ["8102:8000"]

  proxy:
    build: .
    environment:
      - KVBENCH_SERVER__SERVER_TYPE=proxy
      - KVBENCH_PREFILL_ENDPOINTS=prefill-1:8000,prefill-2:8000
      - KVBENCH_DECODE_ENDPOINTS=decode-1:8000,decode-2:8000
    ports: ["8000:8000"]
```

### 6.3 Ansible Playbooks

```yaml
# deployment/ansible/inventory/hosts.yml
all:
  children:
    prefill:
      hosts:
        prefill-1: {ansible_host: 10.0.1.10}
        prefill-2: {ansible_host: 10.0.1.11}
    decode:
      hosts:
        decode-1: {ansible_host: 10.0.2.10}
        decode-2: {ansible_host: 10.0.2.11}
    proxy:
      hosts:
        proxy-1: {ansible_host: 10.0.0.10}
    storage:
      hosts:
        redis-1: {ansible_host: 10.0.3.10}
  vars:
    kvbench_version: "1.0.0"
    kvbench_model: "llama-3.1-8b"
    kvbench_gpu_profile: "H100_SXM"
    redis_url: "redis://10.0.3.10:6379"
```

```yaml
# deployment/ansible/playbooks/deploy.yml
---
- name: Deploy KV-Bench
  hosts: all
  become: yes
  tasks:
    - name: Install dependencies
      apt:
        name: [python3.11, python3.11-venv, python3-pip]
        state: present
    
    - name: Create kvbench user
      user: name=kvbench system=yes
    
    - name: Install KV-Bench
      pip:
        name: kvbench=={{ kvbench_version }}
        virtualenv: /opt/kvbench/venv
    
    - name: Deploy configuration
      template:
        src: kvbench.yaml.j2
        dest: /opt/kvbench/config.yaml
    
    - name: Deploy systemd service
      template:
        src: kvbench.service.j2
        dest: /etc/systemd/system/kvbench.service
      notify: Restart kvbench
    
    - name: Enable and start
      systemd: name=kvbench enabled=yes state=started
  
  handlers:
    - name: Restart kvbench
      systemd: name=kvbench state=restarted
```

```yaml
# deployment/ansible/playbooks/benchmark.yml
---
- name: Run Benchmark
  hosts: proxy
  vars:
    concurrency: 32
    requests: 1000
  tasks:
    - name: Install GenAI-Perf
      pip: name=genai-perf virtualenv=/opt/kvbench/venv
    
    - name: Run benchmark
      shell: |
        source /opt/kvbench/venv/bin/activate
        genai-perf profile -m {{ kvbench_model }} \
          --service-kind openai --endpoint-type chat \
          --url http://localhost:8000 --streaming \
          --synthetic-input-tokens-mean 2000 \
          --output-tokens-mean 200 \
          --concurrency {{ concurrency }} \
          --request-count {{ requests }} \
          --generate-plots
```

### Phase 6 Checklist
- [ ] MkDocs site complete
- [ ] README.md at each directory
- [ ] API reference docs
- [ ] Dockerfile, docker-compose.yml
- [ ] docker-compose.distributed.yml
- [ ] Ansible inventory
- [ ] Ansible playbooks (deploy, configure, benchmark, teardown)
- [ ] Ansible roles
- [ ] Final coverage ≥90%

---

## Summary

### Timeline
```
Week 1-2:   Foundation (config, profiles)
Week 3-5:   Core Engine (latency, tokens)
Week 6-8:   Storage Backends (7 backends)
Week 9-10:  Distributed System (servers, connectors)
Week 11-12: Integration Testing (GenAI-Perf, LMCache)
Week 13-14: Documentation & Deployment
```

### Final Deliverables

| Category | Items |
|----------|-------|
| Storage | 7 backends (Memory, Disk, Redis, NFS, Ceph, Weka, MinIO) |
| Connectors | 3 (LMCache full, Mooncake stub, Dynamo stub) |
| Servers | 4 (Prefill, Decode, Combined, Proxy) |
| Tests | Unit, Integration, E2E (≥90% coverage) |
| Docs | MkDocs site, README at each level |
| Deployment | Docker Compose, Ansible playbooks |
| Scripts | GenAI-Perf test, LMCache test, Coverage report |

### Success Criteria

✅ GenAI-Perf compatibility verified  
✅ LMCache integration functional  
✅ Multi-host distributed deployment working  
✅ All 7 storage backends operational  
✅ Pluggable connector architecture (LMCache/Mooncake/Dynamo)  
✅ Configurable resources (CPU memory, NVMe, external storage)  
✅ Code coverage ≥90% with reports  
✅ Documentation complete  
✅ Deployment automation ready
