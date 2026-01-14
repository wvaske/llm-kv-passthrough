# Distributed Mock LLM Benchmarking System

## Design Document for KV Cache Performance Testing

**Version:** 1.0  
**Date:** January 2026

---

## Executive Summary

This document describes the architecture for a distributed mock LLM serving system that enables benchmarking of KV cache management systems (like LMCache) using NVIDIA's GenAI-Perf tool—**without requiring any GPUs**. The system supports:

- Multi-host deployment with shared storage
- Data-parallel instances where KVs created by one server can be used by another
- Disaggregated prefill/decode architecture emulation
- Configurable GPU performance modeling

---

## 1. Architecture Overview

```
                           ┌─────────────────────────────────────────────────┐
                           │              Load Balancer / Router             │
                           │         (Routes based on request type)          │
                           └──────────────┬──────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │  Prefill Host A │         │  Prefill Host B │         │  Prefill Host Z │
    │  (kv_producer)  │         │  (kv_producer)  │   ...   │  (kv_producer)  │
    │                 │         │                 │         │                 │
    │  ┌───────────┐  │         │  ┌───────────┐  │         │  ┌───────────┐  │
    │  │Mock Server│  │         │  │Mock Server│  │         │  │Mock Server│  │
    │  │  :8000    │  │         │  │  :8000    │  │         │  │  :8000    │  │
    │  └─────┬─────┘  │         │  └─────┬─────┘  │         │  └─────┬─────┘  │
    └────────┼────────┘         └────────┼────────┘         └────────┼────────┘
             │                           │                           │
             └───────────────────────────┼───────────────────────────┘
                                         │
                                         ▼
                           ┌─────────────────────────────────────────────────┐
                           │                                                 │
                           │           Shared KV Cache Storage               │
                           │                                                 │
                           │   Options:                                      │
                           │   • Redis Cluster / Sentinel                    │
                           │   • LMCache Server (CPU-only)                   │
                           │   • Shared NFS/Ceph filesystem                  │
                           │   • S3-compatible object storage                │
                           │                                                 │
                           └─────────────────────────────────────────────────┘
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             │                           │                           │
             ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │  Decode Host 1  │         │  Decode Host 2  │         │  Decode Host N  │
    │  (kv_consumer)  │         │  (kv_consumer)  │   ...   │  (kv_consumer)  │
    │                 │         │                 │         │                 │
    │  ┌───────────┐  │         │  ┌───────────┐  │         │  ┌───────────┐  │
    │  │Mock Server│  │         │  │Mock Server│  │         │  │Mock Server│  │
    │  │  :8001    │  │         │  │  :8001    │  │         │  │  :8001    │  │
    │  └───────────┘  │         │  └───────────┘  │         │  └───────────┘  │
    └─────────────────┘         └─────────────────┘         └─────────────────┘
```

---

## 2. GPU Performance Emulation Model

### 2.1 Why We Need Performance Modeling

Real LLM inference has two distinct phases with very different characteristics:

| Phase | Characteristic | Bottleneck | GPU Utilization |
|-------|---------------|------------|-----------------|
| **Prefill** | Process all input tokens in parallel | Compute-bound | High (near 100%) |
| **Decode** | Generate one token at a time | Memory bandwidth-bound | Low (underutilized) |

To create realistic benchmarks, we must model these timing characteristics accurately.

### 2.2 Roofline-Based Latency Model

The roofline model determines whether an operation is compute-bound or memory-bound:

```
Latency = max(
    FLOPs / Compute_Throughput,     # Compute-bound
    Bytes_Accessed / Memory_Bandwidth  # Memory-bound
)
```

#### GPU Hardware Parameters (Configurable)

```python
@dataclass
class GPUProfile:
    name: str
    # Compute capabilities
    fp16_tflops: float      # FP16 TFLOPS
    bf16_tflops: float      # BF16 TFLOPS
    int8_tops: float        # INT8 TOPS
    
    # Memory characteristics  
    hbm_bandwidth_tb_s: float   # HBM bandwidth in TB/s
    hbm_capacity_gb: float      # HBM capacity in GB
    
    # Interconnect (for multi-GPU)
    nvlink_bandwidth_gb_s: float = 0
    pcie_bandwidth_gb_s: float = 64

# Common GPU profiles
GPU_PROFILES = {
    "H100_SXM": GPUProfile(
        name="NVIDIA H100 SXM",
        fp16_tflops=1979,
        bf16_tflops=1979,
        int8_tops=3958,
        hbm_bandwidth_tb_s=3.35,
        hbm_capacity_gb=80,
        nvlink_bandwidth_gb_s=900,
        pcie_bandwidth_gb_s=128
    ),
    "H100_PCIe": GPUProfile(
        name="NVIDIA H100 PCIe",
        fp16_tflops=1513,
        bf16_tflops=1513,
        int8_tops=3026,
        hbm_bandwidth_tb_s=2.0,
        hbm_capacity_gb=80,
        nvlink_bandwidth_gb_s=0,
        pcie_bandwidth_gb_s=128
    ),
    "A100_SXM": GPUProfile(
        name="NVIDIA A100 SXM",
        fp16_tflops=312,
        bf16_tflops=312,
        int8_tops=624,
        hbm_bandwidth_tb_s=2.0,
        hbm_capacity_gb=80,
        nvlink_bandwidth_gb_s=600,
        pcie_bandwidth_gb_s=64
    ),
    "L4": GPUProfile(
        name="NVIDIA L4",
        fp16_tflops=121,
        bf16_tflops=121,
        int8_tops=242,
        hbm_bandwidth_tb_s=0.3,  # GDDR6
        hbm_capacity_gb=24,
        nvlink_bandwidth_gb_s=0,
        pcie_bandwidth_gb_s=64
    ),
}
```

### 2.3 Model Parameters

```python
@dataclass
class ModelProfile:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int           # For GQA/MQA
    intermediate_size: int      # FFN intermediate
    vocab_size: int
    max_position_embeddings: int
    
    # Derived properties
    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads
    
    @property
    def kv_head_dim(self) -> int:
        return self.hidden_size // self.num_kv_heads
    
    @property
    def bytes_per_token_kv(self) -> int:
        """KV cache bytes per token (bf16)"""
        # 2 (K and V) * num_layers * num_kv_heads * head_dim * 2 (bf16)
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2
    
    @property
    def total_params(self) -> int:
        """Approximate parameter count"""
        # Simplified: embedding + attention + FFN per layer
        embed = self.vocab_size * self.hidden_size
        attn_per_layer = 4 * self.hidden_size * self.hidden_size  # QKV + O
        ffn_per_layer = 3 * self.hidden_size * self.intermediate_size
        return embed + self.num_layers * (attn_per_layer + ffn_per_layer)

MODEL_PROFILES = {
    "llama-3.1-8b": ModelProfile(
        name="Llama-3.1-8B",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_kv_heads=8,  # GQA
        intermediate_size=14336,
        vocab_size=128256,
        max_position_embeddings=131072
    ),
    "llama-3.1-70b": ModelProfile(
        name="Llama-3.1-70B",
        num_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_kv_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        max_position_embeddings=131072
    ),
    "qwen-2.5-32b": ModelProfile(
        name="Qwen-2.5-32B",
        num_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
        num_kv_heads=8,
        intermediate_size=27648,
        vocab_size=152064,
        max_position_embeddings=131072
    ),
}
```

### 2.4 Latency Calculator

```python
class LatencyCalculator:
    def __init__(self, gpu: GPUProfile, model: ModelProfile, 
                 tp_size: int = 1, efficiency: float = 0.7):
        self.gpu = gpu
        self.model = model
        self.tp_size = tp_size
        self.efficiency = efficiency  # Real-world efficiency factor
        
    def prefill_latency_ms(self, num_tokens: int, batch_size: int = 1) -> float:
        """
        Prefill is compute-bound for reasonable sequence lengths.
        
        FLOPs per token ≈ 2 * num_params (forward pass approximation)
        """
        total_tokens = num_tokens * batch_size
        
        # FLOPs for prefill (quadratic in attention, linear in FFN)
        # Simplified: 2 * params * tokens for FFN-dominated
        flops = 2 * self.model.total_params * total_tokens
        
        # Add attention FLOPs (quadratic term for long sequences)
        # attention_flops ≈ 4 * num_layers * num_heads * seq_len^2 * head_dim
        attention_flops = (4 * self.model.num_layers * 
                         self.model.num_attention_heads * 
                         (num_tokens ** 2) * 
                         self.model.head_dim)
        flops += attention_flops
        
        # Scale by tensor parallelism
        flops_per_gpu = flops / self.tp_size
        
        # Calculate compute time
        tflops = self.gpu.bf16_tflops * 1e12 * self.efficiency
        compute_time_s = flops_per_gpu / tflops
        
        # Memory time (loading weights once)
        model_bytes = self.model.total_params * 2  # bf16
        memory_time_s = model_bytes / (self.gpu.hbm_bandwidth_tb_s * 1e12 * self.tp_size)
        
        # Prefill is compute-bound for seq_len > ~512
        latency_s = max(compute_time_s, memory_time_s)
        
        return latency_s * 1000  # Convert to ms
    
    def decode_latency_ms(self, context_length: int, batch_size: int = 1) -> float:
        """
        Decode is memory-bandwidth-bound (one token at a time).
        
        Must load: model weights + KV cache for all context
        """
        # Model weights (loaded every decode step)
        model_bytes = self.model.total_params * 2 / self.tp_size
        
        # KV cache size (grows with context)
        kv_bytes = (self.model.bytes_per_token_kv * context_length * 
                   batch_size / self.tp_size)
        
        # Total bytes to stream from HBM
        total_bytes = model_bytes + kv_bytes
        
        # Memory-bound latency
        bandwidth = self.gpu.hbm_bandwidth_tb_s * 1e12 * self.efficiency
        latency_s = total_bytes / bandwidth
        
        return latency_s * 1000  # Convert to ms
    
    def kv_transfer_latency_ms(self, num_tokens: int) -> float:
        """
        KV cache transfer latency (for disaggregated prefill).
        
        Transfer via network or shared storage.
        """
        kv_bytes = self.model.bytes_per_token_kv * num_tokens / self.tp_size
        
        # Assume network/storage bandwidth (configurable)
        # For NVLink: gpu.nvlink_bandwidth_gb_s
        # For network: ~10-25 GB/s typical
        transfer_bandwidth = 10e9  # 10 GB/s default
        
        return (kv_bytes / transfer_bandwidth) * 1000
```

---

## 3. Multi-Host Data-Parallel Architecture

### 3.1 Shared Storage Backend

For data-parallel instances where KVs created by server 1 can be used by server 2:

```python
# Configuration for shared storage
STORAGE_CONFIG = {
    "type": "redis",  # or "filesystem", "s3", "lmcache_server"
    
    # Redis Cluster configuration
    "redis": {
        "nodes": [
            {"host": "redis-1", "port": 6379},
            {"host": "redis-2", "port": 6379},
            {"host": "redis-3", "port": 6379},
        ],
        "cluster_mode": True,
    },
    
    # Shared filesystem configuration
    "filesystem": {
        "path": "/mnt/shared/kv_cache",
        "type": "nfs",  # or "ceph", "lustre"
    },
    
    # S3-compatible storage
    "s3": {
        "endpoint": "http://minio:9000",
        "bucket": "kv-cache",
        "access_key": "...",
        "secret_key": "...",
    },
    
    # LMCache server
    "lmcache_server": {
        "url": "lm://lmcache-server:8080",
        "serde": "naive",
    }
}
```

### 3.2 Key Format for Cross-Instance Sharing

Using LMCache-compatible key format ensures interoperability:

```python
def make_cache_key(
    model_name: str,
    chunk_hash: str,
    world_size: int,
    worker_id: int,
    suffix: str = "@kv_bytes"
) -> str:
    """
    Generate LMCache-compatible cache key.
    
    Format: format@model_name@world_size@worker_id@chunk_hash@suffix
    
    This format allows:
    - Multiple models to share the same storage
    - Tensor parallelism awareness (world_size, worker_id)
    - Consistent hashing for cache lookups
    """
    return f"vllm@{model_name}@{world_size}@{worker_id}@{chunk_hash}{suffix}"
```

### 3.3 Instance Identity and Coordination

```python
@dataclass
class InstanceConfig:
    instance_id: str              # Unique identifier
    instance_type: str            # "prefill" or "decode"
    host: str
    port: int
    
    # Model parallelism configuration
    tp_size: int = 1              # Tensor parallelism
    pp_stage: int = 0             # Pipeline parallelism stage
    dp_rank: int = 0              # Data parallelism rank
    
    # Performance model
    gpu_profile: str = "H100_SXM"
    model_profile: str = "llama-3.1-8b"
    
    # Storage configuration
    storage_config: dict = None

class InstanceRegistry:
    """Service discovery for distributed instances."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.key_prefix = "mock_llm:instances:"
    
    async def register(self, config: InstanceConfig):
        """Register this instance with the cluster."""
        key = f"{self.key_prefix}{config.instance_id}"
        await self.redis.hset(key, mapping=asdict(config))
        await self.redis.expire(key, 60)  # TTL for health check
    
    async def heartbeat(self, instance_id: str):
        """Keep registration alive."""
        key = f"{self.key_prefix}{instance_id}"
        await self.redis.expire(key, 60)
    
    async def get_prefill_instances(self) -> List[InstanceConfig]:
        """Get all available prefill instances."""
        keys = await self.redis.keys(f"{self.key_prefix}*")
        instances = []
        for key in keys:
            data = await self.redis.hgetall(key)
            if data.get("instance_type") == "prefill":
                instances.append(InstanceConfig(**data))
        return instances
    
    async def get_decode_instances(self) -> List[InstanceConfig]:
        """Get all available decode instances."""
        # Similar to above
        pass
```

---

## 4. Disaggregated Prefill/Decode System

### 4.1 Request Flow

```
┌──────────────┐     ┌────────────────┐     ┌───────────────┐     ┌──────────────┐
│  GenAI-Perf  │────▶│  Proxy/Router  │────▶│ Prefill Host  │────▶│  KV Storage  │
│  (Client)    │     │                │     │ (Producer)    │     │              │
└──────────────┘     └───────┬────────┘     └───────────────┘     └──────┬───────┘
                             │                                           │
                             │  ◀─── First token returned ───────────────┤
                             │                                           │
                             ▼                                           ▼
                     ┌────────────────┐     ┌───────────────┐     ┌──────────────┐
                     │  Proxy/Router  │────▶│  Decode Host  │◀────│  KV Storage  │
                     │  (continues    │     │  (Consumer)   │     │  (Load KV)   │
                     │   streaming)   │     │               │     │              │
                     └────────────────┘     └───────────────┘     └──────────────┘
```

### 4.2 Prefill Server Implementation

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import time
import hashlib
from typing import AsyncGenerator

app = FastAPI()

class PrefillServer:
    def __init__(self, config: InstanceConfig):
        self.config = config
        self.storage = create_storage_client(config.storage_config)
        self.latency_calc = LatencyCalculator(
            gpu=GPU_PROFILES[config.gpu_profile],
            model=MODEL_PROFILES[config.model_profile],
            tp_size=config.tp_size
        )
        self.registry = InstanceRegistry(self.storage.redis)
        self.chunk_size = 256  # tokens per chunk
        
    def tokenize(self, text: str) -> List[int]:
        """Simplified tokenization (use real tokenizer in production)."""
        # For mock purposes, split on whitespace
        # Real implementation would use transformers tokenizer
        return list(range(len(text.split())))
    
    def compute_chunk_hash(self, tokens: List[int]) -> str:
        """Compute SHA-256 hash of token chunk."""
        return hashlib.sha256(str(tokens).encode()).hexdigest()
    
    async def prefill_request(
        self, 
        tokens: List[int],
        request_id: str
    ) -> dict:
        """
        Process prefill and store KV cache.
        
        Returns metadata for decode server.
        """
        num_tokens = len(tokens)
        
        # Check for cache hits on chunks
        chunks = [
            tokens[i:i+self.chunk_size] 
            for i in range(0, num_tokens, self.chunk_size)
        ]
        
        cache_hits = 0
        cache_misses = 0
        chunk_hashes = []
        
        for chunk in chunks:
            chunk_hash = self.compute_chunk_hash(chunk)
            chunk_hashes.append(chunk_hash)
            
            # Check if chunk exists in shared storage
            exists = await self.check_cache_exists(chunk_hash)
            
            if exists:
                cache_hits += 1
            else:
                cache_misses += 1
                # Simulate prefill computation for this chunk
                prefill_time = self.latency_calc.prefill_latency_ms(
                    len(chunk), batch_size=1
                )
                await asyncio.sleep(prefill_time / 1000)
                
                # Store KV cache for this chunk
                await self.store_kv_cache(chunk_hash, len(chunk))
        
        # Calculate KV transfer time
        transfer_time = self.latency_calc.kv_transfer_latency_ms(num_tokens)
        
        return {
            "request_id": request_id,
            "num_tokens": num_tokens,
            "chunk_hashes": chunk_hashes,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "transfer_time_ms": transfer_time,
        }
    
    async def check_cache_exists(self, chunk_hash: str) -> bool:
        """Check if KV cache exists for chunk."""
        for worker_id in range(self.config.tp_size):
            key = make_cache_key(
                self.config.model_profile,
                chunk_hash,
                self.config.tp_size,
                worker_id
            )
            if not await self.storage.exists(key):
                return False
        return True
    
    async def store_kv_cache(self, chunk_hash: str, num_tokens: int):
        """Store fake KV cache data for all TP workers."""
        model = MODEL_PROFILES[self.config.model_profile]
        kv_size_per_worker = (model.bytes_per_token_kv * num_tokens // 
                            self.config.tp_size)
        
        # Store for each tensor parallel worker
        for worker_id in range(self.config.tp_size):
            key = make_cache_key(
                self.config.model_profile,
                chunk_hash,
                self.config.tp_size,
                worker_id
            )
            
            # Store fake KV data
            fake_kv = b"x" * kv_size_per_worker
            await self.storage.put(key, fake_kv)
            
            # Store metadata
            meta_key = key.replace("@kv_bytes", "@metadata")
            await self.storage.put(meta_key, json.dumps({
                "num_tokens": num_tokens,
                "worker_id": worker_id,
                "tp_size": self.config.tp_size,
                "created_by": self.config.instance_id,
                "created_at": time.time(),
            }).encode())

prefill_server = None

@app.on_event("startup")
async def startup():
    global prefill_server
    config = InstanceConfig(
        instance_id=os.environ.get("INSTANCE_ID", "prefill-0"),
        instance_type="prefill",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        tp_size=int(os.environ.get("TP_SIZE", 1)),
        gpu_profile=os.environ.get("GPU_PROFILE", "H100_SXM"),
        model_profile=os.environ.get("MODEL_PROFILE", "llama-3.1-8b"),
    )
    prefill_server = PrefillServer(config)
    await prefill_server.registry.register(config)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    prompt = data["messages"][-1]["content"]
    request_id = f"req-{time.time_ns()}"
    
    tokens = prefill_server.tokenize(prompt)
    
    # Process prefill
    prefill_result = await prefill_server.prefill_request(tokens, request_id)
    
    async def generate():
        # Send first token after prefill completes
        first_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [{
                "delta": {"role": "assistant"},
                "index": 0
            }],
            # Include prefill metadata for debugging
            "prefill_info": prefill_result
        }
        yield f"data: {json.dumps(first_chunk)}\n\n"
        
        # Signal prefill complete - decode server can take over
        yield f"data: {json.dumps({'prefill_complete': True, 'request_id': request_id})}\n\n"
        yield "data: [DONE]\n\n"
    
    if data.get("stream", False):
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    # Non-streaming
    return {"choices": [{"message": {"content": "", "role": "assistant"}}]}
```

### 4.3 Decode Server Implementation

```python
class DecodeServer:
    def __init__(self, config: InstanceConfig):
        self.config = config
        self.storage = create_storage_client(config.storage_config)
        self.latency_calc = LatencyCalculator(
            gpu=GPU_PROFILES[config.gpu_profile],
            model=MODEL_PROFILES[config.model_profile],
            tp_size=config.tp_size
        )
        self.chunk_size = 256
    
    async def load_kv_cache(self, chunk_hashes: List[str]) -> dict:
        """
        Load KV cache from shared storage.
        
        Returns timing and cache statistics.
        """
        total_bytes = 0
        load_start = time.perf_counter()
        
        for chunk_hash in chunk_hashes:
            for worker_id in range(self.config.tp_size):
                key = make_cache_key(
                    self.config.model_profile,
                    chunk_hash,
                    self.config.tp_size,
                    worker_id
                )
                data = await self.storage.get(key)
                if data:
                    total_bytes += len(data)
        
        load_time = time.perf_counter() - load_start
        
        return {
            "total_bytes": total_bytes,
            "load_time_ms": load_time * 1000,
            "num_chunks": len(chunk_hashes),
        }
    
    async def decode_tokens(
        self,
        context_length: int,
        num_output_tokens: int,
        batch_size: int = 1
    ) -> AsyncGenerator[str, None]:
        """
        Generate output tokens with realistic decode latency.
        """
        for i in range(num_output_tokens):
            # Calculate decode latency based on current context
            current_context = context_length + i
            decode_time_ms = self.latency_calc.decode_latency_ms(
                current_context, batch_size
            )
            
            # Simulate decode step
            await asyncio.sleep(decode_time_ms / 1000)
            
            # Yield token
            yield f"token_{i}"

decode_server = None

@app.post("/v1/decode")
async def decode_completions(request: Request):
    """
    Decode endpoint - receives prefill metadata and generates tokens.
    """
    data = await request.json()
    
    # Load KV cache from shared storage
    chunk_hashes = data.get("chunk_hashes", [])
    context_length = data.get("context_length", 0)
    max_tokens = data.get("max_tokens", 100)
    
    # Load KV cache
    load_result = await decode_server.load_kv_cache(chunk_hashes)
    
    async def generate():
        # First, report KV load completion
        yield f"data: {json.dumps({'kv_loaded': True, **load_result})}\n\n"
        
        # Generate tokens
        async for token in decode_server.decode_tokens(
            context_length, max_tokens
        ):
            chunk = {
                "choices": [{"delta": {"content": token + " "}}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 4.4 Proxy/Router for Disaggregated Flow

```python
from fastapi import FastAPI
import httpx

class DisaggregatedProxy:
    """
    Routes requests between prefill and decode instances.
    
    Flow:
    1. Client sends request to proxy
    2. Proxy forwards to prefill instance
    3. Prefill stores KV and returns first token + metadata
    4. Proxy forwards decode request with KV metadata to decode instance
    5. Decode loads KV and continues streaming
    """
    
    def __init__(self):
        self.prefill_instances = []  # Discovered from registry
        self.decode_instances = []
        self.current_prefill_idx = 0
        self.current_decode_idx = 0
    
    def select_prefill_instance(self) -> str:
        """Round-robin selection (could be load-based)."""
        if not self.prefill_instances:
            raise RuntimeError("No prefill instances available")
        instance = self.prefill_instances[self.current_prefill_idx]
        self.current_prefill_idx = (
            (self.current_prefill_idx + 1) % len(self.prefill_instances)
        )
        return f"http://{instance.host}:{instance.port}"
    
    def select_decode_instance(self) -> str:
        """Round-robin selection."""
        if not self.decode_instances:
            raise RuntimeError("No decode instances available")
        instance = self.decode_instances[self.current_decode_idx]
        self.current_decode_idx = (
            (self.current_decode_idx + 1) % len(self.decode_instances)
        )
        return f"http://{instance.host}:{instance.port}"

proxy = DisaggregatedProxy()

@app.post("/v1/chat/completions")
async def proxy_completions(request: Request):
    data = await request.json()
    
    async def disaggregated_stream():
        async with httpx.AsyncClient() as client:
            # Step 1: Send to prefill
            prefill_url = proxy.select_prefill_instance()
            prefill_data = {**data, "stream": True}
            
            prefill_metadata = None
            
            async with client.stream(
                "POST",
                f"{prefill_url}/v1/chat/completions",
                json=prefill_data
            ) as prefill_response:
                async for line in prefill_response.aiter_lines():
                    if line.startswith("data: "):
                        content = line[6:]
                        if content == "[DONE]":
                            break
                        
                        chunk = json.loads(content)
                        
                        # Capture prefill metadata
                        if "prefill_complete" in chunk:
                            prefill_metadata = chunk
                            continue
                        
                        # Forward first token
                        if "choices" in chunk:
                            yield f"data: {json.dumps(chunk)}\n\n"
            
            # Step 2: Continue with decode
            if prefill_metadata:
                decode_url = proxy.select_decode_instance()
                decode_data = {
                    "chunk_hashes": prefill_metadata.get("prefill_info", {}).get("chunk_hashes", []),
                    "context_length": prefill_metadata.get("prefill_info", {}).get("num_tokens", 0),
                    "max_tokens": data.get("max_tokens", 100),
                }
                
                async with client.stream(
                    "POST",
                    f"{decode_url}/v1/decode",
                    json=decode_data
                ) as decode_response:
                    async for line in decode_response.aiter_lines():
                        if line.startswith("data: "):
                            yield line + "\n"
    
    if data.get("stream", False):
        return StreamingResponse(
            disaggregated_stream(),
            media_type="text/event-stream"
        )
    
    # Non-streaming implementation would collect all tokens
    pass
```

---

## 5. Storage Backend Implementations

### 5.1 Redis Cluster Backend

```python
import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster

class RedisStorageBackend:
    def __init__(self, config: dict):
        if config.get("cluster_mode"):
            self.client = RedisCluster(
                startup_nodes=[
                    redis.cluster.ClusterNode(n["host"], n["port"])
                    for n in config["nodes"]
                ],
                decode_responses=False,
            )
        else:
            self.client = redis.from_url(config["url"])
    
    async def get(self, key: str) -> bytes:
        return await self.client.get(key)
    
    async def put(self, key: str, value: bytes, ttl: int = None):
        if ttl:
            await self.client.setex(key, ttl, value)
        else:
            await self.client.set(key, value)
    
    async def exists(self, key: str) -> bool:
        return await self.client.exists(key) > 0
    
    async def delete(self, key: str):
        await self.client.delete(key)
```

### 5.2 Shared Filesystem Backend

```python
import aiofiles
import os
from pathlib import Path

class FilesystemStorageBackend:
    def __init__(self, config: dict):
        self.base_path = Path(config["path"])
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _key_to_path(self, key: str) -> Path:
        # Sanitize key for filesystem
        safe_key = key.replace("@", "/").replace(":", "_")
        return self.base_path / safe_key
    
    async def get(self, key: str) -> bytes:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        async with aiofiles.open(path, "rb") as f:
            return await f.read()
    
    async def put(self, key: str, value: bytes, ttl: int = None):
        path = self._key_to_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "wb") as f:
            await f.write(value)
    
    async def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()
    
    async def delete(self, key: str):
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
```

### 5.3 S3-Compatible Backend

```python
import aioboto3
from botocore.config import Config

class S3StorageBackend:
    def __init__(self, config: dict):
        self.endpoint = config["endpoint"]
        self.bucket = config["bucket"]
        self.session = aioboto3.Session()
        self.config = Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
        )
    
    def _key_to_object(self, key: str) -> str:
        # S3 keys can use / as delimiter
        return key.replace("@", "/")
    
    async def get(self, key: str) -> bytes:
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
            config=self.config
        ) as s3:
            try:
                response = await s3.get_object(
                    Bucket=self.bucket,
                    Key=self._key_to_object(key)
                )
                return await response["Body"].read()
            except s3.exceptions.NoSuchKey:
                return None
    
    async def put(self, key: str, value: bytes, ttl: int = None):
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
            config=self.config
        ) as s3:
            await s3.put_object(
                Bucket=self.bucket,
                Key=self._key_to_object(key),
                Body=value
            )
    
    async def exists(self, key: str) -> bool:
        async with self.session.client(
            "s3",
            endpoint_url=self.endpoint,
            config=self.config
        ) as s3:
            try:
                await s3.head_object(
                    Bucket=self.bucket,
                    Key=self._key_to_object(key)
                )
                return True
            except:
                return False
```

---

## 6. Deployment Configuration

### 6.1 Docker Compose for Multi-Host Simulation

```yaml
version: "3.8"

services:
  # Shared storage
  redis:
    image: redis:7
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
  
  # Prefill instances
  prefill-1:
    build: .
    environment:
      - INSTANCE_ID=prefill-1
      - INSTANCE_TYPE=prefill
      - PORT=8000
      - TP_SIZE=4
      - GPU_PROFILE=H100_SXM
      - MODEL_PROFILE=llama-3.1-70b
      - REDIS_URL=redis://redis:6379
    ports:
      - "8001:8000"
  
  prefill-2:
    build: .
    environment:
      - INSTANCE_ID=prefill-2
      - INSTANCE_TYPE=prefill
      - PORT=8000
      - TP_SIZE=4
      - GPU_PROFILE=H100_SXM
      - MODEL_PROFILE=llama-3.1-70b
      - REDIS_URL=redis://redis:6379
    ports:
      - "8002:8000"
  
  # Decode instances
  decode-1:
    build: .
    environment:
      - INSTANCE_ID=decode-1
      - INSTANCE_TYPE=decode
      - PORT=8000
      - TP_SIZE=4
      - GPU_PROFILE=H100_SXM
      - MODEL_PROFILE=llama-3.1-70b
      - REDIS_URL=redis://redis:6379
    ports:
      - "8101:8000"
  
  decode-2:
    build: .
    environment:
      - INSTANCE_ID=decode-2
      - INSTANCE_TYPE=decode
      - PORT=8000
      - TP_SIZE=4
      - GPU_PROFILE=H100_SXM
      - MODEL_PROFILE=llama-3.1-70b
      - REDIS_URL=redis://redis:6379
    ports:
      - "8102:8000"
  
  # Proxy/Router
  proxy:
    build: 
      context: .
      dockerfile: Dockerfile.proxy
    environment:
      - PREFILL_INSTANCES=prefill-1:8000,prefill-2:8000
      - DECODE_INSTANCES=decode-1:8000,decode-2:8000
      - REDIS_URL=redis://redis:6379
    ports:
      - "8000:8000"

volumes:
  redis-data:
```

### 6.2 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mock-prefill
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mock-llm
      role: prefill
  template:
    metadata:
      labels:
        app: mock-llm
        role: prefill
    spec:
      containers:
      - name: prefill
        image: mock-llm:latest
        env:
        - name: INSTANCE_TYPE
          value: "prefill"
        - name: TP_SIZE
          value: "4"
        - name: GPU_PROFILE
          value: "H100_SXM"
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: shared-storage
          mountPath: /mnt/shared
      volumes:
      - name: shared-storage
        persistentVolumeClaim:
          claimName: kv-cache-pvc
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mock-decode
spec:
  replicas: 5
  selector:
    matchLabels:
      app: mock-llm
      role: decode
  template:
    # Similar to prefill but with role: decode
```

---

## 7. Running Benchmarks

### 7.1 GenAI-Perf Against Mock Cluster

```bash
# Start the benchmark
genai-perf profile \
    -m llama-3.1-70b \
    --service-kind openai \
    --endpoint-type chat \
    --url http://proxy:8000 \
    --streaming \
    --synthetic-input-tokens-mean 2000 \
    --synthetic-input-tokens-stddev 500 \
    --output-tokens-mean 200 \
    --output-tokens-stddev 50 \
    --concurrency 32 \
    --request-count 1000 \
    --request-rate 10 \
    --generate-plots
```

### 7.2 Metrics to Collect

The system should expose Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# Cache metrics
cache_hits = Counter('kv_cache_hits_total', 'KV cache hits', ['instance', 'model'])
cache_misses = Counter('kv_cache_misses_total', 'KV cache misses', ['instance', 'model'])
cache_hit_ratio = Gauge('kv_cache_hit_ratio', 'KV cache hit ratio', ['instance', 'model'])

# Latency metrics
prefill_latency = Histogram('prefill_latency_seconds', 'Prefill latency',
                           ['instance', 'model'], 
                           buckets=[.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10])
decode_latency = Histogram('decode_latency_seconds', 'Per-token decode latency',
                          ['instance', 'model'],
                          buckets=[.001, .005, .01, .025, .05, .1, .25])
kv_transfer_latency = Histogram('kv_transfer_latency_seconds', 'KV transfer latency',
                               ['source', 'destination'],
                               buckets=[.01, .05, .1, .25, .5, 1, 2])

# Storage metrics
storage_ops = Counter('storage_operations_total', 'Storage operations',
                     ['operation', 'backend'])
storage_bytes = Counter('storage_bytes_total', 'Storage bytes transferred',
                       ['operation', 'backend'])
```

---

## 8. Summary

This design enables:

1. **GPU-Free Benchmarking**: All components run on CPU, using calculated sleep times to simulate GPU latency based on roofline models.

2. **Multi-Host Data Parallelism**: Shared storage (Redis/NFS/S3) allows KV caches created by any prefill host to be consumed by any decode host.

3. **Disaggregated Prefill/Decode**: Separate server pools for prefill (compute-bound) and decode (memory-bound) with realistic performance modeling.

4. **Configurable GPU Profiles**: Support for H100, A100, L4, etc. with accurate compute/memory bandwidth parameters.

5. **LMCache Compatibility**: Key format and storage patterns match LMCache conventions for potential integration with real systems.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Sleep-based latency | Simulates GPU timing without requiring GPUs |
| Roofline model | Industry-standard approach for compute vs memory bound analysis |
| Redis/shared storage | Enables cross-host KV sharing for data parallelism |
| Chunk-based hashing | Compatible with LMCache; enables cache reuse |
| Separate prefill/decode | Accurately models disaggregated architecture |
