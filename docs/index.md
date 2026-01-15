# KV-Bench

**Distributed Mock LLM Server for KV Cache Benchmarking**

KV-Bench is a high-fidelity mock LLM serving system designed for benchmarking KV cache management strategies without requiring actual GPUs. It emulates the latency characteristics of real LLM inference while providing full control over storage backends and caching behavior.

## Features

- **GPU Emulation**: Accurate latency modeling based on roofline analysis for H100, H200, A100, L4, and L40S GPUs
- **Model Profiles**: Pre-configured profiles for Llama 3.1, Qwen 2.5, and Mistral model families
- **7 Storage Backends**: Memory, Local Disk, Redis, NFS, Ceph, Weka, and MinIO
- **3 KV Connectors**: LMCache (full), Mooncake (stub), Dynamo (stub)
- **4 Server Types**: Combined, Prefill, Decode, and Proxy for disaggregated serving
- **OpenAI-Compatible API**: Works with GenAI-Perf and other OpenAI-compatible tools

## Quick Start

```bash
# Install KV-Bench
pip install kvbench

# Start the server
kvbench serve --model llama-3.1-8b --gpu H100_SXM

# Test the API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.1-8b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     KV-Bench System                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Prefill   │  │   Decode    │  │   Disaggregated     │  │
│  │   Server    │  │   Server    │  │      Proxy          │  │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │
│         │                │                     │             │
│         └────────────────┼─────────────────────┘             │
│                          │                                   │
│                    ┌─────┴─────┐                             │
│                    │    KV     │                             │
│                    │ Connector │                             │
│                    └─────┬─────┘                             │
│                          │                                   │
│         ┌────────────────┼────────────────┐                  │
│         │                │                │                  │
│    ┌────┴────┐     ┌─────┴────┐    ┌──────┴─────┐           │
│    │ Memory  │     │  Redis   │    │    NFS     │           │
│    │ Backend │     │  Backend │    │  Backend   │  ...      │
│    └─────────┘     └──────────┘    └────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

## Use Cases

1. **Benchmark KV Cache Systems**: Test LMCache, Mooncake, or custom caching without GPU costs
2. **Evaluate Storage Backends**: Compare Redis vs NFS vs distributed storage for KV caching
3. **Test Disaggregated Serving**: Validate prefill/decode separation strategies
4. **Load Testing**: Use GenAI-Perf to stress test your infrastructure
5. **Development**: Build and test LLM serving features without GPU access

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Configuration Reference](getting-started/configuration.md)
- [Architecture Deep Dive](architecture/overview.md)
