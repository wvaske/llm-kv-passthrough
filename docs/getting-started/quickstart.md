# Quick Start

This guide will get you up and running with KV-Bench in minutes.

## Start the Server

```bash
# Start with defaults (combined server, memory storage)
kvbench serve

# Start with specific model and GPU profile
kvbench serve --model llama-3.1-70b --gpu H100_SXM

# Point LMCache at specific storage tiers (disk, Redis, ...)
kvbench serve --lmcache-config lmcache.yaml
```

## Make API Requests

### Non-Streaming Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### Streaming Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

## Check Server Health

```bash
curl http://localhost:8000/health
```

## View Metrics

```bash
curl http://localhost:8000/metrics
```

## Python Client Example

```python
import httpx
import asyncio

async def main():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 50,
            },
        )
        print(response.json())

asyncio.run(main())
```

## Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # KV-Bench doesn't require auth
)

response = client.chat.completions.create(
    model="llama-3.1-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50,
)
print(response.choices[0].message.content)
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize your deployment
- [Architecture Overview](../architecture/overview.md) - Understand the system
- [GenAI-Perf Benchmarking](../benchmarking/genai-perf.md) - Run benchmarks
