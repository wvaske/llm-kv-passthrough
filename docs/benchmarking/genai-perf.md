# GenAI-Perf Benchmarking

KV-Bench is compatible with NVIDIA's [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf) tool for standardized LLM benchmarking.

## Overview

GenAI-Perf provides comprehensive performance metrics for LLM inference:

- **Throughput**: Requests/second, tokens/second
- **Latency**: P50, P90, P99 latencies
- **TTFT**: Time to first token
- **ITL**: Inter-token latency

## Installation

```bash
pip install genai-perf
```

Or with Docker:

```bash
docker pull nvcr.io/nvidia/tritonserver:24.01-py3-sdk
```

## Basic Usage

### Simple Benchmark

```bash
# Start KV-Bench server
kvbench serve --model llama-3.1-8b --gpu H100_SXM

# Run GenAI-Perf
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --concurrency 10 \
  --num-requests 100
```

### Streaming Benchmark

```bash
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --streaming \
  --concurrency 10 \
  --num-requests 100
```

## Configuration Options

### Concurrency Sweep

Test performance across different concurrency levels:

```bash
for concurrency in 1 2 4 8 16 32; do
  genai-perf \
    --endpoint http://localhost:8000/v1/chat/completions \
    --model llama-3.1-8b \
    --concurrency $concurrency \
    --num-requests 100 \
    --output results_c${concurrency}.json
done
```

### Input Length Variation

Test with different prompt lengths:

```bash
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --input-tokens-mean 512 \
  --input-tokens-stddev 64 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 32
```

### Custom Dataset

Use custom prompts for benchmarking:

```bash
# Create dataset file
cat > prompts.jsonl << 'EOF'
{"prompt": "What is machine learning?"}
{"prompt": "Explain the theory of relativity"}
{"prompt": "Write a Python function to sort a list"}
EOF

genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --dataset prompts.jsonl
```

## Benchmark Scenarios

### Single Server

```bash
# Combined mode benchmark
kvbench serve --type combined &

genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --concurrency 16 \
  --num-requests 1000 \
  --output single_server.json
```

### Distributed Deployment

```bash
# Start distributed cluster
docker-compose -f docker-compose.distributed.yml up -d

# Benchmark through proxy
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --concurrency 32 \
  --num-requests 1000 \
  --output distributed.json
```

### Cache Hit/Miss Comparison

```bash
# First run - cache cold
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --dataset shared_prefix_prompts.jsonl \
  --num-requests 100 \
  --output cache_cold.json

# Second run - cache warm
genai-perf \
  --endpoint http://localhost:8000/v1/chat/completions \
  --model llama-3.1-8b \
  --dataset shared_prefix_prompts.jsonl \
  --num-requests 100 \
  --output cache_warm.json
```

## Output Metrics

GenAI-Perf provides detailed metrics:

```json
{
  "request_throughput": 45.2,
  "output_token_throughput": 1824.5,
  "time_to_first_token": {
    "p50": 0.023,
    "p90": 0.045,
    "p99": 0.089
  },
  "inter_token_latency": {
    "p50": 0.012,
    "p90": 0.018,
    "p99": 0.028
  },
  "request_latency": {
    "p50": 0.234,
    "p90": 0.456,
    "p99": 0.789
  }
}
```

## Analysis Scripts

### Compare Results

```python
import json
import pandas as pd

def compare_results(files):
    results = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            results.append({
                'file': f,
                'throughput': data['request_throughput'],
                'ttft_p50': data['time_to_first_token']['p50'],
                'ttft_p99': data['time_to_first_token']['p99'],
            })
    return pd.DataFrame(results)

df = compare_results(['cache_cold.json', 'cache_warm.json'])
print(df.to_markdown())
```

### Generate Report

```bash
# Use built-in script
scripts/genai_perf_test.sh

# Or generate custom report
python -c "
import json
with open('results.json') as f:
    data = json.load(f)
print(f'Throughput: {data[\"request_throughput\"]:.2f} req/s')
print(f'TTFT P50: {data[\"time_to_first_token\"][\"p50\"]*1000:.2f} ms')
print(f'TTFT P99: {data[\"time_to_first_token\"][\"p99\"]*1000:.2f} ms')
"
```

## Best Practices

1. **Warm up**: Run a few requests before benchmarking
2. **Consistent load**: Use fixed concurrency for comparisons
3. **Realistic prompts**: Use production-like input distributions
4. **Multiple runs**: Average results across multiple runs
5. **Monitor resources**: Check CPU/memory during benchmarks

## Troubleshooting

### Connection Errors

```bash
# Check server is running
curl http://localhost:8000/health

# Check endpoint format
genai-perf --endpoint http://localhost:8000/v1/chat/completions ...
```

### Timeout Errors

```bash
# Increase timeout
genai-perf --request-timeout 60 ...

# Reduce concurrency
genai-perf --concurrency 4 ...
```

### Memory Issues

```bash
# Reduce batch size
genai-perf --batch-size 1 ...

# Use streaming
genai-perf --streaming ...
```
