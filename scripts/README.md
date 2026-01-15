# KV-Bench Scripts

Utility scripts for testing and benchmarking KV-Bench.

## Scripts

| Script | Description |
|--------|-------------|
| `coverage_report.sh` | Run tests with coverage analysis |
| `genai_perf_test.sh` | GenAI-Perf integration test |
| `lmcache_test.sh` | LMCache integration test |

## Usage

### Coverage Report

Generate test coverage report:

```bash
./scripts/coverage_report.sh
```

Output includes:
- Console coverage summary
- HTML report in `htmlcov/`
- XML report for CI integration

### GenAI-Perf Test

Run performance benchmarks with GenAI-Perf:

```bash
# Default settings
./scripts/genai_perf_test.sh

# Custom endpoint
KVBENCH_ENDPOINT=http://myserver:8000 ./scripts/genai_perf_test.sh

# Custom concurrency
CONCURRENCY=32 NUM_REQUESTS=500 ./scripts/genai_perf_test.sh
```

### LMCache Test

Test KV cache hit/miss behavior:

```bash
# Default settings
./scripts/lmcache_test.sh

# With real LMCache server
LMCACHE_URL=lm://localhost:8080 ./scripts/lmcache_test.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KVBENCH_ENDPOINT` | `http://localhost:8000` | Server endpoint |
| `KVBENCH_MODEL` | `llama-3.1-8b` | Model to test |
| `CONCURRENCY` | `10` | Concurrent requests |
| `NUM_REQUESTS` | `100` | Total requests |
| `LMCACHE_URL` | - | Optional LMCache server |
