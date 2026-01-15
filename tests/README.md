# KV-Bench Tests

Test suite for KV-Bench.

## Structure

```
tests/
├── unit/                # Unit tests
│   ├── test_config.py   # Configuration tests
│   ├── test_profiles.py # GPU/Model profile tests
│   ├── test_connectors.py # Connector tests
│   ├── test_storage.py  # Storage backend tests
│   └── test_servers.py  # Server component tests
├── e2e/                 # End-to-end tests
│   ├── test_server.py   # HTTP server tests
│   └── test_distributed.py # Distributed deployment tests
└── conftest.py          # Pytest fixtures
```

## Running Tests

### All Tests

```bash
pytest
```

### With Coverage

```bash
pytest --cov=kvbench --cov-report=html
```

Or use the script:

```bash
./scripts/coverage_report.sh
```

### Specific Test File

```bash
pytest tests/unit/test_config.py
```

### Specific Test

```bash
pytest tests/unit/test_config.py::test_config_from_env -v
```

### By Marker

```bash
# Async tests only
pytest -m asyncio

# Skip slow tests
pytest -m "not slow"
```

## Test Categories

| Category | Description | Location |
|----------|-------------|----------|
| Unit | Individual component tests | `tests/unit/` |
| E2E | Full system integration | `tests/e2e/` |

## Writing Tests

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    result = await some_async_function()
    assert result == expected
```

### Using Fixtures

```python
def test_with_storage(memory_storage):
    # memory_storage is provided by conftest.py
    assert memory_storage.stats.used_bytes == 0
```

## Test Coverage Target

- Minimum: 80%
- Target: 90%+
