# KV-Bench Source Code

Main source code for the KV-Bench package.

## Package Structure

```
kvbench/
├── __init__.py          # Package exports
├── cli/                 # Command-line interface
│   └── main.py          # Typer CLI application
├── core/                # Core components
│   ├── config.py        # Configuration management
│   └── profiles.py      # GPU and model profiles
├── connectors/          # KV cache connectors
│   ├── base.py          # Base connector interface
│   ├── lmcache.py       # LMCache connector
│   └── mooncake.py      # Mooncake connector
├── storage/             # Storage backends
│   ├── base.py          # Base storage interface
│   ├── memory.py        # In-memory backend
│   ├── local_disk.py    # Local disk backend
│   ├── redis_backend.py # Redis backend
│   ├── s3.py            # S3/MinIO backend
│   ├── nfs.py           # NFS backend
│   ├── weka.py          # Weka backend
│   └── mooncake.py      # Mooncake backend
└── servers/             # Server implementations
    ├── app.py           # FastAPI application
    ├── openai_compat.py # OpenAI API models
    ├── combined.py      # Combined server
    ├── prefill.py       # Prefill server
    ├── decode.py        # Decode server
    └── proxy.py         # Disaggregated proxy
```

## Key Components

### Core

- **config.py**: Pydantic configuration models with environment variable support
- **profiles.py**: GPU and model specifications for timing calculations

### Connectors

- **base.py**: Abstract interface for KV cache connectors
- **lmcache.py**: LMCache-compatible chunked storage
- **mooncake.py**: Mooncake transfer engine integration

### Storage

- **base.py**: Abstract storage backend interface
- Seven backend implementations for different deployment needs

### Servers

- **app.py**: FastAPI application with OpenAI-compatible endpoints
- **combined.py**: Single-server mode
- **prefill.py/decode.py**: Disaggregated serving
- **proxy.py**: Load balancer for distributed deployment

## Usage

```python
from kvbench import KVBenchConfig
from kvbench.servers.app import KVBenchApp

config = KVBenchConfig()
app = KVBenchApp(config)
```
