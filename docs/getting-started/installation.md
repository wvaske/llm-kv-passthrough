# Installation

## Requirements

- Python 3.11+
- pip or pipx

## Install from PyPI

```bash
pip install kvbench
```

## Install from Source

```bash
git clone https://github.com/your-org/kv-bench.git
cd kv-bench
pip install -e ".[dev]"
```

## Docker Installation

```bash
# Build the image
docker build -t kvbench .

# Run the container
docker run -p 8000:8000 kvbench
```

## Verify Installation

```bash
# Check version
kvbench version

# List available profiles
kvbench list-profiles

# Start server
kvbench serve --help
```

## Optional Dependencies

### Redis Backend
```bash
pip install "kvbench[redis]"
```

### S3/MinIO Backend
```bash
pip install "kvbench[s3]"
```

### All Backends
```bash
pip install "kvbench[all]"
```
