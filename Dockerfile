# KV-Bench Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (README.md is required by the build backend)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build the kvbench wheel and collect all runtime wheels (including the
# LMCache KV management stack, which runs CPU-only)
RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /wheels && \
    pip wheel --no-cache-dir --wheel-dir /wheels "/wheels/$(ls /wheels | head -n1)[lmcache]"

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r kvbench && useradd -r -g kvbench kvbench

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels from builder
COPY --from=builder /wheels/*.whl /wheels/

# Install kvbench with the LMCache stack
RUN pip install --no-cache-dir --no-index --find-links /wheels "kvbench[lmcache]" && \
    rm -rf /wheels

# Directories for LMCache's disk tier and kvbench data
RUN mkdir -p /var/lib/lmcache /var/lib/kvbench && \
    chown -R kvbench:kvbench /var/lib/lmcache /var/lib/kvbench

# Switch to non-root user
USER kvbench

# Environment variables with defaults. Storage is configured through
# LMCache itself: mount a config file and set
# KVBENCH_KV__LMCACHE_CONFIG_FILE, or use LMCACHE_* variables.
ENV KVBENCH_SERVER__HOST=0.0.0.0 \
    KVBENCH_SERVER__PORT=8000 \
    KVBENCH_SERVER__SERVER_TYPE=combined \
    KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b \
    KVBENCH_GPU__GPU_PROFILE=H100_SXM \
    LMCACHE_CHUNK_SIZE=256 \
    LMCACHE_LOCAL_DISK="file:///var/lib/lmcache/" \
    LMCACHE_MAX_LOCAL_DISK_SIZE=20

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
ENTRYPOINT ["kvbench"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
