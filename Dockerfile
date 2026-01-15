# KV-Bench Dockerfile
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r <(echo "uvicorn[standard]>=0.25.0")

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
COPY --from=builder /build/dist/*.whl /wheels/
COPY --from=builder /wheels/*.whl /wheels/

# Install the package
RUN pip install --no-cache-dir /wheels/*.whl && \
    rm -rf /wheels

# Create directories for data
RUN mkdir -p /var/lib/kvbench/nvme /var/lib/kvbench/data && \
    chown -R kvbench:kvbench /var/lib/kvbench

# Switch to non-root user
USER kvbench

# Environment variables with defaults
ENV KVBENCH_SERVER__HOST=0.0.0.0 \
    KVBENCH_SERVER__PORT=8000 \
    KVBENCH_SERVER__SERVER_TYPE=combined \
    KVBENCH_SERVER__MODEL_PROFILE=llama-3.1-8b \
    KVBENCH_GPU__GPU_PROFILE=H100_SXM \
    KVBENCH_STORAGE__BACKEND_TYPE=memory \
    KVBENCH_RESOURCES__NVME_PATH=/var/lib/kvbench/nvme

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
ENTRYPOINT ["kvbench"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
