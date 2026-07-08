"""
KV-Bench Configuration System.

This module provides the configuration system for KV-Bench using Pydantic v2.
Configuration can be loaded from environment variables, YAML files, or programmatically.

Environment variables use the prefix KVBENCH_ and nested delimiter __.
Example: KVBENCH_RESOURCES__CPU_MEMORY_GB=16.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ResourceLimits(BaseModel):
    """Resource limits for CPU memory and NVMe storage.

    Attributes:
        cpu_memory_gb: Maximum CPU memory allocation in gigabytes.
        nvme_storage_gb: Maximum NVMe storage allocation in gigabytes.
        nvme_path: Path to NVMe storage directory.
        memory_allocation: Memory allocation strategy.
    """

    cpu_memory_gb: float = Field(
        default=8.0,
        ge=0.1,
        le=1024.0,
        description="Maximum CPU memory allocation in GB",
    )
    nvme_storage_gb: float = Field(
        default=100.0,
        ge=0.0,
        le=10000.0,
        description="Maximum NVMe storage allocation in GB",
    )
    nvme_path: Path = Field(
        default=Path("/var/lib/kvbench/nvme"),
        description="Path to NVMe storage directory",
    )
    memory_allocation: Literal["greedy", "lazy", "pool"] = Field(
        default="lazy",
        description="Memory allocation strategy",
    )

    @field_validator("nvme_path", mode="before")
    @classmethod
    def convert_nvme_path(cls, v: str | Path) -> Path:
        """Convert string path to Path object."""
        return Path(v) if isinstance(v, str) else v

    @property
    def cpu_memory_bytes(self) -> int:
        """Return CPU memory limit in bytes."""
        return int(self.cpu_memory_gb * 1024**3)

    @property
    def nvme_storage_bytes(self) -> int:
        """Return NVMe storage limit in bytes."""
        return int(self.nvme_storage_gb * 1024**3)


class StorageConfig(BaseModel):
    """Storage backend configuration.

    Attributes:
        backend_type: Type of storage backend to use.
        redis_url: Redis connection URL for redis backend.
        redis_cluster: Whether to use Redis cluster mode.
        filesystem_path: Path for filesystem-based backends (NFS, Weka).
        s3_endpoint: S3-compatible endpoint URL for MinIO.
        s3_bucket: S3 bucket name for MinIO.
        s3_access_key: S3 access key for authentication.
        s3_secret_key: S3 secret key for authentication.
        ceph_pool: Ceph pool name for Ceph backend.
        ceph_conf: Path to Ceph configuration file.
    """

    backend_type: Literal["memory", "local_disk", "redis", "nfs", "ceph", "weka", "minio"] = Field(
        default="memory",
        description="Type of storage backend",
    )
    redis_url: str | None = Field(
        default=None,
        description="Redis connection URL (e.g., redis://localhost:6379)",
    )
    redis_cluster: bool = Field(
        default=False,
        description="Whether to use Redis cluster mode",
    )
    filesystem_path: Path | None = Field(
        default=None,
        description="Path for filesystem-based backends",
    )
    s3_endpoint: str | None = Field(
        default=None,
        description="S3-compatible endpoint URL",
    )
    s3_bucket: str | None = Field(
        default=None,
        description="S3 bucket name",
    )
    s3_access_key: str | None = Field(
        default=None,
        description="S3 access key",
    )
    s3_secret_key: str | None = Field(
        default=None,
        description="S3 secret key",
    )
    ceph_pool: str | None = Field(
        default=None,
        description="Ceph pool name",
    )
    ceph_conf: Path | None = Field(
        default=None,
        description="Path to Ceph configuration file",
    )

    @field_validator("filesystem_path", "ceph_conf", mode="before")
    @classmethod
    def convert_path(cls, v: str | Path | None) -> Path | None:
        """Convert string path to Path object."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_backend_requirements(self) -> StorageConfig:
        """Validate that required fields are set for each backend type."""
        if self.backend_type == "redis" and not self.redis_url:
            raise ValueError("redis_url is required when backend_type is 'redis'")
        if self.backend_type in ("nfs", "weka") and not self.filesystem_path:
            raise ValueError(
                f"filesystem_path is required when backend_type is '{self.backend_type}'"
            )
        if self.backend_type == "minio":
            if not self.s3_endpoint:
                raise ValueError("s3_endpoint is required when backend_type is 'minio'")
            if not self.s3_bucket:
                raise ValueError("s3_bucket is required when backend_type is 'minio'")
        if self.backend_type == "ceph" and not self.ceph_pool:
            raise ValueError("ceph_pool is required when backend_type is 'ceph'")
        return self


class ConnectorConfig(BaseModel):
    """KV connector configuration for cache systems.

    Attributes:
        connector_type: Type of KV connector to use.
        lmcache_chunk_size: Chunk size for LMCache connector.
        lmcache_remote_url: Remote URL for LMCache server.
        mooncake_endpoint: Endpoint for Mooncake connector.
        dynamo_table: DynamoDB table name for Dynamo connector.
    """

    connector_type: Literal["lmcache", "mooncake", "dynamo", "mock"] = Field(
        default="lmcache",
        description="Type of KV connector",
    )
    lmcache_chunk_size: int = Field(
        default=256,
        ge=16,
        le=4096,
        description="Chunk size for LMCache (in tokens)",
    )
    lmcache_remote_url: str | None = Field(
        default=None,
        description="Remote URL for LMCache server (e.g., lm://localhost:8080)",
    )
    mooncake_endpoint: str | None = Field(
        default=None,
        description="Endpoint for Mooncake connector",
    )
    dynamo_table: str | None = Field(
        default=None,
        description="DynamoDB table name for Dynamo connector",
    )


class GPUEmulationConfig(BaseModel):
    """GPU emulation configuration for latency calculations.

    Attributes:
        gpu_profile: Name of the GPU profile to emulate.
        efficiency_factor: GPU efficiency factor (0.1 to 1.0).
        tp_size: Tensor parallelism size.
    """

    gpu_profile: str = Field(
        default="H100_SXM",
        description="Name of the GPU profile to emulate",
    )

    @field_validator("gpu_profile")
    @classmethod
    def validate_gpu_profile(cls, v: str) -> str:
        """Validate that the GPU profile exists in the registry."""
        from kvbench.core.gpu_profiles import GPU_PROFILES

        if v not in GPU_PROFILES:
            available = ", ".join(sorted(GPU_PROFILES.keys()))
            raise ValueError(f"Unknown GPU profile: {v!r}. Available profiles: {available}")
        return v

    efficiency_factor: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="GPU efficiency factor",
    )
    tp_size: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Tensor parallelism size",
    )


class ServerConfig(BaseModel):
    """Server configuration for the KV-Bench HTTP server.

    Attributes:
        host: Host address to bind to.
        port: Port number to listen on.
        server_type: Type of server to run.
        model_profile: Name of the model profile to emulate.
        workers: Number of worker processes.
        log_level: Logging level.
    """

    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind to",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port number to listen on",
    )
    server_type: Literal["prefill", "decode", "combined", "proxy"] = Field(
        default="combined",
        description="Type of server to run",
    )
    model_profile: str = Field(
        default="llama-3.1-8b",
        description="Name of the model profile to emulate",
    )

    @field_validator("model_profile")
    @classmethod
    def validate_model_profile(cls, v: str) -> str:
        """Validate that the model profile exists in the registry."""
        from kvbench.core.models import MODEL_PROFILES

        if v not in MODEL_PROFILES:
            available = ", ".join(sorted(MODEL_PROFILES.keys()))
            raise ValueError(f"Unknown model profile: {v!r}. Available profiles: {available}")
        return v

    workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes",
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Logging level",
    )


class DistributedConfig(BaseModel):
    """Distributed deployment configuration.

    Attributes:
        prefill_endpoints: List of prefill server endpoints for proxy mode.
        decode_endpoints: List of decode server endpoints for proxy mode.
        registry_url: URL of the service registry.
        health_check_interval: Health check interval in seconds.
    """

    prefill_endpoints: list[str] = Field(
        default_factory=list,
        description="List of prefill server endpoints",
    )
    decode_endpoints: list[str] = Field(
        default_factory=list,
        description="List of decode server endpoints",
    )

    @field_validator("prefill_endpoints", "decode_endpoints", mode="before")
    @classmethod
    def parse_endpoint_list(cls, v: object) -> object:
        """Accept endpoint lists given as strings (env vars).

        Supports JSON arrays ('["http://a:8000", "http://b:8000"]') and
        comma-separated values ('http://a:8000,http://b:8000').
        """
        if isinstance(v, str):
            import json

            stripped = v.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return v

    registry_url: str | None = Field(
        default=None,
        description="URL of the service registry",
    )
    health_check_interval: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Health check interval in seconds",
    )


class MetricsConfig(BaseModel):
    """Metrics and monitoring configuration.

    Attributes:
        enabled: Whether metrics collection is enabled.
        prometheus_port: Port for Prometheus metrics endpoint.
        include_histograms: Whether to include histogram metrics.
    """

    enabled: bool = Field(
        default=True,
        description="Whether metrics collection is enabled",
    )
    prometheus_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port for Prometheus metrics endpoint",
    )
    include_histograms: bool = Field(
        default=True,
        description="Whether to include histogram metrics",
    )


class KVBenchConfig(BaseModel):
    """Main KV-Bench configuration.

    This is the root configuration object that contains all sub-configurations.
    Configuration can be loaded from environment variables using the prefix KVBENCH_
    and nested delimiter __.

    Attributes:
        instance_id: Unique identifier for this KV-Bench instance.
        resources: Resource limits configuration.
        storage: Storage backend configuration.
        connector: KV connector configuration.
        gpu: GPU emulation configuration.
        server: Server configuration.
        distributed: Distributed deployment configuration.
        metrics: Metrics configuration.
    """

    instance_id: str = Field(
        default="kvbench-0",
        description="Unique identifier for this instance",
    )
    resources: ResourceLimits = Field(
        default_factory=ResourceLimits,
        description="Resource limits configuration",
    )
    storage: StorageConfig = Field(
        default_factory=StorageConfig,
        description="Storage backend configuration",
    )
    connector: ConnectorConfig = Field(
        default_factory=ConnectorConfig,
        description="KV connector configuration",
    )
    gpu: GPUEmulationConfig = Field(
        default_factory=GPUEmulationConfig,
        description="GPU emulation configuration",
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Server configuration",
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Distributed deployment configuration",
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration",
    )

    @classmethod
    def from_env(cls) -> KVBenchConfig:
        """Load configuration from environment variables.

        Environment variables should be prefixed with KVBENCH_ and use
        double underscores (__) for nested values.

        Returns:
            KVBenchConfig instance populated from environment variables.
        """
        import os

        def get_nested_env(prefix: str = "KVBENCH") -> dict:
            """Extract nested configuration from environment variables."""
            result: dict = {}
            for key, value in os.environ.items():
                if not key.startswith(f"{prefix}_"):
                    continue
                # Remove prefix and split by delimiter
                parts = key[len(prefix) + 1 :].lower().split("__")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result

        env_config = get_nested_env()
        return cls.model_validate(env_config)

    @classmethod
    def from_yaml(cls, path: Path | str) -> KVBenchConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            KVBenchConfig instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is invalid.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML configuration file.
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, default_flow_style=False)
