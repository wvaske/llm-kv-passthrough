"""Unit tests for kvbench.core.config module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from kvbench.core.config import (
    ConnectorConfig,
    DistributedConfig,
    GPUEmulationConfig,
    KVBenchConfig,
    MetricsConfig,
    ResourceLimits,
    ServerConfig,
    StorageConfig,
)


class TestResourceLimits:
    """Tests for ResourceLimits configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = ResourceLimits()
        assert config.cpu_memory_gb == 8.0
        assert config.nvme_storage_gb == 100.0
        assert config.nvme_path == Path("/var/lib/kvbench/nvme")
        assert config.memory_allocation == "lazy"

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = ResourceLimits(
            cpu_memory_gb=16.0,
            nvme_storage_gb=500.0,
            nvme_path=Path("/custom/path"),
            memory_allocation="greedy",
        )
        assert config.cpu_memory_gb == 16.0
        assert config.nvme_storage_gb == 500.0
        assert config.nvme_path == Path("/custom/path")
        assert config.memory_allocation == "greedy"

    def test_path_string_conversion(self) -> None:
        """Test string paths are converted to Path objects."""
        config = ResourceLimits(nvme_path="/string/path")  # type: ignore[arg-type]
        assert config.nvme_path == Path("/string/path")
        assert isinstance(config.nvme_path, Path)

    def test_cpu_memory_bytes_property(self) -> None:
        """Test cpu_memory_bytes property calculation."""
        config = ResourceLimits(cpu_memory_gb=8.0)
        assert config.cpu_memory_bytes == 8 * 1024**3

    def test_nvme_storage_bytes_property(self) -> None:
        """Test nvme_storage_bytes property calculation."""
        config = ResourceLimits(nvme_storage_gb=100.0)
        assert config.nvme_storage_bytes == 100 * 1024**3

    def test_cpu_memory_gb_validation_min(self) -> None:
        """Test cpu_memory_gb minimum validation."""
        with pytest.raises(ValueError):
            ResourceLimits(cpu_memory_gb=0.05)

    def test_cpu_memory_gb_validation_max(self) -> None:
        """Test cpu_memory_gb maximum validation."""
        with pytest.raises(ValueError):
            ResourceLimits(cpu_memory_gb=2000.0)

    def test_nvme_storage_gb_validation_min(self) -> None:
        """Test nvme_storage_gb minimum validation (0 is allowed)."""
        config = ResourceLimits(nvme_storage_gb=0.0)
        assert config.nvme_storage_gb == 0.0

    def test_nvme_storage_gb_validation_max(self) -> None:
        """Test nvme_storage_gb maximum validation."""
        with pytest.raises(ValueError):
            ResourceLimits(nvme_storage_gb=20000.0)

    def test_memory_allocation_literal(self) -> None:
        """Test memory_allocation accepts valid literals."""
        for allocation in ["greedy", "lazy", "pool"]:
            config = ResourceLimits(memory_allocation=allocation)  # type: ignore[arg-type]
            assert config.memory_allocation == allocation

    def test_memory_allocation_invalid(self) -> None:
        """Test memory_allocation rejects invalid values."""
        with pytest.raises(ValueError):
            ResourceLimits(memory_allocation="invalid")  # type: ignore[arg-type]


class TestStorageConfig:
    """Tests for StorageConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = StorageConfig()
        assert config.backend_type == "memory"
        assert config.redis_url is None
        assert config.redis_cluster is False
        assert config.filesystem_path is None

    def test_redis_backend_valid(self) -> None:
        """Test valid Redis configuration."""
        config = StorageConfig(
            backend_type="redis",
            redis_url="redis://localhost:6379",
        )
        assert config.backend_type == "redis"
        assert config.redis_url == "redis://localhost:6379"

    def test_redis_backend_missing_url(self) -> None:
        """Test Redis backend requires redis_url."""
        with pytest.raises(ValueError, match="redis_url is required"):
            StorageConfig(backend_type="redis")

    def test_nfs_backend_valid(self) -> None:
        """Test valid NFS configuration."""
        config = StorageConfig(
            backend_type="nfs",
            filesystem_path=Path("/mnt/nfs"),
        )
        assert config.backend_type == "nfs"
        assert config.filesystem_path == Path("/mnt/nfs")

    def test_nfs_backend_missing_path(self) -> None:
        """Test NFS backend requires filesystem_path."""
        with pytest.raises(ValueError, match="filesystem_path is required"):
            StorageConfig(backend_type="nfs")

    def test_weka_backend_valid(self) -> None:
        """Test valid Weka configuration."""
        config = StorageConfig(
            backend_type="weka",
            filesystem_path=Path("/mnt/weka"),
        )
        assert config.backend_type == "weka"

    def test_weka_backend_missing_path(self) -> None:
        """Test Weka backend requires filesystem_path."""
        with pytest.raises(ValueError, match="filesystem_path is required"):
            StorageConfig(backend_type="weka")

    def test_minio_backend_valid(self) -> None:
        """Test valid MinIO configuration."""
        config = StorageConfig(
            backend_type="minio",
            s3_endpoint="http://localhost:9000",
            s3_bucket="kvbench",
        )
        assert config.backend_type == "minio"
        assert config.s3_endpoint == "http://localhost:9000"
        assert config.s3_bucket == "kvbench"

    def test_minio_backend_missing_endpoint(self) -> None:
        """Test MinIO backend requires s3_endpoint."""
        with pytest.raises(ValueError, match="s3_endpoint is required"):
            StorageConfig(backend_type="minio", s3_bucket="kvbench")

    def test_minio_backend_missing_bucket(self) -> None:
        """Test MinIO backend requires s3_bucket."""
        with pytest.raises(ValueError, match="s3_bucket is required"):
            StorageConfig(backend_type="minio", s3_endpoint="http://localhost:9000")

    def test_ceph_backend_valid(self) -> None:
        """Test valid Ceph configuration."""
        config = StorageConfig(
            backend_type="ceph",
            ceph_pool="kvbench",
        )
        assert config.backend_type == "ceph"
        assert config.ceph_pool == "kvbench"

    def test_ceph_backend_missing_pool(self) -> None:
        """Test Ceph backend requires ceph_pool."""
        with pytest.raises(ValueError, match="ceph_pool is required"):
            StorageConfig(backend_type="ceph")

    def test_path_string_conversion(self) -> None:
        """Test string paths are converted to Path objects."""
        config = StorageConfig(
            backend_type="nfs",
            filesystem_path="/string/path",  # type: ignore[arg-type]
            ceph_conf="/etc/ceph.conf",  # type: ignore[arg-type]
        )
        assert config.filesystem_path == Path("/string/path")
        assert config.ceph_conf == Path("/etc/ceph.conf")

    def test_local_disk_backend(self) -> None:
        """Test local_disk backend (no special requirements)."""
        config = StorageConfig(backend_type="local_disk")
        assert config.backend_type == "local_disk"


class TestConnectorConfig:
    """Tests for ConnectorConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = ConnectorConfig()
        assert config.connector_type == "lmcache"
        assert config.lmcache_chunk_size == 256
        assert config.lmcache_remote_url is None

    def test_lmcache_config(self) -> None:
        """Test LMCache configuration."""
        config = ConnectorConfig(
            connector_type="lmcache",
            lmcache_chunk_size=512,
            lmcache_remote_url="lm://localhost:8080",
        )
        assert config.lmcache_chunk_size == 512
        assert config.lmcache_remote_url == "lm://localhost:8080"

    def test_chunk_size_validation_min(self) -> None:
        """Test chunk_size minimum validation."""
        with pytest.raises(ValueError):
            ConnectorConfig(lmcache_chunk_size=8)

    def test_chunk_size_validation_max(self) -> None:
        """Test chunk_size maximum validation."""
        with pytest.raises(ValueError):
            ConnectorConfig(lmcache_chunk_size=8192)

    def test_mock_connector(self) -> None:
        """Test mock connector type."""
        config = ConnectorConfig(connector_type="mock")
        assert config.connector_type == "mock"


class TestGPUEmulationConfig:
    """Tests for GPUEmulationConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = GPUEmulationConfig()
        assert config.gpu_profile == "H100_SXM"
        assert config.efficiency_factor == 0.7
        assert config.tp_size == 1

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = GPUEmulationConfig(
            gpu_profile="A100_SXM",
            efficiency_factor=0.8,
            tp_size=8,
        )
        assert config.gpu_profile == "A100_SXM"
        assert config.efficiency_factor == 0.8
        assert config.tp_size == 8

    def test_efficiency_factor_validation_min(self) -> None:
        """Test efficiency_factor minimum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(efficiency_factor=0.05)

    def test_efficiency_factor_validation_max(self) -> None:
        """Test efficiency_factor maximum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(efficiency_factor=1.5)

    def test_tp_size_validation_min(self) -> None:
        """Test tp_size minimum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(tp_size=0)

    def test_tp_size_validation_max(self) -> None:
        """Test tp_size maximum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(tp_size=32)


class TestServerConfig:
    """Tests for ServerConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.server_type == "combined"
        assert config.model_profile == "llama-3.1-8b"
        assert config.workers == 1
        assert config.log_level == "info"

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            server_type="prefill",
            model_profile="llama-3.1-70b",
            workers=4,
            log_level="debug",
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.server_type == "prefill"

    def test_port_validation_min(self) -> None:
        """Test port minimum validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=80)

    def test_port_validation_max(self) -> None:
        """Test port maximum validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=70000)

    def test_server_types(self) -> None:
        """Test all server types are valid."""
        for server_type in ["prefill", "decode", "combined", "proxy"]:
            config = ServerConfig(server_type=server_type)  # type: ignore[arg-type]
            assert config.server_type == server_type


class TestDistributedConfig:
    """Tests for DistributedConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = DistributedConfig()
        assert config.prefill_endpoints == []
        assert config.decode_endpoints == []
        assert config.registry_url is None
        assert config.health_check_interval == 10.0

    def test_custom_endpoints(self) -> None:
        """Test custom endpoints configuration."""
        config = DistributedConfig(
            prefill_endpoints=["http://prefill-1:8000", "http://prefill-2:8000"],
            decode_endpoints=["http://decode-1:8000"],
        )
        assert len(config.prefill_endpoints) == 2
        assert len(config.decode_endpoints) == 1


class TestMetricsConfig:
    """Tests for MetricsConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prometheus_port == 9090
        assert config.include_histograms is True

    def test_disabled_metrics(self) -> None:
        """Test metrics can be disabled."""
        config = MetricsConfig(enabled=False)
        assert config.enabled is False


class TestKVBenchConfig:
    """Tests for KVBenchConfig main configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = KVBenchConfig()
        assert config.instance_id == "kvbench-0"
        assert isinstance(config.resources, ResourceLimits)
        assert isinstance(config.storage, StorageConfig)
        assert isinstance(config.connector, ConnectorConfig)
        assert isinstance(config.gpu, GPUEmulationConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.distributed, DistributedConfig)
        assert isinstance(config.metrics, MetricsConfig)

    def test_nested_config(self) -> None:
        """Test nested configuration works correctly."""
        config = KVBenchConfig(
            instance_id="test-instance",
            resources=ResourceLimits(cpu_memory_gb=32.0),
            storage=StorageConfig(
                backend_type="redis",
                redis_url="redis://localhost:6379",
            ),
        )
        assert config.instance_id == "test-instance"
        assert config.resources.cpu_memory_gb == 32.0
        assert config.storage.backend_type == "redis"

    def test_from_env(self) -> None:
        """Test configuration loading from environment variables."""
        env_vars = {
            "KVBENCH_INSTANCE_ID": "env-instance",
            "KVBENCH_RESOURCES__CPU_MEMORY_GB": "64.0",
            "KVBENCH_STORAGE__BACKEND_TYPE": "memory",
            "KVBENCH_GPU__GPU_PROFILE": "A100_SXM",
            "KVBENCH_SERVER__PORT": "9000",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = KVBenchConfig.from_env()
            assert config.instance_id == "env-instance"
            assert config.resources.cpu_memory_gb == 64.0
            assert config.storage.backend_type == "memory"
            assert config.gpu.gpu_profile == "A100_SXM"
            assert config.server.port == 9000

    def test_from_yaml(self) -> None:
        """Test configuration loading from YAML file."""
        yaml_content = """
instance_id: yaml-instance
resources:
  cpu_memory_gb: 128.0
storage:
  backend_type: memory
server:
  port: 8080
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = KVBenchConfig.from_yaml(f.name)
                assert config.instance_id == "yaml-instance"
                assert config.resources.cpu_memory_gb == 128.0
                assert config.server.port == 8080
            finally:
                os.unlink(f.name)

    def test_from_yaml_file_not_found(self) -> None:
        """Test from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            KVBenchConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file(self) -> None:
        """Test from_yaml handles empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                config = KVBenchConfig.from_yaml(f.name)
                # Should use defaults
                assert config.instance_id == "kvbench-0"
            finally:
                os.unlink(f.name)

    def test_to_yaml(self) -> None:
        """Test configuration saving to YAML file."""
        config = KVBenchConfig(instance_id="save-test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)
            assert path.exists()
            # Reload and verify
            loaded = KVBenchConfig.from_yaml(path)
            assert loaded.instance_id == "save-test"

    def test_to_yaml_creates_directories(self) -> None:
        """Test to_yaml creates parent directories."""
        config = KVBenchConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "path" / "config.yaml"
            config.to_yaml(path)
            assert path.exists()

    def test_model_dump(self) -> None:
        """Test model can be dumped to dictionary."""
        config = KVBenchConfig(instance_id="dump-test")
        data = config.model_dump()
        assert data["instance_id"] == "dump-test"
        assert "resources" in data
        assert "storage" in data
