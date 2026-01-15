"""Unit tests for kvbench.storage.factory module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kvbench.core.config import ResourceLimits, StorageConfig
from kvbench.storage.ceph import CephStorageBackend
from kvbench.storage.factory import create_storage_backend
from kvbench.storage.local_disk import LocalDiskStorageBackend
from kvbench.storage.memory import MemoryStorageBackend
from kvbench.storage.minio import MinIOStorageBackend
from kvbench.storage.nfs import NFSStorageBackend
from kvbench.storage.redis_backend import RedisStorageBackend
from kvbench.storage.weka import WekaStorageBackend


def make_config(**kwargs) -> StorageConfig:
    """Create a StorageConfig bypassing validation for testing."""
    defaults = {
        "backend_type": "memory",
        "redis_url": None,
        "redis_cluster": False,
        "filesystem_path": None,
        "s3_endpoint": None,
        "s3_bucket": None,
        "s3_access_key": None,
        "s3_secret_key": None,
        "ceph_pool": None,
        "ceph_conf": None,
    }
    defaults.update(kwargs)
    return StorageConfig.model_construct(**defaults)


class TestCreateStorageBackend:
    """Tests for create_storage_backend factory function."""

    @pytest.fixture
    def resources(self, tmp_path: Path) -> ResourceLimits:
        """Create resource limits for testing."""
        return ResourceLimits(
            cpu_memory_gb=1.0,
            nvme_storage_gb=1.0,
            nvme_path=tmp_path,
        )

    def test_create_memory_backend(self, resources: ResourceLimits) -> None:
        """Test creating a memory storage backend."""
        config = StorageConfig(backend_type="memory")
        backend = create_storage_backend(config, resources)

        assert isinstance(backend, MemoryStorageBackend)
        assert backend.max_size_bytes == resources.cpu_memory_bytes

    def test_create_local_disk_backend(self, resources: ResourceLimits) -> None:
        """Test creating a local disk storage backend."""
        config = StorageConfig(backend_type="local_disk")
        backend = create_storage_backend(config, resources)

        assert isinstance(backend, LocalDiskStorageBackend)
        assert backend.base_path == resources.nvme_path

    def test_create_redis_backend(self, resources: ResourceLimits) -> None:
        """Test creating a Redis storage backend."""
        config = StorageConfig(
            backend_type="redis",
            redis_url="redis://localhost:6379",
        )
        backend = create_storage_backend(config, resources)

        assert isinstance(backend, RedisStorageBackend)
        assert backend.redis_url == "redis://localhost:6379"

    def test_create_redis_backend_cluster(self, resources: ResourceLimits) -> None:
        """Test creating a Redis cluster backend."""
        config = StorageConfig(
            backend_type="redis",
            redis_url="redis://localhost:6379",
            redis_cluster=True,
        )
        backend = create_storage_backend(config, resources)

        assert isinstance(backend, RedisStorageBackend)
        assert backend.cluster_mode is True

    def test_create_redis_backend_missing_url(self, resources: ResourceLimits) -> None:
        """Test that missing redis_url raises error."""
        config = make_config(backend_type="redis", redis_url=None)

        with pytest.raises(ValueError, match="redis_url is required"):
            create_storage_backend(config, resources)

    def test_create_nfs_backend(self, resources: ResourceLimits) -> None:
        """Test creating an NFS storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(
                backend_type="nfs",
                filesystem_path=Path(tmpdir),
            )
            backend = create_storage_backend(config, resources)

            assert isinstance(backend, NFSStorageBackend)

    def test_create_nfs_backend_missing_path(self, resources: ResourceLimits) -> None:
        """Test that missing filesystem_path raises error for NFS."""
        config = make_config(backend_type="nfs", filesystem_path=None)

        with pytest.raises(ValueError, match="filesystem_path is required"):
            create_storage_backend(config, resources)

    def test_create_ceph_backend(self, resources: ResourceLimits) -> None:
        """Test creating a Ceph storage backend."""
        config = make_config(backend_type="ceph", ceph_pool="kvbench")

        backend = create_storage_backend(config, resources)
        assert isinstance(backend, CephStorageBackend)
        assert backend.pool_name == "kvbench"

    def test_create_ceph_backend_missing_pool(self, resources: ResourceLimits) -> None:
        """Test that missing ceph_pool raises error."""
        config = make_config(backend_type="ceph", ceph_pool=None)

        with pytest.raises(ValueError, match="ceph_pool is required"):
            create_storage_backend(config, resources)

    def test_create_weka_backend(self, resources: ResourceLimits) -> None:
        """Test creating a Weka storage backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = StorageConfig(
                backend_type="weka",
                filesystem_path=Path(tmpdir),
            )
            backend = create_storage_backend(config, resources)

            assert isinstance(backend, WekaStorageBackend)

    def test_create_weka_backend_missing_path(self, resources: ResourceLimits) -> None:
        """Test that missing filesystem_path raises error for Weka."""
        config = make_config(backend_type="weka", filesystem_path=None)

        with pytest.raises(ValueError, match="filesystem_path is required"):
            create_storage_backend(config, resources)

    def test_create_minio_backend(self, resources: ResourceLimits) -> None:
        """Test creating a MinIO storage backend."""
        config = make_config(
            backend_type="minio",
            s3_endpoint="http://localhost:9000",
            s3_bucket="kvbench",
        )

        backend = create_storage_backend(config, resources)
        assert isinstance(backend, MinIOStorageBackend)
        assert backend.endpoint_url == "http://localhost:9000"
        assert backend.bucket_name == "kvbench"

    def test_create_minio_backend_missing_endpoint(
        self, resources: ResourceLimits
    ) -> None:
        """Test that missing s3_endpoint raises error."""
        config = make_config(
            backend_type="minio",
            s3_endpoint=None,
            s3_bucket="kvbench",
        )

        with pytest.raises(ValueError, match="s3_endpoint is required"):
            create_storage_backend(config, resources)

    def test_create_minio_backend_missing_bucket(
        self, resources: ResourceLimits
    ) -> None:
        """Test that missing s3_bucket raises error."""
        config = make_config(
            backend_type="minio",
            s3_endpoint="http://localhost:9000",
            s3_bucket=None,
        )

        with pytest.raises(ValueError, match="s3_bucket is required"):
            create_storage_backend(config, resources)

    def test_create_unknown_backend(self, resources: ResourceLimits) -> None:
        """Test that unknown backend type raises error."""
        config = make_config(backend_type="unknown")

        with pytest.raises(ValueError, match="Unknown storage backend type"):
            create_storage_backend(config, resources)

    def test_custom_name(self, resources: ResourceLimits) -> None:
        """Test creating a backend with a custom name."""
        config = StorageConfig(backend_type="memory")
        backend = create_storage_backend(config, resources, name="custom_name")

        assert backend.name == "custom_name"

    def test_default_name(self, resources: ResourceLimits) -> None:
        """Test that default name matches backend type."""
        config = StorageConfig(backend_type="memory")
        backend = create_storage_backend(config, resources)

        assert backend.name == "memory"
