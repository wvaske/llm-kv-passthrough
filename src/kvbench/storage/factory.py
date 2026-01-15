"""
Storage Backend Factory.

This module provides a factory function for creating storage backends
based on configuration.
"""

from __future__ import annotations

from kvbench.core.config import ResourceLimits, StorageConfig
from kvbench.storage.base import StorageBackend
from kvbench.storage.ceph import CephStorageBackend
from kvbench.storage.local_disk import LocalDiskStorageBackend
from kvbench.storage.memory import MemoryStorageBackend
from kvbench.storage.minio import MinIOStorageBackend
from kvbench.storage.nfs import NFSStorageBackend
from kvbench.storage.redis_backend import RedisStorageBackend
from kvbench.storage.weka import WekaStorageBackend


def create_storage_backend(
    config: StorageConfig,
    resources: ResourceLimits,
    name: str | None = None,
) -> StorageBackend:
    """Create a storage backend based on configuration.

    Args:
        config: Storage configuration specifying backend type and settings.
        resources: Resource limits for capacity configuration.
        name: Optional name for the backend instance.

    Returns:
        Configured storage backend instance.

    Raises:
        ValueError: If the backend type is unknown or configuration is invalid.
    """
    backend_type = config.backend_type
    backend_name = name or backend_type

    if backend_type == "memory":
        return MemoryStorageBackend(
            max_size_bytes=resources.cpu_memory_bytes,
            name=backend_name,
        )

    elif backend_type == "local_disk":
        return LocalDiskStorageBackend(
            base_path=resources.nvme_path,
            max_size_bytes=resources.nvme_storage_bytes,
            name=backend_name,
        )

    elif backend_type == "redis":
        if not config.redis_url:
            raise ValueError("redis_url is required for redis backend")
        return RedisStorageBackend(
            redis_url=config.redis_url,
            max_size_bytes=resources.cpu_memory_bytes,
            name=backend_name,
            cluster_mode=config.redis_cluster,
        )

    elif backend_type == "nfs":
        if not config.filesystem_path:
            raise ValueError("filesystem_path is required for nfs backend")
        return NFSStorageBackend(
            nfs_path=config.filesystem_path,
            max_size_bytes=resources.nvme_storage_bytes,
            name=backend_name,
        )

    elif backend_type == "ceph":
        if not config.ceph_pool:
            raise ValueError("ceph_pool is required for ceph backend")
        return CephStorageBackend(
            pool_name=config.ceph_pool,
            max_size_bytes=resources.cpu_memory_bytes,
            name=backend_name,
            conf_file=config.ceph_conf,
        )

    elif backend_type == "weka":
        if not config.filesystem_path:
            raise ValueError("filesystem_path is required for weka backend")
        return WekaStorageBackend(
            weka_path=config.filesystem_path,
            max_size_bytes=resources.nvme_storage_bytes,
            name=backend_name,
        )

    elif backend_type == "minio":
        if not config.s3_endpoint:
            raise ValueError("s3_endpoint is required for minio backend")
        if not config.s3_bucket:
            raise ValueError("s3_bucket is required for minio backend")
        return MinIOStorageBackend(
            endpoint_url=config.s3_endpoint,
            bucket_name=config.s3_bucket,
            max_size_bytes=resources.cpu_memory_bytes,
            name=backend_name,
            access_key=config.s3_access_key,
            secret_key=config.s3_secret_key,
        )

    else:
        raise ValueError(f"Unknown storage backend type: {backend_type}")


__all__ = [
    "create_storage_backend",
    "StorageBackend",
    "MemoryStorageBackend",
    "LocalDiskStorageBackend",
    "RedisStorageBackend",
    "NFSStorageBackend",
    "CephStorageBackend",
    "WekaStorageBackend",
    "MinIOStorageBackend",
]
