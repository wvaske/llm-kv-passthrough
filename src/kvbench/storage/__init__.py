"""
KV-Bench Storage Backends Module.

This module provides storage backend implementations:
- Memory storage (in-memory with LRU eviction)
- Local disk storage (NVMe with sharding)
- Redis storage (standalone, cluster, sentinel)
- NFS storage (async file operations)
- Ceph storage (librados bindings)
- Weka storage (parallel filesystem)
- MinIO storage (S3-compatible)
"""

from kvbench.storage.base import StorageBackend, StorageItem, StorageStats
from kvbench.storage.ceph import CephStorageBackend
from kvbench.storage.factory import create_storage_backend
from kvbench.storage.local_disk import LocalDiskStorageBackend
from kvbench.storage.memory import MemoryStorageBackend
from kvbench.storage.minio import MinIOStorageBackend
from kvbench.storage.nfs import NFSStorageBackend
from kvbench.storage.redis_backend import RedisStorageBackend
from kvbench.storage.weka import WekaStorageBackend

__all__ = [
    # Base classes
    "StorageBackend",
    "StorageItem",
    "StorageStats",
    # Factory
    "create_storage_backend",
    # Implementations
    "MemoryStorageBackend",
    "LocalDiskStorageBackend",
    "RedisStorageBackend",
    "NFSStorageBackend",
    "CephStorageBackend",
    "WekaStorageBackend",
    "MinIOStorageBackend",
]
