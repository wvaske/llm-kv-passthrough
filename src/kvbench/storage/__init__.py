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

__all__: list[str] = []
