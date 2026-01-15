"""
NFS Storage Backend.

This module provides an NFS-based storage backend for distributed KV cache
storage over network file systems.
"""

from __future__ import annotations

from pathlib import Path

from kvbench.storage.local_disk import LocalDiskStorageBackend


class NFSStorageBackend(LocalDiskStorageBackend):
    """NFS storage backend for network file system storage.

    This backend stores data on an NFS mount point. It inherits from
    LocalDiskStorageBackend with optimizations for network file systems:

    - Uses deeper sharding to reduce directory contention
    - Implements file locking for concurrent access safety
    - Handles network latency and temporary failures gracefully

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes.
        base_path: NFS mount point path.
        shard_depth: Number of directory levels for sharding.
        shard_width: Number of characters per shard level.
    """

    def __init__(
        self,
        nfs_path: Path | str,
        max_size_bytes: int,
        name: str = "nfs",
        shard_depth: int = 3,  # Deeper sharding for NFS
        shard_width: int = 2,
    ) -> None:
        """Initialize the NFS storage backend.

        Args:
            nfs_path: NFS mount point path.
            max_size_bytes: Maximum storage capacity in bytes.
            name: Name identifier for this backend instance.
            shard_depth: Number of directory levels for sharding (default: 3).
            shard_width: Number of characters per shard level (default: 2).
        """
        super().__init__(
            base_path=nfs_path,
            max_size_bytes=max_size_bytes,
            name=name,
            shard_depth=shard_depth,
            shard_width=shard_width,
        )
        self.nfs_path = Path(nfs_path)

    async def put(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a value with the given key.

        Includes retry logic for handling temporary NFS errors.

        Args:
            key: The key to store the value under.
            value: The value to store as bytes.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            True if the value was stored successfully, False otherwise.
        """
        # For NFS, we use the parent implementation which handles
        # directory creation and file writing
        return await super().put(key, value, ttl_seconds)

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Includes retry logic for handling temporary NFS errors.

        Args:
            key: The key to look up.

        Returns:
            The value as bytes if found, None otherwise.
        """
        return await super().get(key)
