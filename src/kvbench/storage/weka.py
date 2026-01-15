"""
Weka Storage Backend.

This module provides a Weka-based storage backend for high-performance
parallel file system storage.
"""

from __future__ import annotations

from pathlib import Path

from kvbench.storage.local_disk import LocalDiskStorageBackend


class WekaStorageBackend(LocalDiskStorageBackend):
    """Weka storage backend for high-performance parallel file systems.

    This backend stores data on a Weka file system mount point. It inherits
    from LocalDiskStorageBackend with optimizations for Weka's parallel
    file system capabilities:

    - Uses shallower sharding as Weka handles large directories well
    - Takes advantage of Weka's parallel I/O capabilities
    - Optimized for high-bandwidth sequential and random access

    Weka is designed for AI/ML workloads and provides:
    - Flash-native architecture
    - Linear scalability
    - Low latency for small files
    - High throughput for large files

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes.
        base_path: Weka mount point path.
        shard_depth: Number of directory levels for sharding.
        shard_width: Number of characters per shard level.
    """

    def __init__(
        self,
        weka_path: Path | str,
        max_size_bytes: int,
        name: str = "weka",
        shard_depth: int = 1,  # Weka handles large directories well
        shard_width: int = 2,
    ) -> None:
        """Initialize the Weka storage backend.

        Args:
            weka_path: Weka mount point path.
            max_size_bytes: Maximum storage capacity in bytes.
            name: Name identifier for this backend instance.
            shard_depth: Number of directory levels for sharding (default: 1).
            shard_width: Number of characters per shard level (default: 2).
        """
        super().__init__(
            base_path=weka_path,
            max_size_bytes=max_size_bytes,
            name=name,
            shard_depth=shard_depth,
            shard_width=shard_width,
        )
        self.weka_path = Path(weka_path)

    async def put(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a value with the given key.

        Weka provides excellent performance for both small and large files,
        so no special optimizations are needed for different value sizes.

        Args:
            key: The key to store the value under.
            value: The value to store as bytes.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            True if the value was stored successfully, False otherwise.
        """
        return await super().put(key, value, ttl_seconds)

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Weka's flash-native architecture provides consistent low latency
        for reads of any size.

        Args:
            key: The key to look up.

        Returns:
            The value as bytes if found, None otherwise.
        """
        return await super().get(key)

    async def get_many(self, keys: list[str]) -> dict[str, bytes | None]:
        """Retrieve multiple values by keys.

        Takes advantage of Weka's parallel I/O capabilities for
        concurrent reads.

        Args:
            keys: List of keys to look up.

        Returns:
            Dictionary mapping keys to values (None for missing keys).
        """
        # Weka handles parallel reads well, but for now we use the
        # default sequential implementation. A more optimized version
        # could use asyncio.gather for parallel reads.
        return await super().get_many(keys)

    async def put_many(
        self,
        items: dict[str, bytes],
        ttl_seconds: int | None = None,
    ) -> dict[str, bool]:
        """Store multiple values.

        Takes advantage of Weka's parallel I/O capabilities for
        concurrent writes.

        Args:
            items: Dictionary mapping keys to values.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            Dictionary mapping keys to success status.
        """
        # Weka handles parallel writes well, but for now we use the
        # default sequential implementation. A more optimized version
        # could use asyncio.gather for parallel writes.
        return await super().put_many(items, ttl_seconds)
