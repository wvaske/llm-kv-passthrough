"""
Abstract Storage Backend Interface.

This module defines the abstract interface that all storage backends must implement.
It provides a unified API for storing and retrieving KV cache data across different
storage systems (memory, disk, Redis, NFS, Ceph, Weka, MinIO).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StorageStats:
    """Statistics for a storage backend.

    Attributes:
        total_bytes: Total capacity in bytes.
        used_bytes: Currently used bytes.
        item_count: Number of items stored.
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of items evicted.
        created_at: When the storage was created.
    """

    total_bytes: int
    used_bytes: int = 0
    item_count: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def available_bytes(self) -> int:
        """Return available bytes."""
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def utilization(self) -> float:
        """Return storage utilization as a percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


@dataclass
class StorageItem:
    """Metadata for a stored item.

    Attributes:
        key: The item's key.
        size_bytes: Size of the value in bytes.
        created_at: When the item was created.
        accessed_at: When the item was last accessed.
        ttl_seconds: Time-to-live in seconds (None for no expiry).
        metadata: Optional additional metadata.
    """

    key: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: int | None = None
    metadata: dict | None = None

    @property
    def is_expired(self) -> bool:
        """Check if the item has expired."""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


class StorageBackend(ABC):
    """Abstract base class for all storage backends.

    This class defines the interface that all storage backends must implement.
    It provides async methods for storing, retrieving, and managing KV cache data.

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes.
    """

    def __init__(self, max_size_bytes: int, name: str = "unknown") -> None:
        """Initialize the storage backend.

        Args:
            max_size_bytes: Maximum storage capacity in bytes.
            name: Name identifier for this backend instance.
        """
        self.name = name
        self.max_size_bytes = max_size_bytes
        self._stats = StorageStats(total_bytes=max_size_bytes)
        self._closed = False

    @property
    def stats(self) -> StorageStats:
        """Return current storage statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if the backend has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise an error if the backend is closed."""
        if self._closed:
            raise RuntimeError(f"Storage backend '{self.name}' is closed")

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Args:
            key: The key to look up.

        Returns:
            The value as bytes if found, None otherwise.
        """
        ...

    @abstractmethod
    async def put(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a value with the given key.

        Args:
            key: The key to store the value under.
            value: The value to store as bytes.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            True if the value was stored successfully, False otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        ...

    @abstractmethod
    async def keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter keys (e.g., "prefix:*").

        Returns:
            List of matching keys.
        """
        ...

    @abstractmethod
    async def clear(self) -> int:
        """Clear all stored data.

        Returns:
            Number of items cleared.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the storage backend and release resources."""
        ...

    async def get_many(self, keys: list[str]) -> dict[str, bytes | None]:
        """Retrieve multiple values by keys.

        Default implementation calls get() for each key.
        Backends may override for better performance.

        Args:
            keys: List of keys to look up.

        Returns:
            Dictionary mapping keys to values (None for missing keys).
        """
        self._check_closed()
        result = {}
        for key in keys:
            result[key] = await self.get(key)
        return result

    async def put_many(
        self,
        items: dict[str, bytes],
        ttl_seconds: int | None = None,
    ) -> dict[str, bool]:
        """Store multiple values.

        Default implementation calls put() for each item.
        Backends may override for better performance.

        Args:
            items: Dictionary mapping keys to values.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            Dictionary mapping keys to success status.
        """
        self._check_closed()
        result = {}
        for key, value in items.items():
            result[key] = await self.put(key, value, ttl_seconds)
        return result

    async def delete_many(self, keys: list[str]) -> dict[str, bool]:
        """Delete multiple keys.

        Default implementation calls delete() for each key.
        Backends may override for better performance.

        Args:
            keys: List of keys to delete.

        Returns:
            Dictionary mapping keys to deletion status.
        """
        self._check_closed()
        result = {}
        for key in keys:
            result[key] = await self.delete(key)
        return result

    async def size(self, key: str) -> int | None:
        """Get the size of a stored value in bytes.

        Default implementation retrieves the value and returns its length.
        Backends may override for better performance.

        Args:
            key: The key to check.

        Returns:
            Size in bytes if the key exists, None otherwise.
        """
        self._check_closed()
        value = await self.get(key)
        return len(value) if value is not None else None

    async def iter_keys(self, pattern: str | None = None) -> AsyncIterator[str]:
        """Iterate over keys, optionally filtered by pattern.

        Default implementation uses keys() method.
        Backends may override for memory-efficient streaming.

        Args:
            pattern: Optional glob pattern to filter keys.

        Yields:
            Keys matching the pattern.
        """
        self._check_closed()
        for key in await self.keys(pattern):
            yield key

    async def __aenter__(self) -> StorageBackend:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Async context manager exit."""
        await self.close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"max_size_bytes={self.max_size_bytes}, "
            f"used_bytes={self._stats.used_bytes})"
        )
