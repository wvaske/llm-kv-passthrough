"""
In-Memory Storage Backend with LRU Eviction.

This module provides a memory-based storage backend that uses an OrderedDict
for LRU (Least Recently Used) eviction when capacity is reached.
"""

from __future__ import annotations

import asyncio
import fnmatch
from collections import OrderedDict
from datetime import datetime

from kvbench.storage.base import StorageBackend, StorageItem


class MemoryStorageBackend(StorageBackend):
    """In-memory storage backend with LRU eviction.

    This backend stores all data in memory using an OrderedDict to maintain
    access order for LRU eviction. When storage capacity is reached, the
    least recently used items are evicted to make room for new items.

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes.
    """

    def __init__(
        self,
        max_size_bytes: int,
        name: str = "memory",
    ) -> None:
        """Initialize the memory storage backend.

        Args:
            max_size_bytes: Maximum storage capacity in bytes.
            name: Name identifier for this backend instance.
        """
        super().__init__(max_size_bytes=max_size_bytes, name=name)
        self._data: OrderedDict[str, bytes] = OrderedDict()
        self._metadata: dict[str, StorageItem] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Updates access time and moves key to end for LRU tracking.

        Args:
            key: The key to look up.

        Returns:
            The value as bytes if found, None otherwise.
        """
        self._check_closed()
        async with self._lock:
            if key not in self._data:
                self._stats.misses += 1
                return None

            # Check if expired
            meta = self._metadata.get(key)
            if meta and meta.is_expired:
                await self._delete_internal(key)
                self._stats.misses += 1
                return None

            # Move to end (most recently used)
            self._data.move_to_end(key)

            # Update access time
            if meta:
                meta.accessed_at = datetime.now()

            self._stats.hits += 1
            return self._data[key]

    async def put(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a value with the given key.

        If the key already exists, it is updated. If storage is full,
        LRU eviction is performed to make room.

        Args:
            key: The key to store the value under.
            value: The value to store as bytes.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            True if the value was stored successfully, False otherwise.
        """
        self._check_closed()
        async with self._lock:
            value_size = len(value)

            # Check if value is too large
            if value_size > self.max_size_bytes:
                return False

            # Calculate size change
            old_size = 0
            if key in self._data:
                old_size = len(self._data[key])

            needed_space = value_size - old_size

            # Evict if necessary
            while (
                needed_space > 0
                and self._stats.used_bytes + needed_space > self.max_size_bytes
                and self._data
            ):
                await self._evict_lru()

            # Check if we have enough space
            if self._stats.used_bytes + needed_space > self.max_size_bytes:
                return False

            # Update stats
            if key in self._data:
                self._stats.used_bytes -= old_size
            else:
                self._stats.item_count += 1

            # Store the value
            self._data[key] = value
            self._data.move_to_end(key)
            self._stats.used_bytes += value_size

            # Store metadata
            self._metadata[key] = StorageItem(
                key=key,
                size_bytes=value_size,
                ttl_seconds=ttl_seconds,
            )

            return True

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        self._check_closed()
        async with self._lock:
            return await self._delete_internal(key)

    async def _delete_internal(self, key: str) -> bool:
        """Internal delete without lock (must be called with lock held).

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        if key not in self._data:
            return False

        value_size = len(self._data[key])
        del self._data[key]
        self._metadata.pop(key, None)

        self._stats.used_bytes -= value_size
        self._stats.item_count -= 1

        return True

    async def _evict_lru(self) -> None:
        """Evict the least recently used item.

        Must be called with lock held.
        """
        if not self._data:
            return

        # Get the first (oldest) item
        key = next(iter(self._data))
        await self._delete_internal(key)
        self._stats.evictions += 1

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists and is not expired, False otherwise.
        """
        self._check_closed()
        async with self._lock:
            if key not in self._data:
                return False

            # Check if expired
            meta = self._metadata.get(key)
            if meta and meta.is_expired:
                await self._delete_internal(key)
                return False

            return True

    async def keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter keys (e.g., "prefix:*").

        Returns:
            List of matching keys.
        """
        self._check_closed()
        async with self._lock:
            # Clean up expired items first
            await self._cleanup_expired()

            if pattern is None:
                return list(self._data.keys())

            return [key for key in self._data if fnmatch.fnmatch(key, pattern)]

    async def _cleanup_expired(self) -> None:
        """Remove all expired items.

        Must be called with lock held.
        """
        expired_keys = [
            key
            for key, meta in self._metadata.items()
            if meta.is_expired
        ]
        for key in expired_keys:
            await self._delete_internal(key)

    async def clear(self) -> int:
        """Clear all stored data.

        Returns:
            Number of items cleared.
        """
        self._check_closed()
        async with self._lock:
            count = len(self._data)
            self._data.clear()
            self._metadata.clear()
            self._stats.used_bytes = 0
            self._stats.item_count = 0
            return count

    async def close(self) -> None:
        """Close the storage backend and release resources."""
        if self._closed:
            return
        async with self._lock:
            self._data.clear()
            self._metadata.clear()
            self._closed = True

    async def get_many(self, keys: list[str]) -> dict[str, bytes | None]:
        """Retrieve multiple values by keys.

        Optimized to use a single lock acquisition.

        Args:
            keys: List of keys to look up.

        Returns:
            Dictionary mapping keys to values (None for missing keys).
        """
        self._check_closed()
        result: dict[str, bytes | None] = {}
        async with self._lock:
            for key in keys:
                if key not in self._data:
                    self._stats.misses += 1
                    result[key] = None
                    continue

                meta = self._metadata.get(key)
                if meta and meta.is_expired:
                    await self._delete_internal(key)
                    self._stats.misses += 1
                    result[key] = None
                    continue

                self._data.move_to_end(key)
                if meta:
                    meta.accessed_at = datetime.now()
                self._stats.hits += 1
                result[key] = self._data[key]

        return result

    async def put_many(
        self,
        items: dict[str, bytes],
        ttl_seconds: int | None = None,
    ) -> dict[str, bool]:
        """Store multiple values.

        Optimized to use a single lock acquisition.

        Args:
            items: Dictionary mapping keys to values.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            Dictionary mapping keys to success status.
        """
        self._check_closed()
        result: dict[str, bool] = {}

        async with self._lock:
            for key, value in items.items():
                value_size = len(value)

                if value_size > self.max_size_bytes:
                    result[key] = False
                    continue

                old_size = len(self._data[key]) if key in self._data else 0
                needed_space = value_size - old_size

                while (
                    needed_space > 0
                    and self._stats.used_bytes + needed_space > self.max_size_bytes
                    and self._data
                ):
                    await self._evict_lru()

                if self._stats.used_bytes + needed_space > self.max_size_bytes:
                    result[key] = False
                    continue

                if key in self._data:
                    self._stats.used_bytes -= old_size
                else:
                    self._stats.item_count += 1

                self._data[key] = value
                self._data.move_to_end(key)
                self._stats.used_bytes += value_size

                self._metadata[key] = StorageItem(
                    key=key,
                    size_bytes=value_size,
                    ttl_seconds=ttl_seconds,
                )
                result[key] = True

        return result

    async def size(self, key: str) -> int | None:
        """Get the size of a stored value in bytes.

        Optimized to avoid copying the value.

        Args:
            key: The key to check.

        Returns:
            Size in bytes if the key exists, None otherwise.
        """
        self._check_closed()
        async with self._lock:
            if key not in self._data:
                return None

            meta = self._metadata.get(key)
            if meta and meta.is_expired:
                await self._delete_internal(key)
                return None

            return len(self._data[key])

    def get_metadata(self, key: str) -> StorageItem | None:
        """Get metadata for a stored item.

        Args:
            key: The key to look up.

        Returns:
            StorageItem metadata if found, None otherwise.
        """
        return self._metadata.get(key)
