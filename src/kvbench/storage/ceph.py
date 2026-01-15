"""
Ceph Storage Backend.

This module provides a Ceph RADOS-based storage backend for distributed
KV cache storage using Ceph's native object storage layer.
"""

from __future__ import annotations

import asyncio
import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from kvbench.storage.base import StorageBackend

if TYPE_CHECKING:
    import rados


class CephStorageBackend(StorageBackend):
    """Ceph RADOS storage backend.

    This backend stores data directly in Ceph using the librados library,
    providing high-performance object storage with built-in replication
    and fault tolerance.

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes (advisory).
        pool_name: Ceph pool name for storing objects.
        conf_file: Path to Ceph configuration file.
    """

    def __init__(
        self,
        pool_name: str,
        max_size_bytes: int,
        name: str = "ceph",
        conf_file: Path | str | None = None,
        user: str = "client.admin",
        key_prefix: str = "kvbench:",
    ) -> None:
        """Initialize the Ceph storage backend.

        Args:
            pool_name: Ceph pool name for storing objects.
            max_size_bytes: Maximum storage capacity in bytes (advisory).
            name: Name identifier for this backend instance.
            conf_file: Path to Ceph configuration file.
            user: Ceph user name for authentication.
            key_prefix: Prefix for all object names.
        """
        super().__init__(max_size_bytes=max_size_bytes, name=name)
        self.pool_name = pool_name
        self.conf_file = str(conf_file) if conf_file else "/etc/ceph/ceph.conf"
        self.user = user
        self.key_prefix = key_prefix

        self._cluster: rados.Rados | None = None
        self._ioctx: rados.Ioctx | None = None
        self._lock = asyncio.Lock()
        self._rados_available = False

        # Try to import rados
        try:
            import rados  # noqa: F401
            self._rados_available = True
        except ImportError:
            pass

    async def _ensure_connected(self) -> None:
        """Ensure connection to Ceph cluster is established."""
        if self._ioctx is not None:
            return

        if not self._rados_available:
            raise RuntimeError(
                "Ceph RADOS library (rados) is not available. "
                "Install the 'rados' package or use a different storage backend."
            )

        async with self._lock:
            if self._ioctx is not None:
                return

            import rados

            self._cluster = rados.Rados(
                conffile=self.conf_file,
                name=self.user,
            )
            self._cluster.connect()
            self._ioctx = self._cluster.open_ioctx(self.pool_name)

    def _make_key(self, key: str) -> str:
        """Create a prefixed object name.

        Args:
            key: The original key.

        Returns:
            Prefixed object name for Ceph.
        """
        return f"{self.key_prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        """Remove the prefix from an object name.

        Args:
            key: The prefixed object name.

        Returns:
            Original key without prefix.
        """
        if key.startswith(self.key_prefix):
            return key[len(self.key_prefix):]
        return key

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Args:
            key: The key to look up.

        Returns:
            The value as bytes if found, None otherwise.
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        object_name = self._make_key(key)

        try:
            # Run blocking RADOS operation in thread pool
            loop = asyncio.get_event_loop()
            size = await loop.run_in_executor(
                None, self._ioctx.stat, object_name
            )
            data = await loop.run_in_executor(
                None, self._ioctx.read, object_name, size[0]
            )
            self._stats.hits += 1
            return data
        except Exception:
            self._stats.misses += 1
            return None

    async def put(
        self,
        key: str,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Store a value with the given key.

        Note: Ceph doesn't natively support TTL on objects. TTL would need
        to be implemented using xattrs and a cleanup process.

        Args:
            key: The key to store the value under.
            value: The value to store as bytes.
            ttl_seconds: Optional time-to-live (stored as xattr but not enforced).

        Returns:
            True if the value was stored successfully, False otherwise.
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        object_name = self._make_key(key)
        value_size = len(value)

        try:
            # Check for existing object
            old_size = 0
            try:
                loop = asyncio.get_event_loop()
                stat = await loop.run_in_executor(
                    None, self._ioctx.stat, object_name
                )
                old_size = stat[0]
            except Exception:
                pass

            # Write object
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._ioctx.write_full, object_name, value
            )

            # Store TTL as xattr if provided
            if ttl_seconds is not None:
                await loop.run_in_executor(
                    None,
                    self._ioctx.set_xattr,
                    object_name,
                    "kvbench.ttl",
                    str(ttl_seconds).encode(),
                )

            # Update stats
            if old_size == 0:
                self._stats.item_count += 1
            self._stats.used_bytes += value_size - old_size

            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        object_name = self._make_key(key)

        try:
            # Get size for stats
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(
                None, self._ioctx.stat, object_name
            )
            old_size = stat[0]

            await loop.run_in_executor(
                None, self._ioctx.remove_object, object_name
            )

            self._stats.used_bytes -= old_size
            self._stats.item_count -= 1

            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        object_name = self._make_key(key)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self._ioctx.stat, object_name
            )
            return True
        except Exception:
            return False

    async def keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter keys.

        Returns:
            List of matching keys (without prefix).
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        loop = asyncio.get_event_loop()

        def list_objects() -> list[str]:
            result = []
            for obj in self._ioctx.list_objects():
                obj_name = obj.key
                if not obj_name.startswith(self.key_prefix):
                    continue
                key = self._strip_prefix(obj_name)
                if pattern is None or fnmatch.fnmatch(key, pattern):
                    result.append(key)
            return result

        return await loop.run_in_executor(None, list_objects)

    async def clear(self) -> int:
        """Clear all stored data with the key prefix.

        Returns:
            Number of items cleared.
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        all_keys = await self.keys()
        count = 0

        for key in all_keys:
            if await self.delete(key):
                count += 1

        return count

    async def close(self) -> None:
        """Close the Ceph connection."""
        if self._closed:
            return

        async with self._lock:
            if self._ioctx:
                self._ioctx.close()
                self._ioctx = None

            if self._cluster:
                self._cluster.shutdown()
                self._cluster = None

            self._closed = True

    async def size(self, key: str) -> int | None:
        """Get the size of a stored value in bytes.

        Args:
            key: The key to check.

        Returns:
            Size in bytes if the key exists, None otherwise.
        """
        self._check_closed()
        await self._ensure_connected()
        assert self._ioctx is not None

        object_name = self._make_key(key)

        try:
            loop = asyncio.get_event_loop()
            stat = await loop.run_in_executor(
                None, self._ioctx.stat, object_name
            )
            return stat[0]
        except Exception:
            return None
