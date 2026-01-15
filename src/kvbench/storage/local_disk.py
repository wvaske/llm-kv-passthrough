"""
Local Disk Storage Backend with Sharding.

This module provides a disk-based storage backend that uses sharded directories
for better file system performance with large numbers of files.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import aiofiles
import aiofiles.os

from kvbench.storage.base import StorageBackend


class LocalDiskStorageBackend(StorageBackend):
    """Local disk storage backend with sharded directories.

    This backend stores data on local disk (e.g., NVMe) using a sharded
    directory structure to avoid file system performance issues with
    large numbers of files in a single directory.

    The sharding uses the first N characters of the key's hash to create
    a directory structure like: base_path/ab/cd/key_hash

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes.
        base_path: Base directory for storage.
        shard_depth: Number of directory levels for sharding.
        shard_width: Number of characters per shard level.
    """

    def __init__(
        self,
        base_path: Path | str,
        max_size_bytes: int,
        name: str = "local_disk",
        shard_depth: int = 2,
        shard_width: int = 2,
    ) -> None:
        """Initialize the local disk storage backend.

        Args:
            base_path: Base directory for storage.
            max_size_bytes: Maximum storage capacity in bytes.
            name: Name identifier for this backend instance.
            shard_depth: Number of directory levels for sharding (default: 2).
            shard_width: Number of characters per shard level (default: 2).
        """
        super().__init__(max_size_bytes=max_size_bytes, name=name)
        self.base_path = Path(base_path)
        self.shard_depth = shard_depth
        self.shard_width = shard_width
        self._lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the storage directory is initialized."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Create base directory
            await aiofiles.os.makedirs(self.base_path, exist_ok=True)

            # Load existing stats if available
            await self._load_stats()
            self._initialized = True

    async def _load_stats(self) -> None:
        """Load storage statistics from disk."""
        stats_path = self.base_path / ".kvbench_stats.json"
        try:
            async with aiofiles.open(stats_path) as f:
                data = json.loads(await f.read())
                self._stats.used_bytes = data.get("used_bytes", 0)
                self._stats.item_count = data.get("item_count", 0)
                self._stats.hits = data.get("hits", 0)
                self._stats.misses = data.get("misses", 0)
                self._stats.evictions = data.get("evictions", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            # Scan directory to rebuild stats
            await self._rebuild_stats()

    async def _save_stats(self) -> None:
        """Save storage statistics to disk."""
        stats_path = self.base_path / ".kvbench_stats.json"
        data = {
            "used_bytes": self._stats.used_bytes,
            "item_count": self._stats.item_count,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
        }
        async with aiofiles.open(stats_path, "w") as f:
            await f.write(json.dumps(data))

    async def _rebuild_stats(self) -> None:
        """Rebuild statistics by scanning the directory."""
        self._stats.used_bytes = 0
        self._stats.item_count = 0

        for root, _, files in os.walk(self.base_path):
            for filename in files:
                if filename.startswith("."):
                    continue
                if filename.endswith(".meta"):
                    continue
                filepath = Path(root) / filename
                try:
                    stat = await aiofiles.os.stat(filepath)
                    self._stats.used_bytes += stat.st_size
                    self._stats.item_count += 1
                except OSError:
                    pass

    def _hash_key(self, key: str) -> str:
        """Create a hash of the key for file naming.

        Args:
            key: The key to hash.

        Returns:
            Hex-encoded SHA-256 hash of the key.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def _get_shard_path(self, key_hash: str) -> Path:
        """Get the sharded directory path for a key hash.

        Args:
            key_hash: The hash of the key.

        Returns:
            Path to the shard directory.
        """
        parts = []
        for i in range(self.shard_depth):
            start = i * self.shard_width
            end = start + self.shard_width
            parts.append(key_hash[start:end])

        return self.base_path / Path(*parts)

    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a key.

        Args:
            key: The key.

        Returns:
            Path to the data file.
        """
        key_hash = self._hash_key(key)
        shard_path = self._get_shard_path(key_hash)
        return shard_path / key_hash

    def _get_meta_path(self, key: str) -> Path:
        """Get the metadata file path for a key.

        Args:
            key: The key.

        Returns:
            Path to the metadata file.
        """
        return Path(str(self._get_file_path(key)) + ".meta")

    async def get(self, key: str) -> bytes | None:
        """Retrieve a value by key.

        Args:
            key: The key to look up.

        Returns:
            The value as bytes if found, None otherwise.
        """
        self._check_closed()
        await self._ensure_initialized()

        file_path = self._get_file_path(key)
        meta_path = self._get_meta_path(key)

        try:
            # Check if expired
            if await aiofiles.os.path.exists(meta_path):
                async with aiofiles.open(meta_path) as f:
                    meta_data = json.loads(await f.read())
                    if meta_data.get("ttl_seconds"):
                        created_at = datetime.fromisoformat(meta_data["created_at"])
                        elapsed = (datetime.now() - created_at).total_seconds()
                        if elapsed > meta_data["ttl_seconds"]:
                            await self.delete(key)
                            self._stats.misses += 1
                            return None

                # Update access time
                meta_data["accessed_at"] = datetime.now().isoformat()
                async with aiofiles.open(meta_path, "w") as f:
                    await f.write(json.dumps(meta_data))

            async with aiofiles.open(file_path, "rb") as f:
                self._stats.hits += 1
                return await f.read()
        except FileNotFoundError:
            self._stats.misses += 1
            return None

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
        self._check_closed()
        await self._ensure_initialized()

        value_size = len(value)

        # Check if value is too large
        if value_size > self.max_size_bytes:
            return False

        file_path = self._get_file_path(key)
        meta_path = self._get_meta_path(key)

        async with self._lock:
            # Get old size if updating
            old_size = 0
            try:
                stat = await aiofiles.os.stat(file_path)
                old_size = stat.st_size
            except OSError:
                pass

            # Check capacity
            new_used = self._stats.used_bytes - old_size + value_size
            if new_used > self.max_size_bytes:
                return False

            # Create shard directory
            shard_path = file_path.parent
            await aiofiles.os.makedirs(shard_path, exist_ok=True)

            # Write data
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(value)

            # Write metadata
            now = datetime.now()
            meta = {
                "key": key,
                "size_bytes": value_size,
                "created_at": now.isoformat(),
                "accessed_at": now.isoformat(),
                "ttl_seconds": ttl_seconds,
            }
            async with aiofiles.open(meta_path, "w") as f:
                await f.write(json.dumps(meta))

            # Update stats
            if old_size == 0:
                self._stats.item_count += 1
            self._stats.used_bytes = new_used

            await self._save_stats()
            return True

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        self._check_closed()
        await self._ensure_initialized()

        file_path = self._get_file_path(key)
        meta_path = self._get_meta_path(key)

        async with self._lock:
            try:
                stat = await aiofiles.os.stat(file_path)
                file_size = stat.st_size

                await aiofiles.os.remove(file_path)

                # Remove metadata
                try:
                    await aiofiles.os.remove(meta_path)
                except OSError:
                    pass

                self._stats.used_bytes -= file_size
                self._stats.item_count -= 1

                await self._save_stats()
                return True
            except FileNotFoundError:
                return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists and is not expired, False otherwise.
        """
        self._check_closed()
        await self._ensure_initialized()

        file_path = self._get_file_path(key)
        meta_path = self._get_meta_path(key)

        if not await aiofiles.os.path.exists(file_path):
            return False

        # Check if expired
        try:
            async with aiofiles.open(meta_path) as f:
                meta_data = json.loads(await f.read())
                if meta_data.get("ttl_seconds"):
                    created_at = datetime.fromisoformat(meta_data["created_at"])
                    elapsed = (datetime.now() - created_at).total_seconds()
                    if elapsed > meta_data["ttl_seconds"]:
                        await self.delete(key)
                        return False
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return True

    async def keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered by pattern.

        Note: This operation can be slow for large datasets as it needs
        to read metadata files to recover original keys.

        Args:
            pattern: Optional glob pattern to filter keys.

        Returns:
            List of matching keys.
        """
        self._check_closed()
        await self._ensure_initialized()

        keys_list: list[str] = []

        for root, _, files in os.walk(self.base_path):
            for filename in files:
                if not filename.endswith(".meta"):
                    continue

                meta_path = Path(root) / filename
                try:
                    async with aiofiles.open(meta_path) as f:
                        meta_data = json.loads(await f.read())
                        key = meta_data.get("key")
                        if key is None:
                            continue

                        # Check if expired
                        if meta_data.get("ttl_seconds"):
                            created_at = datetime.fromisoformat(meta_data["created_at"])
                            elapsed = (datetime.now() - created_at).total_seconds()
                            if elapsed > meta_data["ttl_seconds"]:
                                continue

                        if pattern is None or fnmatch.fnmatch(key, pattern):
                            keys_list.append(key)
                except (FileNotFoundError, json.JSONDecodeError):
                    pass

        return keys_list

    async def clear(self) -> int:
        """Clear all stored data.

        Returns:
            Number of items cleared.
        """
        self._check_closed()
        await self._ensure_initialized()

        async with self._lock:
            count = self._stats.item_count

            # Remove all files in subdirectories
            for root, dirs, files in os.walk(self.base_path, topdown=False):
                for filename in files:
                    if filename.startswith(".kvbench"):
                        continue
                    filepath = Path(root) / filename
                    try:
                        await aiofiles.os.remove(filepath)
                    except OSError:
                        pass

                # Remove empty directories
                for dirname in dirs:
                    dirpath = Path(root) / dirname
                    try:
                        await aiofiles.os.rmdir(dirpath)
                    except OSError:
                        pass

            self._stats.used_bytes = 0
            self._stats.item_count = 0
            await self._save_stats()

            return count

    async def close(self) -> None:
        """Close the storage backend and save statistics."""
        if self._closed:
            return

        async with self._lock:
            if self._initialized:
                await self._save_stats()
            self._closed = True

    async def size(self, key: str) -> int | None:
        """Get the size of a stored value in bytes.

        Args:
            key: The key to check.

        Returns:
            Size in bytes if the key exists, None otherwise.
        """
        self._check_closed()
        await self._ensure_initialized()

        file_path = self._get_file_path(key)
        try:
            stat = await aiofiles.os.stat(file_path)
            return stat.st_size
        except FileNotFoundError:
            return None
