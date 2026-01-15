"""
Redis Storage Backend.

This module provides a Redis-based storage backend supporting standalone,
cluster, and sentinel modes for distributed KV cache storage.
"""

from __future__ import annotations

from typing import Literal

import redis.asyncio as redis

from kvbench.storage.base import StorageBackend


class RedisStorageBackend(StorageBackend):
    """Redis storage backend supporting multiple deployment modes.

    This backend stores data in Redis, supporting standalone, cluster,
    and sentinel deployment modes. It uses Redis's native TTL support
    for automatic expiration.

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes (advisory).
        redis_url: Redis connection URL.
        cluster_mode: Whether to use Redis Cluster.
        sentinel_mode: Whether to use Redis Sentinel.
    """

    def __init__(
        self,
        redis_url: str,
        max_size_bytes: int,
        name: str = "redis",
        cluster_mode: bool = False,
        sentinel_master: str | None = None,
        key_prefix: str = "kvbench:",
        mode: Literal["standalone", "cluster", "sentinel"] = "standalone",
    ) -> None:
        """Initialize the Redis storage backend.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379).
            max_size_bytes: Maximum storage capacity in bytes (advisory).
            name: Name identifier for this backend instance.
            cluster_mode: Whether to use Redis Cluster mode.
            sentinel_master: Sentinel master name (required for sentinel mode).
            key_prefix: Prefix for all keys stored in Redis.
            mode: Connection mode (standalone, cluster, or sentinel).
        """
        super().__init__(max_size_bytes=max_size_bytes, name=name)
        self.redis_url = redis_url
        self.cluster_mode = cluster_mode or mode == "cluster"
        self.sentinel_master = sentinel_master
        self.key_prefix = key_prefix
        self.mode = mode

        self._client: redis.Redis | redis.RedisCluster | None = None
        self._sentinel: redis.Sentinel | None = None

    async def _get_client(self) -> redis.Redis | redis.RedisCluster:
        """Get or create the Redis client.

        Returns:
            Redis client instance.
        """
        if self._client is not None:
            return self._client

        if self.mode == "cluster" or self.cluster_mode:
            self._client = redis.RedisCluster.from_url(
                self.redis_url,
                decode_responses=False,
            )
        elif self.mode == "sentinel":
            if not self.sentinel_master:
                raise ValueError("sentinel_master is required for sentinel mode")
            # Parse sentinel URL to get host:port pairs
            # Format: redis+sentinel://host1:port1,host2:port2/master_name
            self._sentinel = redis.Sentinel.from_url(self.redis_url)
            self._client = self._sentinel.master_for(
                self.sentinel_master,
                redis_class=redis.Redis,
            )
        else:
            self._client = redis.Redis.from_url(
                self.redis_url,
                decode_responses=False,
            )

        return self._client

    def _make_key(self, key: str) -> str:
        """Create a prefixed key.

        Args:
            key: The original key.

        Returns:
            Prefixed key for Redis.
        """
        return f"{self.key_prefix}{key}"

    def _strip_prefix(self, key: str | bytes) -> str:
        """Remove the prefix from a key.

        Args:
            key: The prefixed key.

        Returns:
            Original key without prefix.
        """
        if isinstance(key, bytes):
            key = key.decode()
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
        client = await self._get_client()
        redis_key = self._make_key(key)

        result = await client.get(redis_key)
        if result is None:
            self._stats.misses += 1
            return None

        self._stats.hits += 1
        return result if isinstance(result, bytes) else result.encode()

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
        client = await self._get_client()
        redis_key = self._make_key(key)
        value_size = len(value)

        # Check if updating an existing key
        old_size = 0
        old_value = await client.get(redis_key)
        if old_value is not None:
            old_size = len(old_value)

        # Store with optional TTL
        if ttl_seconds:
            await client.setex(redis_key, ttl_seconds, value)
        else:
            await client.set(redis_key, value)

        # Update stats
        if old_size == 0:
            self._stats.item_count += 1
        self._stats.used_bytes += value_size - old_size

        return True

    async def delete(self, key: str) -> bool:
        """Delete a value by key.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False if it didn't exist.
        """
        self._check_closed()
        client = await self._get_client()
        redis_key = self._make_key(key)

        # Get size before deleting for stats
        old_value = await client.get(redis_key)
        if old_value is None:
            return False

        old_size = len(old_value)
        deleted = await client.delete(redis_key)

        if deleted:
            self._stats.used_bytes -= old_size
            self._stats.item_count -= 1

        return deleted > 0

    async def exists(self, key: str) -> bool:
        """Check if a key exists.

        Args:
            key: The key to check.

        Returns:
            True if the key exists, False otherwise.
        """
        self._check_closed()
        client = await self._get_client()
        redis_key = self._make_key(key)
        return await client.exists(redis_key) > 0

    async def keys(self, pattern: str | None = None) -> list[str]:
        """List all keys, optionally filtered by pattern.

        Args:
            pattern: Optional glob pattern to filter keys.

        Returns:
            List of matching keys (without prefix).
        """
        self._check_closed()
        client = await self._get_client()

        # Build Redis pattern
        redis_pattern = f"{self.key_prefix}{pattern}" if pattern else f"{self.key_prefix}*"

        # Get keys
        if self.cluster_mode or self.mode == "cluster":
            # For cluster mode, scan all nodes
            all_keys: list[str] = []
            async for key in client.scan_iter(match=redis_pattern):
                all_keys.append(self._strip_prefix(key))
            return all_keys
        else:
            raw_keys = await client.keys(redis_pattern)
            return [self._strip_prefix(k) for k in raw_keys]

    async def clear(self) -> int:
        """Clear all stored data with the key prefix.

        Returns:
            Number of items cleared.
        """
        self._check_closed()
        client = await self._get_client()

        # Get all keys with our prefix
        keys_to_delete = await self.keys()
        count = len(keys_to_delete)

        if count > 0:
            redis_keys = [self._make_key(k) for k in keys_to_delete]
            if self.cluster_mode or self.mode == "cluster":
                # In cluster mode, delete keys one by one or use pipelines
                for rk in redis_keys:
                    await client.delete(rk)
            else:
                await client.delete(*redis_keys)

        self._stats.used_bytes = 0
        self._stats.item_count = 0

        return count

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._closed:
            return

        if self._client:
            await self._client.aclose()
            self._client = None

        self._closed = True

    async def get_many(self, keys: list[str]) -> dict[str, bytes | None]:
        """Retrieve multiple values by keys.

        Uses Redis MGET for efficient batch retrieval.

        Args:
            keys: List of keys to look up.

        Returns:
            Dictionary mapping keys to values (None for missing keys).
        """
        self._check_closed()
        if not keys:
            return {}

        client = await self._get_client()
        redis_keys = [self._make_key(k) for k in keys]

        if self.cluster_mode or self.mode == "cluster":
            # Cluster mode doesn't support MGET across slots
            result = {}
            for key in keys:
                result[key] = await self.get(key)
            return result

        values = await client.mget(redis_keys)
        result = {}

        for key, value in zip(keys, values, strict=True):
            if value is None:
                self._stats.misses += 1
                result[key] = None
            else:
                self._stats.hits += 1
                result[key] = value if isinstance(value, bytes) else value.encode()

        return result

    async def put_many(
        self,
        items: dict[str, bytes],
        ttl_seconds: int | None = None,
    ) -> dict[str, bool]:
        """Store multiple values.

        Uses Redis MSET for efficient batch storage (without TTL).
        Falls back to individual SET commands when TTL is required.

        Args:
            items: Dictionary mapping keys to values.
            ttl_seconds: Optional time-to-live in seconds.

        Returns:
            Dictionary mapping keys to success status.
        """
        self._check_closed()
        if not items:
            return {}

        client = await self._get_client()
        result = dict.fromkeys(items, True)

        if ttl_seconds or self.cluster_mode or self.mode == "cluster":
            # Use individual SET commands for TTL or cluster mode
            for key, value in items.items():
                try:
                    await self.put(key, value, ttl_seconds)
                except Exception:
                    result[key] = False
        else:
            # Use MSET for efficiency
            redis_items = {self._make_key(k): v for k, v in items.items()}
            await client.mset(redis_items)

            # Update stats
            for _key, value in items.items():
                self._stats.item_count += 1
                self._stats.used_bytes += len(value)

        return result

    async def size(self, key: str) -> int | None:
        """Get the size of a stored value in bytes.

        Uses Redis STRLEN for efficiency.

        Args:
            key: The key to check.

        Returns:
            Size in bytes if the key exists, None otherwise.
        """
        self._check_closed()
        client = await self._get_client()
        redis_key = self._make_key(key)

        if not await client.exists(redis_key):
            return None

        return await client.strlen(redis_key)

    async def ttl(self, key: str) -> int | None:
        """Get the remaining TTL of a key in seconds.

        Args:
            key: The key to check.

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist.
        """
        self._check_closed()
        client = await self._get_client()
        redis_key = self._make_key(key)
        return await client.ttl(redis_key)

    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """Set or update the TTL of an existing key.

        Args:
            key: The key to update.
            ttl_seconds: New TTL in seconds.

        Returns:
            True if the TTL was set, False if the key doesn't exist.
        """
        self._check_closed()
        client = await self._get_client()
        redis_key = self._make_key(key)
        return await client.expire(redis_key, ttl_seconds)

    async def info(self) -> dict:
        """Get Redis server information.

        Returns:
            Dictionary with Redis INFO command output.
        """
        self._check_closed()
        client = await self._get_client()

        if self.cluster_mode or self.mode == "cluster":
            # Return info from a single node
            return {"cluster": True, "mode": "cluster"}

        return await client.info()
