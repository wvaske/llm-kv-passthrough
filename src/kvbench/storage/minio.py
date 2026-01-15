"""
MinIO Storage Backend.

This module provides an S3-compatible storage backend using MinIO
or any S3-compatible object storage service.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

import aioboto3
from botocore.config import Config

from kvbench.storage.base import StorageBackend

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client


class MinIOStorageBackend(StorageBackend):
    """MinIO/S3-compatible storage backend.

    This backend stores data in MinIO or any S3-compatible object storage
    service using the aioboto3 library for async operations.

    Attributes:
        name: Name of the backend instance.
        max_size_bytes: Maximum storage capacity in bytes (advisory).
        endpoint_url: MinIO/S3 endpoint URL.
        bucket_name: S3 bucket name.
        key_prefix: Prefix for all object keys.
    """

    def __init__(
        self,
        endpoint_url: str,
        bucket_name: str,
        max_size_bytes: int,
        name: str = "minio",
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
        key_prefix: str = "kvbench/",
        use_ssl: bool = True,
    ) -> None:
        """Initialize the MinIO storage backend.

        Args:
            endpoint_url: MinIO/S3 endpoint URL (e.g., http://localhost:9000).
            bucket_name: S3 bucket name for storing objects.
            max_size_bytes: Maximum storage capacity in bytes (advisory).
            name: Name identifier for this backend instance.
            access_key: S3 access key (uses env var if not provided).
            secret_key: S3 secret key (uses env var if not provided).
            region: S3 region (default: us-east-1).
            key_prefix: Prefix for all object keys.
            use_ssl: Whether to use SSL/TLS (default: True).
        """
        super().__init__(max_size_bytes=max_size_bytes, name=name)
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self.key_prefix = key_prefix
        self.use_ssl = use_ssl

        self._session: aioboto3.Session | None = None
        self._client: S3Client | None = None
        self._bucket_exists = False

    async def _get_client(self) -> S3Client:
        """Get or create the S3 client.

        Returns:
            S3 client instance.
        """
        if self._client is not None:
            return self._client

        if self._session is None:
            self._session = aioboto3.Session()

        config = Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        # Create client context manager and get client
        client_kwargs = {
            "service_name": "s3",
            "endpoint_url": self.endpoint_url,
            "region_name": self.region,
            "config": config,
            "use_ssl": self.use_ssl,
        }

        if self.access_key and self.secret_key:
            client_kwargs["aws_access_key_id"] = self.access_key
            client_kwargs["aws_secret_access_key"] = self.secret_key

        # We need to manage the client lifecycle ourselves
        self._client_cm = self._session.client(**client_kwargs)
        self._client = await self._client_cm.__aenter__()

        # Ensure bucket exists
        if not self._bucket_exists:
            await self._ensure_bucket()

        return self._client

    async def _ensure_bucket(self) -> None:
        """Ensure the bucket exists, creating it if necessary."""
        if self._bucket_exists:
            return

        client = self._client
        assert client is not None

        try:
            await client.head_bucket(Bucket=self.bucket_name)
            self._bucket_exists = True
        except Exception:
            try:
                await client.create_bucket(Bucket=self.bucket_name)
                self._bucket_exists = True
            except Exception:
                # Bucket may already exist (race condition)
                self._bucket_exists = True

    def _make_key(self, key: str) -> str:
        """Create a prefixed object key.

        Args:
            key: The original key.

        Returns:
            Prefixed object key for S3.
        """
        return f"{self.key_prefix}{key}"

    def _strip_prefix(self, key: str) -> str:
        """Remove the prefix from an object key.

        Args:
            key: The prefixed object key.

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
        client = await self._get_client()
        object_key = self._make_key(key)

        try:
            response = await client.get_object(
                Bucket=self.bucket_name,
                Key=object_key,
            )
            async with response["Body"] as stream:
                data = await stream.read()
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

        Note: S3 doesn't natively support TTL. Lifecycle rules would need
        to be configured on the bucket for automatic expiration.

        Args:
            key: The key to store the value under.
            value: The value to store as bytes.
            ttl_seconds: Optional time-to-live (not directly supported by S3).

        Returns:
            True if the value was stored successfully, False otherwise.
        """
        self._check_closed()
        client = await self._get_client()
        object_key = self._make_key(key)
        value_size = len(value)

        try:
            # Check for existing object
            old_size = 0
            try:
                response = await client.head_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                )
                old_size = response.get("ContentLength", 0)
            except Exception:
                pass

            # Prepare metadata
            metadata = {}
            if ttl_seconds is not None:
                metadata["kvbench-ttl"] = str(ttl_seconds)

            # Upload object
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=value,
                Metadata=metadata,
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
        client = await self._get_client()
        object_key = self._make_key(key)

        try:
            # Get size for stats
            response = await client.head_object(
                Bucket=self.bucket_name,
                Key=object_key,
            )
            old_size = response.get("ContentLength", 0)

            await client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key,
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
        client = await self._get_client()
        object_key = self._make_key(key)

        try:
            await client.head_object(
                Bucket=self.bucket_name,
                Key=object_key,
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
        client = await self._get_client()

        keys_list: list[str] = []
        paginator = client.get_paginator("list_objects_v2")

        async for page in paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=self.key_prefix,
        ):
            for obj in page.get("Contents", []):
                obj_key = obj["Key"]
                key = self._strip_prefix(obj_key)
                if pattern is None or fnmatch.fnmatch(key, pattern):
                    keys_list.append(key)

        return keys_list

    async def clear(self) -> int:
        """Clear all stored data with the key prefix.

        Returns:
            Number of items cleared.
        """
        self._check_closed()
        client = await self._get_client()

        # List all keys
        all_keys = await self.keys()
        count = len(all_keys)

        if count == 0:
            return 0

        # Delete in batches (S3 supports up to 1000 objects per delete)
        batch_size = 1000
        for i in range(0, len(all_keys), batch_size):
            batch = all_keys[i:i + batch_size]
            objects = [{"Key": self._make_key(k)} for k in batch]

            await client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": objects},
            )

        self._stats.used_bytes = 0
        self._stats.item_count = 0

        return count

    async def close(self) -> None:
        """Close the S3 client."""
        if self._closed:
            return

        if self._client is not None and hasattr(self, "_client_cm"):
            await self._client_cm.__aexit__(None, None, None)
            self._client = None

        self._closed = True

    async def size(self, key: str) -> int | None:
        """Get the size of a stored value in bytes.

        Args:
            key: The key to check.

        Returns:
            Size in bytes if the key exists, None otherwise.
        """
        self._check_closed()
        client = await self._get_client()
        object_key = self._make_key(key)

        try:
            response = await client.head_object(
                Bucket=self.bucket_name,
                Key=object_key,
            )
            return response.get("ContentLength", 0)
        except Exception:
            return None

    async def get_many(self, keys: list[str]) -> dict[str, bytes | None]:
        """Retrieve multiple values by keys.

        Uses concurrent requests for efficiency.

        Args:
            keys: List of keys to look up.

        Returns:
            Dictionary mapping keys to values (None for missing keys).
        """
        self._check_closed()
        if not keys:
            return {}

        import asyncio

        async def get_one(key: str) -> tuple[str, bytes | None]:
            return key, await self.get(key)

        results = await asyncio.gather(*[get_one(k) for k in keys])
        return dict(results)

    async def put_many(
        self,
        items: dict[str, bytes],
        ttl_seconds: int | None = None,
    ) -> dict[str, bool]:
        """Store multiple values.

        Uses concurrent requests for efficiency.

        Args:
            items: Dictionary mapping keys to values.
            ttl_seconds: Optional time-to-live.

        Returns:
            Dictionary mapping keys to success status.
        """
        self._check_closed()
        if not items:
            return {}

        import asyncio

        async def put_one(key: str, value: bytes) -> tuple[str, bool]:
            return key, await self.put(key, value, ttl_seconds)

        results = await asyncio.gather(
            *[put_one(k, v) for k, v in items.items()]
        )
        return dict(results)
