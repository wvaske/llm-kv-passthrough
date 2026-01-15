"""
LMCache KV Connector Implementation.

This module provides a KV connector compatible with LMCache key format.
It supports the LMCache key naming convention:
    format@model@world_size@worker_id@hash@suffix

Reference: https://github.com/LMCache/LMCache
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kvbench.connectors.base import ChunkMetadata, KVConnector

if TYPE_CHECKING:
    from kvbench.storage.base import StorageBackend


@dataclass
class LMCacheConfig:
    """Configuration for LMCache connector.

    Attributes:
        model_name: Name of the model for key generation.
        world_size: Total number of workers (tensor parallel size).
        worker_id: ID of this worker (0 to world_size-1).
        chunk_size: Number of tokens per chunk.
        key_format: Format string for the key prefix.
        key_suffix: Suffix for KV cache keys.
        bytes_per_token: Estimated bytes per token for mock data.
    """

    model_name: str = "llama-3.1-8b"
    world_size: int = 1
    worker_id: int = 0
    chunk_size: int = 256
    key_format: str = "lmcache"
    key_suffix: str = "@kv_bytes"
    bytes_per_token: int = 512  # Approximate KV cache size per token


class LMCacheConnector(KVConnector):
    """LMCache-compatible KV cache connector.

    This connector implements the LMCache key format and chunking strategy.
    It uses a consistent key format across all workers to enable shared
    KV cache storage in distributed deployments.

    Key Format: {format}@{model}@{world_size}@{worker_id}@{hash}{suffix}

    Example key:
        lmcache@llama-3.1-8b@4@0@abc123def456@kv_bytes

    Attributes:
        config: LMCache configuration.
    """

    def __init__(
        self,
        storage: StorageBackend,
        config: LMCacheConfig | None = None,
        name: str = "lmcache",
    ) -> None:
        """Initialize the LMCache connector.

        Args:
            storage: Storage backend for persisting KV cache data.
            config: LMCache configuration. Uses defaults if not provided.
            name: Name identifier for this connector instance.
        """
        super().__init__(storage=storage, name=name)
        self.config = config or LMCacheConfig()
        self._metadata: dict[str, ChunkMetadata] = {}

    def make_key(
        self,
        chunk_hash: str,
        worker_id: int | None = None,
        **kwargs,  # noqa: ARG002
    ) -> str:
        """Generate an LMCache-compatible storage key.

        Args:
            chunk_hash: Hash identifying the chunk.
            worker_id: Worker ID override. Uses config.worker_id if not provided.
            **kwargs: Additional arguments (ignored).

        Returns:
            The formatted LMCache key.
        """
        wid = worker_id if worker_id is not None else self.config.worker_id
        return (
            f"{self.config.key_format}@"
            f"{self.config.model_name}@"
            f"{self.config.world_size}@"
            f"{wid}@"
            f"{chunk_hash}"
            f"{self.config.key_suffix}"
        )

    def make_keys_for_all_workers(self, chunk_hash: str) -> list[str]:
        """Generate keys for all workers for a given chunk.

        This is useful for looking up or storing chunks across
        all tensor parallel workers.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            List of keys, one for each worker.
        """
        return [
            self.make_key(chunk_hash, worker_id=wid)
            for wid in range(self.config.world_size)
        ]

    def compute_chunk_hash(self, tokens: list[int]) -> str:
        """Compute a hash for a list of tokens.

        Args:
            tokens: List of token IDs.

        Returns:
            SHA-256 hash of the tokens as a hex string.
        """
        data = ",".join(str(t) for t in tokens).encode()
        return hashlib.sha256(data).hexdigest()

    def _generate_mock_data(self, num_tokens: int) -> bytes:
        """Generate mock KV cache data.

        Args:
            num_tokens: Number of tokens in the chunk.

        Returns:
            Mock data of appropriate size.
        """
        size = num_tokens * self.config.bytes_per_token
        return b"\x00" * size

    async def store(
        self,
        chunk_hash: str,
        num_tokens: int,
        data: bytes | None = None,
    ) -> bool:
        """Store a KV cache chunk.

        Args:
            chunk_hash: Hash identifying the chunk.
            num_tokens: Number of tokens in the chunk.
            data: Optional actual KV cache data. If None, mock data
                  of appropriate size will be generated.

        Returns:
            True if the chunk was stored successfully, False otherwise.
        """
        self._check_closed()

        # Generate mock data if not provided
        if data is None:
            data = self._generate_mock_data(num_tokens)

        key = self.make_key(chunk_hash)
        success = await self.storage.put(key, data)

        if success:
            self._stats.stores += 1
            self._stats.store_bytes += len(data)

            # Store metadata
            self._metadata[chunk_hash] = ChunkMetadata(
                chunk_hash=chunk_hash,
                num_tokens=num_tokens,
                size_bytes=len(data),
                model_name=self.config.model_name,
                worker_id=self.config.worker_id,
            )
        else:
            self._stats.errors += 1

        return success

    async def load(self, chunk_hash: str) -> bytes | None:
        """Load a KV cache chunk.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            The KV cache data as bytes if found, None otherwise.
        """
        self._check_closed()

        key = self.make_key(chunk_hash)
        data = await self.storage.get(key)

        self._stats.loads += 1
        if data is not None:
            self._stats.hits += 1
            self._stats.load_bytes += len(data)
        else:
            self._stats.misses += 1

        return data

    async def exists(self, chunk_hash: str) -> bool:
        """Check if a KV cache chunk exists.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            True if the chunk exists, False otherwise.
        """
        self._check_closed()

        key = self.make_key(chunk_hash)
        return await self.storage.exists(key)

    async def delete(self, chunk_hash: str) -> bool:
        """Delete a KV cache chunk.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            True if the chunk was deleted, False if it didn't exist.
        """
        self._check_closed()

        key = self.make_key(chunk_hash)
        success = await self.storage.delete(key)

        if success and chunk_hash in self._metadata:
            del self._metadata[chunk_hash]

        return success

    async def get_metadata(self, chunk_hash: str) -> ChunkMetadata | None:
        """Get metadata for a KV cache chunk.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            ChunkMetadata if the chunk exists and has metadata, None otherwise.
        """
        self._check_closed()
        return self._metadata.get(chunk_hash)

    async def store_many(
        self,
        chunks: list[tuple[str, int]],
    ) -> dict[str, bool]:
        """Store multiple KV cache chunks efficiently.

        Args:
            chunks: List of (chunk_hash, num_tokens) tuples.

        Returns:
            Dictionary mapping chunk hashes to success status.
        """
        self._check_closed()

        # Prepare all items
        items: dict[str, bytes] = {}
        chunk_info: dict[str, tuple[str, int]] = {}

        for chunk_hash, num_tokens in chunks:
            key = self.make_key(chunk_hash)
            data = self._generate_mock_data(num_tokens)
            items[key] = data
            chunk_info[key] = (chunk_hash, num_tokens)

        # Batch store
        store_results = await self.storage.put_many(items)

        # Process results
        results: dict[str, bool] = {}
        for key, success in store_results.items():
            chunk_hash, num_tokens = chunk_info[key]
            results[chunk_hash] = success

            if success:
                self._stats.stores += 1
                self._stats.store_bytes += len(items[key])
                self._metadata[chunk_hash] = ChunkMetadata(
                    chunk_hash=chunk_hash,
                    num_tokens=num_tokens,
                    size_bytes=len(items[key]),
                    model_name=self.config.model_name,
                    worker_id=self.config.worker_id,
                )
            else:
                self._stats.errors += 1

        return results

    async def load_many(self, chunk_hashes: list[str]) -> dict[str, bytes | None]:
        """Load multiple KV cache chunks efficiently.

        Args:
            chunk_hashes: List of chunk hashes to load.

        Returns:
            Dictionary mapping chunk hashes to data (None for missing).
        """
        self._check_closed()

        # Map keys to chunk hashes
        key_to_hash: dict[str, str] = {}
        keys: list[str] = []
        for chunk_hash in chunk_hashes:
            key = self.make_key(chunk_hash)
            keys.append(key)
            key_to_hash[key] = chunk_hash

        # Batch load
        load_results = await self.storage.get_many(keys)

        # Process results
        results: dict[str, bytes | None] = {}
        for key, data in load_results.items():
            chunk_hash = key_to_hash[key]
            results[chunk_hash] = data

            self._stats.loads += 1
            if data is not None:
                self._stats.hits += 1
                self._stats.load_bytes += len(data)
            else:
                self._stats.misses += 1

        return results

    async def exists_many(self, chunk_hashes: list[str]) -> dict[str, bool]:
        """Check if multiple KV cache chunks exist efficiently.

        Args:
            chunk_hashes: List of chunk hashes to check.

        Returns:
            Dictionary mapping chunk hashes to existence status.
        """
        self._check_closed()

        results: dict[str, bool] = {}
        for chunk_hash in chunk_hashes:
            results[chunk_hash] = await self.exists(chunk_hash)

        return results

    async def list_chunks(self, pattern: str | None = None) -> list[str]:
        """List all stored chunk hashes.

        Args:
            pattern: Optional pattern to filter chunks.

        Returns:
            List of chunk hashes.
        """
        self._check_closed()

        # Get all keys matching the pattern
        key_pattern = f"{self.config.key_format}@{self.config.model_name}@*"
        if pattern:
            key_pattern = pattern

        keys = await self.storage.keys(key_pattern)

        # Extract chunk hashes from keys
        hashes: list[str] = []
        suffix = self.config.key_suffix
        for key in keys:
            # Parse key: format@model@world_size@worker_id@hash@suffix
            if key.endswith(suffix):
                # Remove suffix and extract hash
                key_without_suffix = key[: -len(suffix)]
                parts = key_without_suffix.split("@")
                if len(parts) >= 5:
                    chunk_hash = parts[4]
                    if chunk_hash not in hashes:
                        hashes.append(chunk_hash)

        return hashes

    async def close(self) -> None:
        """Close the connector and release resources."""
        if not self._closed:
            self._metadata.clear()
            await super().close()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"LMCacheConnector("
            f"name={self.name!r}, "
            f"model={self.config.model_name!r}, "
            f"worker={self.config.worker_id}/{self.config.world_size}, "
            f"chunk_size={self.config.chunk_size})"
        )
