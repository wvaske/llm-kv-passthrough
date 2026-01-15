"""
Abstract KV Connector Interface.

This module defines the abstract interface for KV cache connectors.
Connectors handle the storage and retrieval of KV cache data using
different underlying systems (LMCache, Mooncake, Dynamo).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kvbench.storage.base import StorageBackend


@dataclass
class ConnectorStats:
    """Statistics for a KV connector.

    Attributes:
        stores: Number of store operations.
        store_bytes: Total bytes stored.
        loads: Number of load operations.
        load_bytes: Total bytes loaded.
        hits: Number of cache hits.
        misses: Number of cache misses.
        errors: Number of errors encountered.
        created_at: When the connector was created.
    """

    stores: int = 0
    store_bytes: int = 0
    loads: int = 0
    load_bytes: int = 0
    hits: int = 0
    misses: int = 0
    errors: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def total_operations(self) -> int:
        """Return total number of operations."""
        return self.stores + self.loads


@dataclass
class ChunkMetadata:
    """Metadata for a KV cache chunk.

    Attributes:
        chunk_hash: Hash identifying the chunk.
        num_tokens: Number of tokens in the chunk.
        size_bytes: Size of the chunk in bytes.
        created_at: When the chunk was created.
        model_name: Name of the model this chunk belongs to.
        layer_range: Optional tuple of (start_layer, end_layer).
        worker_id: ID of the worker that created this chunk.
    """

    chunk_hash: str
    num_tokens: int
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)
    model_name: str | None = None
    layer_range: tuple[int, int] | None = None
    worker_id: int = 0


class KVConnector(ABC):
    """Abstract base class for KV cache connectors.

    Connectors provide an interface between the mock LLM servers and
    the underlying storage backends. They handle key formatting,
    chunking strategies, and caching logic specific to each system.

    Attributes:
        name: Name of the connector instance.
        storage: The storage backend to use.
    """

    def __init__(
        self,
        storage: StorageBackend,
        name: str = "unknown",
    ) -> None:
        """Initialize the KV connector.

        Args:
            storage: Storage backend for persisting KV cache data.
            name: Name identifier for this connector instance.
        """
        self.name = name
        self.storage = storage
        self._stats = ConnectorStats()
        self._closed = False

    @property
    def stats(self) -> ConnectorStats:
        """Return current connector statistics."""
        return self._stats

    @property
    def is_closed(self) -> bool:
        """Check if the connector has been closed."""
        return self._closed

    def _check_closed(self) -> None:
        """Raise an error if the connector is closed."""
        if self._closed:
            raise RuntimeError(f"KV connector '{self.name}' is closed")

    @abstractmethod
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
        ...

    @abstractmethod
    async def load(self, chunk_hash: str) -> bytes | None:
        """Load a KV cache chunk.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            The KV cache data as bytes if found, None otherwise.
        """
        ...

    @abstractmethod
    async def exists(self, chunk_hash: str) -> bool:
        """Check if a KV cache chunk exists.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            True if the chunk exists, False otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, chunk_hash: str) -> bool:
        """Delete a KV cache chunk.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            True if the chunk was deleted, False if it didn't exist.
        """
        ...

    @abstractmethod
    def make_key(self, chunk_hash: str, **kwargs) -> str:
        """Generate a storage key for a chunk.

        Args:
            chunk_hash: Hash identifying the chunk.
            **kwargs: Additional arguments for key generation.

        Returns:
            The formatted storage key.
        """
        ...

    async def store_many(
        self,
        chunks: list[tuple[str, int]],
    ) -> dict[str, bool]:
        """Store multiple KV cache chunks.

        Default implementation calls store() for each chunk.
        Subclasses may override for better performance.

        Args:
            chunks: List of (chunk_hash, num_tokens) tuples.

        Returns:
            Dictionary mapping chunk hashes to success status.
        """
        self._check_closed()
        results = {}
        for chunk_hash, num_tokens in chunks:
            results[chunk_hash] = await self.store(chunk_hash, num_tokens)
        return results

    async def load_many(self, chunk_hashes: list[str]) -> dict[str, bytes | None]:
        """Load multiple KV cache chunks.

        Default implementation calls load() for each chunk.
        Subclasses may override for better performance.

        Args:
            chunk_hashes: List of chunk hashes to load.

        Returns:
            Dictionary mapping chunk hashes to data (None for missing).
        """
        self._check_closed()
        results = {}
        for chunk_hash in chunk_hashes:
            results[chunk_hash] = await self.load(chunk_hash)
        return results

    async def exists_many(self, chunk_hashes: list[str]) -> dict[str, bool]:
        """Check if multiple KV cache chunks exist.

        Default implementation calls exists() for each chunk.
        Subclasses may override for better performance.

        Args:
            chunk_hashes: List of chunk hashes to check.

        Returns:
            Dictionary mapping chunk hashes to existence status.
        """
        self._check_closed()
        results = {}
        for chunk_hash in chunk_hashes:
            results[chunk_hash] = await self.exists(chunk_hash)
        return results

    async def get_metadata(self, chunk_hash: str) -> ChunkMetadata | None:  # noqa: ARG002
        """Get metadata for a KV cache chunk.

        Default implementation returns None.
        Subclasses may override to provide metadata.

        Args:
            chunk_hash: Hash identifying the chunk.

        Returns:
            ChunkMetadata if the chunk exists and has metadata, None otherwise.
        """
        self._check_closed()
        return None

    async def clear(self) -> int:
        """Clear all stored KV cache data.

        Returns:
            Number of items cleared.
        """
        self._check_closed()
        return await self.storage.clear()

    async def close(self) -> None:
        """Close the connector and release resources."""
        if not self._closed:
            self._closed = True
            await self.storage.close()

    async def __aenter__(self) -> KVConnector:
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
            f"storage={self.storage.name!r}, "
            f"stats={self._stats.total_operations} ops)"
        )
