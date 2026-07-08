"""
KV management stack interface.

A KVStack is the only path between KV-Bench's mock inference servers and
storage. Servers speak in token sequences; the stack owns chunking,
hashing, tiering, and all storage I/O.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field


@dataclass
class KVStackStats:
    """Statistics for KV stack operations.

    Attributes:
        lookups: Number of lookup operations.
        hit_tokens: Total tokens found in cache across lookups.
        lookup_tokens: Total tokens queried across lookups.
        stores: Number of store operations.
        stored_tokens: Total tokens submitted for storage.
        retrieves: Number of retrieve operations.
        retrieved_tokens: Total tokens actually retrieved.
        errors: Number of failed operations.
        start_time: When the stack was started.
    """

    lookups: int = 0
    hit_tokens: int = 0
    lookup_tokens: int = 0
    stores: int = 0
    stored_tokens: int = 0
    retrieves: int = 0
    retrieved_tokens: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def token_hit_rate(self) -> float:
        """Fraction of looked-up tokens found in cache (0.0-1.0)."""
        if self.lookup_tokens == 0:
            return 0.0
        return self.hit_tokens / self.lookup_tokens


class KVStack(abc.ABC):
    """Interface to a KV cache management stack.

    All methods take full token sequences starting at position 0; prefix
    matching and chunk-level deduplication are the stack's responsibility,
    performed with its own (real) hashing and chunking logic.
    """

    stats: KVStackStats

    @property
    @abc.abstractmethod
    def chunk_size(self) -> int:
        """Token chunk size used by the stack (available after start())."""

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize the stack. Must be called before any KV operation."""

    @abc.abstractmethod
    async def lookup(self, tokens: list[int]) -> int:
        """Return the length of the cached token prefix for a sequence.

        Args:
            tokens: Full token sequence from position 0.

        Returns:
            Number of leading tokens present in the cache (chunk-aligned,
            as determined by the stack).
        """

    @abc.abstractmethod
    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        """Store KV cache for a token sequence.

        Args:
            tokens: Full token sequence from position 0.
            skip_leading: Leading tokens already cached (from a prior
                lookup); the stack skips storing them.
        """

    @abc.abstractmethod
    async def retrieve(self, tokens: list[int]) -> int:
        """Read cached KV data for a token sequence (the read path).

        Args:
            tokens: Full token sequence from position 0.

        Returns:
            Number of leading tokens whose KV data was retrieved.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Shut down the stack and release its resources."""

    async def __aenter__(self) -> KVStack:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()
