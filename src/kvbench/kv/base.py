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
from typing import Any


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

    Byte counters cover instantiated ranks only. Logical token counters count
    each sequence once, regardless of rank count. Neither is extrapolated to
    omitted ranks in a representative-rank benchmark.
    """

    lookups: int = 0
    hit_tokens: int = 0
    lookup_tokens: int = 0
    stores: int = 0
    stored_tokens: int = 0
    retrieves: int = 0
    retrieved_tokens: int = 0
    submitted_store_bytes: int = 0
    retrieved_bytes: int = 0
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

    async def lookup_pinned(self, tokens: list[int], lookup_id: str) -> int:  # noqa: ARG002 - optional capability
        """Look up the cached prefix the way a serving engine's connector does
        before scheduling a request: pin the hit chunks until ``unpin`` and
        let the stack refresh their recency. Stacks without pinning fall back
        to ``lookup``."""
        return await self.lookup(tokens)

    async def unpin(self, lookup_id: str) -> None:  # noqa: ARG002 - optional capability
        """Release the pins taken by ``lookup_pinned`` (no-op by default)."""
        return None

    @abc.abstractmethod
    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        """Store KV cache for a token sequence.

        Args:
            tokens: Full token sequence from position 0.
            skip_leading: Leading tokens already cached (from a prior
                lookup); the stack skips storing them.
        """

    @abc.abstractmethod
    async def retrieve(self, tokens: list[int], skip_leading: int = 0) -> int:
        """Read cached KV data for a token sequence (the read path).

        Args:
            tokens: Full token sequence from position 0.

        Returns:
            Number of leading tokens whose KV data was retrieved.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Shut down the stack and release its resources."""

    def capacity_info(self) -> dict[str, Any]:
        """Optional introspection used by warmup and run manifests."""
        raise NotImplementedError("KV stack does not expose capacity_info()")

    def usage_info(self) -> dict[str, dict[str, Any]]:
        return {}

    async def flush(self) -> None:
        """Drain owned pending writes; filesystem/device durability is separate."""
        return None

    async def __aenter__(self) -> KVStack:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()
