"""Test doubles shared across test suites.

FakeKVStack is an in-memory stand-in for a real KV management stack, used
only to unit-test server logic without installing LMCache. It performs no
storage I/O (production code never does either — real stacks own storage).
"""

from __future__ import annotations

from kvbench.kv.base import KVStack, KVStackStats


class FakeKVStack(KVStack):
    """In-memory KV stack with chunk-aligned prefix-caching semantics.

    Mirrors the observable behavior of LMCacheStack: lookups return the
    longest chunk-aligned cached prefix, stores cache full chunks of the
    sequence, and retrieves read the cached prefix.
    """

    def __init__(self, chunk_size: int = 256) -> None:
        self._chunk_size = chunk_size
        self.stats = KVStackStats()
        self.started = False
        self.closed = False
        self._prefixes: set[tuple[int, ...]] = set()

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def _hit_length(self, tokens: list[int]) -> int:
        aligned = (len(tokens) // self._chunk_size) * self._chunk_size
        for k in range(aligned, 0, -self._chunk_size):
            if tuple(tokens[:k]) in self._prefixes:
                return k
        return 0

    async def start(self) -> None:
        self.started = True

    async def lookup(self, tokens: list[int]) -> int:
        hit = self._hit_length(tokens)
        self.stats.lookups += 1
        self.stats.lookup_tokens += len(tokens)
        self.stats.hit_tokens += hit
        return hit

    async def store(self, tokens: list[int], skip_leading: int = 0) -> None:
        for k in range(self._chunk_size, len(tokens) + 1, self._chunk_size):
            self._prefixes.add(tuple(tokens[:k]))
        self.stats.stores += 1
        self.stats.stored_tokens += len(tokens) - skip_leading

    async def retrieve(self, tokens: list[int]) -> int:
        hit = self._hit_length(tokens)
        self.stats.retrieves += 1
        self.stats.retrieved_tokens += hit
        return hit

    async def close(self) -> None:
        self.closed = True
