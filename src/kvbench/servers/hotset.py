"""Host-resident hot set of KV chunks.

Between turns of a multi-turn session the real serving stack keeps the
session's prefix in host memory: LMCache's CPU tier holds what was just
stored *and* re-inserts any chunk it had to read back from disk. A returning
session therefore reads from storage only what has since been evicted, not
its whole prefix. ``HotKVSet`` models that retention as an LRU of chunk
hashes bounded in bytes, so the emulator retrieves through the KV stack only
the non-resident part of a cached prefix. Without it, a session evicted once
re-reads its entire prefix from disk on every later turn (observed 2026-09-27:
12x the real disk reads at the same host-tier size).
"""

from __future__ import annotations

from collections import OrderedDict

_MOD = (1 << 61) - 1
_MUL = 1_000_003


class HotKVSet:
    """LRU of prefix-hashed chunks with a byte capacity (0 disables it)."""

    def __init__(self, capacity_bytes: int, chunk_size: int, chunk_bytes: int) -> None:
        self.capacity_bytes = int(capacity_bytes)
        self.chunk_size = int(chunk_size)
        self.chunk_bytes = int(chunk_bytes)
        self._lru: OrderedDict[int, None] = OrderedDict()
        self.hit_chunks = 0
        self.miss_chunks = 0
        self.evictions = 0

    @property
    def enabled(self) -> bool:
        return self.capacity_bytes > 0 and self.chunk_bytes > 0 and self.chunk_size > 0

    @property
    def resident_bytes(self) -> int:
        return len(self._lru) * self.chunk_bytes

    def _chunk_keys(self, tokens: list[int], n_chunks: int) -> list[int]:
        keys: list[int] = []
        h = 0
        limit = min(n_chunks * self.chunk_size, len(tokens))
        for i in range(limit):
            h = (h * _MUL + tokens[i] + 1) % _MOD
            if (i + 1) % self.chunk_size == 0:
                keys.append(h)
        return keys

    def resident_prefix_chunks(self, tokens: list[int], n_chunks: int) -> int:
        """Leading chunks of ``tokens`` that are resident (touching them)."""
        if not self.enabled or n_chunks <= 0:
            return 0
        n = 0
        for key in self._chunk_keys(tokens, n_chunks):
            if key in self._lru:
                self._lru.move_to_end(key)
                n += 1
            else:
                break
        self.hit_chunks += n
        self.miss_chunks += n_chunks - n
        return n

    def insert(self, tokens: list[int], n_chunks: int) -> None:
        """Mark the first ``n_chunks`` chunks resident, evicting LRU beyond capacity."""
        if not self.enabled or n_chunks <= 0:
            return
        for key in self._chunk_keys(tokens, n_chunks):
            if key in self._lru:
                self._lru.move_to_end(key)
            else:
                self._lru[key] = None
        while len(self._lru) * self.chunk_bytes > self.capacity_bytes and self._lru:
            self._lru.popitem(last=False)
            self.evictions += 1

    def info(self) -> dict[str, int]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "resident_bytes": self.resident_bytes,
            "hit_chunks": self.hit_chunks,
            "miss_chunks": self.miss_chunks,
            "evictions": self.evictions,
        }
