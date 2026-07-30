"""
Steady-state warmup for the KV stack.

Benchmarking KV cache management against storage is only meaningful once
the cache is at steady state: every tier full, so each new store forces an
eviction. This module fills the stack with unique random token sequences
until a byte target derived from the configured tier capacities is reached,
then verifies eviction is active by checking that the earliest stored
sequence is no longer retrievable.

The warmup runs inside the server process because LMCache's local tiers
(CPU pool, local disk with in-memory index) belong to the engine instance —
an external filler process would populate a different cache.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from kvbench.kv.base import KVStack

logger = logging.getLogger(__name__)


class WarmupRequest(BaseModel):
    """Parameters for a warmup run.

    Attributes:
        target_gb: Explicit fill target in GB. When unset, the target is
            fill_factor x the stack's total configured tier capacity.
        fill_factor: Multiple of total tier capacity to store (>1 ensures
            every tier wraps into eviction).
        seq_tokens: Tokens per stored sequence (will be chunk-aligned).
        concurrency: Parallel store workers.
    """

    target_gb: float | None = Field(default=None, gt=0.0)
    fill_factor: float = Field(default=1.25, gt=0.0, le=10.0)
    seq_tokens: int = Field(default=2048, ge=256, le=131072)
    concurrency: int = Field(default=4, ge=1, le=64)


@dataclass
class WarmupState:
    """Progress of the current/last warmup run."""

    state: str = "idle"  # idle | running | done | cancelled | error
    target_bytes: int = 0
    stored_bytes: int = 0
    sequences: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    evicting: bool = False
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        elapsed = (self.finished_at or now) - self.started_at if self.started_at else 0.0
        rate = self.stored_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
        return {
            "state": self.state,
            "target_bytes": self.target_bytes,
            "stored_bytes": self.stored_bytes,
            "progress": (
                min(1.0, self.stored_bytes / self.target_bytes) if self.target_bytes else 0.0
            ),
            "sequences": self.sequences,
            "elapsed_s": round(elapsed, 1),
            "rate_mb_s": round(rate, 1),
            "evicting": self.evicting,
            "error": self.error,
            "params": self.params,
        }


class WarmupController:
    """Drives the KV stack to steady state with unique random sequences."""

    EVICTION_CHECK_ATTEMPTS = 5
    EVICTION_CHECK_DELAY_S = 2.0

    def __init__(self, kv: KVStack) -> None:
        self.kv = kv
        self.status = WarmupState()
        self._task: asyncio.Task | None = None
        self._first_seq: list[int] | None = None
        self._seq_counter = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, params: WarmupRequest) -> WarmupState:
        """Start a warmup run in the background.

        Raises:
            RuntimeError: If a warmup is already running or no byte target
                can be determined.
        """
        if self.running:
            raise RuntimeError("Warmup already running")

        capacity = self._capacity_info()
        chunk_size = capacity.get("chunk_size") or 256
        bytes_per_token = capacity["bytes_per_token"]
        seq_tokens = max(chunk_size, (params.seq_tokens // chunk_size) * chunk_size)

        if params.target_gb is not None:
            target_bytes = int(params.target_gb * 1024**3)
        else:
            total_capacity = capacity.get("total_capacity_bytes") or 0
            if total_capacity <= 0:
                raise RuntimeError(
                    "Cannot derive a warmup target: the KV stack reports no "
                    "configured tier capacity (remote-only backend?). "
                    "Pass an explicit target_gb."
                )
            target_bytes = int(total_capacity * params.fill_factor)

        self.status = WarmupState(
            state="running",
            target_bytes=target_bytes,
            started_at=time.time(),
            params={
                "seq_tokens": seq_tokens,
                "concurrency": params.concurrency,
                "fill_factor": params.fill_factor,
                "target_gb": params.target_gb,
                "bytes_per_token": bytes_per_token,
                "vocab_size": capacity.get("vocab_size", 128256),
            },
        )
        self._first_seq = None
        self._task = asyncio.create_task(self._run())
        logger.info(
            f"Warmup started: target {target_bytes / 1e9:.2f} GB, "
            f"{seq_tokens} tokens/seq ({seq_tokens * bytes_per_token / 1e6:.1f} MB), "
            f"concurrency {params.concurrency}"
        )
        return self.status

    async def cancel(self) -> None:
        """Cancel a running warmup."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            if self.status.state == "running":
                # Task was cancelled before its coroutine ever ran, so the
                # run loop's own cancellation handler never executed.
                self.status.state = "cancelled"
                self.status.finished_at = time.time()

    def _capacity_info(self) -> dict[str, Any]:
        capacity_fn = getattr(self.kv, "capacity_info", None)
        if capacity_fn is None:
            raise RuntimeError("KV stack does not expose capacity_info()")
        return capacity_fn()

    def _next_sequence(self, rng: random.Random, seq_tokens: int, vocab: int) -> list[int]:
        """Generate a unique random token sequence.

        The first tokens encode a per-run counter so no two sequences share
        a chunk-aligned prefix (which would dedupe in the cache and store
        fewer bytes than accounted).
        """
        self._seq_counter += 1
        nonce = self._seq_counter
        tokens = [rng.randrange(vocab) for _ in range(seq_tokens)]
        for i in range(8):
            tokens[i] = (nonce >> (i * 8)) & 0xFF
        return tokens

    async def _run(self) -> None:
        params = self.status.params
        seq_tokens: int = params["seq_tokens"]
        concurrency: int = params["concurrency"]
        bytes_per_token: int = params["bytes_per_token"]
        vocab: int = params["vocab_size"]
        seq_bytes = seq_tokens * bytes_per_token

        try:
            workers = [
                asyncio.create_task(self._worker(i, seq_tokens, seq_bytes, vocab))
                for i in range(concurrency)
            ]
            try:
                await asyncio.gather(*workers)
            except asyncio.CancelledError:
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                raise
            await self._check_eviction()
            self.status.state = "done"
            logger.info(
                f"Warmup done: {self.status.stored_bytes / 1e9:.2f} GB in "
                f"{self.status.sequences} sequences; evicting={self.status.evicting}"
            )
        except asyncio.CancelledError:
            self.status.state = "cancelled"
            logger.info("Warmup cancelled")
        except Exception as e:
            self.status.state = "error"
            self.status.error = str(e)
            logger.exception("Warmup failed")
        finally:
            self.status.finished_at = time.time()

    async def _worker(self, worker_id: int, seq_tokens: int, seq_bytes: int, vocab: int) -> None:
        rng = random.Random(0xBEEF ^ worker_id)
        while self.status.stored_bytes < self.status.target_bytes:
            tokens = self._next_sequence(rng, seq_tokens, vocab)
            if self._first_seq is None:
                self._first_seq = tokens
            await self.kv.store(tokens)
            self.status.stored_bytes += seq_bytes
            self.status.sequences += 1

    async def _check_eviction(self) -> None:
        """Steady state means the earliest stored data has (partly) been evicted.

        Stores return once the CPU tier accepts the data; lower-tier writes
        and evictions drain asynchronously, so the check retries with a
        settling delay before concluding.
        """
        if self._first_seq is None:
            return
        try:
            for _ in range(self.EVICTION_CHECK_ATTEMPTS):
                hit_tokens = await self.kv.lookup(self._first_seq)
                self.status.evicting = hit_tokens < len(self._first_seq)
                if self.status.evicting:
                    return
                await asyncio.sleep(self.EVICTION_CHECK_DELAY_S)
        except Exception as e:
            logger.warning(f"Eviction check failed: {e}")
