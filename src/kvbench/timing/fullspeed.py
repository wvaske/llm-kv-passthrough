"""
Full-speed timing wrapper for KV-Bench.

Wraps any timing strategy so the server never sleeps for emulated GPU
time, while still computing what that time would have been. The server
records the emulated latency (breakdown key ``emulated_ms``) and reports
it as *effective* TTFT and tokens/s for the emulated model and GPU. Use it
to drive the KV storage path as hard as the KV stack allows and still know
what serving rate the represented system would have sustained.
"""

from __future__ import annotations

from kvbench.timing.strategy import TimingResult, TimingStrategy

EMULATED_MS_KEY = "emulated_ms"


class FullSpeedTimingStrategy(TimingStrategy):
    """Zero wall-clock latency; the wrapped strategy's latency is reported, not slept."""

    def __init__(self, inner: TimingStrategy) -> None:
        self.inner = inner

    def _wrap(self, result: TimingResult) -> TimingResult:
        return TimingResult(
            total_ms=0.0,
            breakdown={**result.breakdown, EMULATED_MS_KEY: result.total_ms},
            is_compute_bound=result.is_compute_bound,
        )

    def prefill_latency(
        self, num_tokens: int, batch_size: int = 1, context_tokens: int = 0
    ) -> TimingResult:
        return self._wrap(
            self.inner.prefill_latency(num_tokens, batch_size=batch_size, context_tokens=context_tokens)
        )

    def decode_latency(self, context_length: int, batch_size: int = 1) -> TimingResult:
        return self._wrap(self.inner.decode_latency(context_length, batch_size=batch_size))

    def __repr__(self) -> str:
        return f"FullSpeedTimingStrategy({self.inner!r})"


def emulated_ms(result: TimingResult) -> float:
    """The latency the emulated hardware would have taken for this result."""
    return result.breakdown.get(EMULATED_MS_KEY, result.total_ms)
