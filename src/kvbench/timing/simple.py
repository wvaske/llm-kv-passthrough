"""
Simple Timing Strategy for KV-Bench.

This module provides a simple timing strategy that uses fixed ms-per-token
values for prefill and decode operations. This is useful for:

- Basic benchmarking without GPU modeling complexity
- Testing and development
- Quick performance estimates
- Scenarios where exact GPU timing is not required

For realistic GPU emulation, use RooflineTimingStrategy instead.

The simple timing model:
- Prefill: total_ms = prefill_ms_per_token * num_tokens * batch_size
- Decode: total_ms = decode_ms_per_token * batch_size (per token generated)

Note that decode timing does NOT scale with context_length in simple mode,
and prefill timing ignores the cached-prefix context. This is intentional -
for context-aware timing, use the roofline strategy.
"""

from __future__ import annotations

from kvbench.timing.strategy import TimingResult, TimingStrategy


class SimpleTimingStrategy(TimingStrategy):
    """Simple fixed ms-per-token timing strategy.

    This strategy provides straightforward timing calculations without
    modeling GPU hardware characteristics. It's suitable for basic
    benchmarks and testing scenarios.

    The timing model is linear:
    - Prefill scales linearly with token count and batch size
    - Decode is fixed per token generated (batch size only)

    Attributes:
        prefill_ms_per_token: Milliseconds per token for prefill operations.
        decode_ms_per_token: Milliseconds per token for decode operations.

    Example:
        >>> strategy = SimpleTimingStrategy(
        ...     prefill_ms_per_token=0.1,
        ...     decode_ms_per_token=1.0
        ... )
        >>> # Prefill 1000 tokens
        >>> result = strategy.prefill_latency(1000)
        >>> print(result.total_ms)  # 0.1 * 1000 = 100.0
        100.0
        >>> # Decode one token (context_length doesn't matter in simple mode)
        >>> result = strategy.decode_latency(1000)
        >>> print(result.total_ms)  # 1.0 per token generated
        1.0
    """

    def __init__(
        self,
        prefill_ms_per_token: float,
        decode_ms_per_token: float,
    ) -> None:
        """Initialize the simple timing strategy.

        Args:
            prefill_ms_per_token: Milliseconds per token for prefill.
                                  Must be positive.
            decode_ms_per_token: Milliseconds per token for decode.
                                 Must be positive.

        Raises:
            ValueError: If either timing value is not positive.
        """
        if prefill_ms_per_token <= 0:
            raise ValueError(
                f"prefill_ms_per_token must be positive, got {prefill_ms_per_token}"
            )
        if decode_ms_per_token <= 0:
            raise ValueError(
                f"decode_ms_per_token must be positive, got {decode_ms_per_token}"
            )

        self.prefill_ms_per_token = prefill_ms_per_token
        self.decode_ms_per_token = decode_ms_per_token

    def prefill_latency(
        self,
        num_tokens: int,
        batch_size: int = 1,
        context_tokens: int = 0,  # noqa: ARG002
    ) -> TimingResult:
        """Calculate prefill latency for the given number of new tokens.

        In simple mode, prefill latency scales linearly with the number of
        new tokens and batch size:

            total_ms = prefill_ms_per_token * num_tokens * batch_size

        The cached prefix (context_tokens) is ignored - only the new tokens
        cost time. Prefill is assumed to be compute-bound.

        Args:
            num_tokens: Number of new input tokens to process.
            batch_size: Batch size for the operation (default: 1).
            context_tokens: Already-cached prefix tokens (ignored in simple mode).

        Returns:
            TimingResult with calculated latency.
        """
        total_ms = self.prefill_ms_per_token * num_tokens * batch_size

        return TimingResult(
            total_ms=total_ms,
            breakdown={"simple_prefill": total_ms},
            is_compute_bound=True,  # Prefill is typically compute-bound
        )

    def decode_latency(
        self,
        context_length: int,  # noqa: ARG002
        batch_size: int = 1,
    ) -> TimingResult:
        """Calculate decode latency for generating a single token.

        In simple mode, decode latency is fixed per token generated,
        scaled only by batch size:

            total_ms = decode_ms_per_token * batch_size

        Note: context_length is ignored in simple mode. This is intentional -
        for context-aware timing that models memory bandwidth effects,
        use RooflineTimingStrategy.

        Decode is assumed to be memory-bound (is_compute_bound=False).

        Args:
            context_length: Current context length (ignored in simple mode).
            batch_size: Batch size for the operation (default: 1).

        Returns:
            TimingResult with calculated latency.
        """
        # Note: context_length is ignored in simple mode
        # Decode time is fixed per token, scaled by batch size
        total_ms = self.decode_ms_per_token * batch_size

        return TimingResult(
            total_ms=total_ms,
            breakdown={"simple_decode": total_ms},
            is_compute_bound=False,  # Decode is typically memory-bound
        )

    def __repr__(self) -> str:
        """Return string representation of the strategy."""
        return (
            f"SimpleTimingStrategy("
            f"prefill_ms_per_token={self.prefill_ms_per_token}, "
            f"decode_ms_per_token={self.decode_ms_per_token})"
        )
