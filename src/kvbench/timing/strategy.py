"""
Timing Strategy abstraction for KV-Bench.

This module defines the abstract interface for timing strategies that calculate
latency for LLM inference operations. Different strategies provide different
levels of fidelity:

- SimpleTimingStrategy: Fixed ms-per-token timing for basic benchmarks
- RooflineTimingStrategy: GPU roofline model with TP/PP communication overhead

The TimingStrategy pattern allows users to choose the appropriate timing model
based on their benchmarking needs - from simple testing to GPU-accurate simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TimingResult:
    """Result of a timing calculation.

    Contains the total latency along with a breakdown of component contributions
    and classification of whether the operation is compute-bound or memory-bound.

    Attributes:
        total_ms: Total latency in milliseconds.
        breakdown: Component breakdown mapping names to their contribution in ms.
                   For simple timing, this might be {"simple_prefill": 100.0}.
                   For roofline, this might be {"compute_ms": 5.0, "memory_ms": 3.0,
                   "allreduce_ms": 0.4, "pipeline_ms": 0.1}.
        is_compute_bound: Whether the operation is compute-bound (True) or
                         memory-bound (False). For prefill, typically True.
                         For decode, typically False.
    """

    total_ms: float
    breakdown: dict[str, float] = field(default_factory=dict)
    is_compute_bound: bool = True

    def __post_init__(self) -> None:
        """Validate timing result after initialization."""
        if self.total_ms < 0:
            raise ValueError(f"total_ms must be non-negative, got {self.total_ms}")


class TimingStrategy(ABC):
    """Abstract base class for timing strategies.

    A timing strategy calculates the latency for prefill and decode operations
    in LLM inference. Implementations may use different models:

    - Simple: Fixed ms-per-token values
    - Roofline: GPU compute/memory roofline model with communication overhead

    This abstraction allows the benchmarking system to work with any timing
    model while maintaining a consistent interface.

    Example:
        >>> strategy = SimpleTimingStrategy(prefill_ms_per_token=0.1, decode_ms_per_token=1.0)
        >>> result = strategy.prefill_latency(1000)
        >>> print(f"Prefill latency: {result.total_ms}ms")
        Prefill latency: 100.0ms
    """

    @abstractmethod
    def prefill_latency(
        self, num_tokens: int, batch_size: int = 1, context_tokens: int = 0
    ) -> TimingResult:
        """Calculate latency for prefill operation.

        Prefill processes the new input tokens in parallel, computing causal
        attention over them and any already-cached prefix. This is typically
        compute-bound due to the O(n^2) attention complexity.

        Args:
            num_tokens: Number of new input tokens to process.
            batch_size: Batch size for the operation (default: 1).
            context_tokens: Already-cached prefix tokens the new tokens attend
                to (default: 0). This is the correct cost model for partial
                cache hits: only the miss tokens are prefilled, but they still
                attend to the cached prefix.

        Returns:
            TimingResult containing total latency and breakdown.
        """
        pass

    @abstractmethod
    def decode_latency(self, context_length: int, batch_size: int = 1) -> TimingResult:
        """Calculate latency for decode operation (single token generation).

        Decode generates one token at a time, reading the model weights and
        KV cache for the entire context. This is typically memory-bound as
        it requires loading all model parameters for each token.

        Args:
            context_length: Current context length (all tokens so far).
            batch_size: Batch size for the operation (default: 1).

        Returns:
            TimingResult containing total latency and breakdown.
        """
        pass
