"""
Timing Engine for KV-Bench.

This module provides timing strategies for emulating LLM inference latency
without actual GPU hardware. The timing engine calculates how long operations
would take on real GPUs, enabling realistic benchmarking of KV cache systems.
KV-Bench performs no storage I/O of its own - the timing engine only decides
how long the emulated GPU compute takes.

Timing Strategies:
    - SimpleTimingStrategy: Fixed ms-per-token timing for basic benchmarks
      and testing. Does not model GPU hardware characteristics.

    - RooflineTimingStrategy: Full roofline model with GPU profiles,
      compute/memory bound detection, and TP/PP communication timing.

Usage:
    >>> from kvbench.timing import SimpleTimingStrategy, TimingResult
    >>> strategy = SimpleTimingStrategy(prefill_ms_per_token=0.1, decode_ms_per_token=1.0)
    >>> result = strategy.prefill_latency(num_tokens=1000)
    >>> print(f"Prefill took {result.total_ms}ms")
    Prefill took 100.0ms

The timing engine integrates with the configuration system: the servers build
their strategy via create_timing_strategy(config), so timing mode and TP/PP
communication settings flow from CLI flags, environment variables, or YAML
config into served latency.
"""

from __future__ import annotations

from kvbench.timing.communication import (
    calculate_activation_size,
    calculate_allreduce_message_size,
    calculate_allreduce_ms,
    calculate_pipeline_send_recv_ms,
)
from kvbench.timing.factory import create_timing_strategy
from kvbench.timing.roofline import RooflineTimingStrategy
from kvbench.timing.simple import SimpleTimingStrategy
from kvbench.timing.strategy import TimingResult, TimingStrategy

__all__ = [
    "TimingStrategy",
    "TimingResult",
    "SimpleTimingStrategy",
    "RooflineTimingStrategy",
    "create_timing_strategy",
    "calculate_allreduce_ms",
    "calculate_allreduce_message_size",
    "calculate_activation_size",
    "calculate_pipeline_send_recv_ms",
]
