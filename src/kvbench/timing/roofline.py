"""
Roofline Timing Strategy for KV-Bench.

This module provides GPU-accurate timing using the roofline performance model.
It wraps the LatencyCalculator to provide timing through the TimingStrategy
interface, with optional tensor parallel (TP) and pipeline parallel (PP)
communication overhead.

The roofline model determines whether operations are compute-bound or
memory-bound based on arithmetic intensity, providing realistic latency
estimates for LLM inference.
"""

from __future__ import annotations

from kvbench.core.gpu_profiles import GPUProfile
from kvbench.core.latency import LatencyCalculator
from kvbench.core.models import ModelProfile, get_model_profile
from kvbench.timing.communication import (
    calculate_activation_size,
    calculate_allreduce_message_size,
    calculate_allreduce_ms,
    calculate_pipeline_send_recv_ms,
)
from kvbench.timing.strategy import TimingResult, TimingStrategy

# Fallback interconnect bandwidth when the GPU profile doesn't specify NVLink
DEFAULT_INTERCONNECT_BANDWIDTH_GB_S = 900.0


class RooflineTimingStrategy(TimingStrategy):
    """Timing strategy using roofline model for realistic GPU emulation.

    This strategy wraps the LatencyCalculator to provide roofline-based
    timing through the TimingStrategy interface. Optionally includes TP
    communication (AllReduce) and PP communication (send/recv) overhead.

    The roofline model determines performance based on:
    - Peak compute throughput (TFLOPS)
    - Peak memory bandwidth (TB/s)
    - Arithmetic intensity of the workload (FLOPs/byte)

    Communication overhead from tensor and pipeline parallelism can add
    20-30% to end-to-end latency. This strategy models:
    - AllReduce: After attention and FFN in each layer (TP)
    - Send/Recv: At pipeline stage boundaries (PP)

    The AllReduce and pipeline transfer sizes scale with the number of
    tokens in flight: the full new-token count during prefill, a single
    token during decode.

    Attributes:
        calculator: The underlying LatencyCalculator.
        include_tp_communication: Whether to add AllReduce timing.
        include_pp_communication: Whether to add pipeline send/recv timing.
        pp_size: Pipeline parallelism size.
        interconnect_bandwidth_gb_s: Interconnect bandwidth used for
            communication timing (NVLink from the GPU profile unless overridden).

    Example:
        >>> strategy = RooflineTimingStrategy(
        ...     'H100_SXM', 'llama-3.1-8b',
        ...     tp_size=4, pp_size=2,
        ...     include_tp_communication=True,
        ...     include_pp_communication=True,
        ... )
        >>> result = strategy.prefill_latency(1000)
        >>> print(f"Prefill latency: {result.total_ms}ms")
    """

    def __init__(
        self,
        gpu: str | GPUProfile,
        model: str | ModelProfile,
        tp_size: int = 1,
        pp_size: int = 1,
        efficiency: float = 0.7,
        include_tp_communication: bool = True,
        include_pp_communication: bool = True,
        nvlink_bandwidth_gb_s: float | None = None,
    ) -> None:
        """Initialize the roofline timing strategy.

        Args:
            gpu: GPU profile name or GPUProfile object.
            model: Model profile name or ModelProfile object.
            tp_size: Tensor parallelism size (default: 1).
            pp_size: Pipeline parallelism size (default: 1).
            efficiency: Hardware efficiency factor (default: 0.7).
            include_tp_communication: Whether to add AllReduce timing (default: True).
            include_pp_communication: Whether to add pipeline timing (default: True).
            nvlink_bandwidth_gb_s: Interconnect bandwidth in GB/s. When None
                (the default), the GPU profile's NVLink bandwidth is used,
                falling back to 900 GB/s if the profile has none.
        """
        self._calculator = LatencyCalculator(
            gpu=gpu,
            model=model,
            tp_size=tp_size,
            efficiency=efficiency,
        )

        # Store model for accessing hidden size and layers
        self._model = get_model_profile(model) if isinstance(model, str) else model

        self.pp_size = pp_size
        self.include_tp_communication = include_tp_communication
        self.include_pp_communication = include_pp_communication
        if nvlink_bandwidth_gb_s is not None:
            self.interconnect_bandwidth_gb_s = nvlink_bandwidth_gb_s
        else:
            self.interconnect_bandwidth_gb_s = (
                self._calculator.gpu.nvlink_bandwidth_gb_s
                or DEFAULT_INTERCONNECT_BANDWIDTH_GB_S
            )

    @property
    def calculator(self) -> LatencyCalculator:
        """Access the underlying LatencyCalculator for advanced use."""
        return self._calculator

    @property
    def tp_size(self) -> int:
        """Tensor parallelism size."""
        return self._calculator.tp_size

    def _allreduce_overhead_ms(self, num_tokens: int, batch_size: int) -> float:
        """Calculate AllReduce overhead across all layers.

        AllReduce operations occur:
        - After attention output projection (1 per layer)
        - After FFN down projection (1 per layer)
        Total: 2 AllReduce per layer * num_layers

        The reduced tensor is the hidden states for every token in flight,
        so the message size scales with num_tokens (the new tokens during
        prefill, 1 during decode).

        Args:
            num_tokens: Number of tokens in flight for this forward pass.
            batch_size: Batch size.

        Returns:
            Total AllReduce overhead in milliseconds.
        """
        if not self.include_tp_communication or self.tp_size <= 1:
            return 0.0

        # Hidden-state tensor for all tokens in flight: [batch, tokens, hidden]
        message_size = calculate_allreduce_message_size(
            batch_size=batch_size * num_tokens,
            hidden_size=self._model.hidden,
            dtype_bytes=self._model.bytes_per_element,
        )

        allreduce_per_op_ms = calculate_allreduce_ms(
            message_size_bytes=message_size,
            tp_size=self.tp_size,
            nvlink_bandwidth_gb_s=self.interconnect_bandwidth_gb_s,
        )

        # 2 AllReduce per layer (attention output + FFN down projection)
        total_ops = self._model.layers * 2
        return allreduce_per_op_ms * total_ops

    def _pipeline_overhead_ms(self, num_tokens: int, batch_size: int) -> float:
        """Calculate pipeline send/recv overhead at stage boundaries.

        In pipeline parallelism, activations are transferred between stages:
        - After each stage, activations sent to next stage
        - (pp_size - 1) stage boundaries in total

        Args:
            num_tokens: Number of tokens in flight for this forward pass.
            batch_size: Batch size.

        Returns:
            Total pipeline overhead in milliseconds.
        """
        if not self.include_pp_communication or self.pp_size <= 1:
            return 0.0

        activation_size = calculate_activation_size(
            batch_size=batch_size,
            sequence_length=num_tokens,
            hidden_size=self._model.hidden,
            dtype_bytes=self._model.bytes_per_element,
        )

        send_recv_ms = calculate_pipeline_send_recv_ms(
            activation_size_bytes=activation_size,
            bandwidth_gb_s=self.interconnect_bandwidth_gb_s,
        )

        num_boundaries = self.pp_size - 1
        return send_recv_ms * num_boundaries

    def _build_result(
        self,
        compute_ms: float,
        memory_ms: float,
        base_total_ms: float,
        is_compute_bound: bool,
        allreduce_ms: float,
        pipeline_ms: float,
    ) -> TimingResult:
        """Assemble a TimingResult from roofline and communication components."""
        breakdown = {
            "compute_ms": compute_ms,
            "memory_ms": memory_ms,
        }
        if allreduce_ms > 0:
            breakdown["allreduce_ms"] = allreduce_ms
        if pipeline_ms > 0:
            breakdown["pipeline_ms"] = pipeline_ms

        return TimingResult(
            total_ms=base_total_ms + allreduce_ms + pipeline_ms,
            breakdown=breakdown,
            is_compute_bound=is_compute_bound,
        )

    def prefill_latency(
        self, num_tokens: int, batch_size: int = 1, context_tokens: int = 0
    ) -> TimingResult:
        """Calculate prefill latency with optional communication overhead.

        Prefill is typically compute-bound due to the O(n^2) attention
        complexity. Communication overhead is added when TP > 1 or PP > 1.

        Args:
            num_tokens: Number of new input tokens to process.
            batch_size: Batch size (default: 1).
            context_tokens: Already-cached prefix tokens the new tokens
                attend to (default: 0).

        Returns:
            TimingResult with compute, memory, and communication breakdown.
        """
        base = self._calculator.prefill_latency(
            num_tokens, batch_size, context_tokens=context_tokens
        )

        allreduce_ms = self._allreduce_overhead_ms(num_tokens, batch_size)
        pipeline_ms = self._pipeline_overhead_ms(num_tokens, batch_size)

        return self._build_result(
            compute_ms=base.compute_ms,
            memory_ms=base.memory_ms,
            base_total_ms=base.total_ms,
            is_compute_bound=base.is_compute_bound,
            allreduce_ms=allreduce_ms,
            pipeline_ms=pipeline_ms,
        )

    def decode_latency(self, context_length: int, batch_size: int = 1) -> TimingResult:
        """Calculate decode latency with optional communication overhead.

        Decode is typically memory-bound as it generates one token at a time,
        requiring loading model weights and the entire KV cache.
        Communication overhead is added when TP > 1 or PP > 1; a single
        token's hidden states are in flight per step.

        Args:
            context_length: Current context length (all tokens so far).
            batch_size: Batch size (default: 1).

        Returns:
            TimingResult with compute, memory, and communication breakdown.
        """
        base = self._calculator.decode_latency(context_length, batch_size)

        allreduce_ms = self._allreduce_overhead_ms(1, batch_size)
        pipeline_ms = self._pipeline_overhead_ms(1, batch_size)

        return self._build_result(
            compute_ms=base.compute_ms,
            memory_ms=base.memory_ms,
            base_total_ms=base.total_ms,
            is_compute_bound=base.is_compute_bound,
            allreduce_ms=allreduce_ms,
            pipeline_ms=pipeline_ms,
        )
