"""
Latency Calculator for KV-Bench.

This module provides latency calculations using the roofline model for
emulating LLM inference without actual GPU hardware.

The roofline model determines whether a workload is compute-bound or
memory-bound based on arithmetic intensity (FLOPs / bytes accessed).

Key concepts:
- Prefill: Compute-bound, processes all input tokens in parallel
- Decode: Memory-bound, generates one token at a time
"""

from __future__ import annotations

from dataclasses import dataclass

from kvbench.core.gpu_profiles import GPUProfile, get_gpu_profile
from kvbench.core.models import ModelProfile, get_model_profile


@dataclass
class LatencyBreakdown:
    """Breakdown of latency into compute and memory components.

    Attributes:
        compute_ms: Time spent on compute operations in milliseconds.
        memory_ms: Time spent on memory operations in milliseconds.
        total_ms: Total latency (max of compute and memory) in milliseconds.
        is_compute_bound: Whether the operation is compute-bound.
        flops: Total floating-point operations performed.
        bytes_accessed: Total bytes accessed from memory.
    """

    compute_ms: float
    memory_ms: float
    total_ms: float
    is_compute_bound: bool
    flops: int = 0
    bytes_accessed: int = 0

    @property
    def bottleneck(self) -> str:
        """Return the bottleneck type as a string."""
        return "compute" if self.is_compute_bound else "memory"

    @property
    def arithmetic_intensity(self) -> float:
        """Calculate the arithmetic intensity (FLOPs/byte)."""
        if self.bytes_accessed == 0:
            return float("inf")
        return self.flops / self.bytes_accessed


class LatencyCalculator:
    """Calculator for LLM inference latency using the roofline model.

    This calculator emulates the latency characteristics of running LLM
    inference on specific GPU hardware without requiring actual GPUs.

    The roofline model determines performance based on:
    - Peak compute throughput (TFLOPS)
    - Peak memory bandwidth (TB/s)
    - Arithmetic intensity of the workload (FLOPs/byte)

    Attributes:
        gpu: The GPU profile being emulated.
        model: The model profile being used.
        tp_size: Tensor parallelism size (number of GPUs).
        efficiency: Hardware efficiency factor (0.0 to 1.0).
    """

    def __init__(
        self,
        gpu: str | GPUProfile,
        model: str | ModelProfile,
        tp_size: int = 1,
        efficiency: float = 0.7,
    ):
        """Initialize the latency calculator.

        Args:
            gpu: GPU profile name or GPUProfile object.
            model: Model profile name or ModelProfile object.
            tp_size: Tensor parallelism size (default: 1).
            efficiency: Hardware efficiency factor (default: 0.7).
        """
        self.gpu = get_gpu_profile(gpu) if isinstance(gpu, str) else gpu
        self.model = get_model_profile(model) if isinstance(model, str) else model
        self.tp_size = tp_size
        self.efficiency = efficiency

    @property
    def effective_compute_tflops(self) -> float:
        """Calculate effective compute throughput with efficiency and TP."""
        return self.gpu.bf16_tflops * self.efficiency * self.tp_size

    @property
    def effective_memory_bandwidth_gb_s(self) -> float:
        """Calculate effective memory bandwidth with efficiency and TP."""
        return self.gpu.hbm_bandwidth_gb_s * self.efficiency * self.tp_size

    def _compute_attention_flops(
        self, seq_len: int, batch_size: int, is_prefill: bool
    ) -> int:
        """Calculate FLOPs for attention computation.

        For prefill (full attention):
            - Q @ K^T: 2 * batch * heads * seq * seq * head_dim
            - Softmax: ~5 * batch * heads * seq * seq (approximate)
            - Attn @ V: 2 * batch * heads * seq * seq * head_dim

        For decode (incremental attention):
            - Q @ K^T: 2 * batch * heads * 1 * context * head_dim
            - Softmax: ~5 * batch * heads * context
            - Attn @ V: 2 * batch * heads * 1 * context * head_dim

        Args:
            seq_len: Sequence length (input tokens for prefill, context for decode).
            batch_size: Batch size.
            is_prefill: Whether this is prefill (True) or decode (False).

        Returns:
            Total FLOPs for attention.
        """
        num_heads = self.model.num_heads or 32
        head_dim = self.model.head_dim
        layers = self.model.layers

        if is_prefill:
            # Full attention: O(seq^2)
            # QK^T matmul: 2 * batch * heads * seq * seq * head_dim
            qk_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
            # Softmax: approximately 5 ops per element
            softmax_flops = 5 * batch_size * num_heads * seq_len * seq_len
            # Attention @ V: 2 * batch * heads * seq * seq * head_dim
            av_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
            per_layer = qk_flops + softmax_flops + av_flops
        else:
            # Incremental attention: O(context)
            # QK^T for single query: 2 * batch * heads * 1 * context * head_dim
            qk_flops = 2 * batch_size * num_heads * seq_len * head_dim
            # Softmax: 5 * batch * heads * context
            softmax_flops = 5 * batch_size * num_heads * seq_len
            # Attention @ V: 2 * batch * heads * 1 * context * head_dim
            av_flops = 2 * batch_size * num_heads * seq_len * head_dim
            per_layer = qk_flops + softmax_flops + av_flops

        return per_layer * layers

    def _compute_ffn_flops(self, num_tokens: int, batch_size: int) -> int:
        """Calculate FLOPs for FFN (feed-forward network) computation.

        FFN consists of:
        - Up projection: hidden -> intermediate (2 * hidden * intermediate)
        - Gate projection: hidden -> intermediate (2 * hidden * intermediate)
        - Down projection: intermediate -> hidden (2 * intermediate * hidden)

        For SwiGLU: gate * up, then down projection.

        Args:
            num_tokens: Number of tokens being processed.
            batch_size: Batch size.

        Returns:
            Total FLOPs for FFN.
        """
        hidden = self.model.hidden
        intermediate = self.model.intermediate
        layers = self.model.layers
        total_tokens = num_tokens * batch_size

        # Up + Gate + Down projections: each is 2 * tokens * in_dim * out_dim
        # SwiGLU adds element-wise multiply: tokens * intermediate
        up_flops = 2 * total_tokens * hidden * intermediate
        gate_flops = 2 * total_tokens * hidden * intermediate
        silu_mul_flops = total_tokens * intermediate  # SiLU activation + multiply
        down_flops = 2 * total_tokens * intermediate * hidden

        per_layer = up_flops + gate_flops + silu_mul_flops + down_flops
        return per_layer * layers

    def _compute_projection_flops(self, num_tokens: int, batch_size: int) -> int:
        """Calculate FLOPs for QKV and output projections.

        Args:
            num_tokens: Number of tokens being processed.
            batch_size: Batch size.

        Returns:
            Total FLOPs for projections.
        """
        hidden = self.model.hidden
        num_heads = self.model.num_heads or 32
        kv_heads = self.model.kv_heads
        head_dim = self.model.head_dim
        layers = self.model.layers
        total_tokens = num_tokens * batch_size

        # Q projection: hidden -> num_heads * head_dim
        q_flops = 2 * total_tokens * hidden * (num_heads * head_dim)
        # K projection: hidden -> kv_heads * head_dim
        k_flops = 2 * total_tokens * hidden * (kv_heads * head_dim)
        # V projection: hidden -> kv_heads * head_dim
        v_flops = 2 * total_tokens * hidden * (kv_heads * head_dim)
        # Output projection: num_heads * head_dim -> hidden
        o_flops = 2 * total_tokens * (num_heads * head_dim) * hidden

        per_layer = q_flops + k_flops + v_flops + o_flops
        return per_layer * layers

    def _compute_bytes_for_prefill(self, num_tokens: int, batch_size: int) -> int:
        """Calculate bytes accessed for prefill.

        During prefill, we read:
        - Model weights (once per forward pass)
        - Input activations

        And write:
        - KV cache for all tokens
        - Output activations

        Args:
            num_tokens: Number of input tokens.
            batch_size: Batch size.

        Returns:
            Total bytes accessed.
        """
        # Model weights (read once, divided by TP)
        model_bytes = int(self.model.estimated_params_billions * 1e9 * self.model.bytes_per_element)
        model_bytes = model_bytes // self.tp_size

        # KV cache written
        kv_bytes = self.model.kv_cache_size_bytes(num_tokens) * batch_size

        # Activations (rough estimate: hidden * tokens * 2 bytes per layer)
        activation_bytes = self.model.hidden * num_tokens * batch_size * 2 * self.model.layers

        return model_bytes + kv_bytes + activation_bytes

    def _compute_bytes_for_decode(self, context_length: int, batch_size: int) -> int:
        """Calculate bytes accessed for decode (single token generation).

        During decode, we read:
        - Model weights (once per token)
        - KV cache for entire context

        And write:
        - New KV entry for the generated token

        Args:
            context_length: Current context length.
            batch_size: Batch size.

        Returns:
            Total bytes accessed.
        """
        # Model weights (read once per token, divided by TP)
        model_bytes = int(self.model.estimated_params_billions * 1e9 * self.model.bytes_per_element)
        model_bytes = model_bytes // self.tp_size

        # KV cache read (entire context)
        kv_read_bytes = self.model.kv_cache_size_bytes(context_length) * batch_size

        # KV cache write (one token)
        kv_write_bytes = self.model.total_kv_cache_bytes_per_token * batch_size

        return model_bytes + kv_read_bytes + kv_write_bytes

    def prefill_latency(self, num_tokens: int, batch_size: int = 1) -> LatencyBreakdown:
        """Calculate prefill latency for processing input tokens.

        Prefill is typically compute-bound as it processes all input tokens
        in parallel with quadratic attention complexity.

        Args:
            num_tokens: Number of input tokens to process.
            batch_size: Batch size (default: 1).

        Returns:
            LatencyBreakdown with timing information.
        """
        # Calculate FLOPs
        attention_flops = self._compute_attention_flops(num_tokens, batch_size, is_prefill=True)
        ffn_flops = self._compute_ffn_flops(num_tokens, batch_size)
        projection_flops = self._compute_projection_flops(num_tokens, batch_size)
        total_flops = attention_flops + ffn_flops + projection_flops

        # Calculate bytes accessed
        total_bytes = self._compute_bytes_for_prefill(num_tokens, batch_size)

        # Compute time: FLOPs / (TFLOPS * 1e12) * 1000 for ms
        compute_ms = total_flops / (self.effective_compute_tflops * 1e12) * 1000

        # Memory time: bytes / (GB/s * 1e9) * 1000 for ms
        memory_ms = total_bytes / (self.effective_memory_bandwidth_gb_s * 1e9) * 1000

        # Total is max of compute and memory (roofline model)
        total_ms = max(compute_ms, memory_ms)
        is_compute_bound = compute_ms >= memory_ms

        return LatencyBreakdown(
            compute_ms=compute_ms,
            memory_ms=memory_ms,
            total_ms=total_ms,
            is_compute_bound=is_compute_bound,
            flops=total_flops,
            bytes_accessed=total_bytes,
        )

    def decode_latency(self, context_length: int, batch_size: int = 1) -> LatencyBreakdown:
        """Calculate decode latency for generating a single token.

        Decode is typically memory-bound as it generates one token at a time,
        requiring loading model weights and the entire KV cache.

        Args:
            context_length: Current context length (including all previous tokens).
            batch_size: Batch size (default: 1).

        Returns:
            LatencyBreakdown with timing information.
        """
        # Calculate FLOPs (for single token generation)
        attention_flops = self._compute_attention_flops(context_length, batch_size, is_prefill=False)
        ffn_flops = self._compute_ffn_flops(1, batch_size)  # 1 token
        projection_flops = self._compute_projection_flops(1, batch_size)  # 1 token
        total_flops = attention_flops + ffn_flops + projection_flops

        # Calculate bytes accessed
        total_bytes = self._compute_bytes_for_decode(context_length, batch_size)

        # Compute time
        compute_ms = total_flops / (self.effective_compute_tflops * 1e12) * 1000

        # Memory time
        memory_ms = total_bytes / (self.effective_memory_bandwidth_gb_s * 1e9) * 1000

        # Total is max of compute and memory
        total_ms = max(compute_ms, memory_ms)
        is_compute_bound = compute_ms >= memory_ms

        return LatencyBreakdown(
            compute_ms=compute_ms,
            memory_ms=memory_ms,
            total_ms=total_ms,
            is_compute_bound=is_compute_bound,
            flops=total_flops,
            bytes_accessed=total_bytes,
        )

    def kv_transfer_latency(
        self, num_tokens: int, bandwidth_gb_s: float = 10.0
    ) -> float:
        """Calculate KV cache transfer latency for disaggregated inference.

        In disaggregated prefill/decode architectures, KV cache must be
        transferred between prefill and decode servers.

        Args:
            num_tokens: Number of tokens worth of KV cache to transfer.
            bandwidth_gb_s: Network bandwidth in GB/s (default: 10.0 for 100GbE).

        Returns:
            Transfer latency in milliseconds.
        """
        kv_bytes = self.model.kv_cache_size_bytes(num_tokens)
        transfer_ms = kv_bytes / (bandwidth_gb_s * 1e9) * 1000
        return transfer_ms

    def estimate_ttft(
        self,
        input_tokens: int,
        cache_hit_tokens: int = 0,
        batch_size: int = 1,
    ) -> float:
        """Estimate time to first token (TTFT) with optional cache hits.

        TTFT consists of:
        - Prefill time for cache-miss tokens
        - First decode token generation

        Args:
            input_tokens: Total number of input tokens.
            cache_hit_tokens: Number of tokens with KV cache hits (default: 0).
            batch_size: Batch size (default: 1).

        Returns:
            Estimated TTFT in milliseconds.
        """
        # Tokens that need prefill
        prefill_tokens = input_tokens - cache_hit_tokens
        if prefill_tokens < 0:
            prefill_tokens = 0

        # Prefill latency for miss tokens
        if prefill_tokens > 0:
            prefill = self.prefill_latency(prefill_tokens, batch_size)
            prefill_ms = prefill.total_ms
        else:
            prefill_ms = 0.0

        # First decode latency (context = all input tokens)
        decode = self.decode_latency(input_tokens, batch_size)

        return prefill_ms + decode.total_ms

    def estimate_generation_time(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
        batch_size: int = 1,
    ) -> dict:
        """Estimate total generation time breakdown.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of tokens to generate.
            cache_hit_tokens: Number of tokens with KV cache hits (default: 0).
            batch_size: Batch size (default: 1).

        Returns:
            Dictionary with timing breakdown.
        """
        # Prefill
        prefill_tokens = max(0, input_tokens - cache_hit_tokens)
        if prefill_tokens > 0:
            prefill = self.prefill_latency(prefill_tokens, batch_size)
            prefill_ms = prefill.total_ms
        else:
            prefill_ms = 0.0

        # Decode: sum of latencies for each token
        decode_ms = 0.0
        for i in range(output_tokens):
            context_len = input_tokens + i
            decode = self.decode_latency(context_len, batch_size)
            decode_ms += decode.total_ms

        total_ms = prefill_ms + decode_ms
        tokens_per_second = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0

        return {
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "ttft_ms": self.estimate_ttft(input_tokens, cache_hit_tokens, batch_size),
            "tokens_per_second": tokens_per_second,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit_tokens": cache_hit_tokens,
        }
