"""
Communication Timing Module for KV-Bench.

This module provides timing calculations for inter-GPU communication operations
that occur during tensor parallel (TP) and pipeline parallel (PP) inference.

Communication overhead can add 20-30% to end-to-end latency, so accurate
modeling is essential for realistic latency emulation.

Supported operations:
- AllReduce for tensor parallelism (after attention and FFN layers)
- Point-to-point send/recv for pipeline parallelism (stage boundaries)
"""

from __future__ import annotations


def calculate_allreduce_message_size(
    batch_size: int,
    hidden_size: int,
    dtype_bytes: int = 2,  # BF16
) -> int:
    """Calculate message size for AllReduce of hidden states.

    AllReduce happens after attention and FFN for each layer in tensor
    parallelism. The message size is the output tensor from each layer.

    For a typical transformer layer:
    - After attention: [batch_size, seq_len, hidden_size]
    - After FFN: [batch_size, seq_len, hidden_size]

    This function calculates per-token AllReduce size (for decode phase
    where seq_len=1 typically).

    Args:
        batch_size: Batch size.
        hidden_size: Model hidden dimension.
        dtype_bytes: Bytes per element (default: 2 for BF16).

    Returns:
        Message size in bytes.

    Example:
        >>> calculate_allreduce_message_size(1, 4096, 2)
        8192
    """
    return batch_size * hidden_size * dtype_bytes


def calculate_allreduce_ms(
    message_size_bytes: int,
    tp_size: int,
    nvlink_bandwidth_gb_s: float = 900.0,  # H100 NVLink default
) -> float:
    """Calculate AllReduce latency for tensor parallelism.

    Uses the ring AllReduce algorithm formula:
    - Data moved per GPU: 2 * (N-1) / N * message_size bytes
    - Time = data_moved / bandwidth

    The ring AllReduce algorithm requires 2 * (N-1) communication steps:
    - N-1 reduce-scatter steps (each GPU sends (N-1)/N of data)
    - N-1 all-gather steps (each GPU sends (N-1)/N of data)

    Total data moved per GPU = 2 * (N-1) / N * message_size

    Args:
        message_size_bytes: Size of tensor to AllReduce in bytes.
        tp_size: Tensor parallelism size (number of GPUs).
        nvlink_bandwidth_gb_s: NVLink bandwidth in GB/s (default: 900 for H100).

    Returns:
        AllReduce latency in milliseconds. Returns 0.0 if tp_size <= 1.

    Example:
        >>> # 1MB AllReduce across 2 GPUs at 900 GB/s
        >>> calculate_allreduce_ms(1_000_000, tp_size=2, nvlink_bandwidth_gb_s=900.0)
        0.0011111111111111111
    """
    # No communication needed for single GPU
    if tp_size <= 1:
        return 0.0

    # Ring AllReduce data per GPU: 2 * (N-1) / N * message_size
    data_per_gpu_bytes = 2 * (tp_size - 1) / tp_size * message_size_bytes

    # Convert bandwidth to bytes/ms: GB/s * 1e9 bytes/GB / 1000 ms/s = 1e6 bytes/ms
    bandwidth_bytes_per_ms = nvlink_bandwidth_gb_s * 1e6

    # Time in milliseconds
    time_ms = data_per_gpu_bytes / bandwidth_bytes_per_ms

    return time_ms


def calculate_activation_size(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    dtype_bytes: int = 2,  # BF16
) -> int:
    """Calculate activation tensor size for pipeline boundaries.

    At pipeline stage boundaries, activations are sent/received.
    Shape: [batch_size, sequence_length, hidden_size]

    This is used to determine the data transfer size for pipeline
    parallel point-to-point communication.

    Args:
        batch_size: Batch size.
        sequence_length: Sequence length.
        hidden_size: Model hidden dimension.
        dtype_bytes: Bytes per element (default: 2 for BF16).

    Returns:
        Activation size in bytes.

    Example:
        >>> calculate_activation_size(1, 1024, 4096, 2)
        8388608
    """
    return batch_size * sequence_length * hidden_size * dtype_bytes


def calculate_pipeline_send_recv_ms(
    activation_size_bytes: int,
    bandwidth_gb_s: float = 900.0,  # NVLink or InfiniBand
    latency_overhead_us: float = 5.0,  # Fixed latency per message
) -> float:
    """Calculate point-to-point send/recv latency for pipeline parallelism.

    Pipeline parallel requires sending activations between stages.
    Time = fixed_latency + data_size / bandwidth

    The fixed latency overhead accounts for:
    - CUDA kernel launch overhead
    - Network protocol overhead
    - Memory copy setup

    Args:
        activation_size_bytes: Size of activation tensor in bytes.
        bandwidth_gb_s: Network bandwidth in GB/s (default: 900 for NVLink).
        latency_overhead_us: Fixed latency overhead in microseconds (default: 5).

    Returns:
        Send/recv latency in milliseconds.

    Example:
        >>> # 8MB activation at 900 GB/s with 5us overhead
        >>> calculate_pipeline_send_recv_ms(8_000_000, bandwidth_gb_s=900.0, latency_overhead_us=5.0)
        0.013888888888888888
    """
    # Convert latency overhead from microseconds to milliseconds
    latency_overhead_ms = latency_overhead_us / 1000.0

    # Transfer time: bytes / (GB/s * 1e9 bytes/GB) * 1000 ms/s
    # Simplifies to: bytes / (GB/s * 1e6 bytes/ms)
    transfer_ms = activation_size_bytes / (bandwidth_gb_s * 1e6)

    return latency_overhead_ms + transfer_ms
