"""Unit tests for kvbench.timing.communication module."""

from __future__ import annotations

from kvbench.timing.communication import (
    calculate_activation_size,
    calculate_allreduce_message_size,
    calculate_allreduce_ms,
    calculate_pipeline_send_recv_ms,
)


class TestCalculateAllReduceMs:
    """Tests for calculate_allreduce_ms function."""

    def test_tp_1_returns_zero(self) -> None:
        """Test that TP=1 returns zero (no communication needed)."""
        result = calculate_allreduce_ms(1000000, tp_size=1)
        assert result == 0.0

    def test_tp_0_returns_zero(self) -> None:
        """Test that TP=0 returns zero (edge case)."""
        result = calculate_allreduce_ms(1000000, tp_size=0)
        assert result == 0.0

    def test_tp_2_positive(self) -> None:
        """Test that TP=2 returns positive value."""
        result = calculate_allreduce_ms(1000000, tp_size=2)
        assert result > 0

    def test_scales_with_message_size(self) -> None:
        """Test that timing scales with message size."""
        small = calculate_allreduce_ms(1000000, tp_size=4)
        large = calculate_allreduce_ms(2000000, tp_size=4)
        assert abs(large / small - 2.0) < 0.01

    def test_scales_with_bandwidth(self) -> None:
        """Test that timing inversely scales with bandwidth."""
        slow = calculate_allreduce_ms(1000000, tp_size=4, nvlink_bandwidth_gb_s=450.0)
        fast = calculate_allreduce_ms(1000000, tp_size=4, nvlink_bandwidth_gb_s=900.0)
        assert abs(slow / fast - 2.0) < 0.01

    def test_ring_algorithm_formula(self) -> None:
        """Test ring AllReduce formula: 2*(N-1)/N * size / bandwidth."""
        # 1MB message, TP=4, 900 GB/s
        # data_per_gpu = 2 * (4-1)/4 * 1e6 = 1.5e6 bytes
        # time = 1.5e6 / (900e6 bytes/ms) = ~0.00167 ms
        result = calculate_allreduce_ms(1000000, tp_size=4, nvlink_bandwidth_gb_s=900.0)
        expected_data = 2 * (4 - 1) / 4 * 1000000
        expected_time = expected_data / (900.0 * 1e6)  # GB/s * 1e6 = bytes/ms
        assert abs(result - expected_time) < 0.0001

    def test_more_gpus_more_data_movement(self) -> None:
        """Test that more GPUs means more data movement per ring."""
        tp4 = calculate_allreduce_ms(1000000, tp_size=4)
        tp8 = calculate_allreduce_ms(1000000, tp_size=8)
        # Ring data: 2*(N-1)/N approaches 2 as N increases
        # TP=4: 1.5x, TP=8: 1.75x
        assert tp8 > tp4

    def test_zero_message_size(self) -> None:
        """Test zero message size returns zero time."""
        result = calculate_allreduce_ms(0, tp_size=4)
        assert result == 0.0

    def test_default_bandwidth(self) -> None:
        """Test that default bandwidth is 900 GB/s (H100 NVLink)."""
        # The default should produce consistent results
        result = calculate_allreduce_ms(1000000, tp_size=2)
        expected_data = 2 * (2 - 1) / 2 * 1000000  # 1e6 bytes
        expected_time = expected_data / (900.0 * 1e6)
        assert abs(result - expected_time) < 0.0001


class TestCalculateAllReduceMessageSize:
    """Tests for calculate_allreduce_message_size helper function."""

    def test_basic_calculation(self) -> None:
        """Test basic message size calculation."""
        # batch=4, hidden=4096, BF16
        size = calculate_allreduce_message_size(4, 4096, 2)
        expected = 4 * 4096 * 2  # batch * hidden * dtype_bytes
        assert size == expected

    def test_scales_with_batch(self) -> None:
        """Test that size scales with batch size."""
        s1 = calculate_allreduce_message_size(1, 4096)
        s2 = calculate_allreduce_message_size(2, 4096)
        assert s2 == 2 * s1

    def test_scales_with_hidden(self) -> None:
        """Test that size scales with hidden size."""
        s1 = calculate_allreduce_message_size(1, 4096)
        s2 = calculate_allreduce_message_size(1, 8192)
        assert s2 == 2 * s1

    def test_fp8_vs_bf16(self) -> None:
        """Test FP8 vs BF16 size difference."""
        fp8 = calculate_allreduce_message_size(1, 4096, dtype_bytes=1)
        bf16 = calculate_allreduce_message_size(1, 4096, dtype_bytes=2)
        assert bf16 == 2 * fp8

    def test_default_dtype_bf16(self) -> None:
        """Test that default dtype is BF16 (2 bytes)."""
        size = calculate_allreduce_message_size(1, 4096)
        expected = 1 * 4096 * 2
        assert size == expected

    def test_fp32_dtype(self) -> None:
        """Test FP32 (4 bytes) message size."""
        size = calculate_allreduce_message_size(1, 4096, dtype_bytes=4)
        expected = 1 * 4096 * 4
        assert size == expected

    def test_zero_batch(self) -> None:
        """Test zero batch size returns zero."""
        size = calculate_allreduce_message_size(0, 4096)
        assert size == 0

    def test_large_hidden_size(self) -> None:
        """Test with large hidden size (typical for 70B model)."""
        # Llama 70B has hidden_size=8192
        size = calculate_allreduce_message_size(8, 8192, 2)
        expected = 8 * 8192 * 2  # 131072 bytes
        assert size == expected


class TestCalculateActivationSize:
    """Tests for calculate_activation_size function."""

    def test_basic_calculation(self) -> None:
        """Test basic activation size calculation."""
        # batch=1, seq=1024, hidden=4096, BF16
        size = calculate_activation_size(1, 1024, 4096, 2)
        expected = 1 * 1024 * 4096 * 2
        assert size == expected

    def test_scales_with_batch(self) -> None:
        """Test that size scales with batch size."""
        s1 = calculate_activation_size(1, 1024, 4096)
        s2 = calculate_activation_size(2, 1024, 4096)
        assert s2 == 2 * s1

    def test_scales_with_sequence(self) -> None:
        """Test that size scales with sequence length."""
        s1 = calculate_activation_size(1, 1024, 4096)
        s2 = calculate_activation_size(1, 2048, 4096)
        assert s2 == 2 * s1

    def test_scales_with_hidden(self) -> None:
        """Test that size scales with hidden size."""
        s1 = calculate_activation_size(1, 1024, 4096)
        s2 = calculate_activation_size(1, 1024, 8192)
        assert s2 == 2 * s1

    def test_fp8_vs_bf16(self) -> None:
        """Test FP8 vs BF16 size difference."""
        fp8 = calculate_activation_size(1, 1024, 4096, dtype_bytes=1)
        bf16 = calculate_activation_size(1, 1024, 4096, dtype_bytes=2)
        assert bf16 == 2 * fp8

    def test_default_dtype_bf16(self) -> None:
        """Test that default dtype is BF16 (2 bytes)."""
        size = calculate_activation_size(1, 1024, 4096)
        expected = 1 * 1024 * 4096 * 2
        assert size == expected

    def test_zero_sequence(self) -> None:
        """Test zero sequence length returns zero."""
        size = calculate_activation_size(1, 0, 4096)
        assert size == 0

    def test_large_prefill(self) -> None:
        """Test activation size for large prefill (4096 tokens)."""
        # Common prefill scenario
        size = calculate_activation_size(4, 4096, 4096, 2)
        expected = 4 * 4096 * 4096 * 2  # 134217728 bytes (~128 MB)
        assert size == expected


class TestCalculatePipelineSendRecvMs:
    """Tests for calculate_pipeline_send_recv_ms function."""

    def test_positive_result(self) -> None:
        """Test that result is always positive."""
        result = calculate_pipeline_send_recv_ms(1000000)
        assert result > 0

    def test_includes_latency_overhead(self) -> None:
        """Test that fixed latency overhead is included."""
        # With zero data, should just be latency overhead
        result = calculate_pipeline_send_recv_ms(0, latency_overhead_us=10.0)
        assert abs(result - 0.01) < 0.0001  # 10us = 0.01ms

    def test_scales_with_size(self) -> None:
        """Test that timing scales with activation size."""
        small = calculate_pipeline_send_recv_ms(1000000, latency_overhead_us=0)
        large = calculate_pipeline_send_recv_ms(2000000, latency_overhead_us=0)
        assert abs(large / small - 2.0) < 0.01

    def test_scales_with_bandwidth(self) -> None:
        """Test that timing inversely scales with bandwidth."""
        slow = calculate_pipeline_send_recv_ms(1000000, bandwidth_gb_s=450.0, latency_overhead_us=0)
        fast = calculate_pipeline_send_recv_ms(1000000, bandwidth_gb_s=900.0, latency_overhead_us=0)
        assert abs(slow / fast - 2.0) < 0.01

    def test_formula(self) -> None:
        """Test formula: latency + size/bandwidth."""
        # 1MB at 900 GB/s with 5us overhead
        # time = 5/1000 + 1e6 / 900e6 = 0.005 + ~0.00111 ms
        result = calculate_pipeline_send_recv_ms(
            1000000, bandwidth_gb_s=900.0, latency_overhead_us=5.0
        )
        transfer_ms = 1000000 / (900.0 * 1e6)
        expected = 0.005 + transfer_ms
        assert abs(result - expected) < 0.0001

    def test_default_bandwidth(self) -> None:
        """Test that default bandwidth is 900 GB/s."""
        result = calculate_pipeline_send_recv_ms(1000000, latency_overhead_us=0)
        expected = 1000000 / (900.0 * 1e6)
        assert abs(result - expected) < 0.0001

    def test_default_latency_overhead(self) -> None:
        """Test that default latency overhead is 5us."""
        result = calculate_pipeline_send_recv_ms(0)
        assert abs(result - 0.005) < 0.0001  # 5us = 0.005ms

    def test_zero_activation_has_latency(self) -> None:
        """Test that zero activation still has latency overhead."""
        result = calculate_pipeline_send_recv_ms(0, latency_overhead_us=5.0)
        assert result > 0

    def test_large_activation_prefill(self) -> None:
        """Test with large prefill activation size."""
        # 128MB activation at 900 GB/s
        size = 128 * 1024 * 1024
        result = calculate_pipeline_send_recv_ms(size, bandwidth_gb_s=900.0, latency_overhead_us=5.0)
        transfer_ms = size / (900.0 * 1e6)
        expected = 0.005 + transfer_ms
        assert abs(result - expected) < 0.0001


class TestCommunicationIntegration:
    """Integration tests for communication functions working together."""

    def test_allreduce_with_message_size_helper(self) -> None:
        """Test AllReduce timing using message size helper."""
        # Calculate message size for typical decode
        msg_size = calculate_allreduce_message_size(1, 4096, dtype_bytes=2)

        # Get AllReduce time
        allreduce_time = calculate_allreduce_ms(msg_size, tp_size=4)

        assert allreduce_time > 0
        assert msg_size == 8192  # 1 * 4096 * 2

    def test_pipeline_with_activation_size_helper(self) -> None:
        """Test pipeline timing using activation size helper."""
        # Calculate activation size for decode (seq=1)
        act_size = calculate_activation_size(1, 1, 4096, dtype_bytes=2)

        # Get pipeline time
        pipeline_time = calculate_pipeline_send_recv_ms(act_size)

        assert pipeline_time > 0
        assert act_size == 8192  # 1 * 1 * 4096 * 2

    def test_typical_decode_communication(self) -> None:
        """Test typical decode step communication overhead."""
        batch_size = 1
        hidden_size = 4096
        tp_size = 4
        pp_size = 2
        num_layers = 32

        # AllReduce per layer (2 per layer: attention + FFN)
        msg_size = calculate_allreduce_message_size(batch_size, hidden_size)
        allreduce_per_op = calculate_allreduce_ms(msg_size, tp_size)
        total_allreduce = allreduce_per_op * num_layers * 2

        # Pipeline between stages (pp_size - 1 boundaries)
        act_size = calculate_activation_size(batch_size, 1, hidden_size)
        pipeline_per_boundary = calculate_pipeline_send_recv_ms(act_size)
        total_pipeline = pipeline_per_boundary * (pp_size - 1)

        total_comm = total_allreduce + total_pipeline

        assert total_comm > 0
        # Verify components are reasonable
        assert total_allreduce > 0  # TP=4 has communication
        assert total_pipeline > 0   # PP=2 has one boundary
