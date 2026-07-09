"""Unit tests for kvbench.timing.roofline module."""

from __future__ import annotations

from kvbench.timing import RooflineTimingStrategy, TimingResult


class TestRooflineTimingStrategy:
    """Tests for RooflineTimingStrategy."""

    def test_basic_creation(self) -> None:
        """Test basic creation with string profiles."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        assert strategy.calculator is not None

    def test_with_tp_size(self) -> None:
        """Test creation with tensor parallelism."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", tp_size=4
        )
        assert strategy.tp_size == 4
        assert strategy.calculator.tp_size == 4

    def test_with_pp_size(self) -> None:
        """Test creation with pipeline parallelism."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", pp_size=4
        )
        assert strategy.pp_size == 4

    def test_with_efficiency(self) -> None:
        """Test creation with custom efficiency."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", efficiency=0.8
        )
        assert strategy.calculator.efficiency == 0.8

    def test_default_communication_flags(self) -> None:
        """Test default communication flags are True."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        # Default is to include communication
        assert strategy.include_tp_communication is True
        assert strategy.include_pp_communication is True

    def test_disable_communication(self) -> None:
        """Test that communication can be disabled."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            include_tp_communication=False,
            include_pp_communication=False,
        )
        assert strategy.include_tp_communication is False
        assert strategy.include_pp_communication is False


class TestRooflinePrefillLatency:
    """Tests for RooflineTimingStrategy.prefill_latency."""

    def test_returns_timing_result(self) -> None:
        """Test that prefill returns TimingResult."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.prefill_latency(1000)
        assert isinstance(result, TimingResult)

    def test_positive_latency(self) -> None:
        """Test that latency is positive."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.prefill_latency(1000)
        assert result.total_ms > 0

    def test_increases_with_tokens(self) -> None:
        """Test that latency increases with tokens."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        small = strategy.prefill_latency(100)
        large = strategy.prefill_latency(1000)
        assert large.total_ms > small.total_ms

    def test_typically_compute_bound(self) -> None:
        """Test that prefill is typically compute-bound for long sequences."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.prefill_latency(4096)
        assert result.is_compute_bound is True

    def test_breakdown_has_components(self) -> None:
        """Test that breakdown contains timing components."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.prefill_latency(1000)
        # Should have compute and memory in breakdown
        assert "compute_ms" in result.breakdown
        assert "memory_ms" in result.breakdown

    def test_batch_size_increases_latency(self) -> None:
        """Test that batch size increases latency."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        single = strategy.prefill_latency(1000, batch_size=1)
        double = strategy.prefill_latency(1000, batch_size=2)
        assert double.total_ms > single.total_ms


class TestRooflineDecodeLatency:
    """Tests for RooflineTimingStrategy.decode_latency."""

    def test_returns_timing_result(self) -> None:
        """Test that decode returns TimingResult."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.decode_latency(1000)
        assert isinstance(result, TimingResult)

    def test_positive_latency(self) -> None:
        """Test that latency is positive."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.decode_latency(1000)
        assert result.total_ms > 0

    def test_typically_memory_bound(self) -> None:
        """Test that decode is typically memory-bound."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.decode_latency(1000)
        assert result.is_compute_bound is False

    def test_breakdown_has_components(self) -> None:
        """Test that breakdown contains timing components."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.decode_latency(1000)
        # Should have compute and memory in breakdown
        assert "compute_ms" in result.breakdown
        assert "memory_ms" in result.breakdown

    def test_context_length_affects_latency(self) -> None:
        """Test that context length affects decode latency."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        short = strategy.decode_latency(100)
        long = strategy.decode_latency(1000)
        # Longer context = more KV cache to read
        assert long.total_ms > short.total_ms


class TestRooflineTPCommunication:
    """Tests for TP communication overhead in RooflineTimingStrategy."""

    def test_tp_communication_adds_overhead(self) -> None:
        """Test that TP communication adds overhead when enabled."""
        no_comm = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            include_tp_communication=False,
        )
        with_comm = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            include_tp_communication=True,
        )
        r_no = no_comm.prefill_latency(1000)
        r_with = with_comm.prefill_latency(1000)
        # With communication should be >= without
        assert r_with.total_ms >= r_no.total_ms

    def test_tp1_no_communication(self) -> None:
        """Test that TP=1 has no communication overhead."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=1,
            include_tp_communication=True,
        )
        result = strategy.prefill_latency(1000)
        # AllReduce should be 0 or not present
        allreduce = result.breakdown.get("allreduce_ms", 0)
        assert allreduce == 0

    def test_tp_communication_in_breakdown(self) -> None:
        """Test that TP communication appears in breakdown when enabled."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            include_tp_communication=True,
        )
        result = strategy.prefill_latency(1000)
        # Should have allreduce_ms in breakdown
        assert "allreduce_ms" in result.breakdown
        assert result.breakdown["allreduce_ms"] > 0

    def test_tp_communication_in_decode(self) -> None:
        """Test that TP communication appears in decode timing."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            include_tp_communication=True,
        )
        result = strategy.decode_latency(1000)
        # Should have allreduce_ms in breakdown
        assert "allreduce_ms" in result.breakdown
        assert result.breakdown["allreduce_ms"] > 0


class TestRooflinePPCommunication:
    """Tests for PP communication overhead in RooflineTimingStrategy."""

    def test_pp_communication_adds_overhead(self) -> None:
        """Test that PP communication adds overhead when enabled."""
        no_comm = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            pp_size=4,
            include_pp_communication=False,
        )
        with_comm = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            pp_size=4,
            include_pp_communication=True,
        )
        r_no = no_comm.prefill_latency(1000)
        r_with = with_comm.prefill_latency(1000)
        # With communication should be >= without
        assert r_with.total_ms >= r_no.total_ms

    def test_pp1_no_communication(self) -> None:
        """Test that PP=1 has no communication overhead."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            pp_size=1,
            include_pp_communication=True,
        )
        result = strategy.prefill_latency(1000)
        # Pipeline should be 0 or not present
        pipeline = result.breakdown.get("pipeline_ms", 0)
        assert pipeline == 0

    def test_pp_communication_in_breakdown(self) -> None:
        """Test that PP communication appears in breakdown when enabled."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            pp_size=4,
            include_pp_communication=True,
        )
        result = strategy.prefill_latency(1000)
        # Should have pipeline_ms in breakdown
        assert "pipeline_ms" in result.breakdown
        assert result.breakdown["pipeline_ms"] > 0

    def test_pp_communication_in_decode(self) -> None:
        """Test that PP communication appears in decode timing."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            pp_size=4,
            include_pp_communication=True,
        )
        result = strategy.decode_latency(1000)
        # Should have pipeline_ms in breakdown
        assert "pipeline_ms" in result.breakdown
        assert result.breakdown["pipeline_ms"] > 0


class TestRooflineDifferentGPUs:
    """Tests for different GPU profiles."""

    def test_h100_faster_than_a100(self) -> None:
        """Test that H100 is faster than A100."""
        h100 = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        a100 = RooflineTimingStrategy("A100_SXM", "llama-3.1-8b")
        r_h100 = h100.prefill_latency(1000)
        r_a100 = a100.prefill_latency(1000)
        assert r_h100.total_ms < r_a100.total_ms

    def test_h100_faster_decode(self) -> None:
        """Test that H100 has faster decode than A100."""
        h100 = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        a100 = RooflineTimingStrategy("A100_SXM", "llama-3.1-8b")
        r_h100 = h100.decode_latency(1000)
        r_a100 = a100.decode_latency(1000)
        assert r_h100.total_ms < r_a100.total_ms


class TestRooflineDifferentModels:
    """Tests for different model profiles."""

    def test_larger_model_slower(self) -> None:
        """Test that larger models are slower."""
        small = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        large = RooflineTimingStrategy("H100_SXM", "llama-3.1-70b")
        r_small = small.prefill_latency(1000)
        r_large = large.prefill_latency(1000)
        assert r_large.total_ms > r_small.total_ms

    def test_larger_model_slower_decode(self) -> None:
        """Test that larger models have slower decode."""
        small = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        large = RooflineTimingStrategy("H100_SXM", "llama-3.1-70b")
        r_small = small.decode_latency(1000)
        r_large = large.decode_latency(1000)
        assert r_large.total_ms > r_small.total_ms


class TestRooflineTPScaling:
    """Tests for tensor parallelism scaling."""

    def test_tp_reduces_latency(self) -> None:
        """Test that TP reduces latency."""
        tp1 = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=1,
            include_tp_communication=False,
        )
        tp4 = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            include_tp_communication=False,
        )
        r_tp1 = tp1.prefill_latency(1000)
        r_tp4 = tp4.prefill_latency(1000)
        # TP=4 should be faster (more compute/memory bandwidth)
        assert r_tp4.total_ms < r_tp1.total_ms

    def test_tp_scaling_with_communication(self) -> None:
        """Test TP scaling including communication overhead."""
        tp1 = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=1,
            include_tp_communication=True,
        )
        tp4 = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            include_tp_communication=True,
        )
        r_tp1 = tp1.prefill_latency(1000)
        r_tp4 = tp4.prefill_latency(1000)
        # TP=4 should still be faster even with communication
        assert r_tp4.total_ms < r_tp1.total_ms


class TestRooflineEdgeCases:
    """Tests for edge cases."""

    def test_small_sequence(self) -> None:
        """Test with small sequence (1 token)."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.prefill_latency(1)
        assert result.total_ms > 0

    def test_large_sequence(self) -> None:
        """Test with large sequence (8192 tokens)."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        result = strategy.prefill_latency(8192)
        assert result.total_ms > 0
        assert result.is_compute_bound is True  # Long sequence should be compute-bound

    def test_combined_tp_pp(self) -> None:
        """Test with both TP and PP enabled."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            tp_size=4,
            pp_size=2,
            include_tp_communication=True,
            include_pp_communication=True,
        )
        result = strategy.prefill_latency(1000)
        # Should have both allreduce and pipeline in breakdown
        assert "allreduce_ms" in result.breakdown
        assert "pipeline_ms" in result.breakdown
        assert result.breakdown["allreduce_ms"] > 0
        assert result.breakdown["pipeline_ms"] > 0


class TestRooflineContextTokens:
    """Tests for cached-prefix (context_tokens) handling in prefill."""

    def test_context_increases_latency(self) -> None:
        """Prefilling the same tokens over a cached prefix costs more than
        over an empty context (the new tokens attend to the prefix)."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        no_context = strategy.prefill_latency(1000)
        with_context = strategy.prefill_latency(1000, context_tokens=4000)
        assert with_context.total_ms > no_context.total_ms

    def test_matches_calculator_context_model(self) -> None:
        """Without communication, prefill matches the calculator's
        context-aware roofline result exactly."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b",
            include_tp_communication=False,
            include_pp_communication=False,
        )
        result = strategy.prefill_latency(500, context_tokens=1500)
        base = strategy.calculator.prefill_latency(500, context_tokens=1500)
        assert result.total_ms == base.total_ms

    def test_context_does_not_change_communication(self) -> None:
        """The cached prefix is not re-communicated: comm overhead depends
        only on the new tokens in flight."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", tp_size=4, pp_size=2
        )
        no_context = strategy.prefill_latency(1000)
        with_context = strategy.prefill_latency(1000, context_tokens=4000)
        assert (
            with_context.breakdown["allreduce_ms"]
            == no_context.breakdown["allreduce_ms"]
        )
        assert (
            with_context.breakdown["pipeline_ms"]
            == no_context.breakdown["pipeline_ms"]
        )


class TestRooflineInterconnectBandwidth:
    """Tests for interconnect bandwidth resolution."""

    def test_defaults_to_gpu_profile_nvlink(self) -> None:
        """When not overridden, bandwidth comes from the GPU profile."""
        from kvbench.core.gpu_profiles import get_gpu_profile

        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b")
        profile = get_gpu_profile("H100_SXM")
        expected = profile.nvlink_bandwidth_gb_s or 900.0
        assert strategy.interconnect_bandwidth_gb_s == expected

    def test_explicit_override(self) -> None:
        """An explicit bandwidth overrides the GPU profile."""
        strategy = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", nvlink_bandwidth_gb_s=450.0
        )
        assert strategy.interconnect_bandwidth_gb_s == 450.0

    def test_lower_bandwidth_more_overhead(self) -> None:
        """Slower interconnect means more communication overhead."""
        fast = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", tp_size=4, nvlink_bandwidth_gb_s=900.0
        )
        slow = RooflineTimingStrategy(
            "H100_SXM", "llama-3.1-8b", tp_size=4, nvlink_bandwidth_gb_s=450.0
        )
        r_fast = fast.prefill_latency(1000)
        r_slow = slow.prefill_latency(1000)
        assert r_slow.breakdown["allreduce_ms"] > r_fast.breakdown["allreduce_ms"]


class TestRooflinePrefillCommScalesWithTokens:
    """Prefill communication scales with the number of tokens in flight."""

    def test_allreduce_scales_with_tokens(self) -> None:
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b", tp_size=4)
        small = strategy.prefill_latency(500)
        large = strategy.prefill_latency(1000)
        ratio = large.breakdown["allreduce_ms"] / small.breakdown["allreduce_ms"]
        assert abs(ratio - 2.0) < 0.01

    def test_decode_uses_single_token_message(self) -> None:
        """Decode AllReduce is per-token and independent of context length."""
        strategy = RooflineTimingStrategy("H100_SXM", "llama-3.1-8b", tp_size=4)
        short = strategy.decode_latency(100)
        long = strategy.decode_latency(10000)
        assert short.breakdown["allreduce_ms"] == long.breakdown["allreduce_ms"]
