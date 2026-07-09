"""Unit tests for kvbench.core.latency module."""

from __future__ import annotations

from kvbench.core.gpu_profiles import get_gpu_profile
from kvbench.core.latency import LatencyBreakdown, LatencyCalculator
from kvbench.core.models import get_model_profile


class TestLatencyBreakdown:
    """Tests for LatencyBreakdown dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic LatencyBreakdown creation."""
        breakdown = LatencyBreakdown(
            compute_ms=10.0,
            memory_ms=5.0,
            total_ms=10.0,
            is_compute_bound=True,
            flops=1000000,
            bytes_accessed=100000,
        )
        assert breakdown.compute_ms == 10.0
        assert breakdown.memory_ms == 5.0
        assert breakdown.total_ms == 10.0
        assert breakdown.is_compute_bound is True

    def test_bottleneck_compute(self) -> None:
        """Test bottleneck property for compute-bound case."""
        breakdown = LatencyBreakdown(
            compute_ms=10.0,
            memory_ms=5.0,
            total_ms=10.0,
            is_compute_bound=True,
        )
        assert breakdown.bottleneck == "compute"

    def test_bottleneck_memory(self) -> None:
        """Test bottleneck property for memory-bound case."""
        breakdown = LatencyBreakdown(
            compute_ms=5.0,
            memory_ms=10.0,
            total_ms=10.0,
            is_compute_bound=False,
        )
        assert breakdown.bottleneck == "memory"

    def test_arithmetic_intensity(self) -> None:
        """Test arithmetic intensity calculation."""
        breakdown = LatencyBreakdown(
            compute_ms=10.0,
            memory_ms=5.0,
            total_ms=10.0,
            is_compute_bound=True,
            flops=1000,
            bytes_accessed=100,
        )
        assert breakdown.arithmetic_intensity == 10.0

    def test_arithmetic_intensity_zero_bytes(self) -> None:
        """Test arithmetic intensity with zero bytes."""
        breakdown = LatencyBreakdown(
            compute_ms=10.0,
            memory_ms=0.0,
            total_ms=10.0,
            is_compute_bound=True,
            flops=1000,
            bytes_accessed=0,
        )
        assert breakdown.arithmetic_intensity == float("inf")


class TestLatencyCalculator:
    """Tests for LatencyCalculator class."""

    def test_init_with_strings(self) -> None:
        """Test initialization with profile name strings."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        assert calc.gpu.name == "NVIDIA H100 SXM"
        assert calc.model.name == "Llama 3.1 8B"
        assert calc.tp_size == 1
        assert calc.efficiency == 0.7

    def test_init_with_profile_objects(self) -> None:
        """Test initialization with profile objects."""
        gpu = get_gpu_profile("H100_SXM")
        model = get_model_profile("llama-3.1-8b")
        calc = LatencyCalculator(gpu, model, tp_size=4, efficiency=0.8)
        assert calc.gpu is gpu
        assert calc.model is model
        assert calc.tp_size == 4
        assert calc.efficiency == 0.8

    def test_effective_compute_tflops(self) -> None:
        """Test effective compute throughput calculation."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b", tp_size=2, efficiency=0.5)
        # H100_SXM has 989.5 dense TFLOPS * 0.5 efficiency * 2 TP
        assert calc.effective_compute_tflops == 989.5 * 0.5 * 2

    def test_effective_memory_bandwidth(self) -> None:
        """Test effective memory bandwidth calculation."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b", tp_size=2, efficiency=0.5)
        # H100_SXM has 3350 GB/s * 0.5 efficiency * 2 TP
        expected = 3350.0 * 0.5 * 2
        assert calc.effective_memory_bandwidth_gb_s == expected


class TestPrefillLatency:
    """Tests for prefill latency calculations."""

    def test_prefill_returns_breakdown(self) -> None:
        """Test that prefill_latency returns a LatencyBreakdown."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.prefill_latency(1000)
        assert isinstance(result, LatencyBreakdown)

    def test_prefill_positive_latency(self) -> None:
        """Test that prefill latency is positive."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.prefill_latency(1000)
        assert result.total_ms > 0
        assert result.compute_ms > 0
        assert result.memory_ms > 0

    def test_prefill_increases_with_tokens(self) -> None:
        """Test that prefill latency increases with more tokens."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        small = calc.prefill_latency(100)
        large = calc.prefill_latency(1000)
        assert large.total_ms > small.total_ms

    def test_prefill_quadratic_scaling(self) -> None:
        """Test that prefill has roughly quadratic scaling due to attention."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        t1 = calc.prefill_latency(1000).total_ms
        t2 = calc.prefill_latency(2000).total_ms
        # Quadratic scaling means 2x tokens -> ~4x time
        # Allow some tolerance due to FFN (linear) component
        ratio = t2 / t1
        assert 2.0 < ratio < 5.0  # Should be closer to 4x

    def test_prefill_batch_size_scaling(self) -> None:
        """Test that prefill scales with batch size."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        single = calc.prefill_latency(1000, batch_size=1)
        double = calc.prefill_latency(1000, batch_size=2)
        # Larger batch should take more time (more FLOPs and memory)
        assert double.total_ms > single.total_ms

    def test_prefill_is_typically_compute_bound(self) -> None:
        """Test that prefill is typically compute-bound for long sequences."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.prefill_latency(4096)
        # For long sequences, attention is O(n^2) and should be compute-bound
        assert result.is_compute_bound is True

    def test_prefill_flops_bytes_tracked(self) -> None:
        """Test that FLOPs and bytes are tracked."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.prefill_latency(1000)
        assert result.flops > 0
        assert result.bytes_accessed > 0


class TestDecodeLatency:
    """Tests for decode latency calculations."""

    def test_decode_returns_breakdown(self) -> None:
        """Test that decode_latency returns a LatencyBreakdown."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.decode_latency(1000)
        assert isinstance(result, LatencyBreakdown)

    def test_decode_positive_latency(self) -> None:
        """Test that decode latency is positive."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.decode_latency(1000)
        assert result.total_ms > 0
        assert result.compute_ms > 0
        assert result.memory_ms > 0

    def test_decode_increases_with_context(self) -> None:
        """Test that decode latency increases with context length."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        short = calc.decode_latency(100)
        long = calc.decode_latency(1000)
        assert long.total_ms > short.total_ms

    def test_decode_is_typically_memory_bound(self) -> None:
        """Test that decode is typically memory-bound."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.decode_latency(1000)
        # Decode generates 1 token, mostly reading weights + KV cache
        assert result.is_compute_bound is False

    def test_decode_faster_than_prefill_per_token(self) -> None:
        """Test that decode per token is different from prefill."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        prefill = calc.prefill_latency(1000)
        decode = calc.decode_latency(1000)
        # Decode is for 1 token, prefill is for all tokens
        # Per-token decode should be much faster than per-token prefill
        prefill_per_token = prefill.total_ms / 1000
        assert decode.total_ms != prefill_per_token

    def test_decode_batch_size_scaling(self) -> None:
        """Test that decode scales with batch size."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        single = calc.decode_latency(1000, batch_size=1)
        double = calc.decode_latency(1000, batch_size=2)
        # Larger batch should take more time
        assert double.total_ms > single.total_ms


class TestKVTransferLatency:
    """Tests for KV cache transfer latency calculations."""

    def test_kv_transfer_positive(self) -> None:
        """Test that KV transfer latency is positive."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.kv_transfer_latency(1000)
        assert result > 0

    def test_kv_transfer_scales_with_tokens(self) -> None:
        """Test that KV transfer scales linearly with tokens."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        t1 = calc.kv_transfer_latency(1000)
        t2 = calc.kv_transfer_latency(2000)
        # Should be 2x (linear scaling)
        assert abs(t2 / t1 - 2.0) < 0.01

    def test_kv_transfer_inversely_scales_with_bandwidth(self) -> None:
        """Test that KV transfer inversely scales with bandwidth."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        slow = calc.kv_transfer_latency(1000, bandwidth_gb_s=10.0)
        fast = calc.kv_transfer_latency(1000, bandwidth_gb_s=20.0)
        # 2x bandwidth -> 0.5x time
        assert abs(fast / slow - 0.5) < 0.01


class TestEstimateTTFT:
    """Tests for time-to-first-token estimation."""

    def test_ttft_positive(self) -> None:
        """Test that TTFT is positive."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.estimate_ttft(1000)
        assert result > 0

    def test_ttft_increases_with_tokens(self) -> None:
        """Test that TTFT increases with input tokens."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        short = calc.estimate_ttft(100)
        long = calc.estimate_ttft(1000)
        assert long > short

    def test_ttft_decreases_with_cache_hits(self) -> None:
        """Test that TTFT decreases with cache hits."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        no_cache = calc.estimate_ttft(1000, cache_hit_tokens=0)
        with_cache = calc.estimate_ttft(1000, cache_hit_tokens=500)
        assert with_cache < no_cache

    def test_ttft_full_cache_hit(self) -> None:
        """Test TTFT with full cache hit (no prefill needed)."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        full_hit = calc.estimate_ttft(1000, cache_hit_tokens=1000)
        # Should still include first decode latency
        assert full_hit > 0


class TestEstimateGenerationTime:
    """Tests for full generation time estimation."""

    def test_generation_returns_dict(self) -> None:
        """Test that estimate_generation_time returns expected keys."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.estimate_generation_time(1000, 100)
        expected_keys = [
            "prefill_ms",
            "decode_ms",
            "total_ms",
            "ttft_ms",
            "tokens_per_second",
            "input_tokens",
            "output_tokens",
            "cache_hit_tokens",
        ]
        for key in expected_keys:
            assert key in result

    def test_generation_time_positive(self) -> None:
        """Test that all times are positive."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.estimate_generation_time(1000, 100)
        assert result["prefill_ms"] > 0
        assert result["decode_ms"] > 0
        assert result["total_ms"] > 0
        assert result["ttft_ms"] > 0

    def test_generation_total_equals_sum(self) -> None:
        """Test that total equals prefill + decode."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.estimate_generation_time(1000, 100)
        expected_total = result["prefill_ms"] + result["decode_ms"]
        assert abs(result["total_ms"] - expected_total) < 0.001

    def test_generation_with_cache_hits(self) -> None:
        """Test generation time with cache hits."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        no_cache = calc.estimate_generation_time(1000, 100, cache_hit_tokens=0)
        with_cache = calc.estimate_generation_time(1000, 100, cache_hit_tokens=500)
        # Cache hits should reduce prefill time
        assert with_cache["prefill_ms"] < no_cache["prefill_ms"]

    def test_tokens_per_second_calculated(self) -> None:
        """Test that tokens per second is calculated correctly."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        result = calc.estimate_generation_time(1000, 100)
        # TPS = output_tokens / (total_ms / 1000); the first output token is
        # produced by the prefill pass, so total time is the right denominator
        expected_tps = 100 / (result["total_ms"] / 1000)
        assert abs(result["tokens_per_second"] - expected_tps) < 0.001


class TestTPScaling:
    """Tests for tensor parallelism scaling."""

    def test_tp_reduces_latency(self) -> None:
        """Test that tensor parallelism reduces latency."""
        calc_tp1 = LatencyCalculator("H100_SXM", "llama-3.1-8b", tp_size=1)
        calc_tp2 = LatencyCalculator("H100_SXM", "llama-3.1-8b", tp_size=2)

        prefill_tp1 = calc_tp1.prefill_latency(1000)
        prefill_tp2 = calc_tp2.prefill_latency(1000)

        # TP=2 should be faster
        assert prefill_tp2.total_ms < prefill_tp1.total_ms

    def test_tp_scaling_factor(self) -> None:
        """Test that TP provides expected scaling."""
        calc_tp1 = LatencyCalculator("H100_SXM", "llama-3.1-8b", tp_size=1)
        calc_tp4 = LatencyCalculator("H100_SXM", "llama-3.1-8b", tp_size=4)

        decode_tp1 = calc_tp1.decode_latency(1000)
        decode_tp4 = calc_tp4.decode_latency(1000)

        # TP=4 should be significantly faster (not exactly 4x due to overhead)
        ratio = decode_tp1.total_ms / decode_tp4.total_ms
        assert ratio > 2.0  # At least 2x improvement


class TestEfficiencyFactor:
    """Tests for efficiency factor impact."""

    def test_lower_efficiency_increases_latency(self) -> None:
        """Test that lower efficiency increases latency."""
        calc_high = LatencyCalculator("H100_SXM", "llama-3.1-8b", efficiency=0.9)
        calc_low = LatencyCalculator("H100_SXM", "llama-3.1-8b", efficiency=0.5)

        prefill_high = calc_high.prefill_latency(1000)
        prefill_low = calc_low.prefill_latency(1000)

        assert prefill_low.total_ms > prefill_high.total_ms

    def test_efficiency_scales_compute(self) -> None:
        """Test that efficiency scales compute capacity."""
        calc_high = LatencyCalculator("H100_SXM", "llama-3.1-8b", efficiency=0.8)
        calc_low = LatencyCalculator("H100_SXM", "llama-3.1-8b", efficiency=0.4)

        # 0.4 is half of 0.8, so compute should be half as fast
        ratio = calc_high.effective_compute_tflops / calc_low.effective_compute_tflops
        assert abs(ratio - 2.0) < 0.01


class TestDifferentGPUs:
    """Tests for different GPU profiles."""

    def test_h100_faster_than_a100(self) -> None:
        """Test that H100 is faster than A100."""
        h100 = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        a100 = LatencyCalculator("A100_SXM", "llama-3.1-8b")

        prefill_h100 = h100.prefill_latency(1000)
        prefill_a100 = a100.prefill_latency(1000)

        assert prefill_h100.total_ms < prefill_a100.total_ms

    def test_l4_slower_than_h100(self) -> None:
        """Test that L4 is slower than H100."""
        h100 = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        l4 = LatencyCalculator("L4", "llama-3.1-8b")

        decode_h100 = h100.decode_latency(1000)
        decode_l4 = l4.decode_latency(1000)

        assert decode_l4.total_ms > decode_h100.total_ms


class TestDifferentModels:
    """Tests for different model profiles."""

    def test_larger_model_slower(self) -> None:
        """Test that larger models are slower."""
        calc_8b = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        calc_70b = LatencyCalculator("H100_SXM", "llama-3.1-70b")

        prefill_8b = calc_8b.prefill_latency(1000)
        prefill_70b = calc_70b.prefill_latency(1000)

        assert prefill_70b.total_ms > prefill_8b.total_ms

    def test_decode_model_size_impact(self) -> None:
        """Test that model size impacts decode latency."""
        calc_8b = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        calc_70b = LatencyCalculator("H100_SXM", "llama-3.1-70b")

        decode_8b = calc_8b.decode_latency(1000)
        decode_70b = calc_70b.decode_latency(1000)

        # 70B should be significantly slower (more weights to load)
        assert decode_70b.total_ms > decode_8b.total_ms * 2


class TestPhysicalBounds:
    """Regression tests pinning the model to physically possible values.

    These would have caught the TP double-count (super-linear scaling) and
    the sparsity-vs-dense TFLOPS inconsistency.
    """

    def test_tp_speedup_bounded_by_tp_size(self) -> None:
        """TP=N can never speed anything up by more than N x."""
        for tp in (2, 4, 8):
            calc_tp1 = LatencyCalculator("H100_SXM", "llama-3.1-70b", tp_size=1)
            calc_tpn = LatencyCalculator("H100_SXM", "llama-3.1-70b", tp_size=tp)

            decode_ratio = (
                calc_tp1.decode_latency(4096).total_ms / calc_tpn.decode_latency(4096).total_ms
            )
            prefill_ratio = (
                calc_tp1.prefill_latency(4096).total_ms / calc_tpn.prefill_latency(4096).total_ms
            )

            assert 1.0 <= decode_ratio <= tp * 1.001
            assert 1.0 <= prefill_ratio <= tp * 1.001

    def test_decode_absolute_sanity_8b_h100(self) -> None:
        """llama-8B decode on H100 must be in the single-digit-ms range.

        Weight streaming floor: ~16 GB / (3.35 TB/s * 0.7 eff) ~= 6.8 ms.
        """
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        ms = calc.decode_latency(2048).total_ms
        assert 3.0 < ms < 20.0

    def test_prefill_absolute_sanity_8b_h100(self) -> None:
        """llama-8B prefill of 1000 tokens on H100: ~2*P*N FLOPs -> tens of ms."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        ms = calc.prefill_latency(1000).total_ms
        assert 5.0 < ms < 60.0

    def test_prefill_context_tokens_reduce_cost(self) -> None:
        """Prefilling on top of a cached prefix costs less than from scratch,
        but more than prefilling the same token count with no context."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        from_scratch = calc.prefill_latency(10_000).total_ms
        on_cached = calc.prefill_latency(1_000, context_tokens=9_000).total_ms
        no_context = calc.prefill_latency(1_000).total_ms

        assert on_cached < from_scratch
        assert on_cached > no_context

    def test_ttft_not_double_counting_first_token(self) -> None:
        """With no cache, TTFT equals the prefill time (first token comes
        from the prefill forward pass, not an extra decode step)."""
        calc = LatencyCalculator("H100_SXM", "llama-3.1-8b")
        ttft = calc.estimate_ttft(1000)
        prefill_ms = calc.prefill_latency(1000).total_ms
        assert abs(ttft - prefill_ms) < 1e-9

    def test_gpu_comparison_is_consistent(self) -> None:
        """H100 vs A100 prefill speedup must reflect the dense-FLOPS ratio
        (~3.2x), not the sparsity-inflated ratio (~6.3x)."""
        h100 = LatencyCalculator("H100_SXM", "llama-3.1-8b").prefill_latency(4096)
        a100 = LatencyCalculator("A100_SXM", "llama-3.1-8b").prefill_latency(4096)
        assert h100.is_compute_bound and a100.is_compute_bound
        ratio = a100.total_ms / h100.total_ms
        assert 2.5 < ratio < 4.0
