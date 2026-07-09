"""Unit tests for kvbench.timing.simple module."""

from __future__ import annotations

import pytest

from kvbench.timing import SimpleTimingStrategy, TimingResult


class TestSimpleTimingStrategy:
    """Tests for SimpleTimingStrategy."""

    def test_basic_creation(self) -> None:
        """Test basic creation with valid parameters."""
        strategy = SimpleTimingStrategy(
            prefill_ms_per_token=0.1,
            decode_ms_per_token=1.0,
        )
        assert strategy.prefill_ms_per_token == 0.1
        assert strategy.decode_ms_per_token == 1.0

    def test_invalid_prefill_ms_negative(self) -> None:
        """Test that negative prefill_ms raises error."""
        with pytest.raises(ValueError, match="prefill_ms_per_token must be positive"):
            SimpleTimingStrategy(prefill_ms_per_token=-0.1, decode_ms_per_token=1.0)

    def test_invalid_prefill_ms_zero(self) -> None:
        """Test that zero prefill_ms raises error."""
        with pytest.raises(ValueError, match="prefill_ms_per_token must be positive"):
            SimpleTimingStrategy(prefill_ms_per_token=0.0, decode_ms_per_token=1.0)

    def test_invalid_decode_ms_negative(self) -> None:
        """Test that negative decode_ms raises error."""
        with pytest.raises(ValueError, match="decode_ms_per_token must be positive"):
            SimpleTimingStrategy(prefill_ms_per_token=0.1, decode_ms_per_token=-1.0)

    def test_invalid_decode_ms_zero(self) -> None:
        """Test that zero decode_ms raises error."""
        with pytest.raises(ValueError, match="decode_ms_per_token must be positive"):
            SimpleTimingStrategy(prefill_ms_per_token=0.1, decode_ms_per_token=0.0)

    def test_repr(self) -> None:
        """Test string representation of the strategy."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        repr_str = repr(strategy)
        assert "SimpleTimingStrategy" in repr_str
        assert "0.1" in repr_str
        assert "1.0" in repr_str


class TestSimplePrefillLatency:
    """Tests for SimpleTimingStrategy.prefill_latency."""

    def test_prefill_returns_timing_result(self) -> None:
        """Test that prefill returns TimingResult."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.prefill_latency(1000)
        assert isinstance(result, TimingResult)

    def test_prefill_scales_with_tokens(self) -> None:
        """Test that prefill scales linearly with tokens."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        r1 = strategy.prefill_latency(1000)
        r2 = strategy.prefill_latency(2000)
        assert abs(r2.total_ms - 2 * r1.total_ms) < 0.001

    def test_prefill_calculation(self) -> None:
        """Test exact prefill calculation."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.prefill_latency(1000)
        assert abs(result.total_ms - 100.0) < 0.001  # 0.1 * 1000

    def test_prefill_batch_size(self) -> None:
        """Test prefill with batch_size > 1."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        r1 = strategy.prefill_latency(1000, batch_size=1)
        r2 = strategy.prefill_latency(1000, batch_size=2)
        assert abs(r2.total_ms - 2 * r1.total_ms) < 0.001

    def test_prefill_is_compute_bound(self) -> None:
        """Test that prefill reports as compute-bound."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.prefill_latency(1000)
        assert result.is_compute_bound is True

    def test_prefill_breakdown(self) -> None:
        """Test prefill breakdown has expected key."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.prefill_latency(1000)
        assert "simple_prefill" in result.breakdown
        assert result.breakdown["simple_prefill"] == result.total_ms

    def test_prefill_zero_tokens(self) -> None:
        """Test prefill with zero tokens."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.prefill_latency(0)
        assert result.total_ms == 0.0


class TestSimpleDecodeLatency:
    """Tests for SimpleTimingStrategy.decode_latency."""

    def test_decode_returns_timing_result(self) -> None:
        """Test that decode returns TimingResult."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.decode_latency(1000)
        assert isinstance(result, TimingResult)

    def test_decode_ignores_context_length(self) -> None:
        """Test that decode is constant regardless of context."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        r1 = strategy.decode_latency(100)
        r2 = strategy.decode_latency(10000)
        # In simple mode, context length doesn't matter
        assert abs(r1.total_ms - r2.total_ms) < 0.001

    def test_decode_calculation(self) -> None:
        """Test exact decode calculation."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.decode_latency(1000)
        assert abs(result.total_ms - 1.0) < 0.001  # 1.0 per token

    def test_decode_batch_size(self) -> None:
        """Test decode with batch_size > 1."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        r1 = strategy.decode_latency(1000, batch_size=1)
        r2 = strategy.decode_latency(1000, batch_size=2)
        assert abs(r2.total_ms - 2 * r1.total_ms) < 0.001

    def test_decode_is_memory_bound(self) -> None:
        """Test that decode reports as memory-bound."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.decode_latency(1000)
        assert result.is_compute_bound is False

    def test_decode_breakdown(self) -> None:
        """Test decode breakdown has expected key."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.decode_latency(1000)
        assert "simple_decode" in result.breakdown
        assert result.breakdown["simple_decode"] == result.total_ms

    def test_decode_zero_context(self) -> None:
        """Test decode with zero context length (still produces timing)."""
        strategy = SimpleTimingStrategy(0.1, 1.0)
        result = strategy.decode_latency(0)
        # Decode still takes time even with no context
        assert result.total_ms == 1.0
