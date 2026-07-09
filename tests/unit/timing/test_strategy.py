"""Unit tests for kvbench.timing.strategy module."""

from __future__ import annotations

import pytest

from kvbench.timing import TimingResult, TimingStrategy


class TestTimingResult:
    """Tests for TimingResult dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic TimingResult creation."""
        result = TimingResult(
            total_ms=10.0,
            breakdown={"compute": 8.0, "memory": 2.0},
            is_compute_bound=True,
        )
        assert result.total_ms == 10.0
        assert result.is_compute_bound is True
        assert result.breakdown["compute"] == 8.0

    def test_empty_breakdown(self) -> None:
        """Test TimingResult with empty breakdown."""
        result = TimingResult(total_ms=5.0, breakdown={}, is_compute_bound=False)
        assert result.breakdown == {}

    def test_default_breakdown(self) -> None:
        """Test TimingResult with default breakdown (empty dict)."""
        result = TimingResult(total_ms=5.0, is_compute_bound=False)
        assert result.breakdown == {}

    def test_default_is_compute_bound(self) -> None:
        """Test TimingResult default is_compute_bound (True)."""
        result = TimingResult(total_ms=5.0)
        assert result.is_compute_bound is True

    def test_negative_total_ms_raises(self) -> None:
        """Test that negative total_ms raises ValueError."""
        with pytest.raises(ValueError, match="total_ms must be non-negative"):
            TimingResult(total_ms=-1.0)

    def test_zero_total_ms_allowed(self) -> None:
        """Test that zero total_ms is allowed."""
        result = TimingResult(total_ms=0.0)
        assert result.total_ms == 0.0


class TestTimingStrategy:
    """Tests for TimingStrategy ABC."""

    def test_cannot_instantiate(self) -> None:
        """Test that TimingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError):
            TimingStrategy()  # type: ignore

    def test_is_abstract(self) -> None:
        """Test that TimingStrategy is abstract."""
        from abc import ABC

        assert issubclass(TimingStrategy, ABC)

    def test_has_abstract_methods(self) -> None:
        """Test that TimingStrategy has expected abstract methods."""
        # Check that the methods exist and are abstract
        assert hasattr(TimingStrategy, "prefill_latency")
        assert hasattr(TimingStrategy, "decode_latency")
