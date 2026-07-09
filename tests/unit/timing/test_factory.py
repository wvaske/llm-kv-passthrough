"""Unit tests for kvbench.timing.factory module."""

from __future__ import annotations

from kvbench.core.config import KVBenchConfig
from kvbench.timing import (
    RooflineTimingStrategy,
    SimpleTimingStrategy,
    create_timing_strategy,
)


class TestCreateTimingStrategy:
    """Tests for create_timing_strategy."""

    def test_default_is_roofline(self) -> None:
        """Default config produces a roofline strategy."""
        config = KVBenchConfig()
        strategy = create_timing_strategy(config)
        assert isinstance(strategy, RooflineTimingStrategy)

    def test_simple_mode(self) -> None:
        """simple_mode=True produces a SimpleTimingStrategy with the
        configured per-token latencies."""
        config = KVBenchConfig.model_validate(
            {
                "timing": {
                    "simple_mode": True,
                    "prefill_ms_per_token": 0.2,
                    "decode_ms_per_token": 2.0,
                }
            }
        )
        strategy = create_timing_strategy(config)
        assert isinstance(strategy, SimpleTimingStrategy)
        assert strategy.prefill_ms_per_token == 0.2
        assert strategy.decode_ms_per_token == 2.0

    def test_roofline_uses_gpu_and_model_profiles(self) -> None:
        """Roofline strategy is built from the configured profiles."""
        config = KVBenchConfig.model_validate(
            {
                "gpu": {"gpu_profile": "A100_SXM", "tp_size": 4, "efficiency_factor": 0.8},
                "server": {"model_profile": "llama-3.1-70b"},
                "timing": {"pp_size": 2},
            }
        )
        strategy = create_timing_strategy(config)
        assert isinstance(strategy, RooflineTimingStrategy)
        assert strategy.tp_size == 4
        assert strategy.pp_size == 2
        assert strategy.calculator.efficiency == 0.8
        assert strategy.calculator.gpu.name == "NVIDIA A100 SXM"

    def test_roofline_communication_flags(self) -> None:
        """Communication toggles flow from config into the strategy."""
        config = KVBenchConfig.model_validate(
            {
                "timing": {
                    "include_tp_communication": False,
                    "include_pp_communication": False,
                }
            }
        )
        strategy = create_timing_strategy(config)
        assert isinstance(strategy, RooflineTimingStrategy)
        assert strategy.include_tp_communication is False
        assert strategy.include_pp_communication is False

    def test_roofline_bandwidth_override(self) -> None:
        """nvlink_bandwidth_gb_s in config overrides the GPU profile."""
        config = KVBenchConfig.model_validate(
            {"timing": {"nvlink_bandwidth_gb_s": 450.0}}
        )
        strategy = create_timing_strategy(config)
        assert isinstance(strategy, RooflineTimingStrategy)
        assert strategy.interconnect_bandwidth_gb_s == 450.0
