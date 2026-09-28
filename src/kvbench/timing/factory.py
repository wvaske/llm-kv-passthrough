"""
Timing strategy factory for KV-Bench.

Builds the TimingStrategy the servers use from the loaded KVBenchConfig,
so the timing mode (simple vs roofline) and TP/PP communication settings
flow from configuration into served latency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kvbench.timing.calibration import CalibratedTimingStrategy
from kvbench.timing.fullspeed import FullSpeedTimingStrategy
from kvbench.timing.roofline import RooflineTimingStrategy
from kvbench.timing.simple import SimpleTimingStrategy
from kvbench.timing.strategy import TimingStrategy

if TYPE_CHECKING:
    from kvbench.core.config import KVBenchConfig


def create_timing_strategy(config: KVBenchConfig) -> TimingStrategy:
    """Create the timing strategy described by the configuration.

    Args:
        config: The full KV-Bench configuration. Uses config.timing for the
            mode and communication settings, and config.gpu / config.server
            for the GPU and model profiles in roofline mode.

    Returns:
        A CalibratedTimingStrategy when timing.calibrated is set, else a
        SimpleTimingStrategy when timing.simple_mode is set, otherwise a
        RooflineTimingStrategy with TP/PP communication configured.
    """
    timing = config.timing
    strategy = _create_base_strategy(config)
    if timing.full_speed:
        return FullSpeedTimingStrategy(strategy)
    return strategy


def _create_base_strategy(config: KVBenchConfig) -> TimingStrategy:
    timing = config.timing

    if timing.calibrated:
        return CalibratedTimingStrategy(
            gpu=config.gpu.gpu_profile,
            model=config.server.model_profile,
            tp_size=config.gpu.tp_size,
            pp_size=timing.pp_size,
            include_tp_communication=timing.include_tp_communication,
            include_pp_communication=timing.include_pp_communication,
            nvlink_bandwidth_gb_s=timing.nvlink_bandwidth_gb_s,
            anchors_file=timing.anchors_file,
        )

    if timing.simple_mode:
        return SimpleTimingStrategy(
            prefill_ms_per_token=timing.prefill_ms_per_token,
            decode_ms_per_token=timing.decode_ms_per_token,
        )

    return RooflineTimingStrategy(
        gpu=config.gpu.gpu_profile,
        model=config.server.model_profile,
        tp_size=config.gpu.tp_size,
        pp_size=timing.pp_size,
        efficiency=config.gpu.efficiency_factor,
        include_tp_communication=timing.include_tp_communication,
        include_pp_communication=timing.include_pp_communication,
        nvlink_bandwidth_gb_s=timing.nvlink_bandwidth_gb_s,
    )
