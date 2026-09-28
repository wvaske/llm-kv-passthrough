"""
Calibrated Timing Strategy for KV-Bench.

The roofline model predicts *theoretical* latency from hardware peak specs.
Real deployments land somewhere below peak, and how far below varies with
the serving stack, model, and hardware generation. This module anchors the
roofline to **measured data** — lab experiments and published benchmarks
(e.g. SemiAnalysis InferenceX / InferenceMAX) — and extrapolates to the
model x hardware combination being emulated:

    calibrated_latency(target) = roofline(target, efficiency=1.0) * scale

where ``scale`` is derived from the anchor measurements closest to the
target: measured latency divided by the roofline prediction *for that
anchor's own model and hardware*. Prefill (TTFT-derived, compute-bound) and
decode (TPOT-derived, bandwidth-bound) are calibrated independently,
because compute utilization (MFU) and memory-bandwidth utilization (MBU)
diverge in practice.

Anchor selection, most-specific first:
    1. same model AND same GPU profile (and TP size, when it matches)
    2. same GPU profile          — hardware realism transfers across models
    3. same model                — model behavior transfers across hardware
    4. every usable anchor       — global fallback

Within the chosen tier the geometric mean of the anchors' scales is used.

Users add their own measurements with a YAML/JSON anchors file
(``--anchors`` / ``timing.anchors_file``)::

    - model: llama-3.1-70b
      gpu: H200_SXM
      tp_size: 4
      isl: 4096
      osl: 256
      ttft_ms: 950
      tpot_ms: 11.5
      source: experiment
      citation: "internal run 2026-08-14, vLLM 0.28, TRT backend"

Anchors ideally reflect *warm-storage* measurements (so TTFT is compute,
not KV-fetch); KV-Bench performs its storage I/O for real, so the sleep
this strategy adds represents only the emulated compute.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from kvbench.core.gpu_profiles import GPUProfile
from kvbench.core.models import ModelProfile
from kvbench.timing.roofline import RooflineTimingStrategy
from kvbench.timing.strategy import TimingResult, TimingStrategy

logger = logging.getLogger(__name__)

MIN_SCALE = 0.05   # guard against nonsense anchors (roofline >> measured)
MAX_SCALE = 500.0  # e.g. CPU backends can be 100x+ off GPU-peak rooflines


@dataclass(frozen=True)
class TimingAnchor:
    """One measured (model, hardware) latency point.

    Attributes:
        model: Model profile name the measurement was taken with.
        gpu: GPU profile name of the hardware.
        isl: Input sequence length of the measurement.
        osl: Output sequence length of the measurement.
        ttft_ms: Measured time-to-first-token in ms (None = no prefill
            anchor from this record).
        tpot_ms: Measured time-per-output-token in ms (None = no decode
            anchor from this record).
        tp_size: Tensor parallelism of the measurement.
        source: "experiment" (first-party lab run) or "published".
        citation: Where the number comes from — run id or benchmark URL.
        notes: Free-form context (backend, precision, concurrency, ...).
    """

    model: str
    gpu: str
    isl: int
    osl: int
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    tp_size: int = 1
    source: str = "experiment"
    citation: str = ""
    notes: str = ""
    # Tokens of the prompt that were already cached when TTFT was measured
    # (agentic / multi-turn traffic): prefill covered only isl - cached_tokens
    # new tokens, attending to the cached prefix.
    cached_tokens: int = 0


def _parse_anchor_entries(raw: object, origin: str) -> list[TimingAnchor]:
    if not isinstance(raw, list):
        raise ValueError(f"anchors source {origin} must contain a list of anchors")
    return [TimingAnchor(**entry) for entry in raw]


def _load_anchor_file(path: Path) -> list[TimingAnchor]:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    return _parse_anchor_entries(raw, str(path))


def load_anchors(path: str | Path) -> list[TimingAnchor]:
    """Load anchors from a YAML/JSON file, or every such file in a directory."""
    p = Path(path)
    if p.is_dir():
        anchors: list[TimingAnchor] = []
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".yaml", ".yml", ".json"):
                anchors.extend(_load_anchor_file(f))
        return anchors
    return _load_anchor_file(p)


def _load_builtin_anchors() -> tuple[TimingAnchor, ...]:
    """Load the built-in anchors packaged as config files.

    Calibration data lives in ``kvbench/timing/anchors/*.yaml`` — one file
    per accelerator — so new configurations are added by dropping in a
    file, not by editing code.
    """
    from importlib.resources import files

    anchors: list[TimingAnchor] = []
    root = files("kvbench.timing") / "anchors"
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if entry.name.rsplit(".", 1)[-1].lower() in ("yaml", "yml", "json"):
            import yaml

            raw = yaml.safe_load(entry.read_text())
            anchors.extend(_parse_anchor_entries(raw, entry.name))
    return tuple(anchors)


# Loaded once at import; extend at runtime with timing.anchors_file
# (a file or a directory of files).
BUILTIN_ANCHORS: tuple[TimingAnchor, ...] = _load_builtin_anchors()


@dataclass
class _AnchorScale:
    """A usable anchor with its derived calibration scales."""

    anchor: TimingAnchor
    prefill_scale: float | None
    decode_scale: float | None


def _hardware_class(gpu_name: str) -> str:
    """Coarse hardware class for anchor fallback: 'datacenter' (HBM-class
    bandwidth, >= 1 TB/s) or 'client' (unified-memory parts, CPUs, iGPUs)."""
    from kvbench.core.gpu_profiles import get_gpu_profile

    try:
        return "datacenter" if get_gpu_profile(gpu_name).hbm_bandwidth_tb_s >= 1.0 else "client"
    except KeyError:
        return "unknown"


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


class CalibratedTimingStrategy(TimingStrategy):
    """Roofline timing scaled by measured anchor data.

    Wraps a :class:`RooflineTimingStrategy` at ``efficiency=1.0`` (the
    theoretical envelope) and multiplies its predictions by scales derived
    from the best-matching anchors, so emulated compute time tracks what
    the represented model x hardware actually delivers rather than its
    datasheet peak.
    """

    def __init__(
        self,
        gpu: str | GPUProfile,
        model: str | ModelProfile,
        tp_size: int = 1,
        pp_size: int = 1,
        include_tp_communication: bool = True,
        include_pp_communication: bool = True,
        nvlink_bandwidth_gb_s: float | None = None,
        anchors: list[TimingAnchor] | None = None,
        anchors_file: str | Path | None = None,
    ) -> None:
        self._gpu_name = gpu if isinstance(gpu, str) else gpu.name
        self._model_name = model if isinstance(model, str) else model.name
        self._tp_size = tp_size

        self._backbone = RooflineTimingStrategy(
            gpu=gpu,
            model=model,
            tp_size=tp_size,
            pp_size=pp_size,
            efficiency=1.0,
            include_tp_communication=include_tp_communication,
            include_pp_communication=include_pp_communication,
            nvlink_bandwidth_gb_s=nvlink_bandwidth_gb_s,
        )

        pool = list(anchors) if anchors is not None else list(BUILTIN_ANCHORS)
        if anchors_file:
            pool.extend(load_anchors(anchors_file))

        self._scored = self._score_anchors(pool)
        self.prefill_scale, self._prefill_tier = self._select_scale("prefill")
        self.decode_scale, self._decode_tier = self._select_scale("decode")
        logger.info(
            "Calibrated timing for %s on %s: prefill x%.2f (%s), decode x%.2f (%s)",
            self._model_name,
            self._gpu_name,
            self.prefill_scale,
            self._prefill_tier,
            self.decode_scale,
            self._decode_tier,
        )

    def _score_anchors(self, pool: list[TimingAnchor]) -> list[_AnchorScale]:
        """Compute each usable anchor's measured/theoretical scales."""
        scored: list[_AnchorScale] = []
        for a in pool:
            try:
                backbone = RooflineTimingStrategy(
                    gpu=a.gpu,
                    model=a.model,
                    tp_size=a.tp_size,
                    pp_size=1,
                    efficiency=1.0,
                    include_tp_communication=a.tp_size > 1,
                    include_pp_communication=False,
                )
            except (KeyError, ValueError) as e:
                logger.warning("Skipping anchor %s/%s: %s", a.model, a.gpu, e)
                continue
            pf = df = None
            if a.ttft_ms is not None:
                new_tokens = max(a.isl - a.cached_tokens, 1)
                pred = backbone.prefill_latency(new_tokens, context_tokens=a.cached_tokens).total_ms
                if pred > 0:
                    pf = min(max(a.ttft_ms / pred, MIN_SCALE), MAX_SCALE)
            if a.tpot_ms is not None:
                pred = backbone.decode_latency(a.isl).total_ms
                if pred > 0:
                    df = min(max(a.tpot_ms / pred, MIN_SCALE), MAX_SCALE)
            if pf is not None or df is not None:
                scored.append(_AnchorScale(a, pf, df))
        return scored

    # Anchors whose ISL is within this factor of the request's context are
    # preferred inside a tier (a 2k-token chat anchor should not calibrate an
    # 80k-token agentic turn when an 80k anchor exists, and vice versa).
    ISL_WINDOW = 2.0

    def _select_scale(self, kind: str, isl: int | None = None) -> tuple[float, str]:
        """Pick the calibration scale from the most specific anchor tier.

        When ``isl`` is given, anchors within ISL_WINDOW of it are preferred
        inside each tier; a tier with no such anchors falls back to all of its
        anchors.
        """

        def usable(s: _AnchorScale) -> float | None:
            return s.prefill_scale if kind == "prefill" else s.decode_scale

        def near(s: _AnchorScale) -> bool:
            if not isl:
                return True
            r = s.anchor.isl / max(isl, 1)
            return 1.0 / self.ISL_WINDOW <= r <= self.ISL_WINDOW

        tiers: list[tuple[str, list[float]]] = []
        exact = [
            v
            for s in self._scored
            if (v := usable(s)) is not None
            and s.anchor.model == self._model_name
            and s.anchor.gpu == self._gpu_name
        ]
        same_gpu = [
            v
            for s in self._scored
            if (v := usable(s)) is not None and s.anchor.gpu == self._gpu_name
        ]
        same_model = [
            v
            for s in self._scored
            if (v := usable(s)) is not None
            and s.anchor.model == self._model_name
            and _hardware_class(s.anchor.gpu) == _hardware_class(self._gpu_name)
        ]
        # Cross-hardware fallback stays within the same hardware class, so a
        # datacenter GPU is never calibrated from a CPU or iGPU anchor whose
        # measured/roofline ratio is two orders of magnitude off (and vice
        # versa).
        my_class = _hardware_class(self._gpu_name)
        same_class = [
            v
            for s in self._scored
            if (v := usable(s)) is not None and _hardware_class(s.anchor.gpu) == my_class
        ]
        tiers = [
            ("exact model+gpu anchor", exact),
            ("same-gpu anchors", same_gpu),
            ("same-model anchors", same_model),
            (f"all {my_class}-class anchors", same_class),
        ]
        # Re-derive each tier with the ISL preference applied
        def tier_values(pred) -> list[float]:
            cands = [s for s in self._scored if usable(s) is not None and pred(s)]
            close = [s for s in cands if near(s)]
            chosen = close or cands
            return [usable(s) for s in chosen]

        my_class = _hardware_class(self._gpu_name)
        tiers = [
            ("exact model+gpu anchor", tier_values(lambda s: s.anchor.model == self._model_name and s.anchor.gpu == self._gpu_name)),
            ("same-gpu anchors", tier_values(lambda s: s.anchor.gpu == self._gpu_name)),
            ("same-model anchors", tier_values(lambda s: s.anchor.model == self._model_name and _hardware_class(s.anchor.gpu) == my_class)),
            (f"all {my_class}-class anchors", tier_values(lambda s: _hardware_class(s.anchor.gpu) == my_class)),
        ]
        for name, values in tiers:
            if values:
                return _geomean(values), name
        # No usable anchors at all: fall back to the historical default
        # efficiency (0.7 -> scale 1/0.7).
        return 1.0 / 0.7, "no anchors (default efficiency 0.7)"

    def _apply(self, result: TimingResult, scale: float, tier: str) -> TimingResult:
        return TimingResult(
            total_ms=result.total_ms * scale,
            breakdown={
                **{k: v * scale for k, v in result.breakdown.items()},
                "calibration_scale": scale,
            },
            is_compute_bound=result.is_compute_bound,
        )

    def _scale_for(self, kind: str, isl: int) -> tuple[float, str]:
        """ISL-aware scale, memoized per power-of-two ISL bucket."""
        cache = self.__dict__.setdefault("_scale_cache", {})
        bucket = (kind, max(isl, 1).bit_length())
        if bucket not in cache:
            cache[bucket] = self._select_scale(kind, isl=isl)
        return cache[bucket]

    def prefill_latency(
        self, num_tokens: int, batch_size: int = 1, context_tokens: int = 0
    ) -> TimingResult:
        base = self._backbone.prefill_latency(
            num_tokens, batch_size=batch_size, context_tokens=context_tokens
        )
        scale, tier = self._scale_for("prefill", num_tokens + context_tokens)
        return self._apply(base, scale, tier)

    def decode_latency(self, context_length: int, batch_size: int = 1) -> TimingResult:
        base = self._backbone.decode_latency(context_length, batch_size=batch_size)
        scale, tier = self._scale_for("decode", context_length)
        return self._apply(base, scale, tier)

    def __repr__(self) -> str:
        return (
            f"CalibratedTimingStrategy(model={self._model_name!r}, "
            f"gpu={self._gpu_name!r}, tp={self._tp_size}, "
            f"prefill_scale={self.prefill_scale:.2f} [{self._prefill_tier}], "
            f"decode_scale={self.decode_scale:.2f} [{self._decode_tier}])"
        )
