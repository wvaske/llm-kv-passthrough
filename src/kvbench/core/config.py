"""
KV-Bench Configuration System.

This module provides the configuration system for KV-Bench using Pydantic v2.
Configuration can be loaded from environment variables, YAML files, or programmatically.

Environment variables use the prefix KVBENCH_ and nested delimiter __.
Example: KVBENCH_RESOURCES__CPU_MEMORY_GB=16.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from kvbench.core.gpu_profiles import GPU_PROFILES, GPUProfile, load_gpu_profiles
from kvbench.core.models import MODEL_PROFILES, ModelProfile, load_model_profiles
from kvbench.core.payload import PayloadGenerator
from kvbench.core.ranks import parse_active_ranks, rank_selection_info, resolve_active_ranks


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class HiCacheConfig(ConfigModel):
    """SGLang HiCacheFile L3 and the V1 modeled host-tier policy (GiB/rank)."""

    storage_path: Path | None = None
    page_size: int = Field(default=256, ge=1, le=65536)
    host_size_gb: float = Field(default=0.25, gt=0)
    disk_size_gb: float = Field(default=0.5, gt=0)
    min_free_bytes: int = Field(default=64 * 1024**2, ge=0)
    eviction_ratio: float = Field(default=0.9, gt=0, le=1)
    mem_layout: Literal["page_first", "page_first_direct", "layer_first"] = "page_first_direct"
    write_policy: Literal["write_through", "write_back"] = "write_through"
    write_batch_pages: int = Field(default=32, ge=1, le=128)


class KVStackConfig(ConfigModel):
    """KV management stack configuration.

    KV-Bench never performs storage I/O itself — all KV cache operations go
    through a real KV management stack, and that stack's own application
    configuration controls storage (backends, tier sizes, eviction).
    KV-Bench deliberately has no storage settings of its own.

    Attributes:
        stack: Which KV management stack to use. 'kvbm' is accepted here
            but rejected at stack-creation time with the reason it is not
            yet supported (see kvbench.kv.factory).
        lmcache_config_file: Path to LMCache's own configuration file,
            passed verbatim to LMCacheEngineConfig.from_file(). When unset,
            LMCache is configured from its LMCACHE_* environment variables.
    """

    stack: Literal["lmcache", "hicache", "kvbm", "lmcache_mp"] = Field(
        default="lmcache",
        description="KV management stack; 'lmcache_mp' = LMCache multiprocess storage manager (one per node, L1 + L2 adapters with watermark eviction, as an lmcache server)",
    )
    lmcache_mp_config_file: Path | None = Field(
        default=None,
        description="lmcache_mp stack: JSON with server_args (lmcache server storage flags) and restore_window_chunks",
    )
    hicache: HiCacheConfig = Field(default_factory=HiCacheConfig)
    lmcache_config_file: Path | None = Field(
        default=None,
        description="Path to LMCache's own config file (LMCACHE_* env vars are used when unset)",
    )
    trace_file: Path | None = Field(
        default=None,
        description="Path to a JSONL file recording every logical and physical "
        "storage operation the KV stack performs (for FIO workload derivation)",
    )
    random_fill: bool = Field(
        default=True,
        description="Fill mock KV tensors with random data so stored bytes are "
        "incompressible like real KV cache (storage systems with compression "
        "or dedup report unrealistically good numbers against zero-filled data)",
    )
    payload_generator: PayloadGenerator = Field(
        default="torch", description="Synthetic entropy source: legacy torch bytes, or bounded fast SFC64 words",
    )
    max_cpu_memory_gb: float = Field(default=2.0, gt=0, description="Host guard: maximum aggregate LMCache CPU tier allocation (GiB)")
    max_chunk_mb: int = Field(default=256, ge=1, le=4096, description="Reject a rank chunk larger than this payload guard (MiB)")
    lmcache_reuse_history_entries: int = Field(
        default=65536, ge=1, le=1000000,
        description="Bound each LMCache LRU diagnostic reuse-history table; does not change cache capacity or eviction order",
    )
    seed: int = Field(default=24301, ge=0, le=2**63 - 1)
    active_ranks: list[int] | None = Field(
        default=None,
        description="Logical KV rank IDs to instantiate (null = all). TP/PP sizing and engine rates remain unchanged. "
        "Global rank = pp_rank * tp_size + tp_rank; e.g. [0] exercises one TP=8 shard.",
    )

    @field_validator("active_ranks", mode="before")
    @classmethod
    def parse_rank_selection(cls, value: object) -> list[int] | None:
        return parse_active_ranks(value)

    @field_validator("lmcache_config_file", mode="before")
    @classmethod
    def convert_path(cls, v: str | Path | None) -> Path | None:
        """Convert string path to Path object."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v

    @model_validator(mode="after")
    def validate_config_file_exists(self) -> KVStackConfig:
        """Fail loudly on a nonexistent LMCache config file."""
        if self.lmcache_config_file is not None and not self.lmcache_config_file.exists():
            raise ValueError(f"LMCache config file not found: {self.lmcache_config_file}")
        if self.stack == "hicache":
            if self.hicache.storage_path is None:
                raise ValueError("HiCache requires kv.hicache.storage_path")
            if self.lmcache_config_file is not None:
                raise ValueError("HiCache cannot use lmcache_config_file")
        elif self.hicache != HiCacheConfig():
            raise ValueError("kv.hicache options require kv.stack: hicache")
        return self


class GPUEmulationConfig(ConfigModel):
    """GPU emulation configuration for latency calculations.

    Attributes:
        gpu_profile: Name of the GPU profile to emulate.
        efficiency_factor: GPU efficiency factor (0.1 to 1.0).
        tp_size: Tensor parallelism size.
    """

    gpu_profile: str = Field(
        default="H100_SXM",
        description="Name of the GPU profile to emulate",
    )

    efficiency_factor: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="GPU efficiency factor",
    )
    tp_size: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Tensor parallelism size",
    )


class TimingConfig(ConfigModel):
    """Timing emulation configuration.

    Controls how GPU compute timing is simulated. Two modes available:
    - Simple mode: Fixed ms per token (no GPU modeling)
    - Roofline mode: Uses GPU/model profiles for realistic timing,
      including tensor parallel (AllReduce) and pipeline parallel
      (send/recv) communication overhead.

    Attributes:
        simple_mode: Use simple ms/token timing instead of roofline.
        prefill_ms_per_token: Prefill latency per token (simple mode).
        decode_ms_per_token: Decode latency per token (simple mode).
        include_tp_communication: Include AllReduce timing in roofline mode.
        include_pp_communication: Include pipeline send/recv in roofline mode.
        pp_size: Pipeline parallelism size.
        nvlink_bandwidth_gb_s: Interconnect bandwidth for communication
            timing. Defaults to the GPU profile's NVLink bandwidth when unset.
    """

    simple_mode: bool = Field(
        default=False,
        description="Use simple ms/token timing instead of roofline model",
    )
    prefill_ms_per_token: float = Field(
        default=0.1,
        ge=0.001,
        le=100.0,
        description="Prefill latency per token in ms (simple mode)",
    )
    decode_ms_per_token: float = Field(
        default=1.0,
        ge=0.001,
        le=100.0,
        description="Decode latency per token in ms (simple mode)",
    )
    include_tp_communication: bool = Field(
        default=True,
        description="Include AllReduce timing in roofline mode",
    )
    include_pp_communication: bool = Field(
        default=True,
        description="Include pipeline send/recv in roofline mode",
    )
    pp_size: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Pipeline parallelism size",
    )
    nvlink_bandwidth_gb_s: float | None = Field(
        default=None,
        ge=1.0,
        le=2000.0,
        description="Interconnect bandwidth in GB/s for communication timing "
        "(GPU profile's NVLink bandwidth when unset)",
    )
    calibrated: bool = Field(
        default=False,
        description="Scale the roofline model by measured anchor data "
        "(lab experiments + published benchmarks) instead of a fixed "
        "efficiency factor; takes precedence over simple_mode",
    )
    anchors_file: str | None = Field(
        default=None,
        description="YAML/JSON file (or directory of files) of additional timing anchors "
        "(measured TTFT/TPOT points) merged with the built-ins",
    )
    full_speed: bool = Field(
        default=False,
        description="Do not sleep for emulated GPU time: serve every request as fast "
        "as the KV stack allows, while still computing the emulated latency so the "
        "server can report effective TTFT and tokens/s for the emulated model and GPU",
    )
    hot_kv_gb: float | None = Field(
        default=None,
        ge=0,
        description="Hot set of KV chunks (GB) a returning session does not re-read through "
        "the KV stack, modelling the serving engine's own prefix cache (vLLM "
        "enable_prefix_caching). None or 0 = disabled, matching a serving stack with "
        "prefix caching off, where every turn retrieves its whole cached prefix",
    )
    batching: bool = Field(
        default=False,
        description="Model the serving engine's continuous batching: one step loop over all "
        "in-flight requests (decodes share each step; prefills are chunked into the "
        "remaining token budget), instead of pricing each request as if alone on the GPU",
    )
    batching_eff_bw: float = Field(default=0.90, gt=0, le=1.0, description="Memory-bandwidth efficiency of the step roofline")
    batching_eff_flops: float = Field(default=0.45, gt=0, le=1.0, description="Compute efficiency of the step roofline")
    batching_overhead_ms: float = Field(default=0.0, ge=0, description="Fixed time added to every engine step (ms)")
    batching_token_budget: int = Field(default=8192, ge=16, description="Tokens per step (vLLM max_num_batched_tokens)")
    batching_max_num_seqs: int = Field(default=256, ge=1, description="Max sequences per step (vLLM max_num_seqs)")
    batching_weight_bytes_per_param: float | None = Field(
        default=None, gt=0, description="Weight bytes per parameter read each step (None = the model dtype; e.g. ~0.55 for MXFP4)"
    )
    frontend_us_per_token: float = Field(
        default=1.0, ge=0, description="Serving front-end time per prompt token (templating, tokenization, KV lookup) "
        "before a request reaches the engine, in microseconds; the emulator's own pre-engine work counts against it"
    )
    chars_per_token: float = Field(
        default=4.0, gt=1.0, le=8.0, description="Characters per simulated prompt token; fractional values supported "
        "(3.84 matches the Llama 3 tokenizer on the AgentX traces)"
    )
    kv_load_gb_s: float | None = Field(
        default=12.0, ge=0, description="In-step load rate of the cached prefix into the engine (GB/s); None disables"
    )


class SchedulerConfig(ConfigModel):
    """Fidelity level 1: aggregate rates for one emulated engine/rank group."""

    prefill_tokens_per_second: float = Field(default=10000.0, gt=0)
    decode_tokens_per_second: float = Field(default=1000.0, gt=0)
    prefill_chunk_tokens: int = Field(default=256, ge=1, le=65536)
    max_active_requests: int = Field(default=16, ge=1, le=1024)
    max_pending_requests: int = Field(default=64, ge=0, le=8192)
    max_context_tokens: int = Field(default=131072, ge=1, le=10485760)
    max_output_tokens: int = Field(default=4096, ge=1, le=1048576)
    max_request_bytes: int = Field(default=4 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)
    request_timeout_s: float = Field(default=600, gt=0)


class ServerConfig(ConfigModel):
    """Server configuration for the KV-Bench HTTP server.

    Attributes:
        host: Host address to bind to.
        port: Port number to listen on.
        server_type: Type of server to run.
        model_profile: Name of the model profile to emulate.
        workers: Number of worker processes.
        log_level: Logging level.
    """

    host: str = Field(
        default="0.0.0.0",
        description="Host address to bind to",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port number to listen on",
    )
    server_type: Literal["prefill", "decode", "combined", "proxy"] = Field(
        default="combined",
        description="Type of server to run",
    )
    model_profile: str = Field(
        default="llama-3.1-8b",
        description="Name of the model profile to emulate",
    )

    workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes",
    )
    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Logging level",
    )


class DistributedConfig(ConfigModel):
    """Distributed deployment configuration.

    Attributes:
        prefill_endpoints: List of prefill server endpoints for proxy mode.
        decode_endpoints: List of decode server endpoints for proxy mode.
        registry_url: URL of the service registry.
        health_check_interval: Health check interval in seconds.
    """

    prefill_endpoints: list[str] = Field(
        default_factory=list,
        description="List of prefill server endpoints",
    )
    decode_endpoints: list[str] = Field(
        default_factory=list,
        description="List of decode server endpoints",
    )

    @field_validator("prefill_endpoints", "decode_endpoints", mode="before")
    @classmethod
    def parse_endpoint_list(cls, v: object) -> object:
        """Accept endpoint lists given as strings (env vars).

        Supports JSON arrays ('["http://a:8000", "http://b:8000"]') and
        comma-separated values ('http://a:8000,http://b:8000').
        """
        if isinstance(v, str):
            import json

            stripped = v.strip()
            if stripped.startswith("["):
                return json.loads(stripped)
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return v

    registry_url: str | None = Field(
        default=None,
        description="URL of the service registry",
    )
    health_check_interval: float = Field(
        default=10.0,
        ge=1.0,
        le=300.0,
        description="Health check interval in seconds",
    )


class MetricsConfig(ConfigModel):
    """Metrics and monitoring configuration.

    Attributes:
        enabled: Whether metrics collection is enabled.
        prometheus_port: Port for Prometheus metrics endpoint.
        include_histograms: Whether to include histogram metrics.
    """

    enabled: bool = Field(
        default=True,
        description="Whether metrics collection is enabled",
    )
    prometheus_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Port for Prometheus metrics endpoint",
    )
    include_histograms: bool = Field(
        default=True,
        description="Whether to include histogram metrics",
    )


class KVBenchConfig(ConfigModel):
    """Main KV-Bench configuration.

    This is the root configuration object that contains all sub-configurations.
    Configuration can be loaded from environment variables using the prefix KVBENCH_
    and nested delimiter __.

    Attributes:
        instance_id: Unique identifier for this KV-Bench instance.
        kv: KV management stack configuration.
        gpu: GPU emulation configuration.
        timing: Timing emulation configuration.
        server: Server configuration.
        distributed: Distributed deployment configuration.
        metrics: Metrics configuration.
    """

    instance_id: str = Field(
        default="kvbench-0",
        description="Unique identifier for this instance",
    )
    kv: KVStackConfig = Field(
        default_factory=KVStackConfig,
        description="KV management stack configuration",
    )
    gpu: GPUEmulationConfig = Field(
        default_factory=GPUEmulationConfig,
        description="GPU emulation configuration",
    )
    timing: TimingConfig = Field(
        default_factory=TimingConfig,
        description="Timing emulation configuration",
    )
    server: ServerConfig = Field(
        default_factory=ServerConfig,
        description="Server configuration",
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Distributed deployment configuration",
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration",
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    profiles_file: Path | None = None
    _model: ModelProfile = PrivateAttr()
    _gpu: GPUProfile = PrivateAttr()

    @model_validator(mode="after")
    def resolve_profiles(self) -> KVBenchConfig:
        models, gpus = dict(MODEL_PROFILES), dict(GPU_PROFILES)
        if self.profiles_file:
            models.update(load_model_profiles(self.profiles_file))
            gpus.update(load_gpu_profiles(self.profiles_file))
        if self.server.model_profile not in models:
            raise ValueError(f"Unknown model profile: {self.server.model_profile!r}")
        if self.gpu.gpu_profile not in gpus:
            raise ValueError(f"Unknown GPU profile: {self.gpu.gpu_profile!r}")
        self._model = models[self.server.model_profile]
        self._gpu = gpus[self.gpu.gpu_profile]
        layout = self._model.rank_layout(self.gpu.tp_size, self.timing.pp_size)
        resolve_active_ranks(len(layout), self.kv.active_ranks)
        return self

    @property
    def model(self) -> ModelProfile:
        return self._model

    @property
    def hardware(self) -> GPUProfile:
        return self._gpu

    def validate_serving(self) -> None:
        """Reject legacy modes at the production boundary, not in offline tools."""
        if self.server.server_type != "combined":
            raise ValueError("V1 supports combined serving only; proxy/prefill/decode are deferred. Run independent combined replicas.")
        if self.server.workers != 1:
            raise ValueError("V1 requires workers=1: one process owns one scheduler and KV stack. Use explicit replicas.")
        if self.gpu.efficiency_factor != GPUEmulationConfig().efficiency_factor:
            raise ValueError("gpu.efficiency_factor is an offline analytical option; configure V1 scheduler rates instead")
        legacy = self.timing.model_dump(exclude_defaults=True).keys() - {"full_speed", "chars_per_token", "pp_size"}
        if legacy:
            raise ValueError(f"V1 uses scheduler prefill/decode_tokens_per_second; legacy timing options are offline only: {sorted(legacy)}")
        if self.distributed != DistributedConfig():
            raise ValueError("Distributed configuration is not supported by V1 combined serving")
        if self.metrics.prometheus_port != MetricsConfig().prometheus_port:
            raise ValueError("V1 serves metrics on the HTTP server port; metrics.prometheus_port is a legacy setting")

    def manifest(self) -> dict[str, Any]:
        from dataclasses import asdict
        from importlib.metadata import PackageNotFoundError, version

        versions: dict[str, str | None] = {}
        for package in ("kvbench", "lmcache", "torch", "numpy"):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        return {"fidelity": "rate-v1", "config": self.model_dump(mode="json"),
                "model": asdict(self.model), "hardware": asdict(self.hardware),
                "ranks": self.model.rank_layout(self.gpu.tp_size, self.timing.pp_size),
                "kv_rank_selection": rank_selection_info(
                    self.model.rank_layout(self.gpu.tp_size, self.timing.pp_size), self.kv.active_ranks
                ),
                "versions": versions}

    @classmethod
    def from_env(cls) -> KVBenchConfig:
        """Load configuration from environment variables.

        Environment variables should be prefixed with KVBENCH_ and use
        double underscores (__) for nested values.

        Returns:
            KVBenchConfig instance populated from environment variables.
        """
        import os

        def get_nested_env(prefix: str = "KVBENCH") -> dict[str, Any]:
            """Extract nested configuration from environment variables."""
            result: dict[str, Any] = {}
            for key, value in os.environ.items():
                if not key.startswith(f"{prefix}_"):
                    continue
                if key in {"KVBENCH_CONFIG_FILE", "KVBENCH_REQUEST_LOG"}:
                    continue
                # Remove prefix and split by delimiter
                parts = key[len(prefix) + 1 :].lower().split("__")
                current = result
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return result

        env_config = get_nested_env()
        return cls.model_validate(env_config)

    @classmethod
    def from_yaml(cls, path: Path | str) -> KVBenchConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            KVBenchConfig instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the YAML file is invalid.
        """
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Paths belong to the configuration file, not the caller's cwd.
        if isinstance(data, dict):
            for mapping, key in ((data, "profiles_file"), (data.get("kv", {}), "lmcache_config_file"),
                                 (data.get("kv", {}), "trace_file"), (data.get("timing", {}), "anchors_file"),
                                 (data.get("kv", {}).get("hicache", {}), "storage_path")):
                if isinstance(mapping, dict) and mapping.get(key):
                    value = Path(mapping[key])
                    mapping[key] = str((path.parent / value).resolve()) if not value.is_absolute() else str(value)

        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to a YAML file.

        Args:
            path: Path to save the YAML configuration file.
        """
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, default_flow_style=False)
