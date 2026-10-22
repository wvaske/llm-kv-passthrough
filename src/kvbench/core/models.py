"""Model schema, KV byte geometry, and a file-backed model catalog.

V1 requires KV geometry, not model weights or an executable attention kernel.
The analytical parameter estimates remain available for offline exploration.
"""

from dataclasses import asdict, dataclass
from math import isfinite, lcm
from pathlib import Path
from typing import Any, Literal

from kvbench.core.cache_layout import CacheComponent
from kvbench.core.catalog import BUILTIN_CATALOG, read_catalog


@dataclass(frozen=True)
class ModelProfile:
    name: str
    layers: int
    hidden: int
    kv_heads: int
    intermediate: int
    num_heads: int | None = None
    vocab_size: int = 128256
    max_seq_len: int = 131072
    dtype: Literal["fp16", "bf16", "fp8"] = "bf16"
    kv_dtype: Literal["fp16", "bf16", "fp8"] | None = None
    kv_layers: int | None = None
    head_dim_override: int | None = None
    params_billions_override: float | None = None
    total_params_billions: float | None = None
    # MLA is a single replicated latent, not sharded K/V attention heads.
    kv_latent_dim: int | None = None
    cache_components: tuple[CacheComponent, ...] = ()
    source_repo: str | None = None
    source_revision: str | None = None
    verified_on: str | None = None
    source_urls: tuple[str, ...] = ()
    cache_notes: str = ""
    weight_format: str | None = None
    # Cross-layer sharing beyond a pipeline stage needs a real transfer policy.
    cache_pp_supported: bool = True

    def __post_init__(self) -> None:
        components = tuple(c if isinstance(c, CacheComponent) else CacheComponent(**c)
                           for c in self.cache_components)
        object.__setattr__(self, "cache_components", components)
        object.__setattr__(self, "source_urls", tuple(self.source_urls))
        if type(self.cache_pp_supported) is not bool:
            raise ValueError("cache_pp_supported must be a boolean")
        if components:
            if self.kv_dtype is not None or self.kv_layers is not None or self.kv_latent_dim is not None:
                raise ValueError("cache_components replaces kv_dtype, kv_layers and kv_latent_dim; specify precision/owners per component")
            if len({c.name for c in components}) != len(components):
                raise ValueError("Cache component names must be unique")
            if any(i >= self.layers for c in components for i in c.layer_ids):
                raise ValueError("Cache owner layer is outside the model backbone")
            if not any(c.export for c in components):
                raise ValueError("V1 requires some growing KV payload; fixed-state-only models need a snapshot adapter")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Model name must be nonempty text")
        if self.num_heads is None:
            object.__setattr__(self, "num_heads", max(1, self.hidden // 128))
        for field in ("layers", "hidden", "kv_heads", "intermediate", "num_heads",
                      "vocab_size", "max_seq_len", "kv_layers", "head_dim_override", "kv_latent_dim"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{field} must be a positive integer")
        if self.effective_kv_layers > self.layers:
            raise ValueError("kv_layers cannot exceed layers")
        for value in (self.dtype, self.kv_dtype or self.dtype):
            if value not in ("fp16", "bf16", "fp8"):
                raise ValueError(f"Unsupported dtype: {value}")
        for value in (self.params_billions_override, self.total_params_billions):
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError("Parameter counts must be finite and positive")
        assert self.num_heads is not None
        if not self.head_dim_override and self.hidden % self.num_heads:
            raise ValueError("hidden must divide num_heads unless head_dim_override is provided")

    @property
    def head_dim(self) -> int:
        assert self.num_heads is not None
        return self.head_dim_override or self.hidden // self.num_heads

    @property
    def kv_head_dim(self) -> int:
        return self.head_dim

    @property
    def effective_kv_layers(self) -> int:
        if self.cache_components:
            return len({i for c in self.cache_components if c.export for i in c.layer_ids})
        return self.layers if self.kv_layers is None else self.kv_layers

    @property
    def effective_kv_dtype(self) -> str:
        if self.cache_components:
            return "packed"
        return self.kv_dtype or self.dtype

    @property
    def bytes_per_element(self) -> int:
        return 1 if self.dtype == "fp8" else 2

    @property
    def kv_bytes_per_element(self) -> int:
        return 1 if self.effective_kv_dtype in ("fp8", "packed") else 2

    @property
    def kv_cache_bytes_per_token(self) -> int:
        if self.cache_components:
            raise ValueError("Heterogeneous cache has no single per-layer KV size; use total_kv_cache_bytes_per_token")
        elements = self.kv_latent_dim or 2 * self.kv_heads * self.head_dim
        return elements * self.kv_bytes_per_element

    @property
    def total_kv_cache_bytes_per_token(self) -> int:
        if self.cache_components:
            return sum(c.growth_bytes_per_token() for c in self.cache_components if c.export)
        return self.kv_cache_bytes_per_token * self.effective_kv_layers

    def kv_cache_size_bytes(self, num_tokens: int) -> int:
        if self.cache_components:
            return sum(c.size_bytes(num_tokens) for c in self.cache_components if c.export)
        return self.total_kv_cache_bytes_per_token * num_tokens

    @property
    def cache_token_alignment(self) -> int:
        return lcm(*(c.tokens_per_entry for c in self.cache_components if c.export))

    def cache_footprint(self, num_tokens: int) -> dict[str, Any]:
        """Logical per-sequence payload, excluding allocator/page padding and weights.

        Window/recurrent state is reported, never multiplied by sequence length
        or silently injected into V1's growing-KV offload workload.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")
        if not self.cache_components:
            size = self.kv_cache_size_bytes(num_tokens)
            return {"tokens": num_tokens, "exported_bytes": size, "growing_bytes": size,
                    "window_bytes": 0, "fixed_bytes": 0, "accounted_resident_bytes": size}
        sizes = {scope: sum(c.size_bytes(num_tokens) for c in self.cache_components if c.scope == scope)
                 for scope in ("growing", "window", "fixed")}
        return {"tokens": num_tokens, "exported_bytes": self.kv_cache_size_bytes(num_tokens),
                "growing_bytes": sizes["growing"], "window_bytes": sizes["window"],
                "fixed_bytes": sizes["fixed"], "accounted_resident_bytes": sum(sizes.values()),
                "components": [{"name": c.name, "scope": c.scope, "export": c.export,
                                "bytes": c.size_bytes(num_tokens)} for c in self.cache_components]}

    def cache_identity(self) -> dict[str, Any]:
        """Exclude commentary/provenance URLs from physical cache identity.

        Model revision and actual layouts remain part of the identity. Preserve
        the old identity for profiles without any new layout/source fields.
        """
        identity = asdict(self)
        for key in ("verified_on", "source_urls", "cache_notes", "weight_format", "cache_pp_supported"):
            identity.pop(key)
        for key in ("cache_components", "source_repo", "source_revision"):
            if not identity[key]:
                identity.pop(key)
        for component in identity.get("cache_components", ()):
            component.pop("notes")
        return identity

    def kv_cache_size_gb(self, num_tokens: int) -> float:
        return self.kv_cache_size_bytes(num_tokens) / 1024**3

    def rank_layout(self, tp_size: int = 1, pp_size: int = 1) -> list[dict[str, int]]:
        """Byte geometry per logical rank; PP partitions cache-bearing layers.

        GQA heads shard for TP <= heads, and replicate for TP > heads. MLA
        latent KV replicates on every TP rank. No resharding/collectives run.
        """
        if tp_size < 1 or pp_size < 1 or pp_size > self.effective_kv_layers:
            raise ValueError("Rank counts must be positive; PP cannot exceed KV layers")
        if self.cache_components:
            return self._component_rank_layout(tp_size, pp_size)
        if not self.kv_latent_dim and (
            (tp_size <= self.kv_heads and self.kv_heads % tp_size) or (
                tp_size > self.kv_heads and tp_size % self.kv_heads
            )
        ):
            raise ValueError("TP must divide KV heads or be a multiple of KV heads")
        heads = max(1, self.kv_heads // tp_size)
        # Classic LMCache uses a two-plane payload. MLA's latent is represented
        # by two halves solely for byte sizing (not native MLA interoperability).
        if self.kv_latent_dim and self.kv_latent_dim % 2:
            raise ValueError("Classic byte-layout adapter requires an even MLA latent dimension")
        width = self.kv_latent_dim // 2 if self.kv_latent_dim else heads * self.head_dim
        result = []
        start = 0
        for pp in range(pp_size):
            layers = self.effective_kv_layers // pp_size + (pp < self.effective_kv_layers % pp_size)
            for tp in range(tp_size):
                result.append({"rank": pp * tp_size + tp, "tp_rank": tp, "pp_rank": pp,
                               "layer_start": start, "layers": layers, "width": width,
                               "bytes_per_token": 2 * layers * width * self.kv_bytes_per_element})
            start += layers
        return result

    def _component_rank_layout(self, tp_size: int, pp_size: int) -> list[dict[str, int]]:
        if pp_size > 1 and not self.cache_pp_supported:
            raise ValueError("Cross-layer shared cache profile supports PP=1 only in V1")
        assert self.num_heads is not None
        if self.num_heads % tp_size:
            raise ValueError("TP must divide the model's query attention heads")
        # Validate recurrent-state sharding too, even though it is not offloaded.
        for component in self.cache_components:
            component.local_heads(tp_size)
        result = []
        start = 0
        for pp in range(pp_size):
            end = start + self.layers // pp_size + (pp < self.layers % pp_size)
            owners = {i for c in self.cache_components if c.export for i in c.layer_ids if start <= i < end}
            size = sum(c.growth_bytes_per_token(tp_size, start, end) for c in self.cache_components if c.export)
            if size == 0:
                raise ValueError("PP stage has no growing KV payload; state-only stages need a snapshot adapter")
            for tp in range(tp_size):
                result.append({"rank": pp * tp_size + tp, "tp_rank": tp, "pp_rank": pp,
                               "layer_start": start, "layers": len(owners), "tensor_layers": 1,
                               "planes": 1, "width": size, "bytes_per_token": size})
            start = end
        return result

    @property
    def estimated_params_billions(self) -> float:
        if self.params_billions_override is not None:
            return self.params_billions_override
        assert self.num_heads is not None
        qkv = self.hidden * (self.num_heads + 2 * self.kv_heads) * self.head_dim
        output = self.num_heads * self.head_dim * self.hidden
        per_layer = qkv + output + self.hidden * self.intermediate * 3 + self.hidden * 2
        return (2 * self.vocab_size * self.hidden + per_layer * self.layers + self.hidden) / 1e9

    @property
    def model_size_gb(self) -> float:
        return (self.total_params_billions or self.estimated_params_billions) * self.bytes_per_element


def load_model_profiles(path: Path = BUILTIN_CATALOG) -> dict[str, ModelProfile]:
    return {name: ModelProfile(**spec) for name, spec in read_catalog(path).get("models", {}).items()}


MODEL_PROFILES = load_model_profiles()


def get_model_profile(name: str) -> ModelProfile:
    if name not in MODEL_PROFILES:
        raise KeyError(f"Unknown model profile: {name}. Available profiles: {', '.join(list_model_profiles())}")
    return MODEL_PROFILES[name]


def list_model_profiles() -> list[str]:
    return sorted(MODEL_PROFILES)


def get_model_profile_info(name: str) -> dict[str, Any]:
    p = get_model_profile(name)
    return {**asdict(p), "head_dim": p.head_dim,
            "kv_cache_bytes_per_token": p.total_kv_cache_bytes_per_token,
            "estimated_params_billions": round(p.estimated_params_billions, 2),
            "model_size_gb": round(p.model_size_gb, 2),
            "model_size_basis": f"nominal {p.dtype} parameter bytes, not a mixed-quantization weight footprint",
            "cache_token_alignment": p.cache_token_alignment,
            "cache_transport_dtype": p.effective_kv_dtype,
            "cache_footprint_at_max_context": p.cache_footprint(p.max_seq_len)}


def calculate_kv_cache_requirements(model_name: str, max_batch_size: int,
                                    max_seq_len: int | None = None) -> dict[str, Any]:
    p = get_model_profile(model_name)
    seq = max_seq_len or p.max_seq_len
    size = p.kv_cache_size_bytes(seq)
    resident = p.cache_footprint(seq)["accounted_resident_bytes"]
    return {"model": model_name, "max_batch_size": max_batch_size, "max_seq_len": seq,
            "per_request_gb": size / 1024**3, "total_gb": size * max_batch_size / 1024**3,
            "bytes_per_token": p.total_kv_cache_bytes_per_token,
            "size_scope": "exported growing KV payload; excludes window/fixed state",
            "accounted_resident_per_request_gb": resident / 1024**3,
            "accounted_resident_total_gb": resident * max_batch_size / 1024**3}
