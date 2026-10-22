"""Declarative cache payload accounting, independent of model weight precision.

Each component names the layers that actually OWN its buffers. Shared readers
must not appear as additional owners. Growing KV, bounded windows and recurrent
state have different lifetime/size rules; only growing payload is transported
by the V1 classic LMCache adapter.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class CacheComponent:
    name: str
    layer_ids: tuple[int, ...]
    elements_per_head: int
    bits: int = 16
    heads: int = 1
    partition: Literal["replicated", "kv_heads", "sharded"] = "replicated"
    scope: Literal["growing", "window", "fixed"] = "growing"
    tokens_per_entry: int = 1
    window_tokens: int | None = None
    quant_group_size: int | None = None
    scale_bytes: int = 0
    export: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer_ids", tuple(self.layer_ids))
        if not isinstance(self.name, str) or not self.name or not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("Cache components need a name and unique owning layer_ids")
        if any(type(i) is not int or i < 0 for i in self.layer_ids):
            raise ValueError("Cache layer_ids are zero-based nonnegative integers")
        for value in (self.elements_per_head, self.heads, self.tokens_per_entry):
            if type(value) is not int or value <= 0:
                raise ValueError("Cache dimensions and compression ratios must be positive integers")
        if type(self.bits) is not int or self.bits not in (4, 8, 16, 32, 64):
            raise ValueError("Cache bits must be 4, 8, 16, 32 or 64")
        if type(self.export) is not bool:
            raise ValueError("export must be a boolean")
        if self.partition not in ("replicated", "kv_heads", "sharded"):
            raise ValueError("Unknown cache partition mode")
        if self.scope not in ("growing", "window", "fixed"):
            raise ValueError("Unknown cache lifetime")
        if self.scope == "window":
            if type(self.window_tokens) is not int or self.window_tokens <= 0:
                raise ValueError("Window components need positive window_tokens")
        elif self.window_tokens is not None:
            raise ValueError("window_tokens only applies to window components")
        if self.scope != "growing" and self.export:
            raise ValueError("V1 exports growing KV only; set export: false for window/fixed state")
        if self.scope == "fixed" and self.tokens_per_entry != 1:
            raise ValueError("Fixed state has no token compression ratio")
        if type(self.scale_bytes) is not int or self.scale_bytes < 0:
            raise ValueError("scale_bytes must be a nonnegative integer")
        if self.quant_group_size is not None:
            if type(self.quant_group_size) is not int or self.quant_group_size <= 0 or not self.scale_bytes:
                raise ValueError("Quantization groups require a positive group size and scale_bytes")
        elif self.scale_bytes:
            raise ValueError("scale_bytes requires quant_group_size")

    @property
    def bytes_per_head(self) -> int:
        payload = (self.elements_per_head * self.bits + 7) // 8
        if self.quant_group_size:
            payload += (self.elements_per_head + self.quant_group_size - 1) // self.quant_group_size * self.scale_bytes
        return payload

    def local_heads(self, tp_size: int) -> int:
        if tp_size < 1:
            raise ValueError("TP must be positive")
        if self.partition == "replicated":
            return self.heads
        if self.heads % tp_size == 0:
            return self.heads // tp_size
        if self.partition == "kv_heads" and tp_size % self.heads == 0:
            return 1
        raise ValueError(f"{self.name}: heads={self.heads} cannot use TP={tp_size}")

    def owners(self, start: int, end: int) -> int:
        return sum(start <= i < end for i in self.layer_ids)

    def growth_bytes_per_token(self, tp_size: int = 1, start: int = 0, end: int = 2**31) -> int:
        if self.scope != "growing":
            return 0
        numerator = self.owners(start, end) * self.local_heads(tp_size) * self.bytes_per_head
        if numerator % self.tokens_per_entry:
            raise ValueError(f"{self.name}: fractional byte strides require a different chunk codec; no rounding is allowed")
        return numerator // self.tokens_per_entry

    def size_bytes(self, tokens: int, tp_size: int = 1, start: int = 0, end: int = 2**31) -> int:
        if tokens < 0:
            raise ValueError("tokens cannot be negative")
        if tokens == 0:
            return 0
        if self.scope == "fixed":
            entries = 1
        elif self.scope == "window":
            assert self.window_tokens is not None
            entries = min(tokens, self.window_tokens) // self.tokens_per_entry
        else:
            entries = tokens // self.tokens_per_entry
        return entries * self.owners(start, end) * self.local_heads(tp_size) * self.bytes_per_head
