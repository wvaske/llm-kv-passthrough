"""
Mock GPU connector for LMCache.

LMCache moves KV data between "GPU" memory and its storage tiers through a
GPUConnectorInterface. On a real deployment that interface reads vLLM's
paged GPU memory; here it reads plain CPU tensors that stand in for GPU KV
memory. Everything downstream of this interface — memory objects, tiering,
serialization, disk and remote I/O — is real LMCache code.

The source/destination tensor for each operation is passed per-call via
kwargs (``kv_source`` for stores, ``kv_dest`` for retrieves), so concurrent
requests never share connector state.

Tensor layout is MemoryFormat.KV_2LTD: [2 (K/V), layers, tokens, kv_heads * head_dim].
"""

from __future__ import annotations

import torch
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj


class MockGPUConnector(GPUConnectorInterface):
    """Serves CPU tensors to LMCache in place of GPU KV memory.

    Attributes:
        num_layers: Number of transformer layers.
        hidden_dim: KV hidden dimension per token per layer (kv_heads * head_dim).
        dtype: Torch dtype of the KV elements.
    """

    def __init__(self, num_layers: int, hidden_dim: int, dtype: torch.dtype) -> None:
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dtype = dtype

    def new_kv_tensor(self, num_tokens: int) -> torch.Tensor:
        """Allocate a KV tensor for a token sequence.

        Contents are uninitialized: the benchmark measures I/O, not data
        quality, and skipping the fill keeps tensor creation off the
        measured path. Size is authentic for the model profile.
        """
        return torch.empty(
            (2, self.num_layers, num_tokens, self.hidden_dim), dtype=self.dtype
        )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        kv_dest: torch.Tensor = kwargs["kv_dest"]
        kv_dest[:, :, start:end, :] = memory_obj.tensor

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        kv_source: torch.Tensor = kwargs["kv_source"]
        memory_obj.tensor.copy_(kv_source[:, :, start:end, :])

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs) -> None:
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=True):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(self, memory_objs=None, starts=None, ends=None, **kwargs) -> None:
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=True):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim])

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_format(self) -> MemoryFormat:
        return MemoryFormat.KV_2LTD
