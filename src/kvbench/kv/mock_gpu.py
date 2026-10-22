"""Bounded synthetic payload provider at LMCache's GPU connector boundary.

Fill the memory object LMCache already allocated; never allocate prompt-sized
source/destination tensors. Content is deterministic per prefix, rank and seed,
with distinct high-entropy data across chunks/ranks. Retrieval still traverses
the real stack but the final GPU copy is intentionally omitted in fidelity 1.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

import torch
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

from kvbench.core.payload import PayloadGenerator, fill_sfc64


class MockGPUConnector(GPUConnectorInterface):  # type: ignore[misc]  # LMCache's optional interface is untyped
    def __init__(self, num_layers: int, hidden_dim: int, dtype: torch.dtype,
                 random_fill: bool = True, seed: int = 24301, rank: int = 0,
                 namespace: str = "kvbench", planes: int = 2,
                 payload_generator: PayloadGenerator = "torch"):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.random_fill = random_fill
        self.seed = seed
        self.rank = rank
        self.namespace = namespace
        self.planes = planes
        self.payload_generator = payload_generator

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any) -> None:  # noqa: ARG002 - LMCache interface
        flat = memory_obj.tensor.view(torch.uint8).reshape(-1)
        if not self.random_fill:
            flat.zero_()
            return
        tokens = kwargs["payload_tokens"]
        material = f"{self.namespace}:{self.seed}:{self.rank}:".encode() + tokens[:end].numpy().tobytes()
        digest = hashlib.blake2b(material, digest_size=8).digest()
        # Mix rank into all bits: the CPU generator may only consume the low
        # 32 seed bits, so XORing rank << 32 would duplicate payloads.
        seed = int.from_bytes(digest, "little") & (2**63 - 1)
        if self.payload_generator == "sfc64":
            import numpy as np

            fill_sfc64(flat.numpy(), np.random.SFC64(seed))
        else:
            generator = torch.Generator().manual_seed(seed)
            flat.random_(0, 256, generator=generator)

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs: Any) -> None:  # noqa: ARG002 - no final device copy in V1
        # LMCache fetched/owns the real payload. V1 measures no GPU transfer.
        return None

    def batched_from_gpu(self, memory_objs: Sequence[MemoryObj], starts: Sequence[int], ends: Sequence[int], **kwargs: Any) -> None:
        for obj, start, end in zip(memory_objs, starts, ends, strict=True):
            self.from_gpu(obj, start, end, **kwargs)

    def batched_to_gpu(self, memory_objs: Sequence[MemoryObj] | None = None, starts: Sequence[int] | None = None, ends: Sequence[int] | None = None, **kwargs: Any) -> None:  # noqa: ARG002 - LMCache interface
        return None

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([self.planes, self.num_layers, num_tokens, self.hidden_dim])

    def get_dtype(self) -> torch.dtype:
        return self.dtype

    def get_format(self) -> MemoryFormat:
        return MemoryFormat.KV_2LTD
