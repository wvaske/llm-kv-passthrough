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

import threading

import torch
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj


class MockGPUConnector(GPUConnectorInterface):
    """Serves CPU tensors to LMCache in place of GPU KV memory.

    Attributes:
        num_layers: Number of transformer layers.
        hidden_dim: KV hidden dimension per token per layer (kv_heads * head_dim).
        dtype: Torch dtype of the KV elements.
        random_fill: Fill new KV tensors with random data.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        dtype: torch.dtype,
        random_fill: bool = True,
        random_pool_mb: int = 256,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.random_fill = random_fill
        self._pool: torch.Tensor | None = None
        if random_fill:
            # A pre-generated pool of random bytes copied (at random offsets)
            # into each new KV tensor. Real KV cache is high-entropy float
            # data that neither compresses nor dedupes; zero/uninitialized
            # pages would let storage systems with compression, dedup, or
            # zero-block elision report unrealistically good numbers. Pool
            # slicing keeps the fill to a memcpy instead of an RNG pass.
            generator = torch.Generator().manual_seed(0x5EED)
            self._pool = torch.randint(
                0,
                256,
                (random_pool_mb * 1024 * 1024,),
                dtype=torch.uint8,
                generator=generator,
            )
            self._offset_gen = torch.Generator().manual_seed(0xF111)
            # torch.Generator is not thread-safe; new_kv_tensor runs under
            # asyncio.to_thread with real concurrency, so serialize access.
            self._offset_lock = threading.Lock()

    def new_kv_tensor(self, num_tokens: int) -> torch.Tensor:
        """Allocate a KV tensor for a token sequence.

        Size is authentic for the model profile. Contents are random bytes
        (incompressible, like real KV data) when random_fill is on, else
        uninitialized.
        """
        tensor = torch.empty(
            (2, self.num_layers, num_tokens, self.hidden_dim), dtype=self.dtype
        )
        if self._pool is not None:
            flat = tensor.view(torch.uint8).view(-1)
            pool = self._pool
            pool_size = pool.numel()
            pos = 0
            remaining = flat.numel()
            while remaining > 0:
                with self._offset_lock:
                    offset = int(
                        torch.randint(0, pool_size, (1,), generator=self._offset_gen)
                    )
                length = min(remaining, pool_size - offset)
                flat[pos : pos + length] = pool[offset : offset + length]
                pos += length
                remaining -= length
        return tensor

    # -- shared "GPU" buffers ---------------------------------------------
    # A real engine has a fixed KV pool in GPU memory; it does not allocate
    # and fill a prompt-sized tensor per request. Allocating + random-filling
    # a 2-3.5 GB tensor for every retrieve (whose contents LMCache then
    # overwrites) and every store (of which only the new suffix is read)
    # cost the emulator 0.2-1 s of CPU per 100k-token turn that the real
    # stack does not spend. These views reuse two buffers instead:
    #   source_view: random bytes, viewed at a random token offset per call so
    #     chunks stored by different requests at the same position differ
    #     (no cross-request dedup);
    #   dest_view: scratch (contents irrelevant; concurrent writers race
    #     harmlessly, as they would into disjoint GPU blocks).
    MAX_SHARED_BYTES = 12 << 30

    def _token_bytes(self) -> int:
        return 2 * self.num_layers * self.hidden_dim * torch.empty((), dtype=self.dtype).element_size()

    def _shared(self, name: str, tokens: int, fill: bool) -> torch.Tensor | None:
        lock = self.__dict__.setdefault("_shared_lock", threading.Lock())
        with lock:
            buf = self.__dict__.get(name)
            if buf is not None and buf.shape[2] >= tokens:
                return buf
            want = max(tokens, 2 * (buf.shape[2] if buf is not None else 0), 32768)
            if want * self._token_bytes() > self.MAX_SHARED_BYTES:
                want = tokens
                if want * self._token_bytes() > self.MAX_SHARED_BYTES:
                    return None
            buf = self.new_kv_tensor(want) if fill else torch.empty(
                (2, self.num_layers, want, self.hidden_dim), dtype=self.dtype
            )
            self.__dict__[name] = buf
            return buf

    def source_view(self, num_tokens: int) -> torch.Tensor:
        """KV source for a store: a random-offset view of a shared random buffer."""
        buf = self._shared("_src_buf", 2 * num_tokens, fill=True)
        if buf is None:
            return self.new_kv_tensor(num_tokens)
        span = buf.shape[2] - num_tokens
        if self._pool is not None and span > 0:
            with self._offset_lock:
                off = int(torch.randint(0, span + 1, (1,), generator=self._offset_gen))
        else:
            off = 0
        return buf[:, :, off : off + num_tokens, :]

    def dest_view(self, num_tokens: int) -> torch.Tensor:
        """KV destination for a retrieve: a view of a shared scratch buffer."""
        buf = self._shared("_dst_buf", num_tokens, fill=False)
        if buf is None:
            return torch.empty((2, self.num_layers, num_tokens, self.hidden_dim), dtype=self.dtype)
        return buf[:, :, :num_tokens, :]

    # Retrieve-side copy into the stand-in GPU tensor. Off by default: the
    # bytes are never used, the serving engine's load time is modelled by the
    # batching engine (kv_load_gb_s), and a CPU memcpy of a 2-3 GB prefix
    # (~0.2 s) held LMCache's CPU-tier objects pinned ~7x longer than the
    # real GB10's unified-memory copy (~30 ms), which changed which chunks the
    # 8 GB tier could evict. LMCache still reads every chunk from its tier
    # (disk reads included) before handing it to the connector.
    copy_on_retrieve: bool = False

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        if not self.copy_on_retrieve:
            return
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
