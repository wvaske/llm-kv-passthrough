"""Fast, fresh high-entropy bytes with bounded scratch space.

SFC64 emits 64 random bits per output word instead of one RNG draw per byte.
No repeated random pool, compression-friendly pattern or prompt-sized temporary
is used. This is a statistical PRNG for synthetic payloads, not cryptography.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

PayloadGenerator = Literal["torch", "sfc64"]
SCRATCH_BYTES = 256 * 1024
SFC64_FORMAT = "sfc64-u64le-v1"


def scratch_bytes(generator: PayloadGenerator) -> int:
    return SCRATCH_BYTES if generator == "sfc64" else 0


def fill_sfc64(destination: NDArray[np.uint8], generator: np.random.SFC64) -> None:
    """Fill a contiguous CPU byte view in place; allocate <=256 KiB scratch.

    The caller can reuse a generator for a continuous filler stream, or create
    one per deterministic cache key. Odd byte counts are supported exactly.
    """
    import numpy as np

    if destination.dtype != np.uint8 or not destination.flags.c_contiguous or not destination.flags.writeable:
        raise ValueError("Payload destination must be a writable contiguous uint8 array")
    flat = destination.reshape(-1)
    for offset in range(0, flat.size, SCRATCH_BYTES):
        count = min(SCRATCH_BYTES, flat.size - offset)
        words = generator.random_raw((count + 7) // 8)
        # Define byte order explicitly rather than depending on host endianness.
        if not np.little_endian:
            words.byteswap(inplace=True)
        flat[offset:offset + count] = words.view(np.uint8)[:count]
        del words  # do not overlap consecutive scratch allocations
