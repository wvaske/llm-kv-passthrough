import zlib

import numpy as np
import pytest

from kvbench.core.payload import SCRATCH_BYTES, fill_sfc64


@pytest.mark.parametrize("size", [0, 1, 7, 8, 9, 890, SCRATCH_BYTES, SCRATCH_BYTES + 9])
def test_sfc64_exact_bytes_and_unaligned_tail(size):
    guarded = np.full(size + 10, 77, dtype=np.uint8)
    fill_sfc64(guarded[5:5 + size], np.random.SFC64(42))
    expected = np.random.SFC64(42).random_raw((size + 7) // 8).astype("<u8", copy=False).view(np.uint8)[:size]
    np.testing.assert_array_equal(guarded[5:5 + size], expected)
    assert (guarded[:5] == 77).all() and (guarded[size + 5:] == 77).all()


def test_scratch_bound_and_fresh_incompressible_blocks():
    class Spy:
        def __init__(self):
            self.rng = np.random.SFC64(123)
            self.sizes = []

        def random_raw(self, size):
            self.sizes.append(size * 8)
            return self.rng.random_raw(size)

    spy = Spy()
    data = np.empty(2 * 1024**2, dtype=np.uint8)
    fill_sfc64(data, spy)
    first = data.tobytes()
    assert max(spy.sizes) <= SCRATCH_BYTES
    assert len(zlib.compress(first)) >= len(first) * 0.99
    assert first[:SCRATCH_BYTES] != first[SCRATCH_BYTES:2 * SCRATCH_BYTES]
    fill_sfc64(data, spy)
    assert data.tobytes() != first  # a persistent filler stream does not repeat


def test_rejects_silent_copy_destinations():
    with pytest.raises(ValueError, match="contiguous uint8"):
        fill_sfc64(np.empty(128, dtype=np.uint8)[::2], np.random.SFC64(1))
    with pytest.raises(ValueError, match="contiguous uint8"):
        fill_sfc64(np.empty(128, dtype=np.float64), np.random.SFC64(1))
