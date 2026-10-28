"""Regression test: LMCache 0.5.4 LocalDiskBackend counted re-stored chunks twice.

Storing the same chunk again (agentic traces re-send their parent's prefix) used to
add its size to ``current_cache_size`` a second time without a second dict entry.
The counter drifted above the real usage until the disk tier believed it was full
while nearly empty and stopped storing ("No eviction candidates found. Disk space
under pressure."); racing rewrites also made a later eviction hit a missing file.
``kvbench.kv.lmcache_compat.apply()`` backports the LMCache 0.5.6 fix.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

lmcache = pytest.importorskip("lmcache")

from kvbench.kv import lmcache_compat  # noqa: E402
from kvbench.kv.lmcache_stack import LMCacheStack  # noqa: E402

pytestmark = pytest.mark.integration


def _config(tmp_path: Path, disk_gb: float) -> Path:
    disk = tmp_path / "lmcache-disk"
    disk.mkdir()
    cfg = tmp_path / "lmcache.yaml"
    cfg.write_text(
        f"""\
chunk_size: 256
local_cpu: true
max_local_cpu_size: 0.0625
local_disk: "file://{disk}/"
max_local_disk_size: {disk_gb}
"""
    )
    return cfg


def _disk_backends(stack: LMCacheStack) -> list:
    out = []
    for engine in stack._engines:
        for backend in engine.storage_manager.storage_backends.values():
            if type(backend).__name__ == "LocalDiskBackend":
                out.append(backend)
    return out


def _settle(backend, timeout: float = 30.0) -> None:
    end = time.monotonic() + timeout
    while backend.disk_worker.put_tasks and time.monotonic() < end:
        time.sleep(0.05)


def _drift(backend) -> float:
    return backend.current_cache_size - sum(m.size for m in backend.dict.values())


def test_fix_is_active_on_pinned_lmcache() -> None:
    assert lmcache_compat.apply() or lmcache_compat._lmcache_version() != lmcache_compat.PATCHED_VERSION
    if lmcache_compat._lmcache_version() == lmcache_compat.PATCHED_VERSION:
        assert lmcache_compat.is_active()


@pytest.mark.asyncio
async def test_restored_chunks_do_not_inflate_disk_accounting(tmp_path: Path) -> None:
    """Re-storing the same prefix must not grow the disk tier's size counter."""
    stack = LMCacheStack(model_profile="llama-3.2-1b", instance_id=f"kvbench-restore-{uuid.uuid4().hex[:8]}",
                         config_file=_config(tmp_path, disk_gb=2.0))
    await stack.start()
    try:
        backends = _disk_backends(stack)
        assert backends, "LMCache disk tier not configured"
        chunk = stack.chunk_size
        prefix = list(range(1000, 1000 + 8 * chunk))
        await stack.store(prefix)
        for b in backends:
            _settle(b)
        base_entries = [len(b.dict) for b in backends]
        assert all(n > 0 for n in base_entries)
        # same prefix stored again, as sub-agents re-sending their parent's context do
        for _ in range(5):
            await stack.store(prefix)
        for b, n in zip(backends, base_entries, strict=True):
            _settle(b)
            assert len(b.dict) == n
            assert abs(_drift(b)) < 1, f"disk size counter drifted by {_drift(b)} bytes"
    finally:
        await stack.close()


@pytest.mark.asyncio
async def test_disk_tier_keeps_admitting_after_restores_under_eviction(tmp_path: Path) -> None:
    """With a tiny tier, re-stores plus new chunks must keep evicting and admitting (no stuck tier, no FileNotFoundError)."""
    stack = LMCacheStack(model_profile="llama-3.2-1b", instance_id=f"kvbench-evict-{uuid.uuid4().hex[:8]}",
                         config_file=_config(tmp_path, disk_gb=0.05))
    await stack.start()
    try:
        backends = _disk_backends(stack)
        chunk = stack.chunk_size
        shared = list(range(5000, 5000 + 4 * chunk))
        for i in range(12):
            await stack.store(shared)  # re-stored every round
            await stack.store(shared + list(range(100_000 * (i + 1), 100_000 * (i + 1) + 4 * chunk)))
        for b in backends:
            _settle(b)
            assert abs(_drift(b)) < 1, f"disk size counter drifted by {_drift(b)} bytes"
            assert b.current_cache_size <= b.max_cache_size
            assert len(b.dict) > 0, "disk tier stopped admitting chunks"
            on_disk = {p.name for p in Path(b.path).glob("*.pt")}
            assert {Path(m.path).name for m in b.dict.values()} <= on_disk
        assert stack.stats.errors == 0
    finally:
        await stack.close()
