"""Both real managers accept the fast entropy source without layout/I/O changes."""

import asyncio
from importlib.metadata import PackageNotFoundError, version

import pytest

pytest.importorskip("lmcache")

from kvbench.core.config import KVBenchConfig  # noqa: E402
from kvbench.core.payload import SCRATCH_BYTES  # noqa: E402
from kvbench.kv.factory import create_kv_stack  # noqa: E402
from tests.integration.test_v1_storage import make_config  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("manager", ["lmcache", "hicache"])
async def test_fast_payload_disk_readback_and_namespace_separation(tmp_path, manager):
    if manager == "hicache":
        try:
            if version("sglang") != "0.5.21":
                pytest.skip("pinned SGLang source wheel required")
        except PackageNotFoundError:
            pytest.skip("pinned SGLang source wheel required")
    data = make_config(tmp_path).model_dump(mode="json")
    if manager == "hicache":
        data["kv"].update(stack="hicache", lmcache_config_file=None, hicache={
            "storage_path": str(tmp_path / "disk"), "page_size": 16,
            "host_size_gb": 0.0078125, "disk_size_gb": 0.03125, "min_free_bytes": 0,
        })
    previous_files = set()
    contents = []
    for generator in ("torch", "sfc64"):
        data["kv"]["payload_generator"] = generator
        stack = create_kv_stack(KVBenchConfig.model_validate(data))
        await stack.start()
        try:
            tokens = list(range(32))
            assert await stack.lookup(tokens) == 0
            await stack.store(tokens)
            await stack.flush()
            if manager == "lmcache":
                for engine in stack._engines:
                    await asyncio.to_thread(engine.clear, locations=["LocalCPUBackend"])
            assert await stack.lookup_pinned(tokens, "read") == 32
            assert await stack.retrieve(tokens) == 32
            await stack.unpin("read")
            files = set((tmp_path / "disk").glob("*")) - previous_files
            assert len(files) == 2
            payloads = [p.read_bytes() for p in sorted(files)]
            assert {len(value) for value in payloads} == {16 * 2048}
            assert payloads[0] != payloads[1]
            contents.append(payloads)
            previous_files.update(files)
            assert stack.stats.submitted_store_bytes == stack.stats.retrieved_bytes == 32 * 2048
        finally:
            await stack.unpin("read")
            await stack.close()
    assert contents[0] != contents[1]


async def test_lmcache_fast_scratch_is_included_in_memory_guard(tmp_path):
    data = make_config(tmp_path).model_dump(mode="json")
    data["kv"].update(payload_generator="sfc64", max_cpu_memory_gb=0.0078125)
    stack = create_kv_stack(KVBenchConfig.model_validate(data))
    assert stack.payload_scratch_bytes == SCRATCH_BYTES
    with pytest.raises(ValueError, match="CPU pools"):
        await stack.start()
    assert not stack._engines
