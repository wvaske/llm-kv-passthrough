"""Unit tests for the KV stack factory and stack lifecycle guards."""

from __future__ import annotations

import pytest

from kvbench.core.config import KVBenchConfig, KVStackConfig
from kvbench.kv.factory import create_kv_stack
from kvbench.kv.lmcache_stack import LMCacheStack


class TestCreateKvStack:
    """Tests for create_kv_stack."""

    def test_lmcache_stack_created(self) -> None:
        """The default stack is an (unstarted) LMCacheStack."""
        config = KVBenchConfig(instance_id="factory-test")
        stack = create_kv_stack(config)
        assert isinstance(stack, LMCacheStack)
        assert stack.instance_id == "factory-test"
        assert stack.model_profile == config.server.model_profile

    def test_kvbm_accepted_in_config_rejected_at_creation(self) -> None:
        """'kvbm' is valid configuration but fails stack creation with the
        documented reason (its data plane requires the CUDA driver)."""
        config = KVBenchConfig(kv=KVStackConfig(stack="kvbm"))
        with pytest.raises(ValueError, match="CUDA driver"):
            create_kv_stack(config)

    def test_unknown_stack_rejected_by_config(self) -> None:
        """Stacks that are neither supported nor planned fail validation."""
        with pytest.raises(ValueError):
            KVStackConfig(stack="mooncake")


class TestLMCacheStackGuards:
    """Lifecycle guards that don't require lmcache installed."""

    def test_chunk_size_before_start_raises(self) -> None:
        stack = LMCacheStack(model_profile="llama-3.1-8b")
        with pytest.raises(RuntimeError, match="not started"):
            _ = stack.chunk_size

    @pytest.mark.asyncio
    async def test_ops_before_start_raise(self) -> None:
        stack = LMCacheStack(model_profile="llama-3.1-8b")
        with pytest.raises(RuntimeError, match="not started"):
            await stack.lookup([1, 2, 3])

    @pytest.mark.asyncio
    async def test_start_without_lmcache_gives_install_hint(self) -> None:
        """When the lmcache package is missing, start() fails with an
        actionable install hint rather than a bare ImportError."""
        pytest.importorskip("kvbench")
        try:
            import lmcache  # noqa: F401

            pytest.skip("lmcache is installed in this environment")
        except ImportError:
            pass

        stack = LMCacheStack(model_profile="llama-3.1-8b")
        with pytest.raises(RuntimeError, match="kvbench\\[lmcache\\]"):
            await stack.start()

    @pytest.mark.asyncio
    async def test_close_before_start_is_noop(self) -> None:
        stack = LMCacheStack(model_profile="llama-3.1-8b")
        await stack.close()
