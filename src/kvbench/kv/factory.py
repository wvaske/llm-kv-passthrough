"""
KV stack factory.

Creates the KV management stack the servers talk to. There are no mock or
passthrough stacks here: the benchmark only measures real KV management
stacks driving real storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kvbench.kv.lmcache_stack import LMCacheStack

if TYPE_CHECKING:
    from kvbench.core.config import KVBenchConfig
    from kvbench.kv.base import KVStack

_PLANNED_STACKS = {
    "kvbm": "NVIDIA Dynamo KV Block Manager support is planned but not yet implemented",
}


def create_kv_stack(config: KVBenchConfig) -> KVStack:
    """Create a KV stack from the KV-Bench configuration.

    Args:
        config: KV-Bench configuration; config.kv selects and configures
            the stack, config.server.model_profile sizes the KV tensors.

    Returns:
        An unstarted KVStack (call await stack.start()).

    Raises:
        ValueError: If the stack is unknown or not yet supported.
    """
    stack = config.kv.stack

    if stack == "lmcache":
        return LMCacheStack(
            model_profile=config.server.model_profile,
            instance_id=config.instance_id,
            config_file=config.kv.lmcache_config_file,
        )

    if stack in _PLANNED_STACKS:
        raise ValueError(f"KV stack {stack!r} is not available: {_PLANNED_STACKS[stack]}")

    raise ValueError(f"Unknown KV stack: {stack!r}. Supported: lmcache")
