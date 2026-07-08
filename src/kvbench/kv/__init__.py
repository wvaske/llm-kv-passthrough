"""
KV management stack integrations for KV-Bench.

KV-Bench never performs storage I/O itself. All KV cache operations
(lookup, store, retrieve) go through a real KV management stack, which
owns the storage control plane end to end: chunking, hashing, tiering,
eviction, and every byte written to CPU RAM, disk, or remote backends.

Supported stacks:
- lmcache: the real LMCache library, configured through LMCache's own
  application configuration (config file or LMCACHE_* environment
  variables).

Planned stacks:
- kvbm: NVIDIA Dynamo's KV Block Manager.
"""

from kvbench.kv.base import KVStack, KVStackStats
from kvbench.kv.factory import create_kv_stack

__all__ = [
    "KVStack",
    "KVStackStats",
    "create_kv_stack",
]
