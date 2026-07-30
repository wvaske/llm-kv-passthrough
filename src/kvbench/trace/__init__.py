"""
KV I/O tracing and workload derivation.

This package records the logical operations a KV management stack performs
(put/get/contains/remove, keyed by chunk hash) and the physical file I/O
they turn into (write/read of chunk files, with sizes, threads, and
timings). The resulting JSONL trace is the ground truth for deriving
representative storage workloads — see `kvbench trace2fio`.
"""

from kvbench.trace.recorder import TraceRecorder, install_lmcache_trace

__all__ = ["TraceRecorder", "install_lmcache_trace"]
