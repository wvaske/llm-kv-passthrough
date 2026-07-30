"""
Prometheus metrics for KV-Bench.

Exposes KV activity over time in Prometheus exposition format at /metrics:

- Server request/token counters and request-duration/TTFT histograms.
- KV stack operation counters (lookups, hit tokens, stores, retrieves).
- Live tier usage gauges from the running LMCache engine.
- Physical/logical I/O counters from the trace recorder when tracing is on.
- Warmup progress gauges.

Counters are exported via a custom collector that reads the live stats
objects on scrape — the server's hot path never touches prometheus_client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prometheus_client import CollectorRegistry, Histogram, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.registry import Collector

if TYPE_CHECKING:
    from collections.abc import Iterable

    from prometheus_client import Metric

CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class KVBenchCollector(Collector):
    """Scrape-time collector over the app's live stats objects.

    Attributes:
        source: Object exposing server/kv/warmup accessors (KVBenchApp).
    """

    def __init__(self, source: Any) -> None:
        self.source = source

    def collect(self) -> Iterable[Metric]:  # noqa: C901
        server = getattr(self.source, "_server", None)
        kv = getattr(self.source, "_kv", None)
        warmup = getattr(self.source, "_warmup", None)

        stats = getattr(server, "stats", None)
        if stats is not None:
            requests = CounterMetricFamily(
                "kvbench_requests_total",
                "Requests processed",
                labels=["status"],
            )
            requests.add_metric(["success"], getattr(stats, "requests_success", 0))
            requests.add_metric(["failed"], getattr(stats, "requests_failed", 0))
            yield requests

            tokens = CounterMetricFamily(
                "kvbench_tokens_total",
                "Tokens processed",
                labels=["kind"],
            )
            tokens.add_metric(["prompt"], getattr(stats, "prompt_tokens", 0))
            tokens.add_metric(["completion"], getattr(stats, "completion_tokens", 0))
            yield tokens

            chunks = CounterMetricFamily(
                "kvbench_cache_chunks_total",
                "Chunk-level cache results observed by the server",
                labels=["result"],
            )
            chunks.add_metric(["hit"], getattr(stats, "cache_hits", 0))
            chunks.add_metric(["miss"], getattr(stats, "cache_misses", 0))
            yield chunks

        kv_stats = getattr(kv, "stats", None)
        if kv_stats is not None:
            ops = CounterMetricFamily(
                "kvbench_kv_ops_total",
                "KV stack operations",
                labels=["op"],
            )
            ops.add_metric(["lookup"], kv_stats.lookups)
            ops.add_metric(["store"], kv_stats.stores)
            ops.add_metric(["retrieve"], kv_stats.retrieves)
            ops.add_metric(["error"], kv_stats.errors)
            yield ops

            kv_tokens = CounterMetricFamily(
                "kvbench_kv_tokens_total",
                "Tokens through the KV stack",
                labels=["kind"],
            )
            kv_tokens.add_metric(["lookup"], kv_stats.lookup_tokens)
            kv_tokens.add_metric(["hit"], kv_stats.hit_tokens)
            kv_tokens.add_metric(["stored"], kv_stats.stored_tokens)
            kv_tokens.add_metric(["retrieved"], kv_stats.retrieved_tokens)
            yield kv_tokens

            yield GaugeMetricFamily(
                "kvbench_kv_token_hit_rate",
                "Fraction of looked-up tokens found in cache",
                value=kv_stats.token_hit_rate,
            )

        usage_fn = getattr(kv, "usage_info", None)
        if usage_fn is not None:
            try:
                usage = usage_fn()
            except Exception:
                usage = {}
            if usage:
                usage_bytes = GaugeMetricFamily(
                    "kvbench_kv_tier_usage_bytes",
                    "Live tier usage reported by the KV stack",
                    labels=["backend"],
                )
                tier_keys = GaugeMetricFamily(
                    "kvbench_kv_tier_keys",
                    "Cached keys per tier",
                    labels=["backend"],
                )
                for backend, info in usage.items():
                    if "usage_bytes" in info:
                        usage_bytes.add_metric([backend], info["usage_bytes"])
                    if "keys" in info:
                        tier_keys.add_metric([backend], info["keys"])
                yield usage_bytes
                yield tier_keys

        capacity_fn = getattr(kv, "capacity_info", None)
        if capacity_fn is not None:
            try:
                capacity = capacity_fn()
            except Exception:
                capacity = None
            if capacity:
                cap = GaugeMetricFamily(
                    "kvbench_kv_tier_capacity_bytes",
                    "Configured tier capacity",
                    labels=["tier"],
                )
                cap.add_metric(["local_cpu"], capacity["local_cpu_capacity_bytes"])
                cap.add_metric(["local_disk"], capacity["local_disk_capacity_bytes"])
                yield cap

        recorder = getattr(kv, "trace_recorder", None)
        counters = getattr(recorder, "counters", None)
        if counters:
            io_ops = CounterMetricFamily(
                "kvbench_kv_trace_ops_total",
                "Traced KV operations (logical and physical)",
                labels=["event", "op", "backend"],
            )
            io_bytes = CounterMetricFamily(
                "kvbench_kv_trace_bytes_total",
                "Traced KV bytes (logical and physical)",
                labels=["event", "op", "backend"],
            )
            for (event, op, backend), (op_count, byte_count) in list(counters.items()):
                io_ops.add_metric([event, op, backend], op_count)
                io_bytes.add_metric([event, op, backend], byte_count)
            yield io_ops
            yield io_bytes

        if warmup is not None:
            status = warmup.status
            yield GaugeMetricFamily(
                "kvbench_warmup_running",
                "1 while a warmup run is in progress",
                value=1.0 if warmup.running else 0.0,
            )
            yield GaugeMetricFamily(
                "kvbench_warmup_stored_bytes",
                "Bytes stored by the current/last warmup run",
                value=status.stored_bytes,
            )
            yield GaugeMetricFamily(
                "kvbench_warmup_target_bytes",
                "Byte target of the current/last warmup run",
                value=status.target_bytes,
            )
            yield GaugeMetricFamily(
                "kvbench_warmup_evicting",
                "1 once warmup verified the cache is evicting (steady state)",
                value=1.0 if status.evicting else 0.0,
            )


class MetricsExporter:
    """Owns the registry, custom collector, and request histograms."""

    def __init__(self, source: Any) -> None:
        self.registry = CollectorRegistry()
        self.registry.register(KVBenchCollector(source))
        self.request_duration = Histogram(
            "kvbench_request_duration_seconds",
            "End-to-end chat completion duration",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "kvbench_time_to_first_token_seconds",
            "Time from request start to first streamed content chunk",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
