#!/usr/bin/env python3
"""Maintain a flat JSON feed of KV-Bench AgentX test points for a live dashboard.

The runner (run_agentx_matrix.sh) calls this at every phase boundary so the
feed always reflects what is running right now. Two files live in --out:

  runs.json    one row per test point (model, hardware, workload, phase, results)
  events.json  append-only timestamped log of phase changes and findings

Both are plain JSON arrays of flat objects so any dashboard (claude.ai
Artifact, Grafana JSON datasource, a spreadsheet) can read them directly.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _load(path: Path) -> list[dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1, sort_keys=True))
    os.replace(tmp, path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _diskstats(path: str | None) -> tuple[float, float]:
    """Return (read_bytes, write_bytes) from a saved /proc/diskstats line (512 B sectors)."""
    if not path:
        return 0.0, 0.0
    try:
        fields = Path(path).read_text().split()
        return float(fields[5]) * 512, float(fields[9]) * 512
    except (FileNotFoundError, IndexError, ValueError):
        return 0.0, 0.0


def _aiperf_metrics(artifact_dir: str | None) -> dict[str, Any]:
    """Pull request/latency aggregates out of AIPerf's profile export, whatever its layout."""
    if not artifact_dir:
        return {}
    files = sorted(glob.glob(os.path.join(artifact_dir, "**", "profile_export_aiperf.json"), recursive=True))
    if not files:
        return {}
    doc = _read_json(files[-1])
    out: dict[str, Any] = {}

    def pick(name: str, *keys: str) -> None:
        node: Any = doc
        for key in keys:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node = node.get("avg", node.get("value"))
        if isinstance(node, (int, float)):
            out[name] = node

    pick("aiperf_request_count", "request_count")
    pick("aiperf_request_throughput_rps", "request_throughput")
    pick("aiperf_output_tps", "output_token_throughput")
    pick("aiperf_ttft_ms", "time_to_first_token")
    pick("aiperf_itl_ms", "inter_token_latency")
    pick("aiperf_request_latency_ms", "request_latency")
    pick("aiperf_error_count", "error_request_count")
    return out


def cmd_start(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [r for r in _load(out / "runs.json") if r.get("run_id") != a.run_id]
    row = {
        "run_id": a.run_id, "model": a.model, "gpu": a.gpu, "tp_size": a.tp, "active_ranks": a.ranks,
        "workload": a.workload, "host": a.host, "device": a.device, "mount": a.mount,
        "users": a.users, "duration_s": a.duration, "chunk_tokens": a.chunk,
        "cpu_tier_gib": a.cpu_gib, "disk_tier_gib": a.disk_gib,
        "phase": "starting", "status": "running", "started_at": _now(), "ended_at": None,
        "kv_bytes_per_token_model": None, "kv_bytes_per_token_rank0": None,
        "warmup_gib": None, "requests_total": None, "requests_failed": None,
        "token_hit_rate": None, "stores": None, "retrieves": None,
        "kv_submitted_gib": None, "kv_retrieved_gib": None,
        "disk_read_gib": None, "disk_write_gib": None, "disk_write_amp_vs_kv": None,
        "note": a.note or "",
    }
    runs.append(row)
    _save(out / "runs.json", runs)
    cmd_event(argparse.Namespace(out=a.out, run_id=a.run_id, phase="starting", level="info",
                                 message=f"{a.model} on {a.gpu} x{a.tp} (rank {a.ranks}), {a.workload}"))


def cmd_phase(a: argparse.Namespace) -> None:
    out = Path(a.out)
    runs = _load(out / "runs.json")
    for r in runs:
        if r.get("run_id") == a.run_id:
            r["phase"] = a.phase
            if a.set:
                for kv in a.set:
                    k, _, v = kv.partition("=")
                    try:
                        r[k] = json.loads(v)
                    except json.JSONDecodeError:
                        r[k] = v
    _save(out / "runs.json", runs)
    cmd_event(argparse.Namespace(out=a.out, run_id=a.run_id, phase=a.phase, level="info", message=a.message or a.phase))


def cmd_finish(a: argparse.Namespace) -> None:
    out = Path(a.out)
    runs = _load(out / "runs.json")
    manifest = _read_json(a.manifest)
    state = _read_json(a.state)
    stats = _read_json(a.stats)
    kv = state.get("kv_stats", {})
    cap = state.get("capacity", {})
    r0, w0 = _diskstats(a.disk_before)
    r1, w1 = _diskstats(a.disk_after)
    submitted = float(kv.get("submitted_store_bytes") or 0)
    gib = 2**30
    for r in runs:
        if r.get("run_id") != a.run_id:
            continue
        r.update({
            "phase": "done" if a.status == "ok" else a.status, "status": a.status, "ended_at": _now(),
            "kv_bytes_per_token_model": cap.get("modeled_stored_bytes_per_token") or cap.get("bytes_per_token"),
            "kv_bytes_per_token_rank0": cap.get("active_stored_bytes_per_token"),
            "chunk_bytes_rank0": cap.get("chunk_bytes"),
            "requests_total": stats.get("requests_total"), "requests_failed": stats.get("requests_failed"),
            "token_hit_rate": kv.get("token_hit_rate"), "stores": kv.get("stores"), "retrieves": kv.get("retrieves"),
            "lookup_tokens": kv.get("lookup_tokens"), "hit_tokens": kv.get("hit_tokens"),
            "kv_submitted_gib": round(submitted / gib, 3), "kv_retrieved_gib": round(float(kv.get("retrieved_bytes") or 0) / gib, 3),
            "disk_read_gib": round((r1 - r0) / gib, 3), "disk_write_gib": round((w1 - w0) / gib, 3),
            "disk_write_amp_vs_kv": round((w1 - w0) / submitted, 3) if submitted > 0 else None,
            "warmup_gib": a.warmup_gib,
            "kvbench_version": manifest.get("fidelity"),
            "note": (r.get("note") or "") + (f" {a.note}" if a.note else ""),
        })
        r.update(_aiperf_metrics(a.aiperf_dir))
    _save(out / "runs.json", runs)
    cmd_event(argparse.Namespace(out=a.out, run_id=a.run_id, phase=a.status, level="info" if a.status == "ok" else "error",
                                 message=a.note or f"finished with status {a.status}"))


def cmd_event(a: argparse.Namespace) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    events = _load(out / "events.json")
    events.append({"at": _now(), "run_id": a.run_id, "phase": a.phase, "level": a.level, "message": a.message})
    _save(out / "events.json", events[-500:])


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.set_defaults(fn=cmd_start)
    for name in ("out", "run-id", "model", "gpu", "workload", "host", "device", "mount"):
        s.add_argument(f"--{name}", required=name in ("out", "run-id", "model"), default="")
    s.add_argument("--tp", type=int, default=1); s.add_argument("--ranks", default="all")
    s.add_argument("--users", type=int, default=0); s.add_argument("--duration", type=int, default=0)
    s.add_argument("--chunk", type=int, default=256); s.add_argument("--cpu-gib", type=float, default=0)
    s.add_argument("--disk-gib", type=float, default=0); s.add_argument("--note", default="")
    s = sub.add_parser("phase"); s.set_defaults(fn=cmd_phase)
    s.add_argument("--out", required=True); s.add_argument("--run-id", required=True); s.add_argument("--phase", required=True)
    s.add_argument("--message", default=""); s.add_argument("--set", action="append", help="key=json-value to store on the row")
    s = sub.add_parser("finish"); s.set_defaults(fn=cmd_finish)
    s.add_argument("--out", required=True); s.add_argument("--run-id", required=True); s.add_argument("--status", default="ok")
    for name in ("manifest", "state", "stats", "aiperf-dir", "disk-before", "disk-after"):
        s.add_argument(f"--{name}", default=None)
    s.add_argument("--warmup-gib", type=float, default=None); s.add_argument("--note", default="")
    s = sub.add_parser("event"); s.set_defaults(fn=cmd_event)
    s.add_argument("--out", required=True); s.add_argument("--run-id", default=""); s.add_argument("--phase", default="")
    s.add_argument("--level", default="info"); s.add_argument("--message", required=True)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
