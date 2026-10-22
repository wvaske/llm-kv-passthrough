"""Read-only campaign observer for live Grafana progress, ETA and health.

Runs separately from the frozen benchmark process. Reads bounded metadata and
the atomic NVMe snapshot; never scans KV data files or changes the workload.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

from kvbench.benchmark.nvme import influx_line, write_latest

PROPERTIES = ("LoadState", "ActiveState", "SubState", "Result", "MainPID", "ExecMainStatus",
              "MemoryCurrent", "MemoryPeak", "MemoryMax", "CPUUsageNSec",
              "ExecMainStartTimestampMonotonic", "ExecMainExitTimestampMonotonic")


def read_json(path: Path) -> dict[str, Any]:
    # The benchmark writes small manifests at phase boundaries; tolerate a
    # concurrent rewrite rather than treating a torn read as a failed workload.
    for attempt in range(3):
        try:
            if path.stat().st_size > 1024**2:
                raise ValueError(f"Oversized campaign metadata: {path}")
            result = json.loads(path.read_text())
            if not isinstance(result, dict):
                raise ValueError("Expected campaign metadata object")
            return result
        except json.JSONDecodeError:
            if attempt == 2:
                raise
            time.sleep(0.05)
    raise AssertionError("unreachable")


def snapshot(path: Path) -> dict[str, Any]:
    """Read our numeric/bool NVMe snapshot format, not arbitrary line protocol."""
    if path.stat().st_size > 65536:
        raise ValueError("Oversized NVMe snapshot")
    head, raw_fields, raw_time = path.read_text().strip().rsplit(" ", 2)
    if not head.startswith("kvbench_nvme,"):
        raise ValueError("Expected a kvbench_nvme snapshot")
    tags = dict(part.split("=", 1) for part in head.split(",")[1:])
    fields: dict[str, Any] = {}
    for part in raw_fields.split(","):
        key, value = part.split("=", 1)
        fields[key] = value == "true" if value in ("true", "false") else float(value)
        if not isinstance(fields[key], bool) and not math.isfinite(fields[key]):
            raise ValueError("Nonfinite NVMe snapshot field")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tags["run_id"]):
        raise ValueError("Unsafe telemetry run ID")
    return {"tags": tags, "fields": fields, "timestamp_ns": int(raw_time)}


def unit_info(name: str) -> dict[str, str]:
    if not re.fullmatch(r"kvbench-waf-[A-Za-z0-9_-]+\.service", name):
        raise ValueError("Unexpected campaign unit")
    args = ["systemctl", "show", name, *[f"--property={key}" for key in PROPERTIES]]
    result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not fields:
        raise RuntimeError(f"Cannot inspect campaign unit {name}")
    return {"name": name, **fields}


def select_unit(revision: str | None) -> dict[str, str]:
    result = subprocess.run(["systemctl", "list-units", "--all", "--type=service", "--output=json",
                             "kvbench-waf-*.service"], capture_output=True, text=True, check=True, timeout=5)
    active = [unit_info(row["unit"]) for row in json.loads(result.stdout)
              if row["active"] in ("active", "activating", "deactivating")]
    expected = f"kvbench-waf-{revision[:8]}.service" if revision else None
    for unit in active:
        if unit["name"] == expected:
            return unit
    if active:
        return max(active, key=lambda row: int(row.get("ExecMainStartTimestampMonotonic", "0")))
    return unit_info(expected) if expected else {"name": "unknown", "ActiveState": "inactive"}


def first_sample_time(path: Path, *, last_baseline: bool = False) -> float | None:
    if not path.exists():
        return None
    with path.open("rb") as stream:
        if last_baseline:
            offset = max(0, path.stat().st_size - 256 * 1024)
            stream.seek(offset)
            if offset:
                stream.readline()
            lines = stream.read(256 * 1024).splitlines()
        else:
            lines = [stream.readline(65536)]
    baseline = None
    for line in lines:
        try:
            record = json.loads(line)
            fields = record["fields"]
            if fields.get("sample_valid") and not fields.get("interval_valid") and fields.get("run_host_bytes") == 0:
                baseline = record["sample"]["timestamp_ns"] / 1e9
        except (ValueError, KeyError, TypeError):
            continue
    return baseline


class RollingRate:
    def __init__(self) -> None:
        self.key: object = None
        self.points: deque[tuple[float, float]] = deque()

    def update(self, key: object, timestamp: float, counter: float, fallback: float | None) -> float | None:
        if key != self.key or (self.points and counter < self.points[-1][1]):
            self.points.clear()
            self.key = key
        if not self.points or timestamp > self.points[-1][0]:
            self.points.append((timestamp, counter))
        while len(self.points) > 2 and self.points[1][0] < timestamp - 300:
            self.points.popleft()
        if len(self.points) >= 2 and timestamp - self.points[0][0] >= 20:
            return (counter - self.points[0][1]) / (timestamp - self.points[0][0])
        return fallback


def phase_progress(completed: float, target: float, rate: float | None,
                   *, elapsed: float = 0, min_seconds: float = 0, eta_valid: bool = True) -> dict[str, Any]:
    remaining = max(0.0, target - completed)
    result: dict[str, Any] = {"phase_completed_bytes": max(0.0, completed), "phase_target_bytes": target,
                              "phase_remaining_bytes": remaining,
                              "phase_progress": min(1.0, max(0.0, completed / target)) if target > 0 else 1.0}
    if eta_valid and rate is not None and rate > 0:
        result["phase_eta_seconds"] = max(remaining / rate, min_seconds - elapsed, 0.0)
    return result


class CampaignWatcher:
    def __init__(self, artifacts: Path, telemetry: Path, serial: str, device: str):
        self.artifacts, self.telemetry = artifacts, telemetry
        self.serial, self.device = serial, device
        self.rate = RollingRate()
        self.previous_cpu: tuple[str, float, float] | None = None

    def collect(self) -> dict[str, Any]:
        now, mono = time.time(), time.monotonic()
        fields: dict[str, Any] = {"observer_epoch_seconds": now, "observer_ok": True}
        try:
            sample = snapshot(self.telemetry)
            if sample["tags"].get("serial") != self.serial or sample["tags"].get("device") != self.device:
                raise ValueError("Telemetry drive differs from the watched drive")
            run_id = sample["tags"]["run_id"]
            directory = self.artifacts / run_id
            is_preparation = run_id.startswith("prepare-")
            metadata = read_json(directory / ("campaign.json" if is_preparation else "experiment.json"))
            revision = metadata.get("source_revision")
            unit = select_unit(revision)
            fields.update(self.build(sample, metadata, unit, directory, now, mono, is_preparation))
        except Exception as exc:
            fields.update(observer_ok=False, state_label="OBSERVER ERROR", phase_label="Status unavailable",
                          last_error=str(exc), telemetry_ok=False, qualification_label="Unknown")
        return fields

    def build(self, sample: dict[str, Any], metadata: dict[str, Any], unit: dict[str, str],
              directory: Path, now: float, mono: float, preparing: bool) -> dict[str, Any]:
        config = metadata["experiment" if preparing else "config"]
        order = config["order"]
        data, tags = sample["fields"], sample["tags"]
        timestamp = sample["timestamp_ns"] / 1e9
        age = now - timestamp
        running = unit.get("ActiveState") == "active" and unit.get("SubState") == "running"
        starting = unit.get("ActiveState") == "activating"
        stopping = unit.get("ActiveState") == "deactivating"
        started = int(unit.get("ExecMainStartTimestampMonotonic", "0")) / 1e6
        expected_unit = f"kvbench-waf-{str(metadata.get('source_revision', ''))[:8]}.service"
        matched = (unit["name"] == expected_unit and not starting
                   and not (running and started and timestamp < now - (mono - started) - 2))
        fresh = -2 <= age <= max(30, config["interval_seconds"] * 3)
        valid = fresh and data.get("sample_valid", False) and data.get("run_valid", False) and matched
        outcome = metadata.get("status")
        complete = outcome == "complete"
        failed = outcome == "failed" or unit.get("ActiveState") == "failed"
        state = ("COMPLETE" if complete else "FAILED" if failed else "RUNNING" if running
                 else "STARTING" if starting else "STOPPING" if stopping else "STOPPED")
        if running and not valid:
            state = "RUNNING — TELEMETRY STALE/INVALID"
        fields: dict[str, Any] = {
            "unit": unit["name"], "unit_running": running, "state_label": state,
            "source_revision": str(metadata.get("source_revision", "unknown")),
            "telemetry_run_id": tags["run_id"], "telemetry_age_seconds": max(0, age),
            "telemetry_ok": bool(valid), "artifact_path": str(directory),
            "last_error": metadata.get("error", ""), "planned_arms": len(order),
            "completed_arms": len(metadata.get("arms", [])),
            "qualification_label": "QUALIFIED" if complete and metadata.get("qualified") else
                                   "SMOKE — NOT QUALIFIED" if config.get("mode") == "smoke" else
                                   "FAILED" if failed else "PENDING — NO ENDURANCE RESULT",
            "workload_streams": config["concurrency"],
        }
        if failed and not fields["last_error"]:
            fields["last_error"] = f"systemd result: {unit.get('Result', 'unknown')}; exit status: {unit.get('ExecMainStatus', 'unknown')}"
        elif running and not valid:
            fields["last_error"] = (f"Latest telemetry is {max(0, age):.0f} seconds old" if not fresh
                                    else "NVMe sample/counter validation failed")
        for prop, name in (("MemoryCurrent", "memory_bytes"), ("MemoryPeak", "memory_peak_bytes"), ("MemoryMax", "memory_limit_bytes")):
            value = unit.get(prop, "")
            if value.isdecimal() and int(value) < 2**64 - 1:
                fields[name] = int(value)
        cpu = unit.get("CPUUsageNSec", "")
        if cpu.isdecimal() and int(cpu) < 2**64 - 1:
            used = int(cpu) / 1e9
            if self.previous_cpu and self.previous_cpu[0] == unit["name"]:
                _, previous_time, previous_used = self.previous_cpu
                if mono > previous_time and used >= previous_used:
                    fields["cpu_cores"] = (used - previous_used) / (mono - previous_time)
            self.previous_cpu = (unit["name"], mono, used)
        exited = int(unit.get("ExecMainExitTimestampMonotonic", "0")) / 1e6
        if started:
            fields["campaign_elapsed_seconds"] = max(0, (exited or mono) - started)
        if not matched:
            fields.update(phase_label="Waiting for the new campaign's telemetry", next_phase_label="Waiting",
                          state_label="RUNNING — WAITING FOR TELEMETRY" if running else unit.get("ActiveState", "unknown").upper(),
                          last_error="Latest telemetry belongs to a previous campaign", qualification_label="PENDING")
            return fields

        # Only publish live rates/WAF from a fresh, valid sample. Historical
        # progress can remain visible after a stop, with explicit state/age.
        if valid and running:
            for key in ("host_bytes_per_second", "nand_bytes_per_second", "waf_interval", "waf_run",
                        "host_bytes_total", "nand_bytes_total", "run_host_bytes", "run_nand_bytes",
                        "drive_fills", "temperature_celsius", "occupancy"):
                if key in data:
                    fields[key] = data[key]
        device = metadata["device"]
        capacity = device["capacity_bytes"]
        desired_occupancy = max(0.85, config["min_occupancy"])
        fields["occupancy_target"] = desired_occupancy
        done = len(metadata.get("arms", []))
        arm = min(done, len(order) - 1)
        raw_phase = tags["phase"]
        if not preparing and raw_phase != "idle-baseline" and done and not (directory / f"{arm:02d}-{order[arm]}").exists():
            arm = done - 1
        phase = raw_phase.removesuffix("-drain")
        draining = raw_phase.endswith("-drain")
        fields["current_arm"] = 0 if preparing else arm + 1
        fields["phase_label"] = (f"Preparing drive to {desired_occupancy:.0%} occupancy" if preparing else
                                 f"Arm {arm + 1}/{len(order)} · {order[arm]} · {raw_phase}")
        fields["next_phase_label"] = (f"Arm 1/{len(order)} · {order[0]} · precondition" if preparing else
                                      "Measured workload" if phase == "precondition" else
                                      "Next arm / campaign completion" if phase == "measure" else "Preconditioning")
        for index, adapter in enumerate(order):
            status = "Complete" if index < done else ("Active" if running else state.title()) if not preparing and index == arm else "Waiting"
            fields[f"arm_{index + 1:02d}_status"] = f"{adapter} — {status}"

        rate_key = (tags["run_id"], arm, phase)
        if preparing:
            target = max(0, desired_occupancy * device["filesystem_bytes"] - device["used_bytes"])
            fields["phase_target_bytes"] = target
            if "occupancy" in data and data.get("sample_valid"):
                completed = max(0, data["occupancy"] * device["filesystem_bytes"] - device["used_bytes"])
                rate = self.rate.update(rate_key, timestamp, completed, data.get("host_bytes_per_second")) if valid else None
                fields.update(phase_progress(completed, target, rate, eta_valid=bool(valid and running)))
            phase_start = first_sample_time(directory / "samples.jsonl")
            fields["phase_goal_label"] = f"{desired_occupancy:.0%} filesystem occupancy"
            fields["comparison_progress"] = 0.0
        else:
            phase_file = directory / f"{arm:02d}-{order[arm]}" / f"{phase}.jsonl"
            if phase == "idle-baseline":
                phase_file = directory / "idle.jsonl"
            phase_start = first_sample_time(phase_file, last_baseline=phase == "idle-baseline")
            elapsed = max(0, now - phase_start) if phase_start is not None else 0
            if phase in ("precondition", "measure"):
                fills = config["precondition_drive_fills" if phase == "precondition" else "drive_fills"]
                target = config["smoke_host_bytes"] if config.get("mode") == "smoke" else capacity * fills
                fields["phase_target_bytes"] = target
                if "run_host_bytes" in data and data.get("run_valid"):
                    completed = data["run_host_bytes"]
                    rate = self.rate.update(rate_key, timestamp, completed, data.get("host_bytes_per_second")) if valid else None
                    minimum = config["smoke_seconds"] if config.get("mode") == "smoke" else config["min_seconds"]
                    fields.update(phase_progress(completed, target, rate, elapsed=elapsed, min_seconds=minimum,
                                                 eta_valid=bool(valid and running and not draining and phase_start)))
                fields["phase_goal_label"] = f"{fills:g} observed host drive fills" if config.get("mode") != "smoke" else "Bounded smoke check"
                fields["phase_target_drive_fills"] = fills
            elif phase == "idle-baseline":
                fields["phase_goal_label"] = f"{config['idle_seconds']:g} seconds quiet baseline"
                if phase_start is not None:
                    fields["phase_progress"] = min(1.0, elapsed / config["idle_seconds"])
                    if valid and running:
                        fields["phase_eta_seconds"] = max(0, config["idle_seconds"] - elapsed)
            if draining:
                fields["phase_goal_label"] = f"Drain pending I/O, sync filesystem, then {config['drain_seconds']:g}s idle tail"
            total_goal = len(order) * (config["precondition_drive_fills"] + config["drive_fills"]) * capacity
            accounted = 0.0
            for index, adapter in enumerate(order):
                for stage in ("precondition", "measure"):
                    goal = capacity * config["precondition_drive_fills" if stage == "precondition" else "drive_fills"]
                    path = directory / f"{index:02d}-{adapter}" / f"{stage}-summary.json"
                    if path.exists() and read_json(path).get("qualified"):
                        accounted += goal
                    elif index == arm and stage == phase and data.get("run_valid"):
                        accounted += min(goal, max(0, data.get("run_host_bytes", 0)))
            fields["comparison_progress"] = accounted / total_goal if total_goal else 0
        if phase_start is not None:
            fields["phase_elapsed_seconds"] = max(0, (now if running else timestamp) - phase_start)
        if complete:
            fields.update(phase_label="Campaign complete", next_phase_label="Review final artifacts",
                          phase_progress=1.0, phase_eta_seconds=0.0)
            if metadata.get("qualified"):
                fields["comparison_progress"] = 1.0
        if "phase_eta_seconds" in fields:
            fields["phase_eta_unix_ms"] = (timestamp if complete else now + fields["phase_eta_seconds"]) * 1000
        fields["last_error"] = fields["last_error"] or "No errors"
        return fields


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--device", default="nvme6")
    parser.add_argument("--output", required=True, type=Path, help="Atomic status JSON; companion .stream feeds Telegraf")
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("interval must be >=1 second")
    watcher = CampaignWatcher(args.artifacts, args.telemetry, args.serial, args.device)
    tags = {"host": socket.gethostname().split(".")[0], "device": args.device, "serial": args.serial}
    previous = None
    while True:
        started = time.monotonic()
        fields = watcher.collect()
        now = time.time_ns()
        write_latest(args.output, json.dumps({"timestamp_ns": now, "tags": tags, "fields": fields}, indent=2))
        with args.output.with_suffix(".stream").open("a") as stream:
            stream.write(influx_line(fields, tags, now, measurement="kvbench_campaign", include_strings=True))
        state = (fields.get("unit"), fields.get("state_label"), fields.get("phase_label"), fields.get("last_error"))
        if state != previous:
            print(json.dumps({"state": state, "progress": fields.get("phase_progress"),
                              "eta_seconds": fields.get("phase_eta_seconds")}), flush=True)
            previous = state
        if args.once:
            break
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
