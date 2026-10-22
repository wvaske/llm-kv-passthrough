"""Read-only NVMe/OCP counter collection with integer-safe WAF deltas.

Standard SMART data units are 512,000 bytes; OCP physical media units written
are bytes (128 bit). Neither Linux diskstats nor host writes substitutes for
the NAND counter. Unknown schemas/units fail closed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

OCP_GUID = "afd514c97c6f4f9ca4f2bfea2810afc5"


def integer(value: Any) -> int:
    if type(value) is int:
        result = value
    elif isinstance(value, str):
        text = value.replace(",", "").strip()
        result = int(text, 16 if text.startswith("0x") else 10)
    else:
        raise ValueError("Counter must be an exact integer, not a float or boolean")
    if result < 0:
        raise ValueError("Negative counter")
    return result


def decode_counters(smart: dict[str, Any], ocp: dict[str, Any]) -> tuple[int, int]:
    host = integer(smart["data_units_written"]) * 512000
    guid = str(ocp.get("Log page GUID", "")).lower().removeprefix("0x")
    if guid != OCP_GUID or integer(ocp["Log page version"]) not in (2, 3, 4):
        raise ValueError("Unknown OCP SMART log GUID/version; NAND units are not established")
    value = ocp["Physical media units written"]
    if isinstance(value, dict):
        hi, lo = integer(value["hi"]), integer(value["lo"])
        if max(hi, lo) >= 2**64:
            raise ValueError("Invalid OCP 128-bit counter halves")
        nand = (hi << 64) | lo
    else:
        nand = integer(value)
    if nand >= 2**128:
        raise ValueError("OCP counter exceeds 128 bits")
    return host, nand


def command_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=20)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object from {args[0]}")
    return value


@dataclass
class Sample:
    timestamp_ns: int
    monotonic_ns: int
    serial: str
    model: str
    firmware: str
    host_bytes: int
    nand_bytes: int
    power_cycles: int
    capacity_bytes: int
    occupancy: float | None
    temperature_kelvin: int
    raw_smart: dict[str, Any]
    raw_ocp: dict[str, Any]


class NVMeProbe:
    def __init__(self, device: str, mount: Path | None = None, sudo: bool = False, expected_serial: str | None = None):
        if not re.fullmatch(r"/dev/nvme[0-9]+", device):
            raise ValueError("Use an NVMe controller, e.g. /dev/nvme6; all counters are controller-wide")
        self.device, self.mount, self.sudo = device, mount, sudo
        self.expected_serial = expected_serial

    def command(self, *args: str) -> dict[str, Any]:
        prefix = ["sudo", "-n"] if self.sudo else []
        return command_json([*prefix, "nvme", *args, self.device, "-o", "json"])

    def sample(self) -> Sample:
        identity = self.command("id-ctrl")
        if self.expected_serial is not None and identity["sn"].strip() != self.expected_serial:
            raise ValueError("NVMe controller serial changed or differs from expected_serial")
        smart = self.command("smart-log")
        ocp = self.command("ocp", "smart-add-log")
        host, nand = decode_counters(smart, ocp)
        occupancy = None
        if self.mount is not None:
            usage = shutil.disk_usage(self.mount)
            occupancy = usage.used / usage.total
        return Sample(time.time_ns(), time.monotonic_ns(), identity["sn"].strip(), identity["mn"].strip(),
                      identity["fr"].strip(), host, nand, integer(smart["power_cycles"]),
                      integer(identity["tnvmcap"]), occupancy, integer(smart["temperature"]), smart, ocp)


class CounterTracker:
    def __init__(self, max_gap_seconds: float):
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be positive and finite")
        self.max_gap_seconds = max_gap_seconds
        self.previous: Sample | None = None
        self.baseline: Sample | None = None
        self.run_valid = True

    def invalidate(self) -> None:
        self.previous = None
        self.run_valid = False

    def update(self, sample: Sample) -> dict[str, Any]:
        previous = self.previous
        if self.baseline is None:
            self.baseline = sample
        baseline = self.baseline
        fields: dict[str, Any] = {
            "host_bytes_total": sample.host_bytes, "nand_bytes_total": sample.nand_bytes,
            "capacity_bytes": sample.capacity_bytes, "sample_valid": True,
            "interval_valid": False, "run_valid": self.run_valid,
            "temperature_celsius": sample.temperature_kelvin - 273.15,
        }
        if sample.occupancy is not None:
            fields["occupancy"] = sample.occupancy
        if previous is not None:
            dt = (sample.monotonic_ns - previous.monotonic_ns) / 1e9
            dh, dn = sample.host_bytes - previous.host_bytes, sample.nand_bytes - previous.nand_bytes
            identity = (sample.serial, sample.firmware, sample.capacity_bytes, sample.power_cycles)
            old_identity = (previous.serial, previous.firmware, previous.capacity_bytes, previous.power_cycles)
            valid = identity == old_identity and 0 < dt <= self.max_gap_seconds and dh >= 0 and dn >= 0
            fields["interval_seconds"] = dt
            fields["interval_valid"] = valid
            if valid:
                fields.update(host_bytes_delta=dh, nand_bytes_delta=dn,
                              host_bytes_per_second=dh / dt, nand_bytes_per_second=dn / dt)
                if dh > 0:
                    fields["waf_interval"] = dn / dh
            else:
                self.run_valid = False
                fields["invalid_reason"] = "counter reset, device/firmware/power-cycle change, or sampling gap"
        fields["run_valid"] = self.run_valid
        if self.run_valid:
            dh, dn = sample.host_bytes - baseline.host_bytes, sample.nand_bytes - baseline.nand_bytes
            fields.update(run_host_bytes=dh, run_nand_bytes=dn, drive_fills=dh / sample.capacity_bytes)
            if dh > 0:
                fields["waf_run"] = dn / dh
        self.previous = sample
        return fields


def _tag(value: str) -> str:
    if "\n" in value or "\r" in value:
        raise ValueError("Newline in metric tag")
    return value.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def influx_line(fields: dict[str, Any], tags: dict[str, str], timestamp_ns: int, *,
                measurement: str = "kvbench_nvme", include_strings: bool = False) -> str:
    # Raw JSON retains exact 128-bit integers. Numeric dashboard fields use
    # floats; all deltas/ratios are computed BEFORE conversion to float.
    metrics = []
    for key, value in fields.items():
        if isinstance(value, bool):
            metrics.append(f"{key}={'true' if value else 'false'}")
        elif isinstance(value, (int, float)) and math.isfinite(value):
            metrics.append(f"{key}={float(value)}")
        elif include_strings and isinstance(value, str):
            # Influx line protocol is not JSON: Telegraf rejects a \n escape
            # in a quoted field. Flatten display text; exact multiline errors
            # remain in the JSON status/artifacts. Only quote/backslash escape.
            single_line = " ".join(value.splitlines())
            single_line = "".join(" " if ord(char) < 32 or ord(char) == 127 else char for char in single_line)
            escaped = single_line.replace("\\", "\\\\").replace('"', '\\"')
            metrics.append(f'{key}="{escaped}"')
    labels = ",".join(f"{_tag(k)}={_tag(v)}" for k, v in sorted(tags.items()))
    return f"{_tag(measurement)},{labels} {','.join(metrics)} {timestamp_ns}\n"


def write_latest(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


class Collector:
    def __init__(self, probe: NVMeProbe, output: Path, interval: float, *, run_id: str,
                 adapter: str = "monitor", phase: str = "monitor", influx_file: Path | None = None):
        self.probe, self.output, self.interval = probe, output, interval
        self.tags = {"host": socket.gethostname().split(".")[0], "device": probe.device.rsplit("/", 1)[-1],
                     "run_id": run_id, "adapter": adapter, "phase": phase}
        self.influx_file = influx_file
        self.tracker = CounterTracker(interval * 2.5 + 1)
        self.output.parent.mkdir(parents=True, exist_ok=True)

    def collect(self) -> dict[str, Any]:
        try:
            sample = self.probe.sample()
            fields = self.tracker.update(sample)
            self.tags.update(serial=sample.serial, firmware=sample.firmware)
            record: dict[str, Any] = {"sample": asdict(sample), "fields": fields, "tags": dict(self.tags)}
            timestamp_ns = sample.timestamp_ns
        except Exception as exc:
            self.tracker.invalidate()
            timestamp_ns = time.time_ns()
            fields = {"sample_valid": False, "interval_valid": False, "run_valid": False}
            record = {"timestamp_ns": timestamp_ns, "error": str(exc), "fields": fields, "tags": dict(self.tags)}
        with self.output.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        if self.influx_file:
            line = influx_line(fields, self.tags, timestamp_ns)
            write_latest(self.influx_file, line)
            # Telegraf tails the companion stream so every sample is delivered,
            # including phase boundaries and sub-collection-interval samples.
            with self.influx_file.with_suffix(".stream").open("a") as stream:
                stream.write(line)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mount", type=Path)
    parser.add_argument("--expected-serial", help="Fail rather than follow a renumbered controller to another drive")
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="Append raw samples and derived deltas as JSONL")
    parser.add_argument("--influx-file", type=Path, help="Atomic latest sample plus a .stream companion for Telegraf inputs.tail")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--samples", type=int, default=0, help="0 means until interrupted")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1 or args.samples < 0:
        parser.error("interval must be >=1 second and samples >=0")
    collector = Collector(NVMeProbe(args.device, args.mount, args.sudo, args.expected_serial), args.output, args.interval,
                          run_id=args.run_id or f"monitor-{time.time_ns()}", influx_file=args.influx_file)
    count = 0
    try:
        while args.samples == 0 or count < args.samples:
            start = time.monotonic()
            record = collector.collect()
            print(json.dumps({"tags": record["tags"], "fields": record["fields"]}), flush=True)
            count += 1
            if args.samples == 0 or count < args.samples:
                time.sleep(max(0, args.interval - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
