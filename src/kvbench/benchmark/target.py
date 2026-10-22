"""Qualify one direct NVMe filesystem and prepare real, bounded filler files."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from kvbench.benchmark.nvme import NVMeProbe, command_json
from kvbench.core.payload import SFC64_FORMAT, fill_sfc64


@contextmanager
def device_lease(controller: str) -> Iterator[None]:
    """Prevent this harness's fill/compare jobs from overlapping on a controller."""
    if not re.fullmatch(r"/dev/nvme\d+", controller):
        raise ValueError("Expected an NVMe controller path")
    path = Path("/tmp") / ("kvbench-" + controller.rsplit("/", 1)[-1] + ".lock")
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another KV-Bench drive experiment/preparation owns this controller") from exc
        yield
    finally:
        os.close(fd)


def backing_disk(tree: dict[str, Any]) -> dict[str, Any]:
    nodes = []

    def visit(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node.get("children", []):
            visit(child)

    for node in tree["blockdevices"]:
        visit(node)
    disks = [node for node in nodes if node["type"] == "disk"]
    if len(disks) != 1 or any(node["type"] not in ("disk", "part") for node in nodes):
        raise ValueError("Test filesystem must have one direct disk, without RAID, LVM, ZFS or device mapper")
    disk = disks[0]
    if disk.get("tran") != "nvme" or not re.fullmatch(r"/dev/nvme\d+n\d+", disk["name"]):
        raise ValueError("Test disk must be a directly attached NVMe namespace")
    return disk


def qualify_target(target: Path, probe: NVMeProbe, expected_serial: str,
                   min_occupancy: float = 0.8, max_occupancy: float = 0.95) -> dict[str, Any]:
    target = target.resolve(strict=True)
    mounts = command_json(["findmnt", "-J", "-T", str(target), "-o", "SOURCE,FSTYPE,TARGET,OPTIONS"])["filesystems"]
    if len(mounts) != 1:
        raise ValueError("Ambiguous filesystem backing")
    mount = mounts[0]
    if mount["fstype"] not in ("xfs", "ext4") or mount["target"] in ("/", "/boot", "/boot/efi"):
        raise ValueError("Use a non-system, direct ext4/XFS test filesystem")
    if "rw" not in mount["options"].split(","):
        raise ValueError("Test filesystem is not writable")
    disk = backing_disk(command_json(["lsblk", "-J", "-b", "-p", "-s", "-o", "NAME,TYPE,SIZE,TRAN,SERIAL", mount["source"]]))
    match = re.fullmatch(r"(/dev/nvme\d+)n(\d+)", disk["name"])
    assert match is not None
    if match[1] != probe.device or disk["serial"] != expected_serial:
        raise ValueError("Controller/serial does not match the filesystem's backing drive")
    namespaces = probe.command("list-ns", "-a")["nsid_list"]
    if [n["nsid"] for n in namespaces] != [int(match[2])]:
        raise ValueError("WAF qualification requires exactly one allocated namespace on the controller")
    sample = probe.sample()
    if sample.serial != expected_serial:
        raise ValueError("SMART controller identity differs from the requested drive")
    size = int(disk["size"])
    usage = shutil.disk_usage(target)
    if usage.total < 0.9 * size or not (0.99 * sample.capacity_bytes <= size <= sample.capacity_bytes):
        raise ValueError("Test filesystem/namespace must cover nearly the entire physical drive")
    occupancy = usage.used / usage.total
    if not min_occupancy <= occupancy <= max_occupancy:
        raise ValueError(f"Drive occupancy {occupancy:.1%} outside required {min_occupancy:.0%}..{max_occupancy:.0%}")
    return {"target": str(target), "mount": mount, "disk": disk, "serial": sample.serial,
            "firmware": sample.firmware, "capacity_bytes": size, "filesystem_bytes": usage.total,
            "used_bytes": usage.used, "free_bytes": usage.free, "occupancy": occupancy,
            "filesystem_device_id": target.stat().st_dev, "raw_initial_sample": sample.__dict__}


def fill_target(target: Path, probe: NVMeProbe, expected_serial: str, *, occupancy: float = 0.85,
                rate_mib: float = 128, max_bytes: int | None = None,
                stop: threading.Event | None = None,
                generator: Literal["system", "sfc64"] = "system",
                fsync_bytes: int = 64 * 1024**2) -> dict[str, Any]:
    """Create a NEW dedicated filler directory; never truncate existing files.

    Writes real random payload, fsyncs at the configured interval (64 MiB by
    default), and preserves the files. A
    max_bytes bound may intentionally leave preparation incomplete. No format,
    trim, discard, sparse allocation or deletion is performed.
    """
    if not math.isfinite(occupancy) or not 0.8 <= occupancy <= 0.9:
        raise ValueError("Filler target occupancy must be between 80% and 90%")
    if not math.isfinite(rate_mib) or rate_mib <= 0 or (max_bytes is not None and max_bytes <= 0):
        raise ValueError("Filler rate and byte bound must be positive")
    if generator not in ("system", "sfc64") or fsync_bytes <= 0:
        raise ValueError("Invalid filler generator or fsync interval")
    report = qualify_target(target.parent, probe, expected_serial, min_occupancy=0, max_occupancy=0.95)
    target.mkdir(exist_ok=False)
    manifest = target / "filler-manifest.json"
    base = {"target_occupancy": occupancy, "device": report, "generator": generator,
            "rate_mib": rate_mib, "fsync_bytes": fsync_bytes, "method": "fresh random buffered writes"}
    buffer, rng = None, None
    if generator == "sfc64":
        import numpy as np

        seed = int.from_bytes(os.urandom(16), "little")
        rng = np.random.SFC64(seed)
        buffer = np.empty(8 * 1024**2, dtype=np.uint8)
        base.update(seed=seed, generator_format=SFC64_FORMAT, numpy_version=np.__version__)
    manifest.write_text(json.dumps({**base, "status": "writing", "bytes_written": 0}, indent=2))
    written = 0
    stop = stop or threading.Event()
    started = time.monotonic()
    try:
        file_index = 0
        while True:
            usage = shutil.disk_usage(target)
            needed = max(0, int(occupancy * usage.total) - usage.used)
            if needed == 0 or (max_bytes is not None and written >= max_bytes):
                break
            if target.stat().st_dev != report["filesystem_device_id"]:
                raise RuntimeError("Target filesystem changed")
            file_bytes = min(8 * 1024**3, needed, max_bytes - written if max_bytes is not None else needed)
            with (target / f"filler-{file_index:06d}.bin").open("xb", buffering=0) as stream:
                remaining = file_bytes
                unflushed = 0
                while remaining:
                    if stop.is_set():
                        raise InterruptedError("Filler cancelled")
                    usage = shutil.disk_usage(target)
                    if usage.free < max(64 * 1024**2, int(usage.total * 0.05)):
                        raise RuntimeError("Filler stopped at 5% free-space floor")
                    count_bytes = min(8 * 1024**2, remaining)
                    if buffer is not None and rng is not None:
                        fill_sfc64(buffer[:count_bytes], rng)
                        view = buffer[:count_bytes].data
                    else:
                        view = memoryview(os.urandom(count_bytes))
                    while view:
                        count = stream.write(view)
                        if not count:
                            raise OSError("Short filler write")
                        view = view[count:]
                        written += count
                        remaining -= count
                        unflushed += count
                    if unflushed >= fsync_bytes:
                        os.fsync(stream.fileno())
                        unflushed = 0
                    stop.wait(max(0, written / (rate_mib * 1024**2) - (time.monotonic() - started)))
                os.fsync(stream.fileno())
            file_index += 1
            manifest.write_text(json.dumps({**base, "status": "writing", "bytes_written": written}, indent=2))
    finally:
        usage = shutil.disk_usage(target)
        achieved = usage.used / usage.total
        result = {**base, "status": "prepared" if achieved >= occupancy - 1e-6 else "incomplete",
                  "bytes_written": written, "occupancy": achieved, "target_occupancy": occupancy,
                  "elapsed_seconds": time.monotonic() - started}
        manifest.write_text(json.dumps(result, indent=2))
    return result
