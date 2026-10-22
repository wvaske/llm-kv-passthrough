"""Bounded block-issue samples: distinguish file layout from actual LBA order."""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def validate_trace_log(text: str) -> None:
    if "failed" in text.lower():
        raise RuntimeError("blktrace reported an error despite its exit status; inspect blktrace.log")
    dropped = re.findall(r"dropped\s+(\d+)", text, flags=re.IGNORECASE)
    if not dropped or any(int(count) for count in dropped):
        raise RuntimeError("Missing or nonzero blktrace dropped-event count; LBA sample is incomplete")


def summarize_issues(path: Path) -> dict[str, Any]:
    """Parse blkparse's '%T.%t %a %d %S %n' on ONE namespace.

    Sector units in the Linux block trace ABI are always 512 bytes, irrespective
    of the namespace's formatted logical block size. This measures issue order,
    not physical NAND placement or per-file stream adjacency.
    """
    count = total = sequential = backwards = run_bytes = max_run = 0
    end_sector: int | None = None
    sizes: Counter[int] = Counter()
    # A disk-backed ordered set avoids retaining millions of LBA intervals in
    # Python RAM. The primary key supplies ordered traversal without a temp
    # sort; scratch files stay beside the off-device trace and are owned here.
    with TemporaryDirectory(prefix="kvbench-lba-", dir=path.parent) as scratch, closing(
        sqlite3.connect(Path(scratch) / "coverage.sqlite")
    ) as database:
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("PRAGMA cache_size=-8192")
        database.execute("PRAGMA mmap_size=0")
        database.execute("CREATE TABLE spans (start INTEGER, end INTEGER, PRIMARY KEY(start, end)) WITHOUT ROWID")
        pending: list[tuple[int, int]] = []
        with path.open() as stream:
            for line in stream:
                fields = line.split()
                if len(fields) != 5 or fields[1] != "D" or "W" not in fields[2]:
                    continue
                sector, sectors = int(fields[3]), int(fields[4])
                if sector < 0 or sectors < 0 or sector + sectors >= 2**63:
                    raise ValueError("Invalid/out-of-range block write span")
                if sectors == 0:  # flush/barrier commands carry no payload
                    continue
                size = sectors * 512
                count += 1
                total += size
                sizes[size] += 1
                if end_sector is not None and sector == end_sector:
                    sequential += 1
                    run_bytes += size
                else:
                    if end_sector is not None and sector < end_sector:
                        backwards += 1
                    run_bytes = size
                max_run = max(max_run, run_bytes)
                end_sector = sector + sectors
                pending.append((sector, end_sector))
                if len(pending) >= 4096:
                    database.executemany("INSERT OR IGNORE INTO spans VALUES (?, ?)", pending)
                    database.commit()
                    pending.clear()
        database.executemany("INSERT OR IGNORE INTO spans VALUES (?, ?)", pending)
        database.commit()
        covered = 0
        covered_end = 0
        for start, end in database.execute("SELECT start, end FROM spans ORDER BY start, end"):
            covered += max(0, end - max(start, covered_end))
            covered_end = max(covered_end, end)
        unique_bytes = covered * 512
    return {"scope": "sampled block write issue order; not NAND placement or per-file streams",
            "write_ops": count, "write_bytes": total, "write_size_counts": dict(sizes),
            "adjacent_write_transitions": sequential, "backward_write_transitions": backwards,
            "sequential_transition_fraction": sequential / (count - 1) if count > 1 else None,
            "max_contiguous_run_bytes": max_run,
            "unique_written_lba_bytes": unique_bytes,
            "revisited_write_bytes": total - unique_bytes,
            "revisited_write_fraction": (total - unique_bytes) / total if total else None,
            "reuse_scope": "LBA overlap within this sample, including metadata/reissues; not NAND rewrites"}
