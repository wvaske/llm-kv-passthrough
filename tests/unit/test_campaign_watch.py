import json

import pytest

from kvbench.benchmark.campaign_watch import CampaignWatcher, RollingRate, phase_progress, snapshot
from kvbench.benchmark.nvme import influx_line


def fixture(tmp_path, *, preparing=True):
    config = {"order": ["lmcache", "hicache", "hicache", "lmcache"], "concurrency": 16,
              "interval_seconds": 5, "min_occupancy": 0.8, "precondition_drive_fills": 1,
              "drive_fills": 3, "min_seconds": 1, "idle_seconds": 60, "drain_seconds": 300,
              "mode": "endurance", "smoke_host_bytes": 50, "smoke_seconds": 1}
    meta = {"source_revision": "abcdefgh12345678", "experiment" if preparing else "config": config,
            "device": {"capacity_bytes": 1000, "filesystem_bytes": 1000, "used_bytes": 200}}
    data = {"sample_valid": True, "run_valid": True, "occupancy": 0.5,
            "host_bytes_per_second": 10, "nand_bytes_per_second": 12, "run_host_bytes": 300,
            "run_nand_bytes": 360, "waf_run": 1.2}
    tags = {"run_id": "prepare-a" if preparing else "run-a", "phase": "preparation" if preparing else "measure",
            "adapter": "filler" if preparing else "lmcache"}
    sample = {"fields": data, "tags": tags, "timestamp_ns": 1000 * 10**9}
    unit = {"name": "kvbench-waf-abcdefgh.service", "ActiveState": "active", "SubState": "running",
            "MemoryCurrent": "1024", "MemoryMax": "4096", "CPUUsageNSec": "1000000000",
            "ExecMainStartTimestampMonotonic": "1000000"}
    watcher = CampaignWatcher(tmp_path, tmp_path / "latest.influx", "serial", "nvme0")
    baseline(tmp_path / "samples.jsonl", 900)
    return watcher, sample, meta, unit


def baseline(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sample": {"timestamp_ns": seconds * 10**9},
                               "fields": {"sample_valid": True, "interval_valid": False, "run_host_bytes": 0}}) + "\n")


def test_preparation_progress_and_eta_uses_remaining_fill_not_lifetime_bytes(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path)
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 100, True)
    assert fields["state_label"] == "RUNNING"
    assert fields["phase_progress"] == pytest.approx(300 / 650)
    assert fields["phase_remaining_bytes"] == 350
    assert fields["phase_eta_seconds"] == 35
    assert fields["phase_eta_unix_ms"] == 1045 * 1000
    assert fields["completed_arms"] == 0
    assert fields["qualification_label"] == "PENDING — NO ENDURANCE RESULT"
    assert fields["memory_bytes"] == 1024


def test_stale_or_invalid_telemetry_never_publishes_live_eta_or_rates(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path)
    fields = watcher.build(sample, meta, unit, tmp_path, 1100, 100, True)
    assert not fields["telemetry_ok"]
    assert "STALE/INVALID" in fields["state_label"]
    assert "phase_eta_seconds" not in fields
    assert "host_bytes_per_second" not in fields
    sample["fields"]["run_valid"] = False
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 110, True)
    assert "phase_eta_seconds" not in fields
    del sample["fields"]["occupancy"]
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 120, True)
    assert "phase_progress" not in fields  # missing telemetry must not look like zero progress


def test_stopped_unit_is_not_mistaken_for_completed_work(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path)
    unit.update(ActiveState="inactive", SubState="dead", Result="success")
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 100, True)
    assert fields["state_label"] == "STOPPED"
    assert fields["qualification_label"] != "QUALIFIED"
    assert "phase_eta_seconds" not in fields
    assert "host_bytes_per_second" not in fields


def test_new_campaign_does_not_inherit_old_progress(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path)
    unit["name"] = "kvbench-waf-otherrev.service"
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 100, True)
    assert fields["phase_label"].startswith("Waiting for the new campaign")
    assert "phase_progress" not in fields
    assert not fields["telemetry_ok"]


def test_restarting_same_revision_does_not_show_previous_run_as_complete(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path)
    meta.update(status="complete", qualified=True)
    unit["ExecMainStartTimestampMonotonic"] = "195000000"
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 200, True)
    assert fields["state_label"] == "RUNNING — WAITING FOR TELEMETRY"
    assert fields["qualification_label"] == "PENDING"
    assert "phase_progress" not in fields


def test_comparison_auto_follows_arm_and_counts_qualified_phases(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path, preparing=False)
    meta.update(status="running", arms=[{"arm": "lmcache"}])
    sample["tags"].update(adapter="hicache")
    sample["fields"].update(run_host_bytes=1500, host_bytes_per_second=100)
    baseline(tmp_path / "01-hicache/measure.jsonl", 900)
    for name in ("00-lmcache/precondition-summary.json", "00-lmcache/measure-summary.json",
                 "01-hicache/precondition-summary.json"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('{"qualified":true}')
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 100, False)
    assert fields["current_arm"] == 2
    assert fields["phase_progress"] == 0.5
    assert fields["phase_eta_seconds"] == 15
    assert fields["comparison_progress"] == 6500 / 16000
    assert fields["arm_01_status"].endswith("Complete")
    assert fields["arm_02_status"].endswith("Active")
    sample["tags"]["phase"] = "measure-drain"
    fields = watcher.build(sample, meta, unit, tmp_path, 1010, 110, False)
    assert "phase_eta_seconds" not in fields  # cannot time outstanding I/O from tail duration alone


def test_completed_campaign_is_distinct_from_stopped_and_smoke(tmp_path):
    watcher, sample, meta, unit = fixture(tmp_path, preparing=False)
    meta.update(status="complete", qualified=True, arms=[{}] * 4)
    unit.update(ActiveState="inactive", SubState="dead")
    fields = watcher.build(sample, meta, unit, tmp_path, 1100, 100, False)
    assert fields["state_label"] == "COMPLETE"
    assert fields["qualification_label"] == "QUALIFIED"
    assert fields["comparison_progress"] == fields["phase_progress"] == 1
    assert fields["phase_eta_unix_ms"] == sample["timestamp_ns"] / 1e6
    meta["qualified"] = False
    meta["config"]["mode"] = "smoke"
    fields = watcher.build(sample, meta, unit, tmp_path, 1100, 110, False)
    assert fields["qualification_label"] == "SMOKE — NOT QUALIFIED"


def test_rolling_rate_uses_counter_deltas_and_resets_between_phases():
    rate = RollingRate()
    assert rate.update("a", 100, 100, 99) == 99
    rate.update("a", 100, 100, 99)
    assert len(rate.points) == 1
    assert rate.update("a", 120, 200, 99) == 5
    assert rate.update("b", 125, 0, 8) == 8
    assert rate.update("b", 145, 0, 8) == 0
    assert "phase_eta_seconds" not in phase_progress(0, 100, 0)
    assert phase_progress(100, 100, 10, elapsed=10, min_seconds=60)["phase_eta_seconds"] == 50


def test_snapshot_reader_and_string_metric_encoding(tmp_path):
    path = tmp_path / "sample.influx"
    path.write_text(influx_line({"sample_valid": True, "run_host_bytes": 123},
                               {"run_id": "prepare-a", "device": "nvme0", "serial": "x"}, 123456))
    assert snapshot(path)["fields"]["run_host_bytes"] == 123
    line = influx_line({"phase_label": 'hello "phase"\nnext', "observer_ok": True},
                       {"host": "dev"}, 123, measurement="kvbench_campaign", include_strings=True)
    assert line.startswith("kvbench_campaign,")
    assert 'phase_label="hello \\"phase\\" next"' in line
    assert line.count("\n") == 1
