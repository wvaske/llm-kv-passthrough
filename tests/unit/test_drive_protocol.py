import json
from pathlib import Path

import pytest

from kvbench.benchmark.drive import (
    Experiment,
    comparison_signature,
    goal_met,
    load_experiment,
    validate_arms,
)
from kvbench.benchmark.target import backing_disk, device_lease, fill_target
from kvbench.core.config import KVBenchConfig


def test_device_topology_rejects_raid_and_mapper():
    disk = {"name": "/dev/nvme6n1", "type": "disk", "size": 10000, "tran": "nvme", "serial": "s"}
    part = {"name": "/dev/nvme6n1p1", "type": "part", "children": [disk]}
    assert backing_disk({"blockdevices": [part]}) == disk
    for kind in ("raid10", "lvm", "crypt", "mpath"):
        with pytest.raises(ValueError, match="one direct disk"):
            backing_disk({"blockdevices": [{"type": kind, "children": [disk]}]})
    with pytest.raises(ValueError, match="one direct disk"):
        backing_disk({"blockdevices": [disk, disk]})
    with pytest.raises(ValueError, match="NVMe"):
        backing_disk({"blockdevices": [{**disk, "tran": "sata"}]})


def test_goals_use_observed_host_bytes_and_duration_not_nand_or_submissions():
    assert not goal_met({"run_valid": True, "run_host_bytes": 99, "run_nand_bytes": 900}, 100,
                        min_seconds=60, host_bytes=100)
    assert not goal_met({"run_valid": True, "run_host_bytes": 1000}, 59, min_seconds=60, host_bytes=100)
    assert not goal_met({"run_valid": False, "run_host_bytes": 1000}, 100, min_seconds=60, host_bytes=100)
    assert goal_met({"run_valid": True, "run_host_bytes": 100}, 60, min_seconds=60, host_bytes=100)


@pytest.mark.parametrize("generator", ["system", "sfc64"])
def test_filler_is_real_data_bounded_and_does_not_overwrite(tmp_path, monkeypatch, generator):
    import kvbench.benchmark.target as module

    monkeypatch.setattr(module, "qualify_target", lambda *_a, **_k: {"filesystem_device_id": tmp_path.stat().st_dev})
    existing = tmp_path / "existing.dat"
    existing.write_bytes(b"keep me")
    output = tmp_path / "filler"
    result = fill_target(output, None, "test", max_bytes=4096, rate_mib=128, generator=generator)
    assert result["bytes_written"] == 4096
    assert result["status"] == "incomplete"
    assert result["generator"] == generator
    file = output / "filler-000000.bin"
    assert file.stat().st_size == 4096
    assert file.stat().st_blocks * 512 >= 4096
    assert any(file.read_bytes())
    assert existing.read_bytes() == b"keep me"
    assert json.loads((output / "filler-manifest.json").read_text())["bytes_written"] == 4096
    with pytest.raises(FileExistsError):
        fill_target(output, None, "test", max_bytes=4096)


def test_config_demands_multiple_fills_and_resolves_paths(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("""target: /mnt/test
controller: /dev/nvme0
expected_serial: test
artifacts: artifacts
configs: {lmcache: lm.yaml, hicache: hi.yaml}
""")
    cfg = load_experiment(path)
    assert cfg.artifacts == tmp_path / "artifacts"
    assert cfg.configs["lmcache"] == tmp_path / "lm.yaml"
    assert cfg.drive_fills == 3
    assert cfg.precondition_drive_fills == 1
    data = cfg.model_dump()
    data["drive_fills"] = 1
    with pytest.raises(ValueError):
        Experiment.model_validate(data)
    data["drive_fills"] = 3
    data["configs"] = {"../oops": Path("x")}
    with pytest.raises(ValueError, match="Arm names"):
        Experiment.model_validate(data)


def test_comparison_signature_detects_byte_and_rate_mismatches():
    cfg = KVBenchConfig()
    capacity = {"chunk_size": 256, "stored_bytes_per_token": 131072,
                "local_cpu_capacity_bytes": 1024, "local_disk_capacity_bytes": 2048}
    first = comparison_signature(cfg, capacity)
    assert comparison_signature(cfg, {**capacity, "chunk_size": 128}) != first
    cfg.kv.payload_generator = "sfc64"
    assert comparison_signature(cfg, capacity) != first
    cfg.kv.payload_generator = "torch"
    cfg.scheduler.decode_tokens_per_second *= 2
    assert comparison_signature(cfg, capacity) != first


def test_shipped_endurance_arms_match_before_any_writes():
    experiment = load_experiment(Path(__file__).resolve().parents[2] / "examples/drive-compare.yaml")
    experiment.block_trace_seconds = 0  # optional system tools are tested on the real device
    validate_arms(experiment)
    assert experiment.concurrency == 16
    for path in experiment.configs.values():
        config = KVBenchConfig.from_yaml(path)
        assert config.timing.full_speed is True
        assert config.scheduler.max_active_requests == 32


def test_controller_lease_prevents_overlapping_experiments():
    with device_lease("/dev/nvme999999"), pytest.raises(RuntimeError, match="Another KV-Bench"), device_lease("/dev/nvme999999"):
        pass
    with device_lease("/dev/nvme999999"):
        pass


def test_cancelled_filler_stops_without_background_writes(tmp_path, monkeypatch):
    import threading

    import kvbench.benchmark.target as module

    monkeypatch.setattr(module, "qualify_target", lambda *_a, **_k: {"filesystem_device_id": tmp_path.stat().st_dev})
    stop = threading.Event()
    stop.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        fill_target(tmp_path / "filler", None, "test", max_bytes=4096, stop=stop)
    manifest = json.loads((tmp_path / "filler/filler-manifest.json").read_text())
    assert manifest["bytes_written"] == 0
    assert manifest["status"] == "incomplete"


async def test_campaign_telemetry_failure_stops_filler_thread(tmp_path, monkeypatch):
    import threading

    import kvbench.benchmark.drive as module

    stopped = threading.Event()

    def fill(*_args, stop, **_kwargs):
        stop.wait(10)
        stopped.set()
        raise InterruptedError("test stopped")

    class FakeCollector:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def collect(self):
            self.calls += 1
            return {"fields": {"sample_valid": self.calls == 1, "run_valid": self.calls == 1}}

    monkeypatch.setattr(module, "validate_arms", lambda _e: None)
    monkeypatch.setattr(module, "qualify_target", lambda *_a: {"filesystem_device_id": -1})
    monkeypatch.setattr(module, "Collector", FakeCollector)
    monkeypatch.setattr(module, "fill_target", fill)
    cfg = Experiment(target=tmp_path, controller="/dev/nvme0", expected_serial="test", artifacts=tmp_path / "artifacts",
                     configs={"lmcache": tmp_path / "a", "hicache": tmp_path / "b"}, interval_seconds=1)
    with pytest.raises(RuntimeError, match="telemetry failed"):
        await module.campaign(cfg, 128)
    assert stopped.is_set()


async def test_phase_preserves_request_error_when_drain_also_fails(tmp_path, monkeypatch):
    import httpx

    import kvbench.benchmark.drive as module

    class FakeCollector:
        def __init__(self, *_args, **_kwargs):
            self.tags = {}

        def collect(self):
            return {"fields": {"sample_valid": True, "run_valid": True, "occupancy": 0.85,
                               "run_host_bytes": 0}}

    class Client:
        async def post(self, path, **_kwargs):
            message = "MemoryError" if path == "/v1/chat/completions" else "flush timeout"
            return httpx.Response(500, request=httpx.Request("POST", "http://test" + path),
                                  json={"error": {"message": message}})

    monkeypatch.setattr(module, "Collector", FakeCollector)
    cfg = Experiment(target=tmp_path, controller="/dev/nvme0", expected_serial="test", artifacts=tmp_path,
                     configs={"a": tmp_path / "a"}, concurrency=1, interval_seconds=1,
                     min_artifact_free_bytes=64 * 1024**2)
    with pytest.raises(RuntimeError, match="MemoryError"):
        await module.workload_phase(Client(), KVBenchConfig(), cfg, None,
                                    {"capacity_bytes": 1000, "filesystem_device_id": tmp_path.stat().st_dev},
                                    tmp_path, "lmcache", "precondition", 1, "test")
    result = json.loads((tmp_path / "precondition-summary.json").read_text())
    assert not result["qualified"]
    assert "MemoryError" in result["errors"][0]
    assert any("Drain failed" in error for error in result["errors"])
