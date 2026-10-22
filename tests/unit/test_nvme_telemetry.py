import json
from dataclasses import replace

import pytest

from kvbench.benchmark.nvme import (
    OCP_GUID,
    Collector,
    CounterTracker,
    NVMeProbe,
    Sample,
    decode_counters,
    influx_line,
)


def sample(host=100, nand=200, seconds=1, **kwargs):
    return Sample(timestamp_ns=seconds * 10**9, monotonic_ns=seconds * 10**9,
                  serial="disk1", model="test", firmware="v1", host_bytes=host, nand_bytes=nand,
                  power_cycles=1, capacity_bytes=1000, occupancy=0.85, temperature_kelvin=300,
                  raw_smart={}, raw_ocp={}, **kwargs)


def test_units_and_128_bit_counter_preserve_precision():
    smart = {"data_units_written": "486,061,721"}
    ocp = {"Log page GUID": "0x" + OCP_GUID, "Log page version": 3,
           "Physical media units written": {"hi": 3, "lo": 2**64 - 1}}
    assert decode_counters(smart, ocp) == (486061721 * 512000, 4 * 2**64 - 1)
    ocp["Physical media units written"] = float(2**64)
    with pytest.raises(ValueError, match="exact integer"):
        decode_counters(smart, ocp)
    ocp["Log page GUID"] = "unknown"
    with pytest.raises(ValueError, match="Unknown OCP"):
        decode_counters(smart, ocp)


def test_delta_waf_is_not_lifetime_waf_and_zero_host_is_missing():
    tracker = CounterTracker(10)
    assert "waf_interval" not in tracker.update(sample(10000, 100000, 1))
    fields = tracker.update(sample(11000, 101500, 6))
    assert fields["waf_interval"] == fields["waf_run"] == 1.5
    assert fields["drive_fills"] == 1
    assert fields["host_bytes_per_second"] == 200
    idle = tracker.update(sample(11000, 101900, 11))
    assert "waf_interval" not in idle  # idle GC still increments numerator
    assert idle["nand_bytes_delta"] == 400
    assert idle["waf_run"] == 1.9


@pytest.mark.parametrize("changed", [
    sample(99, 220, 6), sample(110, 199, 6), sample(110, 220, 30),
    replace(sample(110, 220, 6), serial="other"), replace(sample(110, 220, 6), power_cycles=2),
    replace(sample(110, 220, 6), firmware="v2"), sample(110, 220, 1),
])
def test_resets_gaps_and_identity_changes_invalidate_run(changed):
    tracker = CounterTracker(10)
    tracker.update(sample())
    fields = tracker.update(changed)
    assert fields["interval_valid"] is False
    assert fields["run_valid"] is False
    assert "waf_run" not in fields and "waf_interval" not in fields


def test_large_counters_small_deltas_remain_exact():
    tracker = CounterTracker(10)
    tracker.update(sample(2**100, 2**101, 1))
    fields = tracker.update(sample(2**100 + 3, 2**101 + 7, 6))
    assert fields["host_bytes_delta"] == 3
    assert fields["nand_bytes_delta"] == 7
    assert fields["waf_interval"] == 7 / 3


def test_missing_nand_records_error_not_zero(tmp_path):
    class Probe:
        device = "/dev/nvme6"

        def sample(self):
            raise KeyError("Physical media units written")

    collector = Collector(Probe(), tmp_path / "raw.jsonl", 5, run_id="test", influx_file=tmp_path / "latest.influx")
    record = collector.collect()
    assert record["fields"]["sample_valid"] is False
    assert "nand_bytes_total" not in record["fields"]
    assert "waf" not in (tmp_path / "latest.influx").read_text()
    assert (tmp_path / "latest.stream").read_text() == (tmp_path / "latest.influx").read_text()
    collector.collect()
    assert len((tmp_path / "latest.stream").read_text().splitlines()) == 2
    assert json.loads((tmp_path / "raw.jsonl").read_text().splitlines()[0])["error"]


def test_influx_escaping_and_unknown_waf_not_emitted():
    line = influx_line({"host_bytes_delta": 3, "waf_interval": None, "sample_valid": True},
                       {"model": "hello, world=1"}, 123)
    assert line == 'kvbench_nvme,model=hello\\,\\ world\\=1 host_bytes_delta=3.0,sample_valid=true 123\n'
    with pytest.raises(ValueError, match="Newline"):
        influx_line({}, {"run_id": "a\nb"}, 1)


def test_probe_does_not_follow_controller_renumbering(monkeypatch):
    probe = NVMeProbe("/dev/nvme6", expected_serial="expected")
    monkeypatch.setattr(probe, "command", lambda *_args: {"sn": "different"})
    with pytest.raises(ValueError, match="serial changed"):
        probe.sample()
