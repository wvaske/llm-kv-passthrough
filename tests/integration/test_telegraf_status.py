"""Exercise the actual parser, not just our own line-protocol expectations."""

import shutil
import subprocess

import pytest

from kvbench.benchmark.nvme import influx_line

pytestmark = pytest.mark.integration


def test_failed_campaign_multiline_error_is_accepted_by_telegraf(tmp_path):
    executable = shutil.which("telegraf")
    if executable is None:
        pytest.skip("Telegraf is optional outside the monitoring host")
    sample = tmp_path / "status.influx"
    error = "Server error '500 Internal Server Error' for url 'http://127.0.0.1/flush'\nFor details: \"failure\"\tend; literal \\n"
    sample.write_text(influx_line({"observer_ok": True, "state_label": "FAILED", "last_error": error},
                                 {"host": "test"}, 123456789, measurement="kvbench_campaign", include_strings=True))
    config = tmp_path / "telegraf.conf"
    config.write_text(f'[[inputs.file]]\nfiles = ["{sample}"]\ndata_format = "influx"\n')
    result = subprocess.run([executable, "--config", str(config), "--test"],
                            capture_output=True, text=True, timeout=20, check=True)
    assert "metric parse error" not in result.stderr
    assert 'state_label="FAILED"' in result.stdout
    assert "For details:" in result.stdout
    assert 'observer_ok=true' in result.stdout
