"""Unit tests for kvbench.core.config module."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from kvbench.core.config import (
    DistributedConfig,
    GPUEmulationConfig,
    KVBenchConfig,
    KVStackConfig,
    MetricsConfig,
    ServerConfig,
)


class TestKVStackConfig:
    """Tests for KVStackConfig."""

    def test_default_values(self) -> None:
        """Default stack is LMCache configured from its own env/defaults."""
        config = KVStackConfig()
        assert config.stack == "lmcache"
        assert config.lmcache_config_file is None

    def test_config_file_string_conversion(self, tmp_path: Path) -> None:
        """String paths convert to Path objects."""
        f = tmp_path / "lmcache.yaml"
        f.write_text("chunk_size: 256\n")
        config = KVStackConfig(lmcache_config_file=str(f))
        assert isinstance(config.lmcache_config_file, Path)

    def test_missing_config_file_fails_loudly(self) -> None:
        """A nonexistent LMCache config file must fail at config time."""
        with pytest.raises(ValueError, match="not found"):
            KVStackConfig(lmcache_config_file="/nonexistent/lmcache.yaml")

    def test_unknown_stack_rejected(self) -> None:
        """Only implemented stacks are accepted."""
        with pytest.raises(ValueError):
            KVStackConfig(stack="mooncake")


class TestGPUEmulationConfig:
    """Tests for GPUEmulationConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = GPUEmulationConfig()
        assert config.gpu_profile == "H100_SXM"
        assert config.efficiency_factor == 0.7
        assert config.tp_size == 1

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = GPUEmulationConfig(
            gpu_profile="A100_SXM",
            efficiency_factor=0.8,
            tp_size=8,
        )
        assert config.gpu_profile == "A100_SXM"
        assert config.efficiency_factor == 0.8
        assert config.tp_size == 8

    def test_efficiency_factor_validation_min(self) -> None:
        """Test efficiency_factor minimum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(efficiency_factor=0.05)

    def test_efficiency_factor_validation_max(self) -> None:
        """Test efficiency_factor maximum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(efficiency_factor=1.5)

    def test_tp_size_validation_min(self) -> None:
        """Test tp_size minimum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(tp_size=0)

    def test_tp_size_validation_max(self) -> None:
        """Test tp_size maximum validation."""
        with pytest.raises(ValueError):
            GPUEmulationConfig(tp_size=32)


class TestServerConfig:
    """Tests for ServerConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.server_type == "combined"
        assert config.model_profile == "llama-3.1-8b"
        assert config.workers == 1
        assert config.log_level == "info"

    def test_custom_values(self) -> None:
        """Test custom values are accepted."""
        config = ServerConfig(
            host="127.0.0.1",
            port=9000,
            server_type="prefill",
            model_profile="llama-3.1-70b",
            workers=4,
            log_level="debug",
        )
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.server_type == "prefill"

    def test_port_validation_min(self) -> None:
        """Test port minimum validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=80)

    def test_port_validation_max(self) -> None:
        """Test port maximum validation."""
        with pytest.raises(ValueError):
            ServerConfig(port=70000)

    def test_server_types(self) -> None:
        """Test all server types are valid."""
        for server_type in ["prefill", "decode", "combined", "proxy"]:
            config = ServerConfig(server_type=server_type)  # type: ignore[arg-type]
            assert config.server_type == server_type


class TestDistributedConfig:
    """Tests for DistributedConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = DistributedConfig()
        assert config.prefill_endpoints == []
        assert config.decode_endpoints == []
        assert config.registry_url is None
        assert config.health_check_interval == 10.0

    def test_custom_endpoints(self) -> None:
        """Test custom endpoints configuration."""
        config = DistributedConfig(
            prefill_endpoints=["http://prefill-1:8000", "http://prefill-2:8000"],
            decode_endpoints=["http://decode-1:8000"],
        )
        assert len(config.prefill_endpoints) == 2
        assert len(config.decode_endpoints) == 1


class TestMetricsConfig:
    """Tests for MetricsConfig configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = MetricsConfig()
        assert config.enabled is True
        assert config.prometheus_port == 9090
        assert config.include_histograms is True

    def test_disabled_metrics(self) -> None:
        """Test metrics can be disabled."""
        config = MetricsConfig(enabled=False)
        assert config.enabled is False


class TestKVBenchConfig:
    """Tests for KVBenchConfig main configuration."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        config = KVBenchConfig()
        assert config.instance_id == "kvbench-0"
        assert isinstance(config.kv, KVStackConfig)
        assert isinstance(config.gpu, GPUEmulationConfig)
        assert isinstance(config.server, ServerConfig)
        assert isinstance(config.distributed, DistributedConfig)
        assert isinstance(config.metrics, MetricsConfig)

    def test_nested_config(self) -> None:
        """Test nested configuration works correctly."""
        config = KVBenchConfig(
            instance_id="test-instance",
            kv=KVStackConfig(stack="lmcache"),
            gpu=GPUEmulationConfig(gpu_profile="A100_SXM"),
        )
        assert config.instance_id == "test-instance"
        assert config.kv.stack == "lmcache"
        assert config.gpu.gpu_profile == "A100_SXM"

    def test_from_env(self) -> None:
        """Test configuration loading from environment variables."""
        env_vars = {
            "KVBENCH_INSTANCE_ID": "env-instance",
            "KVBENCH_KV__STACK": "lmcache",
            "KVBENCH_GPU__GPU_PROFILE": "A100_SXM",
            "KVBENCH_SERVER__PORT": "9000",
        }
        with mock.patch.dict(os.environ, env_vars, clear=False):
            config = KVBenchConfig.from_env()
            assert config.instance_id == "env-instance"
            assert config.kv.stack == "lmcache"
            assert config.gpu.gpu_profile == "A100_SXM"
            assert config.server.port == 9000

    def test_from_yaml(self) -> None:
        """Test configuration loading from YAML file."""
        yaml_content = """
instance_id: yaml-instance
kv:
  stack: lmcache
server:
  port: 8080
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                config = KVBenchConfig.from_yaml(f.name)
                assert config.instance_id == "yaml-instance"
                assert config.kv.stack == "lmcache"
                assert config.server.port == 8080
            finally:
                os.unlink(f.name)

    def test_from_yaml_file_not_found(self) -> None:
        """Test from_yaml raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            KVBenchConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_empty_file(self) -> None:
        """Test from_yaml handles empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            try:
                config = KVBenchConfig.from_yaml(f.name)
                # Should use defaults
                assert config.instance_id == "kvbench-0"
            finally:
                os.unlink(f.name)

    def test_to_yaml(self) -> None:
        """Test configuration saving to YAML file."""
        config = KVBenchConfig(instance_id="save-test")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)
            assert path.exists()
            # Reload and verify
            loaded = KVBenchConfig.from_yaml(path)
            assert loaded.instance_id == "save-test"

    def test_to_yaml_creates_directories(self) -> None:
        """Test to_yaml creates parent directories."""
        config = KVBenchConfig()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "path" / "config.yaml"
            config.to_yaml(path)
            assert path.exists()

    def test_model_dump(self) -> None:
        """Test model can be dumped to dictionary."""
        config = KVBenchConfig(instance_id="dump-test")
        data = config.model_dump()
        assert data["instance_id"] == "dump-test"
        assert "kv" in data
        assert "server" in data
