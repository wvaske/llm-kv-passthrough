"""Unit tests for kvbench.servers.app module.

These tests exercise the FastAPI HTTP layer and — critically — the
configuration wiring: the original implementation constructed the app from
default configuration regardless of CLI flags, env vars, or YAML files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import kvbench.servers.app as app_module
from kvbench.core.config import (
    GPUEmulationConfig,
    KVBenchConfig,
    ServerConfig,
)
from kvbench.servers.app import create_app, load_config
from kvbench.servers.warmup import WarmupController
from tests.fakes import FakeKVStack


@pytest.fixture(autouse=True)
def fast_eviction_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(WarmupController, "EVICTION_CHECK_ATTEMPTS", 1)
    monkeypatch.setattr(WarmupController, "EVICTION_CHECK_DELAY_S", 0.0)


@pytest.fixture(autouse=True)
def reset_lazy_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the lazy app singleton and scrub KVBENCH_* env vars."""
    monkeypatch.setattr(app_module, "_app", None)
    import os

    for key in list(os.environ):
        if key.startswith("KVBENCH_"):
            monkeypatch.delenv(key)


@pytest.fixture(autouse=True)
def fake_kv_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """Substitute the real LMCache stack with an in-memory fake.

    The HTTP-layer tests exercise the app, not LMCache; the real stack is
    covered by tests/integration/test_lmcache_stack.py.
    """
    monkeypatch.setattr(app_module, "create_kv_stack", lambda _config: FakeKVStack())


def make_config(**server_kwargs: object) -> KVBenchConfig:
    """Build a small test configuration."""
    return KVBenchConfig(
        instance_id="test-app",
        server=ServerConfig(**server_kwargs),  # type: ignore[arg-type]
    )


class TestConfigWiring:
    """The app must reflect the configuration it is given."""

    def test_create_app_uses_provided_config(self) -> None:
        config = KVBenchConfig(
            instance_id="wired-1",
            server=ServerConfig(model_profile="llama-3.1-70b", server_type="combined"),
            gpu=GPUEmulationConfig(gpu_profile="A100_SXM"),
        )
        with TestClient(create_app(config)) as client:
            health = client.get("/health").json()
            assert health["instance_id"] == "wired-1"
            assert health["model"] == "llama-3.1-70b"
            assert health["server_type"] == "combined"

    def test_create_app_server_type_is_respected(self) -> None:
        config = make_config(server_type="prefill")
        with TestClient(create_app(config)) as client:
            health = client.get("/health").json()
            assert health["server_type"] == "prefill"

    def test_get_app_reads_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`uvicorn kvbench.servers.app:app` deployments configure via env."""
        monkeypatch.setenv("KVBENCH_INSTANCE_ID", "env-test-42")
        monkeypatch.setenv("KVBENCH_SERVER__MODEL_PROFILE", "qwen-2.5-7b")
        monkeypatch.setenv("KVBENCH_SERVER__SERVER_TYPE", "prefill")

        with TestClient(app_module.get_app()) as client:
            health = client.get("/health").json()
            assert health["instance_id"] == "env-test-42"
            assert health["model"] == "qwen-2.5-7b"
            assert health["server_type"] == "prefill"

    def test_load_config_prefers_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The CLI hands its resolved config over via KVBENCH_CONFIG_FILE."""
        config_path = tmp_path / "config.yaml"
        KVBenchConfig(
            instance_id="from-file",
            server=ServerConfig(model_profile="llama-3.1-70b"),
        ).to_yaml(config_path)
        monkeypatch.setenv("KVBENCH_CONFIG_FILE", str(config_path))
        # A stray env var must not override the resolved file
        monkeypatch.setenv("KVBENCH_INSTANCE_ID", "should-be-ignored")

        config = load_config()
        assert config.instance_id == "from-file"
        assert config.server.model_profile == "llama-3.1-70b"

    def test_load_config_falls_back_to_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KVBENCH_GPU__GPU_PROFILE", "L4")
        config = load_config()
        assert config.gpu.gpu_profile == "L4"

    def test_invalid_profile_env_var_fails_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A typo'd profile must fail at startup, not silently use defaults."""
        monkeypatch.setenv("KVBENCH_GPU__GPU_PROFILE", "H100_SMX")  # typo
        with pytest.raises(Exception, match="Unknown GPU profile"):
            load_config()

    def test_endpoint_lists_parse_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "KVBENCH_DISTRIBUTED__PREFILL_ENDPOINTS",
            "http://prefill-1:8000,http://prefill-2:8000",
        )
        monkeypatch.setenv(
            "KVBENCH_DISTRIBUTED__DECODE_ENDPOINTS",
            '["http://decode-1:8000", "http://decode-2:8000"]',
        )
        config = load_config()
        assert config.distributed.prefill_endpoints == [
            "http://prefill-1:8000",
            "http://prefill-2:8000",
        ]
        assert config.distributed.decode_endpoints == [
            "http://decode-1:8000",
            "http://decode-2:8000",
        ]


class TestHttpEndpoints:
    """Exercise the HTTP layer end-to-end through TestClient."""

    @pytest.fixture
    def client(self) -> TestClient:
        with TestClient(create_app(make_config())) as client:
            yield client

    def test_health(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_models(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        assert response.json()["data"][0]["id"] == "llama-3.1-8b"

    def test_chat_completion_roundtrip(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 2,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"]
        assert body["usage"]["completion_tokens"] == 2

    def test_chat_completion_streaming(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 2,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            payload = "".join(response.iter_text())
        assert "chat.completion.chunk" in payload
        assert payload.rstrip().endswith("data: [DONE]")

    def test_metrics_counts_requests(self, client: TestClient) -> None:
        client.post(
            "/v1/chat/completions",
            json={
                "model": "llama-3.1-8b",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
            },
        )
        metrics = client.get("/stats").json()
        assert metrics["requests_total"] == 1
        assert metrics["requests_success"] == 1


class TestObservabilityEndpoints:
    """Prometheus metrics, JSON stats, and KV state endpoints."""

    def test_metrics_is_prometheus_format(self) -> None:
        with TestClient(create_app(make_config())) as client:
            response = client.get("/metrics")
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/plain")
            body = response.text
            assert "kvbench_kv_ops_total" in body
            assert "kvbench_request_duration_seconds" in body

    def test_metrics_reflect_kv_activity(self) -> None:
        with TestClient(create_app(make_config())) as client:
            client.post(
                "/v1/chat/completions",
                json={
                    "model": "llama-3.1-8b",
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hello " * 300}],
                },
            )
            body = client.get("/metrics").text
            assert 'kvbench_kv_ops_total{op="lookup"} 1.0' in body
            assert 'kvbench_requests_total{status="success"} 1.0' in body

    def test_stats_is_json(self) -> None:
        with TestClient(create_app(make_config())) as client:
            data = client.get("/stats").json()
            assert "requests_total" in data
            assert "cache_hits" in data

    def test_kv_state_reports_capacity_and_usage(self) -> None:
        with TestClient(create_app(make_config())) as client:
            state = client.get("/kvbench/state").json()
            assert state["capacity"]["total_capacity_bytes"] == 65536
            assert "FakeBackend" in state["usage"]
            assert state["kv_stats"]["lookups"] == 0


class TestWarmupEndpoints:
    """Warmup lifecycle over HTTP."""

    def test_warmup_runs_to_completion(self) -> None:
        import time as _time

        with TestClient(create_app(make_config())) as client:
            response = client.post("/kvbench/warmup", json={"seq_tokens": 256, "concurrency": 2})
            assert response.status_code == 202
            assert response.json()["state"] == "running"

            for _ in range(100):
                status = client.get("/kvbench/warmup").json()
                if status["state"] != "running":
                    break
                _time.sleep(0.05)
            assert status["state"] == "done"
            assert status["stored_bytes"] >= 65536

    def test_warmup_status_idle_initially(self) -> None:
        with TestClient(create_app(make_config())) as client:
            assert client.get("/kvbench/warmup").json()["state"] == "idle"

    def test_warmup_unavailable_on_proxy(self) -> None:
        config = KVBenchConfig(
            instance_id="proxy-test",
            server=ServerConfig(server_type="proxy"),
        )
        with TestClient(create_app(config)) as client:
            assert client.post("/kvbench/warmup", json={}).status_code == 404
            assert client.get("/kvbench/state").status_code == 404
