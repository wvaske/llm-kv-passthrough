"""Real CLI -> uvicorn -> HTTP/SSE -> LMCache -> disk smoke test."""

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys

import httpx
import pytest

pytest.importorskip("lmcache")

from tests.integration.test_v1_storage import make_config  # noqa: E402

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize("tp,active", [(1, None), (8, [7])])
async def test_live_cli_stream_usage_followup_and_disk(tmp_path, tp, active):
    if not shutil.which("prlimit"):
        pytest.skip("Linux prlimit is required for the real-server memory guard")
    cfg = make_config(tmp_path)
    cfg.timing.full_speed = False
    cfg.scheduler.decode_tokens_per_second = 100
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    config_path = tmp_path / "run.yaml"
    cfg.to_yaml(config_path)
    env = {k: v for k, v in os.environ.items() if not k.startswith("KVBENCH_")}
    env.update(MALLOC_ARENA_MAX="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               LMCACHE_LOG_LEVEL="ERROR", PYTHONDONTWRITEBYTECODE="1")
    rank_args = ["--tp-size", str(tp)]
    if active is not None:
        rank_args.extend(["--kv-ranks", ",".join(map(str, active))])
    expected_ranks = active if active is not None else list(range(tp))
    process = subprocess.Popen(
        ["prlimit", "--as=4294967296", "--", sys.executable, "-m", "kvbench.cli.main",
         "serve", "--config", str(config_path), "--host", "127.0.0.1", "--port", str(port), *rank_args],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=15) as client:
            async with asyncio.timeout(40):
                while True:
                    if process.poll() is not None:
                        raise AssertionError(process.communicate()[0])
                    try:
                        if (await client.get("/health")).status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    await asyncio.sleep(0.1)
            manifest_response = await client.get("/kvbench/manifest")
            assert manifest_response.status_code == 200, manifest_response.text
            manifest = manifest_response.json()
            assert manifest["fidelity"] == "rate-v1"
            assert manifest["lmcache"]["chunk_size"] == 16
            assert manifest["config"]["scheduler"]["decode_tokens_per_second"] == 100
            assert manifest["config"]["gpu"]["tp_size"] == tp
            assert manifest["kv_rank_selection"]["active_rank_ids"] == expected_ranks
            assert manifest["kv_rank_selection"]["modeled_rank_count"] == tp
            messages = [{"role": "user", "content": "real network test " * 16}]
            events = []
            async with client.stream("POST", "/v1/chat/completions", json={
                "model": "storage-test", "messages": messages, "max_completion_tokens": 33,
                "stream": True, "stream_options": {"include_usage": True},
            }) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if line.startswith("data: {"):
                        events.append(json.loads(line[6:]))
            assert events[-1]["usage"]["completion_tokens"] == 33
            content = "".join(c["delta"].get("content") or "" for e in events for c in e["choices"])
            messages.extend([{"role": "assistant", "content": content}, {"role": "user", "content": "continue"}])
            response = await client.post("/v1/chat/completions", json={
                "model": "storage-test", "messages": messages, "max_tokens": 1,
            })
            assert response.status_code == 200
            state = (await client.get("/kvbench/state")).json()
            assert state["capacity"]["active_rank_ids"] == expected_ranks
            assert state["capacity"]["modeled_rank_count"] == tp
            assert state["capacity"]["active_rank_count"] == len(expected_ranks)
            assert state["kv_stats"]["hit_tokens"] > events[-1]["usage"]["prompt_tokens"]
            assert (await client.get("/stats")).json()["requests_success"] == 2
            metrics = (await client.get("/metrics")).text
            assert f'kvbench_kv_rank_count{{scope="modeled"}} {tp}.0' in metrics
            files = list((tmp_path / "disk").glob("*.pt"))
            assert {int(f.name.split("@")[2]) for f in files} == set(expected_ranks)
    finally:
        process.terminate()
        try:
            output, _ = await asyncio.to_thread(process.communicate, timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = await asyncio.to_thread(process.communicate)
        assert process.returncode in (0, -15), output
