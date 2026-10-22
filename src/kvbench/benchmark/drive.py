"""Single-NVMe qualification, real filler and sustained cache comparisons.

Run as a normal user with read-only nvme sudo access. Servers are separate,
memory-capped subprocesses; control logs/traces live off the measured drive.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from pydantic import Field, field_validator

from kvbench.benchmark.blocktrace import summarize_issues, validate_trace_log
from kvbench.benchmark.nvme import Collector, NVMeProbe
from kvbench.benchmark.target import device_lease, fill_target, qualify_target
from kvbench.core.config import ConfigModel, KVBenchConfig


class Experiment(ConfigModel):
    target: Path
    controller: str
    expected_serial: str
    artifacts: Path
    configs: dict[str, Path]
    order: list[str] = Field(default_factory=lambda: ["lmcache", "hicache"])
    mode: Literal["endurance", "smoke"] = "endurance"
    min_occupancy: float = Field(default=0.8, ge=0.8, le=0.9)
    max_occupancy: float = Field(default=0.95, gt=0.9, le=0.95)
    drive_fills: float = Field(default=3, ge=2)
    precondition_drive_fills: float = Field(default=1, ge=1)
    min_seconds: float = Field(default=600, ge=1)
    max_seconds: float = Field(default=14 * 86400, ge=1)
    interval_seconds: float = Field(default=5, ge=1, le=60)
    idle_seconds: float = Field(default=60, ge=1)
    drain_seconds: float = Field(default=300, ge=1)
    max_idle_host_bytes_per_second: float = Field(default=1024**2, ge=0)
    concurrency: int = Field(default=8, ge=1, le=64)
    prompt_chars: int = Field(default=65536, ge=32, le=1048576)
    output_tokens: int = Field(default=128, ge=1, le=4096)
    seed: int = Field(default=24301, ge=0)
    process_memory_bytes: int = Field(default=4294967296, ge=512 * 1024**2, le=4294967296)
    min_artifact_free_bytes: int = Field(default=16 * 1024**3, ge=64 * 1024**2)
    smoke_host_bytes: int = Field(default=64 * 1024**2, ge=1)
    smoke_seconds: float = Field(default=10, ge=1)
    smoke_max_requests: int = Field(default=16, ge=1, le=1024)
    influx_file: Path | None = None
    block_trace_seconds: int = Field(default=60, ge=0, le=300)
    sudo: bool = True

    @field_validator("configs")
    @classmethod
    def safe_names(cls, configs: dict[str, Path]) -> dict[str, Path]:
        import re

        if not configs or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in configs):
            raise ValueError("Arm names must be nonempty alphanumeric/dash/underscore names")
        return configs


def load_experiment(path: Path) -> Experiment:
    data = yaml.safe_load(path.read_text())
    for name in ("target", "artifacts", "influx_file"):
        if data.get(name):
            data[name] = str((path.parent / data[name]).resolve())
    data["configs"] = {name: (path.parent / value).resolve() for name, value in data["configs"].items()}
    experiment = Experiment.model_validate(data)
    if not experiment.order or any(name not in experiment.configs for name in experiment.order):
        raise ValueError("Every ordered arm must name an available config")
    if len(set(experiment.order)) < 2:
        raise ValueError("Comparison requires at least two distinct arms")
    if experiment.max_seconds < experiment.min_seconds:
        raise ValueError("max_seconds must be >= min_seconds")
    return experiment


def prepared_config(base: KVBenchConfig, disk: Path, output: Path, port: int) -> KVBenchConfig:
    data = base.model_dump(mode="json")
    data["instance_id"] = output.name
    data["server"].update(host="127.0.0.1", port=port)
    data["kv"]["trace_file"] = str(output / "io.jsonl")
    if base.kv.stack == "hicache":
        data["kv"]["hicache"]["storage_path"] = str(disk)
    elif base.kv.stack == "lmcache":
        if base.kv.lmcache_config_file is None:
            raise ValueError("Comparison requires an explicit LMCache configuration file")
        storage = yaml.safe_load(base.kv.lmcache_config_file.read_text())
        if storage.get("remote_url") or not storage.get("local_disk"):
            raise ValueError("Comparison supports local-disk LMCache only")
        storage["local_disk"] = "file://" + str(disk) + "/"
        path = output / "lmcache.yaml"
        path.write_text(yaml.safe_dump(storage))
        data["kv"]["lmcache_config_file"] = str(path)
    else:
        raise ValueError("Comparison supports lmcache and hicache only")
    result = KVBenchConfig.model_validate(data)
    result.validate_serving()
    result.to_yaml(output / "server.yaml")
    return result


def comparison_signature(config: KVBenchConfig, capacity: dict[str, Any]) -> dict[str, Any]:
    return {"model": config.model.cache_identity(), "topology": config.manifest()["kv_rank_selection"],
            "scheduler": config.scheduler.model_dump(), "timing": config.timing.model_dump(),
            "random_fill": config.kv.random_fill, "seed": config.kv.seed,
            "payload_generator": config.kv.payload_generator,
            "capacity": {k: capacity[k] for k in ("chunk_size", "stored_bytes_per_token",
                                                 "local_cpu_capacity_bytes", "local_disk_capacity_bytes")}}


def goal_met(fields: dict[str, Any], elapsed: float, *, min_seconds: float, host_bytes: int) -> bool:
    return bool(fields.get("run_valid") and elapsed >= min_seconds and fields.get("run_host_bytes", 0) >= host_bytes)


def validate_arms(experiment: Experiment) -> None:
    """Catch unfair/unsupported arms before spending even one drive fill."""
    from kvbench.kv.factory import create_kv_stack

    expected = None
    for name in dict.fromkeys(experiment.order):
        config = KVBenchConfig.from_yaml(experiment.configs[name])
        config.validate_serving()
        create_kv_stack(config)  # validates supported rank/profile families without allocation
        selection = config.manifest()["kv_rank_selection"]
        count = selection["active_rank_count"]
        if config.kv.stack == "hicache":
            opts = config.kv.hicache
            chunk, cpu, disk = opts.page_size, opts.host_size_gb, opts.disk_size_gb
        else:
            if config.kv.lmcache_config_file is None:
                raise ValueError("Comparison requires an explicit LMCache config")
            storage = yaml.safe_load(config.kv.lmcache_config_file.read_text())
            if not storage.get("local_cpu") or not storage.get("local_disk") or storage.get("remote_url"):
                raise ValueError("Comparison requires CPU + local disk tiers only")
            chunk = storage["chunk_size"]
            cpu, disk = storage["max_local_cpu_size"], storage["max_local_disk_size"]
        cap = {"chunk_size": chunk, "stored_bytes_per_token": selection["active_stored_bytes_per_token"],
               "local_cpu_capacity_bytes": int(cpu * 1024**3) * count,
               "local_disk_capacity_bytes": int(disk * 1024**3) * count}
        current = comparison_signature(config, cap)
        if expected is not None and current != expected:
            raise ValueError("Comparison arms differ in model, topology, rates, page geometry or tier sizes")
        expected = current
    if experiment.block_trace_seconds and (not shutil.which("blktrace") or not shutil.which("blkparse")):
        raise ValueError("Install blktrace/blkparse before requesting block samples")


def source_revision() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        found = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                               capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        if Path(found).resolve() != root:
            return None
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


async def sample_idle(probe: NVMeProbe, output: Path, experiment: Experiment, run_id: str) -> None:
    collector = Collector(probe, output / "idle.jsonl", experiment.interval_seconds,
                          run_id=run_id, phase="idle-baseline", influx_file=experiment.influx_file)
    started = time.monotonic()
    while True:
        record = await asyncio.to_thread(collector.collect)
        fields = record["fields"]
        if not fields["sample_valid"] or not fields["run_valid"]:
            raise RuntimeError("Invalid baseline NVMe telemetry")
        elapsed = time.monotonic() - started
        if elapsed >= experiment.idle_seconds:
            rate = fields.get("run_host_bytes", 0) / elapsed
            if rate > experiment.max_idle_host_bytes_per_second:
                raise RuntimeError(f"Other writers active: idle host rate {rate:g} B/s exceeds limit")
            return
        await asyncio.sleep(experiment.interval_seconds)


async def workload_phase(client: httpx.AsyncClient, config: KVBenchConfig, experiment: Experiment,
                         probe: NVMeProbe, report: dict[str, Any], output: Path, arm: str,
                         phase: str, drive_fills: float, run_id: str) -> dict[str, Any]:
    collector = Collector(probe, output / f"{phase}.jsonl", experiment.interval_seconds,
                          run_id=run_id, adapter=arm, phase=phase, influx_file=experiment.influx_file)
    first = await asyncio.to_thread(collector.collect)
    if not first["fields"]["sample_valid"]:
        raise RuntimeError("NAND/host telemetry unavailable before workload")
    latest = first
    stop_load, stop_samples = asyncio.Event(), asyncio.Event()
    errors: list[str] = []
    counts = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
    claims = 0
    started = time.monotonic()

    async def sample_loop() -> None:
        nonlocal latest
        while not stop_samples.is_set():
            await asyncio.sleep(experiment.interval_seconds)
            latest = await asyncio.to_thread(collector.collect)
            fields = latest["fields"]
            if not fields["sample_valid"] or not fields["run_valid"]:
                errors.append("Invalid/reset/gapped NVMe telemetry")
                stop_load.set()
            if experiment.mode == "endurance" and not experiment.min_occupancy <= fields.get("occupancy", 0) <= experiment.max_occupancy:
                errors.append("Drive occupancy left the qualified range")
                stop_load.set()
            if experiment.target.stat().st_dev != report["filesystem_device_id"]:
                errors.append("Target filesystem changed")
                stop_load.set()
            if shutil.disk_usage(output).free < experiment.min_artifact_free_bytes:
                errors.append("Artifact filesystem reached its free-space reserve")
                stop_load.set()

    async def requests(worker: int) -> None:
        nonlocal claims
        rng = random.Random(experiment.seed + worker)
        index = 0
        while not stop_load.is_set():
            if experiment.mode == "smoke" and claims >= experiment.smoke_max_requests:
                stop_load.set()
                return
            claims += 1
            # Per-worker sequence is reproducible across arms. Put unique data at
            # the start to prevent an ever-growing shared-prefix hit workload.
            content = f"{phase}:{worker}:{index}:" + rng.randbytes((experiment.prompt_chars + 1) // 2).hex()
            content = content[:experiment.prompt_chars]
            try:
                response = await client.post("/v1/chat/completions", json={
                    "model": config.server.model_profile,
                    "messages": [{"role": "user", "content": content}], "max_tokens": experiment.output_tokens,
                })
                response.raise_for_status()
                usage = response.json()["usage"]
                counts["requests"] += 1
                counts["prompt_tokens"] += usage["prompt_tokens"]
                counts["completion_tokens"] += usage["completion_tokens"]
            except Exception as exc:
                detail = f"; response={exc.response.text[:2048]}" if isinstance(exc, httpx.HTTPStatusError) else ""
                errors.append(f"Workload request failed: {type(exc).__name__}: {exc}{detail}")
                stop_load.set()
                return
            index += 1

    sampler = asyncio.create_task(sample_loop())
    workers = [asyncio.create_task(requests(i)) for i in range(experiment.concurrency)]
    requested_bytes = (int(report["capacity_bytes"] * drive_fills) if experiment.mode == "endurance"
                       else experiment.smoke_host_bytes)
    minimum = experiment.min_seconds if experiment.mode == "endurance" else experiment.smoke_seconds
    achieved = False
    try:
        while not stop_load.is_set():
            await asyncio.sleep(experiment.interval_seconds)
            if sampler.done():
                sampler.result()
                raise RuntimeError("Telemetry sampler stopped")
            elapsed = time.monotonic() - started
            if goal_met(latest["fields"], elapsed, min_seconds=minimum, host_bytes=requested_bytes):
                achieved = True
                break
            if elapsed >= experiment.max_seconds:
                errors.append("Timed out before required observed host drive fills and duration")
                break
    except Exception as exc:
        errors.append(f"Phase control failed: {type(exc).__name__}: {exc}")
    finally:
        stop_load.set()
        collector.tags["phase"] = phase + "-drain"
        try:
            await asyncio.wait_for(asyncio.gather(*workers), timeout=config.scheduler.request_timeout_s + 10)
            response = await client.post("/kvbench/flush", timeout=150)
            response.raise_for_status()
            await asyncio.to_thread(subprocess.run, ["sync", "-f", str(experiment.target)], check=True, timeout=120)
            # Continue collecting idle NAND writes after dirty host pages drain.
            await asyncio.sleep(experiment.drain_seconds)
        except Exception as exc:
            errors.append(f"Drain failed: {type(exc).__name__}: {exc}")
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
        except BaseException:
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        finally:
            stop_samples.set()
            try:
                await sampler
            except Exception as exc:
                errors.append(f"Telemetry sampler failed: {type(exc).__name__}: {exc}")
    # A bounded smoke workload may reach the device only during syncfs. That
    # validates counter plumbing, but never qualifies as an endurance result.
    if experiment.mode == "smoke":
        achieved = goal_met(latest["fields"], time.monotonic() - started,
                            min_seconds=minimum, host_bytes=requested_bytes)
    result = {"phase": phase, "arm": arm, "mode": experiment.mode, "goal_met": achieved,
              "qualified": experiment.mode == "endurance" and achieved and not errors,
              "requested_host_bytes": requested_bytes, "elapsed_seconds": time.monotonic() - started,
              "counts": counts, "final_fields": latest["fields"], "errors": errors,
              "includes_drain_seconds": experiment.drain_seconds}
    (output / f"{phase}-summary.json").write_text(json.dumps(result, indent=2))
    if errors or not achieved:
        raise RuntimeError(f"Unqualified {arm}/{phase}: {errors}")
    return result


async def compare(experiment: Experiment) -> dict[str, Any]:
    # All qualification is read-only and precedes any benchmark file writes.
    validate_arms(experiment)
    probe = NVMeProbe(experiment.controller, experiment.target, experiment.sudo, experiment.expected_serial)
    report = await asyncio.to_thread(qualify_target, experiment.target, probe, experiment.expected_serial,
                                     experiment.min_occupancy if experiment.mode == "endurance" else 0,
                                     experiment.max_occupancy)
    experiment.artifacts.mkdir(parents=True, exist_ok=True)
    if experiment.artifacts.stat().st_dev == report["filesystem_device_id"]:
        raise ValueError("Artifacts/traces/telemetry must be off the measured filesystem")
    if shutil.disk_usage(experiment.artifacts).free < experiment.min_artifact_free_bytes:
        raise ValueError("Insufficient free space for experiment artifacts")
    if experiment.influx_file:
        experiment.influx_file.parent.mkdir(parents=True, exist_ok=True)
        if experiment.influx_file.parent.stat().st_dev == report["filesystem_device_id"]:
            raise ValueError("Influx telemetry output must be off the measured filesystem")
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output = experiment.artifacts / run_id
    output.mkdir()
    summary: dict[str, Any] = {"run_id": run_id, "status": "running", "qualified": False,
                               "source_revision": source_revision(),
                               "config": experiment.model_dump(mode="json"), "device": report, "arms": []}
    summary_path = output / "experiment.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Experiment artifacts: {output}", flush=True)
    signature = None
    try:
        for sequence, arm in enumerate(experiment.order):
            await sample_idle(probe, output, experiment, run_id)
            arm_output = output / f"{sequence:02d}-{arm}"
            arm_output.mkdir()
            disk = experiment.target / f"kvbench-{run_id}-{sequence:02d}-{arm}"
            disk.mkdir(exist_ok=False)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            config = prepared_config(KVBenchConfig.from_yaml(experiment.configs[arm]), disk, arm_output, port)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("KVBENCH_", "LMCACHE_", "SGLANG_"))}
            env.update(MALLOC_ARENA_MAX="2", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
                       ROCR_VISIBLE_DEVICES="", LMCACHE_TRACK_USAGE="false", LMCACHE_LOG_LEVEL="ERROR")
            with (arm_output / "server.log").open("w") as log:
                process = subprocess.Popen(
                    ["prlimit", f"--as={experiment.process_memory_bytes}", "--", sys.executable,
                     "-m", "kvbench.cli.main", "serve", "--config", str(arm_output / "server.yaml")],
                    stdout=log, stderr=subprocess.STDOUT, env=env,
                )
                try:
                    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}",
                                                  timeout=config.scheduler.request_timeout_s + 10) as client:
                        async with asyncio.timeout(90):
                            while True:
                                if process.poll() is not None:
                                    raise RuntimeError(f"Server exited; see {arm_output / 'server.log'}")
                                try:
                                    if (await client.get("/health", timeout=2)).status_code == 200:
                                        break
                                except httpx.HTTPError:
                                    pass
                                await asyncio.sleep(0.2)
                        manifest = (await client.get("/kvbench/manifest")).json()
                        state = (await client.get("/kvbench/state")).json()
                        (arm_output / "manifest.json").write_text(json.dumps(manifest, indent=2))
                        current = comparison_signature(config, state["capacity"])
                        if signature is not None and signature != current:
                            raise ValueError("Comparison arms differ in workload rates, model, ranks, page bytes or tier capacities")
                        signature = current
                        quota = state["capacity"]["local_disk_capacity_bytes"]
                        usage = shutil.disk_usage(experiment.target)
                        if quota > usage.free - usage.total * (1 - experiment.max_occupancy):
                            raise ValueError("Cache quota would consume the required free-space reserve")
                        stages = []
                        if experiment.mode == "endurance":
                            stages.append(await workload_phase(client, config, experiment, probe, report, arm_output,
                                                               arm, "precondition", experiment.precondition_drive_fills, run_id))
                        block_stats = None
                        tracer = None
                        with (arm_output / "blktrace.log").open("w") as trace_log:
                            if experiment.block_trace_seconds:
                                command = (["sudo", "-n"] if experiment.sudo else []) + [
                                    "blktrace", "-d", report["disk"]["name"], "-a", "issue",
                                    "-w", str(experiment.block_trace_seconds), "-b", "64", "-n", "2",
                                    "-D", str(arm_output), "-o", "block",
                                ]
                                tracer = subprocess.Popen(command, stdout=trace_log, stderr=subprocess.STDOUT)
                                await asyncio.sleep(0.2)
                                if tracer.poll() is not None and tracer.returncode != 0:
                                    raise RuntimeError("blktrace failed; inspect blktrace.log")
                            try:
                                stages.append(await workload_phase(client, config, experiment, probe, report, arm_output,
                                                                   arm, "measure", experiment.drive_fills, run_id))
                            finally:
                                if tracer:
                                    # The kernel trace has its own bounded -w lifetime,
                                    # including when requests fail or this task cancels.
                                    await asyncio.to_thread(tracer.wait, timeout=experiment.block_trace_seconds + 10)
                                    if tracer.returncode != 0:
                                        raise RuntimeError("blktrace failed; inspect blktrace.log")
                            if tracer:
                                trace_log.flush()
                                trace_messages = (arm_output / "blktrace.log").read_text()
                                validate_trace_log(trace_messages)
                                await asyncio.to_thread(subprocess.run, [
                                    "blkparse", "-q", "-i", str(arm_output / "block"), "-f", "%T.%t %a %d %S %n\\n",
                                    "-o", str(arm_output / "block-issues.txt"),
                                ], check=True, stdout=trace_log, stderr=subprocess.STDOUT, timeout=120)
                                block_stats = summarize_issues(arm_output / "block-issues.txt")
                                (arm_output / "block-summary.json").write_text(json.dumps(block_stats, indent=2))
                        final_state = (await client.get("/kvbench/state")).json()
                        (arm_output / "final-state.json").write_text(json.dumps(final_state, indent=2))
                        summary["arms"].append({"arm": arm, "sequence": sequence, "stages": stages,
                                                "block_sample": block_stats})
                        summary_path.write_text(json.dumps(summary, indent=2))
                finally:
                    process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait)
                        raise RuntimeError("Server did not drain/stop cleanly") from None
                    if process.returncode not in (0, -15):
                        raise RuntimeError(f"Server shutdown failed with exit code {process.returncode}")
        summary.update(status="complete", qualified=experiment.mode == "endurance")
    except BaseException as exc:
        summary.update(status="failed", error=str(exc), qualified=False)
        raise
    finally:
        summary_path.write_text(json.dumps(summary, indent=2))
    return summary


async def campaign(experiment: Experiment, rate_mib: float, *,
                   filler_generator: Literal["system", "sfc64"] = "system",
                   fsync_bytes: int = 64 * 1024**2) -> dict[str, Any]:
    """Real filler -> qualified A/B campaign, with durable preparation telemetry."""
    if experiment.mode != "endurance":
        raise ValueError("campaign prepares a drive for endurance, not smoke mode")
    validate_arms(experiment)
    probe = NVMeProbe(experiment.controller, experiment.target, experiment.sudo, experiment.expected_serial)
    report = await asyncio.to_thread(qualify_target, experiment.target, probe, experiment.expected_serial, 0, experiment.max_occupancy)
    experiment.artifacts.mkdir(parents=True, exist_ok=True)
    if experiment.artifacts.stat().st_dev == report["filesystem_device_id"]:
        raise ValueError("Preparation telemetry must live off the test drive")
    if shutil.disk_usage(experiment.artifacts).free < experiment.min_artifact_free_bytes:
        raise ValueError("Insufficient artifact free space before preparation")
    if experiment.influx_file:
        experiment.influx_file.parent.mkdir(parents=True, exist_ok=True)
        if experiment.influx_file.parent.stat().st_dev == report["filesystem_device_id"]:
            raise ValueError("Preparation Influx output must live off the test drive")
    name = f"prepare-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    output = experiment.artifacts / name
    output.mkdir()
    (output / "campaign.json").write_text(json.dumps({"source_revision": source_revision(),
        "experiment": experiment.model_dump(mode="json"), "device": report, "rate_mib": rate_mib,
        "filler_generator": filler_generator, "fsync_bytes": fsync_bytes}, indent=2))
    collector = Collector(probe, output / "samples.jsonl", experiment.interval_seconds,
                          run_id=name, adapter="filler", phase="preparation", influx_file=experiment.influx_file)
    record = await asyncio.to_thread(collector.collect)
    if not record["fields"]["sample_valid"]:
        raise RuntimeError("Preparation requires functioning NAND telemetry")
    desired_occupancy = max(0.85, experiment.min_occupancy)
    stop_filler = threading.Event()
    task = asyncio.create_task(asyncio.to_thread(fill_target, experiment.target / name, probe,
                                               experiment.expected_serial, occupancy=desired_occupancy,
                                               rate_mib=rate_mib, stop=stop_filler, generator=filler_generator,
                                               fsync_bytes=fsync_bytes))
    print(f"Preparation telemetry: {output}; filler rate {rate_mib:g} MiB/s", flush=True)
    try:
        while not task.done():
            await asyncio.sleep(experiment.interval_seconds)
            record = await asyncio.to_thread(collector.collect)
            if not record["fields"]["sample_valid"] or not record["fields"]["run_valid"]:
                raise RuntimeError("Preparation telemetry failed; stopping filler")
        prepared = await asyncio.shield(task)
    finally:
        stop_filler.set()
        await asyncio.gather(task, return_exceptions=True)
    (output / "preparation.json").write_text(json.dumps(prepared, indent=2))
    await asyncio.to_thread(collector.collect)
    if prepared["status"] != "prepared":
        raise RuntimeError("Drive preparation incomplete")
    return await compare(experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight", help="Read-only backing-device/counter/occupancy qualification")
    fill = commands.add_parser("fill", help="Create real, paced filler in a NEW directory; preserve existing data")
    for command in (check, fill):
        command.add_argument("--target", type=Path, required=True)
        command.add_argument("--controller", required=True)
        command.add_argument("--expected-serial", required=True)
        command.add_argument("--sudo", action="store_true")
    check.add_argument("--min-occupancy", type=float, default=0.8)
    fill.add_argument("--occupancy", type=float, default=0.85)
    fill.add_argument("--rate-mib", type=float, default=128)
    fill.add_argument("--max-bytes", type=int)
    fill.add_argument("--generator", choices=("system", "sfc64"), default="system")
    fill.add_argument("--fsync-mib", type=int, default=64)
    run = commands.add_parser("compare", help="Run each configured cache arm until real host-write goals are met")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--mode", choices=("endurance", "smoke"))
    full = commands.add_parser("campaign", help="Fill to 85% with paced real data, then run the endurance comparison")
    full.add_argument("--config", type=Path, required=True)
    full.add_argument("--rate-mib", type=float, default=128)
    full.add_argument("--filler-generator", choices=("system", "sfc64"), default="system")
    full.add_argument("--fsync-mib", type=int, default=64)
    args = parser.parse_args()
    if args.command in ("compare", "campaign"):
        experiment = load_experiment(args.config)
        if getattr(args, "mode", None):
            experiment.mode = args.mode
        with device_lease(experiment.controller):
            result = asyncio.run(campaign(experiment, args.rate_mib, filler_generator=args.filler_generator,
                                           fsync_bytes=args.fsync_mib * 1024**2)
                                 if args.command == "campaign" else compare(experiment))
    else:
        probe = NVMeProbe(args.controller, args.target if args.command == "preflight" else args.target.parent, args.sudo, args.expected_serial)
        if args.command == "preflight":
            result = qualify_target(args.target, probe, args.expected_serial, min_occupancy=args.min_occupancy)
        else:
            with device_lease(args.controller):
                result = fill_target(args.target, probe, args.expected_serial, occupancy=args.occupancy,
                                     rate_mib=args.rate_mib, max_bytes=args.max_bytes, generator=args.generator,
                                     fsync_bytes=args.fsync_mib * 1024**2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
