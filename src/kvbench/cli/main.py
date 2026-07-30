"""
KV-Bench Command Line Interface.

This module provides the main CLI entry point for KV-Bench.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from kvbench.core import (
    KVBenchConfig,
    get_gpu_profile_info,
    get_model_profile_info,
    list_gpu_profiles,
    list_model_profiles,
)

app = typer.Typer(
    name="kvbench",
    help="KV-Bench: Distributed KV Cache Benchmarking System",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: int | None = typer.Option(None, "--port", "-p", help="Port to listen on"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model profile to emulate"),
    gpu: str | None = typer.Option(None, "--gpu", "-g", help="GPU profile to emulate"),
    server_type: str | None = typer.Option(
        None, "--type", "-t", help="Server type: combined, prefill, decode, proxy"
    ),
    lmcache_config: str | None = typer.Option(
        None,
        "--lmcache-config",
        "-s",
        help="Path to LMCache's own config file (storage backends, tier sizes); "
        "LMCACHE_* env vars are used when unset",
    ),
    trace_file: str | None = typer.Option(
        None,
        "--trace-file",
        help="Record every KV storage operation (logical + file I/O) to this "
        "JSONL file, for `kvbench trace2fio`",
    ),
    random_fill: bool | None = typer.Option(
        None,
        "--random-fill/--no-random-fill",
        help="Fill KV tensors with incompressible random data (default: on)",
    ),
    workers: int | None = typer.Option(None, "--workers", "-w", help="Number of worker processes"),
    tp_size: int | None = typer.Option(
        None, "--tp-size", help="Tensor parallelism size for the emulated GPUs"
    ),
    pp_size: int | None = typer.Option(
        None, "--pp-size", help="Pipeline parallelism size for the emulated GPUs"
    ),
    simple_timing: bool | None = typer.Option(
        None,
        "--simple-timing/--roofline-timing",
        help="Use fixed ms/token timing instead of the roofline model",
    ),
    prefill_ms_per_token: float | None = typer.Option(
        None,
        "--prefill-ms-per-token",
        help="Prefill latency per token in ms (simple timing mode)",
    ),
    decode_ms_per_token: float | None = typer.Option(
        None,
        "--decode-ms-per-token",
        help="Decode latency per token in ms (simple timing mode)",
    ),
    config: str | None = typer.Option(
        None, "--config", "-c", help="Path to config file (explicit CLI flags override it)"
    ),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
) -> None:
    """Start the KV-Bench server.

    Configuration precedence: explicit CLI flags > --config file >
    KVBENCH_* environment variables > defaults.
    """
    import os
    import tempfile

    console.print("[bold blue]KV-Bench Server[/bold blue]")

    # Base configuration: --config file, else environment, else defaults
    if config:
        console.print(f"Config: {config}")
        cfg = KVBenchConfig.from_yaml(config)
    else:
        cfg = KVBenchConfig.from_env()

    # Overlay explicit CLI flags onto the base configuration
    updates: dict = {}
    server_updates = {
        key: value
        for key, value in {
            "host": host,
            "port": port,
            "model_profile": model,
            "server_type": server_type,
            "workers": workers,
        }.items()
        if value is not None
    }
    if server_updates:
        updates["server"] = cfg.server.model_copy(update=server_updates)
    gpu_updates = {
        key: value
        for key, value in {"gpu_profile": gpu, "tp_size": tp_size}.items()
        if value is not None
    }
    if gpu_updates:
        updates["gpu"] = cfg.gpu.model_copy(update=gpu_updates)
    timing_updates = {
        key: value
        for key, value in {
            "simple_mode": simple_timing,
            "prefill_ms_per_token": prefill_ms_per_token,
            "decode_ms_per_token": decode_ms_per_token,
            "pp_size": pp_size,
        }.items()
        if value is not None
    }
    if timing_updates:
        updates["timing"] = cfg.timing.model_copy(update=timing_updates)
    kv_updates = {
        key: value
        for key, value in {
            "lmcache_config_file": lmcache_config,
            "trace_file": trace_file,
            "random_fill": random_fill,
        }.items()
        if value is not None
    }
    if kv_updates:
        updates["kv"] = cfg.kv.model_copy(update=kv_updates)
    if updates:
        # Re-validate the merged configuration (model_copy skips validators)
        cfg = KVBenchConfig.model_validate(cfg.model_copy(update=updates).model_dump(mode="json"))

    console.print("\n[green]Configuration loaded:[/green]")
    console.print(f"  Instance ID: {cfg.instance_id}")
    console.print(f"  Server Type: {cfg.server.server_type}")
    console.print(f"  KV Stack: {cfg.kv.stack}")
    console.print(
        f"  LMCache Config: {cfg.kv.lmcache_config_file or 'LMCACHE_* env / defaults'}"
    )
    console.print(f"  Model: {cfg.server.model_profile}")
    console.print(f"  GPU: {cfg.gpu.gpu_profile}")
    if cfg.timing.simple_mode:
        console.print(
            f"  Timing: simple ({cfg.timing.prefill_ms_per_token} ms/token prefill, "
            f"{cfg.timing.decode_ms_per_token} ms/token decode)"
        )
    else:
        console.print(
            f"  Timing: roofline (TP={cfg.gpu.tp_size}, PP={cfg.timing.pp_size}, "
            f"comm: TP {'on' if cfg.timing.include_tp_communication else 'off'}, "
            f"PP {'on' if cfg.timing.include_pp_communication else 'off'})"
        )

    # Persist the fully-resolved config and point the app factory at it.
    # This survives uvicorn worker processes and --reload subprocesses,
    # which re-import the app module in a fresh interpreter.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="kvbench-config-", delete=False
    ) as tmp:
        resolved_config_path = tmp.name
    cfg.to_yaml(resolved_config_path)
    os.environ["KVBENCH_CONFIG_FILE"] = resolved_config_path

    bind_host = cfg.server.host
    bind_port = cfg.server.port
    console.print(f"\n[green]Starting server at http://{bind_host}:{bind_port}[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Start uvicorn server
    import uvicorn

    try:
        uvicorn.run(
            "kvbench.servers.app:get_app",
            factory=True,
            host=bind_host,
            port=bind_port,
            workers=cfg.server.workers if not reload else 1,
            reload=reload,
            log_level=cfg.server.log_level,
        )
    finally:
        try:
            os.unlink(resolved_config_path)
        except OSError:
            pass


@app.command()
def warmup(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="KV-Bench server URL"),
    target_gb: float | None = typer.Option(
        None,
        "--target-gb",
        help="Explicit fill target in GB (default: fill-factor x configured tier capacity)",
    ),
    fill_factor: float = typer.Option(
        1.25,
        "--fill-factor",
        help="Multiple of total tier capacity to store (>1 forces eviction)",
    ),
    seq_tokens: int = typer.Option(
        2048, "--seq-tokens", help="Tokens per stored sequence (chunk-aligned)"
    ),
    concurrency: int = typer.Option(4, "--concurrency", help="Parallel store workers"),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Poll until the warmup finishes"
    ),
) -> None:
    """Fill the server's KV cache to steady state (all tiers full, evicting).

    Warmup runs inside the server process (local cache tiers belong to the
    engine instance); this command starts it and reports progress.
    """
    import time as _time

    import httpx

    body: dict = {
        "fill_factor": fill_factor,
        "seq_tokens": seq_tokens,
        "concurrency": concurrency,
    }
    if target_gb is not None:
        body["target_gb"] = target_gb

    with httpx.Client(base_url=url, timeout=30.0) as client:
        response = client.post("/kvbench/warmup", json=body)
        if response.status_code == 409:
            console.print(f"[red]{response.json().get('detail')}[/red]")
            raise typer.Exit(1)
        response.raise_for_status()
        status = response.json()
        console.print(
            f"[green]Warmup started:[/green] target "
            f"{status['target_bytes'] / 1e9:.2f} GB, params {status['params']}"
        )

        if not wait:
            console.print("Poll with: GET /kvbench/warmup")
            return

        from rich.progress import (
            BarColumn,
            Progress,
            TextColumn,
            TimeElapsedColumn,
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TextColumn("{task.fields[rate]} MB/s"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task("Filling KV cache", total=status["target_bytes"], rate="0.0")
            while True:
                _time.sleep(2)
                status = client.get("/kvbench/warmup").json()
                progress.update(
                    bar,
                    completed=min(status["stored_bytes"], status["target_bytes"]),
                    rate=f"{status['rate_mb_s']:.0f}",
                )
                if status["state"] != "running":
                    break

    if status["state"] == "done":
        evicting = status.get("evicting")
        console.print(
            f"[green]Warmup complete:[/green] {status['stored_bytes'] / 1e9:.2f} GB "
            f"in {status['sequences']} sequences ({status['elapsed_s']}s)"
        )
        if evicting:
            console.print(
                "[green]Steady state confirmed:[/green] earliest stored data has been evicted"
            )
        else:
            console.print(
                "[yellow]Warning: earliest stored data is still cached — the cache "
                "may not be full yet. Increase --fill-factor or --target-gb.[/yellow]"
            )
    else:
        console.print(f"[red]Warmup {status['state']}: {status.get('error') or ''}[/red]")
        raise typer.Exit(1)


@app.command()
def trace2fio(
    trace: str = typer.Argument(..., help="KV trace JSONL file (from serve --trace-file)"),
    output: str = typer.Option("kv_workload.fio", "--output", "-o", help="FIO job file to write"),
    backend: str = typer.Option(
        "LocalDiskBackend", "--backend", "-b", help="Backend class to model"
    ),
    directory: str = typer.Option(
        "/mnt/kvcache", "--directory", "-d", help="Target directory in the FIO job"
    ),
    runtime: int = typer.Option(300, "--runtime", help="FIO runtime in seconds"),
    paced: bool = typer.Option(
        False,
        "--paced",
        help="Cap FIO at the observed throughput (reproduce intensity, not just shape)",
    ),
) -> None:
    """Derive an FIO job file from a recorded KV I/O trace.

    Analyzes the logical and physical operations LMCache performed (chunk
    writes, reads, evictions, parallelism) and emits an FIO job that
    reproduces that workload shape on a raw filesystem.
    """
    from kvbench.trace.analyze import list_backends, load_events, summarize
    from kvbench.trace.fio import generate_fio_job

    events = load_events(trace)
    if not events:
        console.print(f"[red]No events found in {trace}[/red]")
        raise typer.Exit(1)

    available = list_backends(events)
    if backend not in available:
        console.print(
            f"[red]Backend {backend!r} not in trace. Backends present: {available}[/red]"
        )
        raise typer.Exit(1)

    summary = summarize(events, backend=backend)

    table = Table(title=f"KV workload summary ({backend})", show_header=True)
    table.add_column("Metric", style="green")
    table.add_column("Write")
    table.add_column("Read")
    write, read = summary.io_write, summary.io_read
    table.add_row("Operations", str(write.count), str(read.count))
    table.add_row("Bytes", f"{write.total_bytes / 1e9:.2f} GB", f"{read.total_bytes / 1e9:.2f} GB")
    table.add_row(
        "Dominant size",
        f"{(write.dominant_size or 0) / 1e6:.1f} MB",
        f"{(read.dominant_size or 0) / 1e6:.1f} MB",
    )
    table.add_row("Max concurrency", str(write.max_concurrency), str(read.max_concurrency))
    table.add_row("Threads", str(len(write.threads)), str(len(read.threads)))
    table.add_row(
        "Median latency",
        f"{write.median_dur_ms or 0:.2f} ms",
        f"{read.median_dur_ms or 0:.2f} ms",
    )
    table.add_row(
        "Throughput",
        f"{write.bytes_per_sec / 1e6:.1f} MB/s",
        f"{read.bytes_per_sec / 1e6:.1f} MB/s",
    )
    console.print(table)
    console.print(
        f"Evictions (deletes): {summary.logical_remove.count}; "
        f"steady-state files: {summary.steady_state_files}; "
        f"O_DIRECT: {summary.use_odirect}"
    )

    try:
        job = generate_fio_job(
            summary, directory=directory, runtime_s=runtime, paced=paced
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None

    with open(output, "w") as f:
        f.write(job)
    console.print(f"\n[green]FIO job written to {output}[/green]")
    console.print(f"Run with: fio {output}  (adjust directory= first)")


@app.command()
def list_profiles() -> None:
    """List available GPU and model profiles."""
    console.print("[bold blue]Available GPU Profiles[/bold blue]\n")

    gpu_table = Table(show_header=True, header_style="bold cyan")
    gpu_table.add_column("Name", style="green")
    gpu_table.add_column("BF16 TFLOPS")
    gpu_table.add_column("HBM BW (TB/s)")
    gpu_table.add_column("HBM Capacity (GB)")
    gpu_table.add_column("TDP (W)")

    for name in list_gpu_profiles():
        info = get_gpu_profile_info(name)
        gpu_table.add_row(
            name,
            str(info["bf16_tflops"]),
            str(info["hbm_bandwidth_tb_s"]),
            str(info["hbm_capacity_gb"]),
            str(info["tdp_watts"] or "N/A"),
        )

    console.print(gpu_table)

    console.print("\n[bold blue]Available Model Profiles[/bold blue]\n")

    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("Name", style="green")
    model_table.add_column("Layers")
    model_table.add_column("Hidden")
    model_table.add_column("KV Heads")
    model_table.add_column("Est. Params (B)")
    model_table.add_column("Size (GB)")

    for name in list_model_profiles():
        info = get_model_profile_info(name)
        model_table.add_row(
            name,
            str(info["layers"]),
            str(info["hidden"]),
            str(info["kv_heads"]),
            str(info["estimated_params_billions"]),
            str(info["model_size_gb"]),
        )

    console.print(model_table)


@app.command()
def info(
    model: str = typer.Option(None, "--model", "-m", help="Model profile to show"),
    gpu: str = typer.Option(None, "--gpu", "-g", help="GPU profile to show"),
) -> None:
    """Show detailed information about a profile."""
    if gpu:
        try:
            info_dict = get_gpu_profile_info(gpu)
            console.print(f"\n[bold blue]GPU Profile: {gpu}[/bold blue]\n")
            for key, value in info_dict.items():
                console.print(f"  {key}: {value}")
        except KeyError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from None

    if model:
        try:
            info_dict = get_model_profile_info(model)
            console.print(f"\n[bold blue]Model Profile: {model}[/bold blue]\n")
            for key, value in info_dict.items():
                console.print(f"  {key}: {value}")
        except KeyError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from None

    if not gpu and not model:
        console.print("[yellow]Please specify --gpu or --model[/yellow]")
        raise typer.Exit(1)


@app.command()
def version() -> None:
    """Show version information."""
    from kvbench import __version__

    console.print(f"[bold blue]KV-Bench[/bold blue] version {__version__}")


if __name__ == "__main__":
    app()
