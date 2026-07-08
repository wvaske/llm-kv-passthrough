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
    workers: int | None = typer.Option(None, "--workers", "-w", help="Number of worker processes"),
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
    if gpu is not None:
        updates["gpu"] = cfg.gpu.model_copy(update={"gpu_profile": gpu})
    if lmcache_config is not None:
        updates["kv"] = cfg.kv.model_copy(update={"lmcache_config_file": lmcache_config})
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
