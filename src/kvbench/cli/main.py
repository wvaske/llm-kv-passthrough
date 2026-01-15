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
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    model: str = typer.Option("llama-3.1-8b", "--model", "-m", help="Model profile to emulate"),
    gpu: str = typer.Option("H100_SXM", "--gpu", "-g", help="GPU profile to emulate"),
    server_type: str = typer.Option("combined", "--type", "-t", help="Server type: combined, prefill, decode, proxy"),
    storage: str = typer.Option("memory", "--storage", "-s", help="Storage backend: memory, local_disk, redis"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of worker processes"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
) -> None:
    """Start the KV-Bench server."""
    import os

    console.print("[bold blue]KV-Bench Server[/bold blue]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Model: {model}")
    console.print(f"GPU: {gpu}")
    console.print(f"Server Type: {server_type}")
    console.print(f"Storage: {storage}")

    if config:
        console.print(f"Config: {config}")
        cfg = KVBenchConfig.from_yaml(config)
    else:
        from kvbench.core.config import GPUEmulationConfig, ServerConfig, StorageConfig

        cfg = KVBenchConfig(
            server=ServerConfig(
                host=host,
                port=port,
                model_profile=model,
                server_type=server_type,  # type: ignore[arg-type]
                workers=workers,
            ),
            gpu=GPUEmulationConfig(gpu_profile=gpu),
            storage=StorageConfig(backend_type=storage),  # type: ignore[arg-type]
        )

    console.print("\n[green]Configuration loaded:[/green]")
    console.print(f"  Instance ID: {cfg.instance_id}")
    console.print(f"  Server Type: {cfg.server.server_type}")
    console.print(f"  Storage: {cfg.storage.backend_type}")
    console.print(f"  Model: {cfg.server.model_profile}")
    console.print(f"  GPU: {cfg.gpu.gpu_profile}")

    # Set environment variables for the app
    os.environ["KVBENCH_SERVER__HOST"] = host
    os.environ["KVBENCH_SERVER__PORT"] = str(port)
    os.environ["KVBENCH_SERVER__MODEL_PROFILE"] = model
    os.environ["KVBENCH_SERVER__SERVER_TYPE"] = server_type
    os.environ["KVBENCH_GPU__GPU_PROFILE"] = gpu
    os.environ["KVBENCH_STORAGE__BACKEND_TYPE"] = storage

    console.print(f"\n[green]Starting server at http://{host}:{port}[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    # Start uvicorn server
    import uvicorn

    uvicorn.run(
        "kvbench.servers.app:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info",
    )


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
