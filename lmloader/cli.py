"""CLI tool for LM Loader.

Architecture: the server (`lmloader serve`) manages llama-server processes.
Commands that mutate state (load/unload/switch) talk to the running server via HTTP.
Read-only commands (list, scan, engine-info) work standalone via the database/config.
"""

import httpx
import typer
import asyncio
from typing import Optional
from rich.console import Console
from rich.table import Table

from lmloader.config import settings
from lmloader.infra.database import db
from lmloader.infra.scanner import scanner
from lmloader.engine.registry import engine_registry

app = typer.Typer(help="掬水 CLI - Manage local AI models")
console = Console(width=140)


def _server_url() -> str:
    return f"http://{settings.server.host}:{settings.server.port}"


async def _server_alive() -> bool:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(2.0)) as client:
            resp = await client.get(f"{_server_url()}/health")
            return resp.status_code == 200
    except Exception:
        return False


def _require_server():
    async def _check():
        if not await _server_alive():
            console.print("[red][FAIL][/red] 掬水 server is not running.")
            console.print("Start it first with: [bold]lmloader serve[/bold]")
            raise typer.Exit(1)
    asyncio.run(_check())


@app.command()
def scan():
    """Scan for models in configured directories."""
    async def _scan():
        await db.init()
        result = await scanner.scan_and_register()
        console.print(
            f"[green][OK][/green] Registered {result['scanned']} new model(s), "
            f"removed {len(result['removed'])} stale record(s)"
        )

    asyncio.run(_scan())


@app.command(name="list")
def list_models(status: Optional[str] = typer.Option(None, "--status", "-s")):
    """List registered models."""
    async def _list():
        await db.init()
        models = await db.list_models(status)

        if not models:
            console.print("[yellow]No models found[/yellow]")
            return

        table = Table(title="Models")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Architecture")
        table.add_column("Quantization")
        table.add_column("Parameters")
        table.add_column("Size", justify="right")
        table.add_column("Status", style="bold")

        for model in models:
            size_mb = model.size / (1024 * 1024)
            table.add_row(
                str(model.id),
                model.name,
                model.architecture,
                model.quantization,
                model.parameters,
                f"{size_mb:.0f}MB",
                model.status,
            )

        console.print(table)

    asyncio.run(_list())


@app.command()
def load(
    model_name: str = typer.Argument(..., help="Model name to load"),
    keep_previous: bool = typer.Option(False, "--keep", "-k"),
):
    """Load a model (starts llama-server via the running server)."""
    _require_server()

    async def _load():
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{_server_url()}/api/models/load",
                json={"model_name": model_name, "keep_previous": keep_previous},
            )
            if resp.status_code != 200:
                detail = resp.json().get("detail", "Unknown error")
                console.print(f"[red][FAIL][/red] {detail}")
                raise typer.Exit(1)

            console.print(f"[green][OK][/green] Loaded {model_name}")

            loaded_resp = await client.get(f"{_server_url()}/api/models/loaded")
            if loaded_resp.status_code == 200:
                for item in loaded_resp.json().get("loaded", []):
                    if item["model"]["name"] == model_name:
                        console.print(f"  Port: {item['port']} | PID: {item['pid']} | URL: {item['base_url']}")

    asyncio.run(_load())


@app.command()
def unload(model_name: str = typer.Argument(..., help="Model name to unload")):
    """Unload a model (stops llama-server)."""
    _require_server()

    async def _unload():
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
            resp = await client.post(
                f"{_server_url()}/api/models/unload",
                params={"model_name": model_name},
            )
            if resp.status_code != 200:
                detail = resp.json().get("detail", "Unknown error")
                console.print(f"[red][FAIL][/red] {detail}")
                raise typer.Exit(1)

            console.print(f"[green][OK][/green] Unloaded {model_name}")

    asyncio.run(_unload())


@app.command()
def switch(
    model_name: str = typer.Argument(..., help="Model name to switch to"),
    keep_previous: bool = typer.Option(False, "--keep", "-k"),
):
    """Switch to a different model."""
    _require_server()

    async def _switch():
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{_server_url()}/api/models/switch",
                json={"model_name": model_name, "keep_previous": keep_previous},
            )
            if resp.status_code != 200:
                detail = resp.json().get("detail", "Unknown error")
                console.print(f"[red][FAIL][/red] {detail}")
                raise typer.Exit(1)

            console.print(f"[green][OK][/green] Switched to {model_name}")

    asyncio.run(_switch())


@app.command(name="status")
def status():
    """Show current status."""
    async def _status():
        console.print("[bold]Engines:[/bold]")
        engines = await engine_registry.detect_all()

        if not engines:
            console.print("  [red][FAIL][/red] No engines available")
        else:
            for name, engine in engines.items():
                console.print(f"  [green][{name}][/green] llama-server {engine.version}")
                console.print(f"    Path: {engine.binary_path}")
                if engine.cuda_available:
                    console.print(f"    [green][OK][/green] CUDA available")
                else:
                    console.print(f"    [yellow][WARN][/yellow] CUDA not found")

        console.print("\n[bold]Server:[/bold]")
        if await _server_alive():
            console.print(f"  [green][OK][/green] Running at {_server_url()}")

            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{_server_url()}/api/models/loaded")
                if resp.status_code == 200:
                    loaded = resp.json().get("loaded", [])
                    console.print("\n[bold]Loaded Models:[/bold]")
                    if not loaded:
                        console.print("  [yellow]No models loaded[/yellow]")
                    else:
                        for item in loaded:
                            model = item["model"]
                            port = item["port"]
                            pid = item["pid"]
                            engine_slot = item.get("engine_slot", "unknown")
                            console.print(f"  [green][*][/green] {model['name']} (port={port}, pid={pid}, engine={engine_slot})")
        else:
            console.print("  [yellow]Not running[/yellow] (start with: lmloader serve)")

    asyncio.run(_status())


@app.command()
def engine_info():
    """Show engine (llama-server) info."""
    async def _info():
        engines = await engine_registry.detect_all()
        if not engines:
            console.print("[red][FAIL][/red] No engines found")
            raise typer.Exit(1)

        for name, engine in engines.items():
            console.print(f"\n[bold][{name}][/bold] llama-server {engine.version}")
            console.print(f"  Binary: {engine.binary_path}")
            console.print(f"  CUDA:   {'available' if engine.cuda_available else 'not found'}")

        console.print(f"\n[bold]Default launch params (primary):[/bold]")
        d = settings.engines.primary.defaults
        console.print(f"  GPU layers: {d.n_gpu_layers}")
        console.print(f"  Context:    {d.ctx_size}")
        console.print(f"  KV cache:   {d.cache_type_k}/{d.cache_type_v}")
        console.print(f"  Flash attn: {d.flash_attn}")
        console.print(f"  Threads:    {d.threads}")
        console.print(f"  Port range: {settings.port_range.start}-{settings.port_range.end}")

    asyncio.run(_info())


@app.command()
def pull(
    model_spec: str = typer.Argument(..., help="Model spec: 'repo/model:tag' or HuggingFace URL"),
):
    """Download a model from HuggingFace."""
    _require_server()

    async def _pull():
        console.print(f"[bold]Pulling:[/bold] {model_spec}")
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            async with client.stream(
                "POST",
                f"{_server_url()}/api/pull",
                params={"name": model_spec, "stream": True},
            ) as resp:
                if resp.status_code != 200:
                    console.print(f"[red][FAIL][/red] Pull failed: {resp.status_code}")
                    raise typer.Exit(1)

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        import json
                        event = json.loads(line)
                        status = event.get("status", "")

                        if status == "downloading":
                            progress = event.get("progress", 0)
                            speed = event.get("speed", 0)
                            speed_mb = speed / 1024 / 1024
                            bar_len = 30
                            filled = int(bar_len * progress / 100)
                            bar = "█" * filled + "░" * (bar_len - filled)
                            console.print(f"\r  [{bar}] {progress:.1f}%  {speed_mb:.1f} MB/s", end="")
                        elif status == "resuming":
                            resumed = event.get("resumed_bytes", 0)
                            console.print(f"  Resuming from {resumed / 1024 / 1024:.1f} MB")
                        elif status == "completed":
                            console.print(f"\n[green][OK][/green] Downloaded: {event.get('path', '')}")
                        elif status == "error":
                            console.print(f"\n[red][FAIL][/red] {event.get('error', 'Unknown error')}")
                            raise typer.Exit(1)
                    except (json.JSONDecodeError, KeyError):
                        continue

    asyncio.run(_pull())


@app.command()
def run(
    model_name: str = typer.Argument(..., help="Model name to load"),
    keep_previous: bool = typer.Option(False, "--keep", "-k"),
):
    """Auto-start server if needed, then load a model (daemon mode)."""
    async def _run():
        if not await _server_alive():
            console.print("[yellow]Server not running, starting...[/yellow]")
            import subprocess
            import sys
            from lmloader.infra.proc_util import no_window_kwargs
            proc = subprocess.Popen(
                [sys.executable, "-m", "lmloader.cli", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **no_window_kwargs(),
            )
            for _ in range(30):
                await asyncio.sleep(0.5)
                if await _server_alive():
                    break
            else:
                console.print("[red][FAIL][/red] Server failed to start")
                raise typer.Exit(1)
            console.print("[green][OK][/green] Server started")

        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                f"{_server_url()}/api/models/load",
                json={"model_name": model_name, "keep_previous": keep_previous},
            )
            if resp.status_code != 200:
                detail = resp.json().get("detail", "Unknown error")
                console.print(f"[red][FAIL][/red] {detail}")
                raise typer.Exit(1)

            console.print(f"[green][OK][/green] Loaded {model_name}")

            loaded_resp = await client.get(f"{_server_url()}/api/models/loaded")
            if loaded_resp.status_code == 200:
                for item in loaded_resp.json().get("loaded", []):
                    if item["model"]["name"] == model_name:
                        console.print(f"  Port: {item['port']} | PID: {item['pid']} | URL: {item['base_url']}")

    asyncio.run(_run())


@app.command()
def ps():
    """Show loaded models with port/PID info."""
    _require_server()

    async def _ps():
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            resp = await client.get(f"{_server_url()}/api/models/loaded")
            if resp.status_code != 200:
                console.print("[red][FAIL][/red] Failed to get loaded models")
                raise typer.Exit(1)

            loaded = resp.json().get("loaded", [])
            if not loaded:
                console.print("[yellow]No models loaded[/yellow]")
                return

            table = Table(title="Loaded Models")
            table.add_column("Model", style="green")
            table.add_column("Port", style="cyan")
            table.add_column("PID", style="cyan")
            table.add_column("Engine", style="yellow")
            table.add_column("URL", style="blue")

            for item in loaded:
                model = item["model"]
                table.add_row(
                    model["name"],
                    str(item["port"]),
                    str(item["pid"]),
                    item.get("engine_slot", "unknown"),
                    item["base_url"],
                )

            console.print(table)

    asyncio.run(_ps())


@app.command()
def stop():
    """Stop the 掬水 server gracefully."""
    async def _stop():
        if not await _server_alive():
            console.print("[yellow]Server is not running[/yellow]")
            return

        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.post(f"{_server_url()}/api/shutdown")
            if resp.status_code == 200:
                console.print("[green][OK][/green] Server shutting down")
            else:
                console.print("[red][FAIL][/red] Failed to stop server")
                raise typer.Exit(1)

    asyncio.run(_stop())


@app.command()
def serve():
    """Start the 掬水 server."""
    import uvicorn
    from lmloader.infra.engine_store import ensure_engines

    migrated = ensure_engines()
    for slot, info in migrated.items():
        if info.get("action") == "bundled":
            console.print(f"[green][OK][/green] Engine '{slot}' bundled into {info['path']}")
        elif info.get("action") == "fallback":
            console.print(f"[green][OK][/green] Engine '{slot}' fell back to bundled {info['path']}")
        elif info.get("action") == "missing":
            console.print(f"[red][FAIL][/red] Engine '{slot}' binary missing: {info['path']}")

    console.print(f"[green][OK][/green] Starting server on {settings.server.host}:{settings.server.port}")
    console.print(f"  Web GUI:  http://{settings.server.host}:{settings.server.port}/")
    console.print(f"  API docs: http://{settings.server.host}:{settings.server.port}/docs")
    uvicorn.run(
        "lmloader.api.app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
    )


if __name__ == "__main__":
    app()
