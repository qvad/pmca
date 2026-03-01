"""CLI entry point for PMCA — Portable Modular Coding Agent."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pmca.models.config import Config
from pmca.models.setup import OllamaSetup
from pmca.orchestrator import Orchestrator
from pmca.utils.logger import setup_logging

console = Console()


def _load_config(config_path: str | None) -> Config:
    if config_path:
        return Config.from_yaml(Path(config_path))
    return Config.default()


@click.group()
@click.option("--config", "-c", "config_path", default=None, help="Path to config YAML")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, config_path: str | None, verbose: bool) -> None:
    """PMCA — Portable Modular Coding Agent.

    A fully local, hierarchical coding agent that uses Ollama models
    to decompose, implement, and verify code.
    """
    ctx.ensure_object(dict)
    config = _load_config(config_path)
    ctx.obj["config"] = config

    log_level = "DEBUG" if verbose else config.logging.level
    setup_logging(level=log_level, log_file=config.logging.file)


@main.command()
@click.argument("request", nargs=-1, required=True)
@click.option("--workspace", "-w", default=None, help="Workspace directory path")
@click.pass_context
def run(ctx: click.Context, request: tuple[str, ...], workspace: str | None) -> None:
    """Run the coding agent with a task description.

    Example: pmca run "Build a CLI calculator with add, subtract, multiply, divide"
    """
    config: Config = ctx.obj["config"]
    user_request = " ".join(request)

    ws_path = Path(workspace) if workspace else Path(config.workspace.path)
    ws_path = ws_path.resolve()

    console.print(Panel(
        f"[bold cyan]PMCA — Portable Modular Coding Agent[/bold cyan]\n\n"
        f"Task: {user_request}\n"
        f"Workspace: {ws_path}",
        title="Starting",
    ))

    orchestrator = Orchestrator(config, ws_path)
    result = asyncio.run(orchestrator.run(user_request))

    console.print()
    orchestrator.print_tree()

    console.print()
    summary = orchestrator.task_tree.summary()
    table = Table(title="Task Summary")
    table.add_column("Status", style="bold")
    table.add_column("Count")
    for status, count in sorted(summary.items()):
        table.add_row(status, str(count))
    console.print(table)


@main.command()
@click.option("--workspace", "-w", default=None, help="Workspace directory path")
@click.pass_context
def status(ctx: click.Context, workspace: str | None) -> None:
    """Show the current task tree status."""
    config: Config = ctx.obj["config"]
    ws_path = Path(workspace) if workspace else Path(config.workspace.path)
    ws_path = ws_path.resolve()

    orchestrator = Orchestrator.load_state(config, ws_path)
    orchestrator.print_tree()

    summary = orchestrator.task_tree.summary()
    if summary:
        table = Table(title="Task Summary")
        table.add_column("Status", style="bold")
        table.add_column("Count")
        for s, count in sorted(summary.items()):
            table.add_row(s, str(count))
        console.print(table)
    else:
        console.print("[dim]No saved state found.[/dim]")


@main.command()
@click.option("--workspace", "-w", default=None, help="Workspace directory path")
@click.pass_context
def resume(ctx: click.Context, workspace: str | None) -> None:
    """Resume a previously interrupted task."""
    config: Config = ctx.obj["config"]
    ws_path = Path(workspace) if workspace else Path(config.workspace.path)
    ws_path = ws_path.resolve()

    orchestrator = Orchestrator.load_state(config, ws_path)
    root = orchestrator.task_tree.root

    if root is None:
        console.print("[red]No saved task found to resume.[/red]")
        return

    if root.is_complete:
        console.print("[green]Task is already complete![/green]")
        orchestrator.print_tree()
        return

    console.print(Panel(
        f"[bold]Resuming:[/bold] {root.title}\n"
        f"Status: {root.status.value}",
        title="Resume",
    ))

    result = asyncio.run(orchestrator.cascade(root))
    orchestrator.print_tree()


@main.command()
@click.pass_context
def models(ctx: click.Context) -> None:
    """Check model availability and status."""
    config: Config = ctx.obj["config"]

    table = Table(title="Model Configuration")
    table.add_column("Role", style="bold")
    table.add_column("Model")
    table.add_column("Context Window")
    table.add_column("Temperature")

    for role, model_cfg in config.models.items():
        table.add_row(
            role.value,
            model_cfg.name,
            str(model_cfg.context_window),
            str(model_cfg.temperature),
        )
    console.print(table)

    console.print("\n[bold]Checking Ollama availability...[/bold]")
    from pmca.models.manager import ModelManager
    manager = ModelManager(config)
    available = asyncio.run(manager.check_available())
    asyncio.run(manager.close())

    avail_table = Table(title="Model Availability")
    avail_table.add_column("Model", style="bold")
    avail_table.add_column("Status")

    for model_name, is_avail in available.items():
        status_str = "[green]Available[/green]" if is_avail else "[red]Not found[/red]"
        avail_table.add_row(model_name, status_str)
    console.print(avail_table)


@main.command()
@click.pass_context
def setup(ctx: click.Context) -> None:
    """Set up Ollama and pull required models."""
    config: Config = ctx.obj["config"]

    console.print(Panel("[bold]Running PMCA Setup[/bold]", style="cyan"))

    ollama_setup = OllamaSetup(config)
    success = asyncio.run(ollama_setup.full_setup())

    if success:
        console.print("[bold green]Setup complete! All models ready.[/bold green]")
    else:
        console.print("[bold red]Setup encountered issues. Check logs for details.[/bold red]")


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to listen on")
@click.option("--workspace", "-w", default=None, help="Workspace directory path")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, workspace: str | None) -> None:
    """Start the OpenAI-compatible API server.

    Example: pmca serve --port 8000 --workspace ./workspace
    """
    import uvicorn

    from pmca.api.server import create_app

    config: Config = ctx.obj["config"]
    ws_path = Path(workspace) if workspace else Path(config.workspace.path)
    ws_path = ws_path.resolve()

    app = create_app(config=config, workspace_path=ws_path)

    console.print(Panel(
        f"[bold cyan]PMCA API Server[/bold cyan]\n\n"
        f"  Base URL:   http://{host}:{port}/v1\n"
        f"  Health:     http://{host}:{port}/health\n"
        f"  Workspace:  {ws_path}\n\n"
        f"  Connect from OpenCode, aider, Continue, or Cursor\n"
        f'  using base URL http://{host}:{port}/v1 and model "pmca"',
        title="Starting",
    ))

    uvicorn.run(app, host=host, port=port, log_level="info")


@main.command("rag-index")
@click.argument("docs_path")
@click.pass_context
def rag_index(ctx: click.Context, docs_path: str) -> None:
    """Index documentation files for RAG retrieval.

    Example: pmca rag-index ./docs/library-api
    """
    from pmca.models.config import RAGConfig
    from pmca.utils.rag import RAGManager

    config: Config = ctx.obj["config"]

    # Use config's RAG settings, but override docs_path
    rag_config = config.rag
    if not rag_config.enabled:
        # Allow indexing even if RAG isn't enabled in config
        rag_config = RAGConfig(
            enabled=True,
            docs_path=docs_path,
            embedding_model=config.rag.embedding_model,
            n_results=config.rag.n_results,
            persist_dir=config.rag.persist_dir,
        )

    manager = RAGManager(rag_config)
    if not manager.available:
        console.print(
            "[red]RAG dependencies not installed.[/red]\n"
            "Install with: pip install pmca[rag]"
        )
        return

    docs = Path(docs_path).resolve()
    console.print(f"Indexing docs from: {docs}")
    count = manager.index_directory(docs)
    console.print(f"[green]Indexed {count} chunks successfully.[/green]")
    if not config.rag.enabled:
        console.print(
            "[yellow]Note: RAG is not enabled in your config. "
            "Set 'rag.enabled: true' to use indexed docs during code generation.[/yellow]"
        )
    manager.close()


@main.command()
@click.option("--workspace", "-w", default=None, help="Workspace directory path")
@click.pass_context
def mcp(ctx: click.Context, workspace: str | None) -> None:
    """Start the MCP server (stdio transport).

    Example: pmca mcp --workspace ./workspace
    """
    config: Config = ctx.obj["config"]
    ws_path = Path(workspace) if workspace else Path(config.workspace.path)
    ws_path = ws_path.resolve()

    try:
        from pmca.mcp.server import create_mcp_server

        console.print(Panel(
            f"[bold cyan]PMCA MCP Server[/bold cyan]\n\n"
            f"  Server name:  {config.mcp.server_name}\n"
            f"  Workspace:    {ws_path}\n"
            f"  Transport:    stdio\n\n"
            f"  Add to Claude Desktop or VS Code MCP settings",
            title="Starting",
        ))

        server = create_mcp_server(config, ws_path)
        server.run()
    except ImportError:
        console.print(
            "[red]MCP dependencies not installed.[/red]\n"
            "Install with: pip install pmca[mcp]"
        )
        return


if __name__ == "__main__":
    main()
