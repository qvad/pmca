"""MCP (Model Context Protocol) server for PMCA.

Exposes PMCA capabilities as MCP tools for use with Claude Desktop,
VS Code, and other MCP-compatible clients.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pmca.models.config import Config
from pmca.utils.logger import get_logger

log = get_logger("mcp.server")


def create_mcp_server(config: Config, workspace_path: Path):
    """Create and configure an MCP server with PMCA tools.

    Returns a configured mcp Server instance ready to run.
    Raises ImportError if the mcp package is not installed.
    """
    from mcp.server import Server
    from mcp.types import Resource, TextContent, Tool

    server_name = config.mcp.server_name
    server = Server(server_name)

    # Lock to serialize tool calls (orchestrator is not thread-safe)
    _tool_lock = asyncio.Lock()

    # Create a fresh orchestrator per generate_code call since
    # Orchestrator.run() closes the model manager in its finally block.
    def _make_orchestrator():
        from pmca.orchestrator import Orchestrator
        return Orchestrator(config, workspace_path)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="generate_code",
                description=(
                    "Generate code from a natural language request. "
                    "Runs the full PMCA cascade: design, code, review, verify."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "request": {
                            "type": "string",
                            "description": "Natural language description of the code to generate",
                        },
                    },
                    "required": ["request"],
                },
            ),
            Tool(
                name="review_code",
                description="Review code against a specification for correctness and quality.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to review",
                        },
                        "spec": {
                            "type": "string",
                            "description": "The specification to review against",
                        },
                    },
                    "required": ["code", "spec"],
                },
            ),
            Tool(
                name="run_tests",
                description="Run tests in the PMCA workspace and return results.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workspace": {
                            "type": "string",
                            "description": "Path to workspace directory (uses default if omitted)",
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        if arguments is None:
            arguments = {}
        async with _tool_lock:
            try:
                if name == "generate_code":
                    return await _handle_generate_code(arguments)
                elif name == "review_code":
                    return await _handle_review_code(arguments)
                elif name == "run_tests":
                    return await _handle_run_tests(arguments)
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
            except Exception as e:
                log.error(f"Tool '{name}' failed: {e}")
                return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_generate_code(arguments: dict) -> list[TextContent]:
        request = arguments.get("request", "")
        if not request:
            return [TextContent(type="text", text="Error: 'request' is required")]

        # Fresh orchestrator each time — run() closes the model manager
        orchestrator = _make_orchestrator()
        result = await orchestrator.run(request)

        generated_files = orchestrator.get_generated_code()
        output = {
            "status": result.status.value,
            "files": generated_files,
            "task_title": result.title,
        }
        return [TextContent(type="text", text=json.dumps(output, indent=2))]

    async def _handle_review_code(arguments: dict) -> list[TextContent]:
        code = arguments.get("code", "")
        spec = arguments.get("spec", "")
        if not code or not spec:
            return [TextContent(type="text", text="Error: 'code' and 'spec' are required")]

        orchestrator = _make_orchestrator()
        try:
            review = await orchestrator._reviewer.verify_code(code, spec)
            output = {
                "passed": review.passed,
                "issues": review.issues,
                "suggestions": review.suggestions,
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]
        finally:
            await orchestrator._model_manager.close()

    async def _handle_run_tests(arguments: dict) -> list[TextContent]:
        from pmca.tasks.tree import TaskNode

        ws = arguments.get("workspace")
        ws_path = Path(ws).resolve() if ws else workspace_path

        # Validate the workspace is under the configured workspace
        if ws and not str(ws_path).startswith(str(workspace_path.resolve())):
            return [TextContent(
                type="text",
                text="Error: workspace must be within the configured workspace directory",
            )]

        # Find test files in workspace
        test_files = [
            str(p.relative_to(ws_path))
            for p in ws_path.rglob("test_*.py")
        ]
        if not test_files:
            return [TextContent(type="text", text="No test files found in workspace")]

        orchestrator = _make_orchestrator()
        try:
            task = TaskNode(title="mcp-test-run")
            task.test_files = test_files

            test_result = await orchestrator._watcher.run_tests(task)
            output = {
                "passed": test_result.passed,
                "total": test_result.total,
                "failures": test_result.failures,
                "output": test_result.output[:2000],
            }
            return [TextContent(type="text", text=json.dumps(output, indent=2))]
        finally:
            await orchestrator._model_manager.close()

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="pmca://status",
                name="PMCA Task Status",
                description="Current task tree status as JSON",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        if str(uri) == "pmca://status":
            from pmca.orchestrator import Orchestrator
            orch = Orchestrator.load_state(config, workspace_path)
            tree = orch.task_tree
            root = tree.root
            if root is None:
                return json.dumps({"status": "idle", "tasks": []})

            tasks = []
            for node in tree.walk():
                tasks.append({
                    "id": node.id,
                    "title": node.title,
                    "status": node.status.value,
                    "depth": node.depth,
                    "code_files": node.code_files,
                    "test_files": node.test_files,
                })
            return json.dumps({"status": "active", "tasks": tasks}, indent=2)

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server
