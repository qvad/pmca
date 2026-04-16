"""FastAPI application — OpenAI-compatible API server for PMCA."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from pmca.api.events import CascadeEvent, EventBus, EventType
from pmca.api.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    DeltaMessage,
    ModelInfo,
    ModelListResponse,
    UsageInfo,
)
from pmca.models.config import Config
from pmca.orchestrator import Orchestrator


def _format_event_as_text(event: CascadeEvent) -> str:
    """Convert a CascadeEvent to a human-readable markdown line for streaming."""
    etype = event.event_type

    if etype == EventType.CASCADE_START:
        return f"**Starting PMCA cascade** for: {event.task_title}\n\n"

    if etype == EventType.PHASE_START:
        return f"**[{event.phase.upper()}]** {event.task_title}\n"

    if etype == EventType.TASK_DECOMPOSED:
        subtasks = event.data.get("subtasks", [])
        lines = f"Decomposed into {len(subtasks)} subtasks:\n"
        for st in subtasks:
            lines += f"  - {st}\n"
        return lines

    if etype == EventType.CODE_GENERATED:
        files = event.data.get("files", [])
        return f"Generated {len(files)} files: {', '.join(files)}\n"

    if etype == EventType.REVIEW_RESULT:
        passed = event.data.get("passed", False)
        if passed:
            return "Review: **passed**\n"
        issues = event.data.get("issues", [])
        return f"Review: **failed** — {', '.join(issues[:3])}\n"

    if etype == EventType.TEST_RESULT:
        passed = event.data.get("passed", False)
        return f"Tests: **{'passed' if passed else 'failed'}**\n"

    if etype == EventType.RETRY:
        return f"Retrying ({event.data.get('attempt', '?')}/{event.data.get('max_retries', '?')})...\n"

    if etype == EventType.TASK_COMPLETE:
        return f"Task verified: {event.task_title}\n"

    if etype == EventType.TASK_FAILED:
        return f"Task **FAILED**: {event.task_title}\n"

    if etype == EventType.CASCADE_COMPLETE:
        return "**Cascade complete.**\n"

    if etype == EventType.CASCADE_ERROR:
        return f"**Cascade error:** {event.message}\n"

    if etype == EventType.PHASE_COMPLETE:
        return ""

    return f"{event.message}\n"


def _format_final_result(orchestrator: Orchestrator, root_title: str) -> str:
    """Format generated code + summary as markdown for the final response."""
    code_files = orchestrator.get_generated_code()
    if not code_files:
        return "\n---\n**No files were generated.**\n"

    parts = ["\n---\n## Generated Files\n"]
    for path, content in sorted(code_files.items()):
        ext = Path(path).suffix.lstrip(".")
        lang = ext if ext else "text"
        parts.append(f"### `{path}`\n```{lang}\n{content}\n```\n")

    summary = orchestrator.task_tree.summary()
    if summary:
        parts.append("## Summary\n")
        for status, count in sorted(summary.items()):
            parts.append(f"- {status}: {count}\n")

    return "\n".join(parts)


def _make_chunk(
    content: str | None,
    chunk_id: str,
    model: str,
    finish_reason: str | None = None,
    role: str | None = None,
) -> str:
    """Build a single SSE data line with an OpenAI-format chunk."""
    delta = DeltaMessage(role=role, content=content)
    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        choices=[ChunkChoice(delta=delta, finish_reason=finish_reason)],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def create_app(
    config: Config | None = None,
    workspace_path: str | Path = "./workspace",
) -> FastAPI:
    """Factory that builds a configured FastAPI application."""

    app = FastAPI(title="PMCA API", version="0.1.0")
    ws_path = Path(workspace_path).resolve()

    # Track whether a cascade is currently running (VRAM constraint)
    _lock: dict[str, bool] = {"running": False}

    if config is None:
        config = Config.default()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[ModelInfo(id="pmca")],
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest):
        # Extract last user message
        user_message: str | None = None
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                user_message = msg.content
                break

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found in request")

        if _lock["running"]:
            raise HTTPException(
                status_code=429,
                detail="A cascade is already running. PMCA supports one request at a time.",
            )

        if request.stream:
            return StreamingResponse(
                _stream_cascade(user_message, request, config, ws_path, _lock),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await _run_cascade(user_message, request, config, ws_path, _lock)

    return app


async def _stream_cascade(
    user_message: str,
    request: ChatCompletionRequest,
    config: Config,
    ws_path: Path,
    lock: dict[str, bool],
) -> AsyncIterator[str]:
    """Run the cascade in a background task, yield SSE chunks from EventBus."""
    lock["running"] = True
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = request.model
    bus = EventBus()

    orchestrator = Orchestrator(config, ws_path, event_callback=bus.emit)

    # Role chunk first
    yield _make_chunk(None, chunk_id, model, role="assistant")

    # Background task runs the cascade
    async def _run() -> None:
        try:
            await orchestrator.run(user_message)
        except Exception:
            pass
        finally:
            bus.finish()

    task = asyncio.create_task(_run())

    try:
        async for event in bus:
            text = _format_event_as_text(event)
            if text:
                yield _make_chunk(text, chunk_id, model)

        # Final result with generated code
        final_text = _format_final_result(orchestrator, user_message)
        if final_text:
            yield _make_chunk(final_text, chunk_id, model)

        # Stop chunk
        yield _make_chunk(None, chunk_id, model, finish_reason="stop")
        yield "data: [DONE]\n\n"
    finally:
        lock["running"] = False
        if not task.done():
            task.cancel()


async def _run_cascade(
    user_message: str,
    request: ChatCompletionRequest,
    config: Config,
    ws_path: Path,
    lock: dict[str, bool],
) -> JSONResponse:
    """Run the cascade synchronously and return a complete ChatCompletion."""
    lock["running"] = True
    try:
        orchestrator = Orchestrator(config, ws_path)
        root = await orchestrator.run(user_message)

        # Build full response content
        parts: list[str] = []
        if root.is_complete:
            parts.append(f"**PMCA cascade completed** for: {user_message}\n")
        else:
            parts.append(f"**PMCA cascade ended** with status: {root.status.value}\n")

        parts.append(_format_final_result(orchestrator, user_message))
        content = "\n".join(parts)

        response = ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(user_message.split()),
                completion_tokens=len(content.split()),
                total_tokens=len(user_message.split()) + len(content.split()),
            ),
        )
        return JSONResponse(content=response.model_dump())
    finally:
        lock["running"] = False
