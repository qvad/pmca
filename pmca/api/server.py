"""FastAPI application — OpenAI-compatible API server for PMCA.

Three request routing modes:
  1. **Tool result ack** — opencode sends tool execution results back;
     respond immediately without triggering a cascade.
  2. **Lightweight pass-through** — title/summarizer requests forwarded
     to Ollama directly (no cascade, no VRAM lock).
  3. **Agent mode** — requests with tool definitions run the full cascade
     and return verified code as streaming ``tool_calls``.
  4. **Direct mode** — no tools, substantial prompt → cascade → markdown.
"""

from __future__ import annotations

import asyncio
import copy
import json as _json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from pmca.api.events import CascadeEvent, EventBus, EventType
from pmca.api.models import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    DeltaMessage,
    ModelInfo,
    ModelListResponse,
    ToolCall,
    ToolFunction,
)
from pmca.models.config import Config
from pmca.orchestrator import Orchestrator

log = logging.getLogger("pmca.api")

# ---------------------------------------------------------------------------
# Event formatting — maps EventType → human-readable markdown line
# ---------------------------------------------------------------------------

_EVENT_FORMATTERS: dict[EventType, str] = {
    EventType.CASCADE_START: "**Starting PMCA cascade** for: {task_title}\n\n",
    EventType.PHASE_START: "**[{phase_upper}]** {task_title}\n",
    EventType.CASCADE_COMPLETE: "**Cascade complete.**\n",
    EventType.CASCADE_ERROR: "**Cascade error:** {message}\n",
    EventType.TASK_COMPLETE: "Task verified: {task_title}\n",
    EventType.TASK_FAILED: "Task **FAILED**: {task_title}\n",
    EventType.PHASE_COMPLETE: "",
}


def _format_event_as_text(event: CascadeEvent) -> str:
    """Convert a CascadeEvent to a human-readable markdown line."""
    template = _EVENT_FORMATTERS.get(event.event_type)
    if template is not None:
        return template.format(
            task_title=event.task_title,
            phase_upper=(event.phase or "").upper(),
            message=event.message,
        )

    etype = event.event_type
    if etype == EventType.TASK_DECOMPOSED:
        subtasks = event.data.get("subtasks", [])
        return f"Decomposed into {len(subtasks)} subtasks:\n" + "".join(
            f"  - {s}\n" for s in subtasks
        )
    if etype == EventType.CODE_GENERATED:
        files = event.data.get("files", [])
        return f"Generated {len(files)} files: {', '.join(files)}\n"
    if etype == EventType.REVIEW_RESULT:
        if event.data.get("passed"):
            return "Review: **passed**\n"
        return f"Review: **failed** — {', '.join(event.data.get('issues', [])[:3])}\n"
    if etype == EventType.TEST_RESULT:
        return f"Tests: **{'passed' if event.data.get('passed') else 'failed'}**\n"
    if etype == EventType.RETRY:
        return (
            f"Retrying ({event.data.get('attempt', '?')}/"
            f"{event.data.get('max_retries', '?')})...\n"
        )
    return f"{event.message}\n"


# ---------------------------------------------------------------------------
# Request classification helpers
# ---------------------------------------------------------------------------


def _extract_user_message(raw: dict) -> str | None:
    """Pull the last user message, handling opencode's ``[{"text":...}]`` format."""
    for msg in reversed(raw.get("messages", [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            text = " ".join(parts).strip()
        elif isinstance(content, str):
            text = content.strip()
        else:
            continue
        if text:
            return text
    return None


def _has_tools(raw: dict) -> bool:
    """True when the request carries tool definitions (agent mode)."""
    return bool(raw.get("tools"))


def _has_tool_results(raw: dict) -> bool:
    """True when the conversation contains ``role: tool`` messages."""
    return any(m.get("role") == "tool" for m in raw.get("messages", []))


def _is_lightweight(raw: dict) -> bool:
    """True for title/summarizer requests that should bypass the cascade.

    Heuristic: no tools AND (system prompt < 500 chars OR no system prompt).
    """
    if _has_tools(raw):
        return False
    for msg in raw.get("messages", []):
        if msg.get("role") == "system":
            sys_content = msg.get("content", "")
            if isinstance(sys_content, str) and len(sys_content) < 500:
                return True
            return False  # long system prompt → not lightweight
    return True  # no system prompt → lightweight


# ---------------------------------------------------------------------------
# SSE chunk builders
# ---------------------------------------------------------------------------


def _make_chunk(
    content: str | None,
    chunk_id: str,
    model: str,
    finish_reason: str | None = None,
    role: str | None = None,
) -> str:
    """Build a single SSE ``data:`` line with an OpenAI-format chunk."""
    chunk = ChatCompletionChunk(
        id=chunk_id,
        model=model,
        choices=[
            ChunkChoice(
                delta=DeltaMessage(role=role, content=content),
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def _make_tool_call_chunk(
    chunk_id: str, model: str, idx: int, tc: ToolCall,
) -> str:
    """Build an SSE chunk carrying one tool_call delta."""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": idx,
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }],
            },
            "finish_reason": None,
        }],
    }
    return f"data: {_json.dumps(chunk)}\n\n"


def _make_finish_chunk(chunk_id: str, model: str, reason: str) -> str:
    """Build the final SSE chunk with ``finish_reason``."""
    chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    }
    return f"data: {_json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Code-file collector
# ---------------------------------------------------------------------------


def _build_tool_calls(orchestrator: Orchestrator) -> list[ToolCall]:
    """Convert PMCA generated files (code + tests) into ``write`` tool_calls."""
    all_files: dict[str, str] = {}
    for node in orchestrator.task_tree.walk():
        for path, content in node.code_files.items():
            all_files[path] = content
        for path, content in node.test_files.items():
            all_files[path] = content

    return [
        ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            function=ToolFunction(
                name="write",
                arguments=_json.dumps({"file_path": path, "content": content}),
            ),
        )
        for path, content in sorted(all_files.items())
    ]


def _format_final_result(orchestrator: Orchestrator) -> str:
    """Format generated code as markdown for direct-mode responses."""
    code_files = orchestrator.get_generated_code()
    if not code_files:
        return "\n---\n**No files were generated.**\n"
    parts = ["\n---\n## Generated Files\n"]
    for path, content in sorted(code_files.items()):
        ext = Path(path).suffix.lstrip(".")
        parts.append(f"### `{path}`\n```{ext or 'text'}\n{content}\n```\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(
    config: Config | None = None,
    workspace_path: str | Path = "./workspace",
) -> FastAPI:
    """Build a configured FastAPI application."""
    app = FastAPI(title="PMCA API", version="0.1.0")
    ws_path = Path(workspace_path).resolve()
    _cascade_lock = asyncio.Lock()

    if config is None:
        config = Config.default()

    ollama_base = os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    ).rstrip("/")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelListResponse:
        return ModelListResponse(data=[ModelInfo(id="pmca")])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(raw: dict):
        log.info(
            "Incoming: keys=%s tools=%s lightweight=%s",
            list(raw.keys()), _has_tools(raw), _is_lightweight(raw),
        )

        # 1. Tool result follow-up — acknowledge, don't cascade
        if _has_tool_results(raw):
            return StreamingResponse(
                _stream_ack(raw),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        user_message = _extract_user_message(raw)
        if not user_message:
            raise HTTPException(400, "No user message found")

        # 2. Lightweight pass-through (title, summarizer)
        if _is_lightweight(raw):
            result = await _passthrough(raw, config, ollama_base)
            if result is not None:
                return result
            # Ollama unavailable — fall through to cascade

        # 3. Agent mode: cascade → streaming tool_calls
        if _has_tools(raw):
            if _cascade_lock.locked():
                raise HTTPException(429, "Cascade already running")
            return StreamingResponse(
                _stream_cascade_agent(user_message, raw, config, ws_path, _cascade_lock),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # 4. Direct mode: cascade → markdown
        if _cascade_lock.locked():
            raise HTTPException(429, "Cascade already running")
        if raw.get("stream", False):
            return StreamingResponse(
                _stream_cascade_direct(user_message, raw, config, ws_path, _cascade_lock),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        async with _cascade_lock:
            return await _run_cascade_direct(user_message, raw, config, ws_path)

    return app


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


async def _stream_ack(raw: dict) -> AsyncIterator[str]:
    """Acknowledge tool results with a short text response."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = raw.get("model", "pmca")
    yield _make_chunk(
        "Files written successfully by PMCA cascade.",
        chunk_id, model, role="assistant",
    )
    yield _make_finish_chunk(chunk_id, model, "stop")
    yield "data: [DONE]\n\n"


async def _passthrough(
    raw: dict, config: Config, ollama_base: str,
) -> JSONResponse | None:
    """Forward lightweight requests to Ollama. Returns None on failure."""
    model_name = "gemma4:e2b"
    if config.models:
        first_role = next(iter(config.models))
        model_name = config.models[first_role].name

    # Deep-copy to avoid mutating the caller's data
    ollama_req = copy.deepcopy(raw)
    ollama_req["model"] = model_name
    ollama_req["stream"] = False
    for key in ("reasoning_effort", "max_completion_tokens", "stream_options"):
        ollama_req.pop(key, None)
    # Flatten content lists to strings (Ollama expects strings)
    for msg in ollama_req.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            msg["content"] = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{ollama_base}/v1/chat/completions", json=ollama_req,
            )
            if resp.status_code == 200:
                return JSONResponse(content=resp.json(), status_code=200)
            log.warning("Ollama pass-through returned %d", resp.status_code)
            return None
    except Exception as exc:
        log.warning("Ollama pass-through failed: %s", exc)
        return None


async def _stream_cascade_agent(
    user_message: str,
    raw: dict,
    config: Config,
    ws_path: Path,
    lock: asyncio.Lock,
) -> AsyncIterator[str]:
    """Run cascade, then stream tool_calls in OpenAI SSE format."""
    async with lock:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = raw.get("model", "pmca")

        orchestrator = Orchestrator(config, ws_path)
        await orchestrator.run(user_message)

        tool_calls = _build_tool_calls(orchestrator)
        if not tool_calls:
            yield _make_chunk(
                "PMCA cascade produced no files.",
                chunk_id, model, role="assistant",
            )
            yield _make_finish_chunk(chunk_id, model, "stop")
            yield "data: [DONE]\n\n"
            return

        for idx, tc in enumerate(tool_calls):
            yield _make_tool_call_chunk(chunk_id, model, idx, tc)

        yield _make_finish_chunk(chunk_id, model, "tool_calls")
        yield "data: [DONE]\n\n"


async def _run_cascade_direct(
    user_message: str, raw: dict, config: Config, ws_path: Path,
) -> JSONResponse:
    """Run cascade and return markdown with code blocks."""
    orchestrator = Orchestrator(config, ws_path)
    root = await orchestrator.run(user_message)

    status = "completed" if root.is_complete else root.status.value
    content = (
        f"**PMCA cascade {status}** for: {user_message}\n"
        + _format_final_result(orchestrator)
    )

    return JSONResponse(content=ChatCompletionResponse(
        model=raw.get("model", "pmca"),
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content=content),
            finish_reason="stop",
        )],
    ).model_dump())


async def _stream_cascade_direct(
    user_message: str,
    raw: dict,
    config: Config,
    ws_path: Path,
    lock: asyncio.Lock,
) -> AsyncIterator[str]:
    """Run cascade as streaming SSE chunks with event progress."""
    async with lock:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = raw.get("model", "pmca")
        bus = EventBus()
        orchestrator = Orchestrator(config, ws_path, event_callback=bus.emit)

        yield _make_chunk(None, chunk_id, model, role="assistant")

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

            final = _format_final_result(orchestrator)
            if final:
                yield _make_chunk(final, chunk_id, model)

            yield _make_finish_chunk(chunk_id, model, "stop")
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
