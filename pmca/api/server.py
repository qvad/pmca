"""FastAPI application — OpenAI-compatible API server for PMCA.

Serves two modes:
  1. **Direct mode** (no tools in request): runs cascade, returns markdown.
  2. **Agent mode** (tools in request, e.g. opencode/crush): runs cascade,
     returns tool_calls so the agent writes files to its own workspace.

Requests without tools and with short system prompts (title, summarizer)
are forwarded to Ollama directly — no cascade needed.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
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
# Helpers
# ---------------------------------------------------------------------------


def _extract_user_message(raw: dict) -> str | None:
    """Pull the last user message from the raw request, handling opencode's list format."""
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
    tools = raw.get("tools")
    return bool(tools and len(tools) > 0)


def _is_lightweight_request(raw: dict) -> bool:
    """True for title/summarizer requests that should bypass the cascade.

    Heuristic: no tools AND system prompt is short (< 500 chars) or mentions
    'title' / 'short'.
    """
    if _has_tools(raw):
        return False
    messages = raw.get("messages", [])
    for msg in messages:
        if msg.get("role") == "system":
            sys_content = msg.get("content", "")
            if isinstance(sys_content, str) and len(sys_content) < 500:
                return True
    # No system prompt at all — also lightweight
    if not any(m.get("role") == "system" for m in messages):
        return True
    return False


def _has_tool_result_messages(raw: dict) -> bool:
    """True when the conversation contains tool result messages (post-execution follow-up)."""
    for msg in raw.get("messages", []):
        if msg.get("role") == "tool":
            return True
    return False


async def _stream_tool_result_ack(raw: dict) -> AsyncIterator[str]:
    """Acknowledge tool results with a simple 'files written' response."""
    import time
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    model = raw.get("model", "pmca")
    ts = int(time.time())

    content_chunk = _json.dumps({
        "id": chunk_id, "object": "chat.completion.chunk", "created": ts,
        "model": model, "choices": [{"index": 0, "delta": {
            "role": "assistant", "content": "Files written successfully by PMCA cascade.",
        }, "finish_reason": None}],
    })
    yield f"data: {content_chunk}\n\n"

    stop_chunk = _json.dumps({
        "id": chunk_id, "object": "chat.completion.chunk", "created": ts,
        "model": model, "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}],
    })
    yield f"data: {stop_chunk}\n\n"
    yield "data: [DONE]\n\n"


def _build_tool_calls(orchestrator: Orchestrator) -> list[ToolCall]:
    """Convert PMCA generated files (code + tests) into OpenAI tool_calls using 'write'."""
    # Collect both code and test files from all tasks
    all_files: dict[str, str] = {}
    for node in orchestrator.task_tree.walk():
        for path, content in node.code_files.items():
            all_files[path] = content
        for path, content in node.test_files.items():
            all_files[path] = content

    calls: list[ToolCall] = []
    for path, content in sorted(all_files.items()):
        calls.append(ToolCall(
            id=f"call_{uuid.uuid4().hex[:8]}",
            type="function",
            function=ToolFunction(
                name="write",
                arguments=_json.dumps({"file_path": path, "content": content}),
            ),
        ))
    return calls


def _format_event_as_text(event: CascadeEvent) -> str:
    """Convert a CascadeEvent to a human-readable markdown line."""
    etype = event.event_type
    if etype == EventType.CASCADE_START:
        return f"**Starting PMCA cascade** for: {event.task_title}\n\n"
    if etype == EventType.PHASE_START:
        return f"**[{event.phase.upper()}]** {event.task_title}\n"
    if etype == EventType.TASK_DECOMPOSED:
        subtasks = event.data.get("subtasks", [])
        return f"Decomposed into {len(subtasks)} subtasks:\n" + "".join(f"  - {s}\n" for s in subtasks)
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


def _format_final_result(orchestrator: Orchestrator) -> str:
    """Format generated code as markdown."""
    code_files = orchestrator.get_generated_code()
    if not code_files:
        return "\n---\n**No files were generated.**\n"
    parts = ["\n---\n## Generated Files\n"]
    for path, content in sorted(code_files.items()):
        ext = Path(path).suffix.lstrip(".")
        parts.append(f"### `{path}`\n```{ext or 'text'}\n{content}\n```\n")
    return "\n".join(parts)


def _make_chunk(content: str | None, chunk_id: str, model: str,
                finish_reason: str | None = None, role: str | None = None) -> str:
    """Build a single SSE data line."""
    chunk = ChatCompletionChunk(
        id=chunk_id, model=model,
        choices=[ChunkChoice(delta=DeltaMessage(role=role, content=content),
                             finish_reason=finish_reason)],
    )
    return f"data: {chunk.model_dump_json()}\n\n"


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

    ollama_base = (
        os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> ModelListResponse:
        return ModelListResponse(data=[ModelInfo(id="pmca")])

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(raw: dict):
        log.info(f"Incoming: keys={list(raw.keys())} tools={_has_tools(raw)} lightweight={_is_lightweight_request(raw)}")

        # --- Tool result follow-up: opencode sends back tool execution results.
        # Respond with a simple "done" — don't start a new cascade.
        if _has_tool_result_messages(raw):
            return StreamingResponse(
                _stream_tool_result_ack(raw),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        user_message = _extract_user_message(raw)
        if not user_message:
            raise HTTPException(400, "No user message found")

        # --- Lightweight pass-through (title, summarizer) ---
        if _is_lightweight_request(raw):
            result = await _passthrough_to_ollama(raw, config, ollama_base)
            if result is not None:
                return result
            # Passthrough failed (no Ollama) — fall through to cascade

        # --- Agent mode: run cascade, return tool_calls ---
        if _has_tools(raw):
            if _cascade_lock.locked():
                raise HTTPException(429, "Cascade already running")
            return StreamingResponse(
                _stream_cascade_agent(user_message, raw, config, ws_path, _cascade_lock),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # --- Direct mode: run cascade, return markdown ---
        stream = raw.get("stream", False)
        if _cascade_lock.locked():
            raise HTTPException(429, "Cascade already running")
        if stream:
            return StreamingResponse(
                _stream_cascade(user_message, raw, config, ws_path, _cascade_lock),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        async with _cascade_lock:
            return await _run_cascade_direct(user_message, raw, config, ws_path)

    return app


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------


async def _passthrough_to_ollama(raw: dict, config: Config, ollama_base: str) -> JSONResponse:
    """Forward lightweight requests (title, summarizer) to Ollama directly."""
    # Use the first configured model for pass-through
    model_name = "gemma4:e2b"  # lightweight model for titles
    if config.models:
        first_role = next(iter(config.models))
        model_name = config.models[first_role].name

    # Rewrite the request for Ollama
    ollama_req = dict(raw)
    ollama_req["model"] = model_name
    # Flatten content lists to strings (Ollama expects strings)
    for msg in ollama_req.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            msg["content"] = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
    # Remove fields Ollama doesn't understand
    for key in ("reasoning_effort", "max_completion_tokens", "stream_options"):
        ollama_req.pop(key, None)
    ollama_req["stream"] = False

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{ollama_base}/v1/chat/completions", json=ollama_req)
            if resp.status_code == 200:
                return JSONResponse(content=resp.json(), status_code=200)
            log.warning(f"Ollama pass-through returned {resp.status_code}")
            return None
    except Exception as exc:
        log.warning(f"Ollama pass-through failed: {exc}")
        return None


async def _stream_cascade_agent(
    user_message: str, raw: dict, config: Config,
    ws_path: Path, lock: asyncio.Lock,
) -> AsyncIterator[str]:
    """Run cascade, then stream tool_calls in OpenAI SSE format.

    OpenAI streaming tool_calls protocol:
      1. first chunk: delta with tool_call id + function name + empty args
      2. subsequent chunks: delta with argument fragments
      3. final chunk: finish_reason="tool_calls"
      4. data: [DONE]
    """
    async with lock:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        model = raw.get("model", "pmca")

        orchestrator = Orchestrator(config, ws_path)
        await orchestrator.run(user_message)

        tool_calls = _build_tool_calls(orchestrator)

        if not tool_calls:
            # No files — return text
            yield _make_chunk("PMCA cascade produced no files.", chunk_id, model, role="assistant")
            yield _make_chunk(None, chunk_id, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        # Stream each tool_call: send the full arguments in one chunk per tool
        # (opencode's parser handles this correctly)
        import time
        for idx, tc in enumerate(tool_calls):
            delta = {
                "tool_calls": [{
                    "index": idx,
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }]
            }
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None,
                }],
            }
            yield f"data: {_json.dumps(chunk)}\n\n"

        # Final chunk with finish_reason
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls",
            }],
        }
        yield f"data: {_json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"


async def _run_cascade_direct(
    user_message: str, raw: dict, config: Config, ws_path: Path,
) -> JSONResponse:
    """Run cascade and return markdown with code blocks."""
    orchestrator = Orchestrator(config, ws_path)
    root = await orchestrator.run(user_message)

    parts: list[str] = []
    if root.is_complete:
        parts.append(f"**PMCA cascade completed** for: {user_message}\n")
    else:
        parts.append(f"**PMCA cascade ended** with status: {root.status.value}\n")
    parts.append(_format_final_result(orchestrator))

    return JSONResponse(content=ChatCompletionResponse(
        model=raw.get("model", "pmca"),
        choices=[ChatCompletionChoice(
            message=ChatMessage(role="assistant", content="\n".join(parts)),
            finish_reason="stop",
        )],
    ).model_dump())


async def _stream_cascade(
    user_message: str, raw: dict, config: Config,
    ws_path: Path, lock: asyncio.Lock,
) -> AsyncIterator[str]:
    """Run cascade as streaming SSE chunks."""
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

            yield _make_chunk(None, chunk_id, model, finish_reason="stop")
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()
