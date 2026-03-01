"""Tests for the PMCA OpenAI-compatible API layer."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pmca.api.events import CascadeEvent, EventBus, EventType
from pmca.api.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelListResponse,
)
from pmca.api.server import _format_event_as_text, _format_final_result, create_app
from pmca.models.config import AgentRole, Config, ModelConfig
from pmca.tasks.state import CodeFile, ReviewResult, TaskStatus, TaskType
from pmca.tasks.tree import TaskNode


@pytest.fixture
def config():
    return Config(
        models={
            AgentRole.ARCHITECT: ModelConfig(name="test:14b", temperature=0.3),
            AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.2),
            AgentRole.REVIEWER: ModelConfig(name="test:14b", temperature=0.1),
            AgentRole.WATCHER: ModelConfig(name="test:7b", temperature=0.1),
        }
    )


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def app(config, workspace):
    return create_app(config=config, workspace_path=workspace)


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealth:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestListModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "pmca"
        assert data["data"][0]["object"] == "model"


class TestChatCompletionsNonStreaming:
    @patch("pmca.api.server.Orchestrator")
    def test_chat_completions_non_streaming(self, MockOrch, client):
        """Non-streaming request returns a full ChatCompletion."""
        # Set up mock orchestrator
        mock_orch = MagicMock()
        MockOrch.return_value = mock_orch

        root_node = TaskNode(title="hello", status=TaskStatus.VERIFIED)
        mock_orch.run = AsyncMock(return_value=root_node)
        mock_orch.get_generated_code.return_value = {
            "src/hello.py": "print('hello')",
        }
        mock_orch.task_tree.summary.return_value = {"verified": 1}

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pmca",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "hello.py" in data["choices"][0]["message"]["content"]


class TestChatCompletionsStreaming:
    @patch("pmca.api.server.Orchestrator")
    def test_chat_completions_streaming(self, MockOrch, client):
        """Streaming request returns SSE chunks."""
        mock_orch = MagicMock()
        MockOrch.return_value = mock_orch

        root_node = TaskNode(title="hello", status=TaskStatus.VERIFIED)

        # The run method needs to emit events via the bus, then complete
        async def fake_run(msg):
            # Emit events via the callback that was passed at construction
            callback = MockOrch.call_args[1].get("event_callback") or MockOrch.call_args[0][2] if len(MockOrch.call_args[0]) > 2 else None
            if callback is None:
                # Try kwargs
                for key, val in MockOrch.call_args.kwargs.items():
                    if key == "event_callback":
                        callback = val
                        break
            if callback:
                callback(CascadeEvent(
                    event_type=EventType.CASCADE_START,
                    task_title=msg,
                    message=f"Starting PMCA cascade for: {msg}",
                ))
                callback(CascadeEvent(
                    event_type=EventType.CASCADE_COMPLETE,
                    task_title=msg,
                    message="Cascade completed",
                ))
            return root_node

        mock_orch.run = AsyncMock(side_effect=fake_run)
        mock_orch.get_generated_code.return_value = {}
        mock_orch.task_tree.summary.return_value = {"verified": 1}

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pmca",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE lines
        lines = resp.text.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 2  # At least role chunk + [DONE]

        # Last data line should be [DONE]
        assert data_lines[-1] == "data: [DONE]"

        # Second to last should have finish_reason=stop
        stop_line = data_lines[-2]
        stop_data = json.loads(stop_line[len("data: "):])
        assert stop_data["choices"][0]["finish_reason"] == "stop"


class TestNoUserMessage:
    def test_no_user_message_returns_400(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pmca",
                "messages": [{"role": "system", "content": "You are a helper"}],
                "stream": False,
            },
        )
        assert resp.status_code == 400
        assert "No user message" in resp.json()["detail"]

    def test_empty_messages_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "pmca",
                "messages": [],
                "stream": False,
            },
        )
        # Empty messages should still work — we check for user content
        # FastAPI will accept an empty list, then our code returns 400
        assert resp.status_code == 400


class TestEventFormatting:
    """Unit tests for _format_event_as_text with each EventType."""

    def test_cascade_start(self):
        event = CascadeEvent(
            event_type=EventType.CASCADE_START,
            task_title="Build a calculator",
            message="Starting",
        )
        text = _format_event_as_text(event)
        assert "Starting PMCA cascade" in text
        assert "Build a calculator" in text

    def test_phase_start(self):
        event = CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title="Add function",
            phase="design",
        )
        text = _format_event_as_text(event)
        assert "[DESIGN]" in text
        assert "Add function" in text

    def test_task_decomposed(self):
        event = CascadeEvent(
            event_type=EventType.TASK_DECOMPOSED,
            task_title="Calculator",
            data={"subtasks": ["Add", "Subtract", "Multiply"]},
        )
        text = _format_event_as_text(event)
        assert "3 subtasks" in text
        assert "Add" in text

    def test_code_generated(self):
        event = CascadeEvent(
            event_type=EventType.CODE_GENERATED,
            task_title="Add",
            data={"files": ["src/add.py", "tests/test_add.py"]},
        )
        text = _format_event_as_text(event)
        assert "2 files" in text
        assert "src/add.py" in text

    def test_review_result_passed(self):
        event = CascadeEvent(
            event_type=EventType.REVIEW_RESULT,
            task_title="Add",
            data={"passed": True, "issues": []},
        )
        text = _format_event_as_text(event)
        assert "passed" in text

    def test_review_result_failed(self):
        event = CascadeEvent(
            event_type=EventType.REVIEW_RESULT,
            task_title="Add",
            data={"passed": False, "issues": ["Missing error handling"]},
        )
        text = _format_event_as_text(event)
        assert "failed" in text
        assert "Missing error handling" in text

    def test_test_result_passed(self):
        event = CascadeEvent(
            event_type=EventType.TEST_RESULT,
            task_title="Add",
            data={"passed": True},
        )
        text = _format_event_as_text(event)
        assert "passed" in text

    def test_test_result_failed(self):
        event = CascadeEvent(
            event_type=EventType.TEST_RESULT,
            task_title="Add",
            data={"passed": False},
        )
        text = _format_event_as_text(event)
        assert "failed" in text

    def test_retry(self):
        event = CascadeEvent(
            event_type=EventType.RETRY,
            task_title="Add",
            data={"attempt": 2, "max_retries": 3},
        )
        text = _format_event_as_text(event)
        assert "2/3" in text

    def test_task_complete(self):
        event = CascadeEvent(
            event_type=EventType.TASK_COMPLETE,
            task_title="Add",
        )
        text = _format_event_as_text(event)
        assert "verified" in text.lower() or "Add" in text

    def test_task_failed(self):
        event = CascadeEvent(
            event_type=EventType.TASK_FAILED,
            task_title="Add",
        )
        text = _format_event_as_text(event)
        assert "FAILED" in text

    def test_cascade_complete(self):
        event = CascadeEvent(event_type=EventType.CASCADE_COMPLETE)
        text = _format_event_as_text(event)
        assert "complete" in text.lower()

    def test_cascade_error(self):
        event = CascadeEvent(
            event_type=EventType.CASCADE_ERROR,
            message="Connection lost",
        )
        text = _format_event_as_text(event)
        assert "error" in text.lower()
        assert "Connection lost" in text

    def test_phase_complete_empty(self):
        event = CascadeEvent(
            event_type=EventType.PHASE_COMPLETE,
            task_title="Add",
            phase="design",
        )
        text = _format_event_as_text(event)
        assert text == ""


class TestEventBus:
    @pytest.mark.asyncio
    async def test_event_bus_basic(self):
        bus = EventBus()
        event = CascadeEvent(event_type=EventType.CASCADE_START, task_title="test")
        bus.emit(event)
        bus.finish()

        events = []
        async for e in bus:
            events.append(e)

        assert len(events) == 1
        assert events[0].event_type == EventType.CASCADE_START

    @pytest.mark.asyncio
    async def test_event_bus_multiple(self):
        bus = EventBus()
        bus.emit(CascadeEvent(event_type=EventType.CASCADE_START))
        bus.emit(CascadeEvent(event_type=EventType.PHASE_START, phase="design"))
        bus.emit(CascadeEvent(event_type=EventType.CASCADE_COMPLETE))
        bus.finish()

        events = []
        async for e in bus:
            events.append(e)

        assert len(events) == 3
        assert events[0].event_type == EventType.CASCADE_START
        assert events[1].event_type == EventType.PHASE_START
        assert events[2].event_type == EventType.CASCADE_COMPLETE

    @pytest.mark.asyncio
    async def test_event_bus_empty(self):
        bus = EventBus()
        bus.finish()

        events = []
        async for e in bus:
            events.append(e)

        assert len(events) == 0
