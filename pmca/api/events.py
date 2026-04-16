"""Event system for streaming cascade progress to API clients."""

from __future__ import annotations

import asyncio
import enum
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime


class EventType(str, enum.Enum):
    CASCADE_START = "cascade_start"
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    TASK_DECOMPOSED = "task_decomposed"
    CODE_GENERATED = "code_generated"
    REVIEW_RESULT = "review_result"
    TEST_RESULT = "test_result"
    RETRY = "retry"
    TASK_COMPLETE = "task_complete"
    TASK_FAILED = "task_failed"
    CASCADE_COMPLETE = "cascade_complete"
    CASCADE_ERROR = "cascade_error"


@dataclass
class CascadeEvent:
    event_type: EventType
    task_title: str = ""
    task_id: str = ""
    phase: str = ""
    message: str = ""
    data: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class EventBus:
    """Per-request event queue that bridges sync orchestrator callbacks to async SSE stream."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[CascadeEvent | None] = asyncio.Queue()

    def emit(self, event: CascadeEvent) -> None:
        """Put an event on the queue (called from orchestrator callback)."""
        self._queue.put_nowait(event)

    def finish(self) -> None:
        """Signal end-of-stream with a None sentinel."""
        self._queue.put_nowait(None)

    async def __aiter__(self) -> AsyncIterator[CascadeEvent]:
        """Async iteration over events until sentinel."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event
