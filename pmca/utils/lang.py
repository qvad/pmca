"""Language detection and mapping utilities."""

from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmca.tasks.tree import TaskNode

def detect_language(task: TaskNode) -> str:
    """Detect programming language using word boundaries and file extensions."""
    text = (task.title + (task.spec or "")).lower()
    if re.search(r"\b(ts|typescript)\b", text) or ".ts" in text:
        return "typescript"
    if re.search(r"\b(go|golang)\b", text) or ".go" in text:
        return "go"
    if re.search(r"\b(js|javascript)\b", text) or ".js" in text:
        return "javascript"
    return "python"

def get_extension(lang: str) -> str:
    """Map language name to primary file extension."""
    mapping = {
        "typescript": ".ts",
        "javascript": ".js",
        "go": ".go",
        "python": ".py"
    }
    return mapping.get(lang, ".py")
