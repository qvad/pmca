"""Context window management — build context that fits within model limits."""

from __future__ import annotations

import re

from pmca.tasks.state import TaskStatus
from pmca.tasks.tree import TaskNode, TaskTree
from pmca.utils.logger import get_logger

log = get_logger("utils.context")

# Rough token-to-char ratio for English text/code
CHARS_PER_TOKEN = 4

# Markers / regex for interface and metadata sections
_INTERFACE_MARKER = "[INTERFACE]"
_TARGET_FILE_RE = re.compile(r"TARGET_FILE:\s*(\S+)")
_EXPORTS_RE = re.compile(r"EXPORTS:\s*(.+)")
_DEPENDS_RE = re.compile(r"DEPENDS_ON:\s*(.+)")


def _extract_interface_section(spec: str) -> str:
    """Return the text after the ``[INTERFACE]`` marker, or empty string."""
    idx = spec.find(_INTERFACE_MARKER)
    if idx < 0:
        return ""
    return spec[idx + len(_INTERFACE_MARKER):].strip()


def _extract_metadata_lines(spec: str) -> str:
    """Return just the TARGET_FILE / EXPORTS / DEPENDS_ON lines from a spec."""
    lines: list[str] = []
    for line in spec.splitlines():
        stripped = line.strip()
        if (
            _TARGET_FILE_RE.match(stripped)
            or _EXPORTS_RE.match(stripped)
            or _DEPENDS_RE.match(stripped)
        ):
            lines.append(stripped)
    return "\n".join(lines)


def _extract_target_file(spec: str) -> str:
    """Extract TARGET_FILE value from spec, e.g. 'src/models.py'.

    Normalizes deep package paths (e.g. ``src/pkg/models.py`` → ``src/models.py``).
    """
    m = _TARGET_FILE_RE.search(spec)
    if not m:
        return ""
    from pmca.utils.assembler import _normalize_target_path
    return _normalize_target_path(m.group(1))


def _extract_exports(spec: str) -> list[str]:
    """Extract EXPORTS as a list of names from spec."""
    m = _EXPORTS_RE.search(spec)
    if not m:
        return []
    return [name.strip() for name in m.group(1).split(",") if name.strip()]


def _build_import_hint(target_file: str, exports: list[str]) -> str:
    """Build import hint line like 'Import: from models import Task, Category'."""
    if not target_file:
        return ""
    # Convert 'src/models.py' -> 'models'
    module = target_file.replace(".py", "").split("/")[-1]
    if exports:
        return f"  Import: from {module} import {', '.join(exports)}\n"
    return f"  Module: {module}\n"


class ContextManager:
    """Builds context strings that fit within model context windows."""

    def __init__(
        self,
        task_tree: TaskTree,
        project_mode: bool = False,
        rag_manager=None,
    ) -> None:
        self._tree = task_tree
        self._project_mode = project_mode
        self._rag_manager = rag_manager

    def build_context(self, task: TaskNode, max_tokens: int = 8192) -> str:
        """Build context string that fits within the model's effective window.

        Priority (highest to lowest):
        1. Task's own spec
        2. Parent spec
        3. Sibling specs (summaries — with interfaces when available)
        4. Existing code context

        Lower-priority items are truncated first.
        """
        max_chars = max_tokens * CHARS_PER_TOKEN
        sections: list[tuple[str, str, int]] = []  # (label, content, priority)

        # Priority 1: Task spec
        if task.spec:
            sections.append(("Task Specification", task.spec, 1))

        # Priority 2: Parent spec
        if task.parent_id:
            try:
                parent = self._tree.get_node(task.parent_id)
                if parent.spec:
                    sections.append(("Parent Specification", parent.spec, 2))
            except KeyError:
                pass

        # Priority 3: Library documentation from RAG
        if self._rag_manager is not None:
            query_text = task.spec or task.title
            rag_chunks = self._rag_manager.query(query_text)
            if rag_chunks:
                rag_text = "\n\n---\n\n".join(rag_chunks)
                sections.append(("Library Documentation", rag_text, 3))

        # Priority 4: Sibling summaries (enhanced for project mode)
        if task.parent_id:
            siblings = self._tree.get_siblings(task.id)
            if siblings:
                sibling_text = self._build_sibling_text(siblings)
                sections.append(("Sibling Tasks", sibling_text, 4))

        # Build the context, truncating lower-priority items if needed
        return self._assemble(sections, max_chars)

    def _build_sibling_text(self, siblings: list[TaskNode]) -> str:
        """Build sibling context with interface-awareness for project mode."""
        parts: list[str] = []
        for s in siblings:
            if self._project_mode:
                iface = _extract_interface_section(s.spec)
                target_file = _extract_target_file(s.spec)
                exports = _extract_exports(s.spec)
                if s.status == TaskStatus.VERIFIED and iface:
                    # Completed sibling — include import path and interface
                    import_hint = _build_import_hint(target_file, exports)
                    parts.append(
                        f"- {s.title} [DONE]:\n"
                        f"{import_hint}"
                        f"{iface}"
                    )
                else:
                    # Pending sibling — just metadata
                    meta = _extract_metadata_lines(s.spec)
                    if meta:
                        parts.append(f"- {s.title} [{s.status.value}]:\n{meta}")
                    else:
                        parts.append(f"- {s.title} [{s.status.value}]")
            else:
                # Original behaviour: 200-char spec summary
                if len(s.spec) > 200:
                    parts.append(f"- {s.title}: {s.spec[:200]}...")
                else:
                    parts.append(f"- {s.title}: {s.spec}")
        return "\n".join(parts)

    def build_integration_context(self, task: TaskNode, max_tokens: int = 8192) -> str:
        """Build context for integration review of a task's children."""
        max_chars = max_tokens * CHARS_PER_TOKEN
        sections: list[tuple[str, str, int]] = []

        # Parent spec
        sections.append(("Parent Specification", task.spec, 1))

        # Each child's spec and status
        children = self._tree.get_children(task.id)
        for child in children:
            child_info = f"Status: {child.status.value}\n"
            child_info += f"Spec:\n{child.spec}\n"
            if child.code_files:
                child_info += f"Code files: {', '.join(child.code_files)}\n"
            sections.append((f"Child: {child.title}", child_info, 2))

        return self._assemble(sections, max_chars)

    def _assemble(self, sections: list[tuple[str, str, int]], max_chars: int) -> str:
        """Assemble sections into a context string, respecting max length."""
        # Sort by priority (lower number = higher priority)
        sections.sort(key=lambda s: s[2])

        result_parts: list[str] = []
        chars_used = 0

        for label, content, priority in sections:
            section_text = f"## {label}\n{content}\n"
            section_len = len(section_text)

            if chars_used + section_len <= max_chars:
                result_parts.append(section_text)
                chars_used += section_len
            else:
                # Truncate this section to fit
                remaining = max_chars - chars_used
                if remaining > 100:  # Only include if we can fit something meaningful
                    truncated = content[: remaining - len(f"## {label}\n\n[truncated]")]
                    result_parts.append(f"## {label}\n{truncated}\n[truncated]")
                break  # Skip lower-priority items

        return "\n".join(result_parts)
