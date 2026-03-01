"""File assembler — merges leaf-task snippets into complete project files."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

from pmca.tasks.tree import TaskNode, TaskTree
from pmca.utils.logger import get_logger
from pmca.workspace.file_manager import FileManager

log = get_logger("utils.assembler")

# Regex to pull TARGET_FILE from task spec/description
_TARGET_FILE_RE = re.compile(r"TARGET_FILE:\s*(\S+)")


class FileAssembler:
    """Merges per-leaf code snippets into complete project files."""

    def __init__(self, file_manager: FileManager) -> None:
        self._fm = file_manager

    def assemble(
        self,
        parent_task: TaskNode,
        task_tree: TaskTree,
        snippet_store: dict[str, str],
    ) -> list[str]:
        """Merge all descendant snippets and write assembled files.

        Args:
            parent_task: The parent whose children produced snippets.
            task_tree: The full task tree (for walking descendants).
            snippet_store: ``{task_id:filepath: code}`` mapping.

        Returns:
            List of assembled file paths (relative to workspace).
        """
        # Group snippets by target filepath
        by_file: dict[str, list[str]] = defaultdict(list)
        for key, code in snippet_store.items():
            # key format: "task_id:filepath"
            parts = key.split(":", 1)
            if len(parts) != 2:
                continue
            filepath = parts[1]
            by_file[filepath].append(code)

        assembled_paths: list[str] = []
        for filepath, snippets in sorted(by_file.items()):
            if len(snippets) == 1:
                merged = snippets[0]
            else:
                merged = self._merge_snippets(snippets, filepath)
            self._fm.write_file(filepath, merged)
            assembled_paths.append(filepath)
            log.info(f"Assembled {filepath} from {len(snippets)} snippet(s)")

        # Ensure __init__.py exists for all package directories
        self.ensure_package_init_files(assembled_paths)

        return assembled_paths

    def _merge_snippets(self, snippets: list[str], filepath: str) -> str:
        """Merge multiple code snippets targeting the same file.

        Strategy:
        1. Split each snippet into imports vs body.
        2. Deduplicate imports (set-based, sorted output).
        3. Extract named definitions (functions/classes) from each body.
        4. Deduplicate by name (later definition wins).
        5. Reassemble: imports -> blank line -> definitions.
        """
        all_imports: set[str] = set()
        definitions: dict[str, str] = {}  # name -> full definition text

        for snippet in snippets:
            imports, body = self._split_imports(snippet)
            all_imports.update(imports)
            defs = self._extract_definitions(body)
            definitions.update(defs)  # later wins

        parts: list[str] = []
        if all_imports:
            parts.append("\n".join(sorted(all_imports)))
            parts.append("")  # blank line after imports
        if definitions:
            parts.append("\n\n".join(definitions.values()))

        return "\n".join(parts) + "\n" if parts else ""

    @staticmethod
    def _split_imports(code: str) -> tuple[list[str], str]:
        """Split code into import lines and the rest (body)."""
        import_lines: list[str] = []
        body_lines: list[str] = []
        in_body = False

        for line in code.splitlines():
            stripped = line.strip()
            if not in_body and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
            ):
                import_lines.append(stripped)
            elif not in_body and stripped == "":
                # skip blank lines between imports
                continue
            else:
                in_body = True
                body_lines.append(line)

        return import_lines, "\n".join(body_lines)

    @staticmethod
    def _extract_definitions(body: str) -> dict[str, str]:
        """Extract top-level class/function definitions from body text.

        Returns ``{name: full_text}`` for each definition.
        """
        try:
            tree = ast.parse(body)
        except SyntaxError:
            # Can't parse — return the whole body under a synthetic key
            return {"__unparseable__": body} if body.strip() else {}

        defs: dict[str, str] = {}
        lines = body.splitlines()

        nodes = [
            n for n in ast.iter_child_nodes(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        ]

        for i, node in enumerate(nodes):
            start = node.lineno - 1  # 0-indexed
            if i + 1 < len(nodes):
                end = nodes[i + 1].lineno - 1
            else:
                end = len(lines)
            # Trim trailing blank lines
            while end > start and not lines[end - 1].strip():
                end -= 1
            defs[node.name] = "\n".join(lines[start:end])

        # If there's non-definition body code (e.g. module-level statements),
        # capture it under a special key
        def_line_ranges = set()
        for i, node in enumerate(nodes):
            start = node.lineno - 1
            if i + 1 < len(nodes):
                end = nodes[i + 1].lineno - 1
            else:
                end = len(lines)
            for ln in range(start, end):
                def_line_ranges.add(ln)

        remaining = []
        for idx, line in enumerate(lines):
            if idx not in def_line_ranges and line.strip():
                remaining.append(line)
        if remaining:
            defs["__module_body__"] = "\n".join(remaining)

        return defs

    def ensure_package_init_files(self, paths: list[str]) -> None:
        """Create ``__init__.py`` in all parent directories of assembled files."""
        init_dirs: set[str] = set()
        for p in paths:
            parts = Path(p).parts
            # Walk up from the file's directory toward root
            for i in range(1, len(parts)):
                dir_path = str(Path(*parts[:i]))
                init_dirs.add(dir_path)

        for d in sorted(init_dirs):
            init_path = f"{d}/__init__.py"
            if not self._fm.file_exists(init_path):
                self._fm.write_file(init_path, "")
                log.info(f"Created {init_path}")


def _normalize_target_path(path: str) -> str:
    """Normalize TARGET_FILE path by stripping extra package directories.

    Models like deepseek generate ``src/taskboard/models.py`` instead of
    ``src/models.py``.  Normalize to ``src/<filename>.py`` or
    ``tests/<filename>.py`` so files land in the expected flat layout.
    """
    parts = Path(path).parts
    if len(parts) <= 2:
        return path  # already "src/models.py" or "models.py"
    # Keep the first directory (src/ or tests/) and the filename
    prefix = parts[0]   # "src" or "tests"
    filename = parts[-1] # "models.py"
    if prefix in ("src", "tests"):
        return f"{prefix}/{filename}"
    return path


def parse_target_file(spec: str) -> str | None:
    """Extract TARGET_FILE value from a task spec/description."""
    m = _TARGET_FILE_RE.search(spec)
    return _normalize_target_path(m.group(1)) if m else None
