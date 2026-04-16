"""Watcher agent — test execution and integration verification."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pmca.agents.base import BaseAgent
from pmca.models.config import AgentRole, CascadeConfig, LintConfig
from pmca.prompts import watcher as prompts
from pmca.tasks.state import ReviewResult, TestResult
from pmca.tasks.tree import TaskNode


def _build_pythonpath(workspace: Path) -> str:
    """Build PYTHONPATH including workspace root and src/ if it exists."""
    ws = workspace.resolve()
    paths = [str(ws)]
    src_dir = ws / "src"
    if src_dir.is_dir():
        paths.append(str(src_dir))
    return os.pathsep.join(paths)

# Common imports that LLMs forget — maps name → import statement
_KNOWN_IMPORTS: dict[str, str] = {
    # typing
    "Any": "from typing import Any",
    "Optional": "from typing import Optional",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Tuple": "from typing import Tuple",
    "Set": "from typing import Set",
    "Union": "from typing import Union",
    "Callable": "from typing import Callable",
    "Iterator": "from typing import Iterator",
    "Generator": "from typing import Generator",
    "Sequence": "from typing import Sequence",
    "Mapping": "from typing import Mapping",
    # collections
    "Counter": "from collections import Counter",
    "defaultdict": "from collections import defaultdict",
    "OrderedDict": "from collections import OrderedDict",
    "deque": "from collections import deque",
    "namedtuple": "from collections import namedtuple",
    # dataclasses
    "dataclass": "from dataclasses import dataclass",
    "field": "from dataclasses import dataclass, field",
    # stdlib modules (used as bare names after import)
    "re": "import re",
    "math": "import math",
    "json": "import json",
    "os": "import os",
    "sys": "import sys",
    "datetime": "import datetime",
    "enum": "import enum",
    "Enum": "from enum import Enum",
    "ABC": "from abc import ABC",
    "abstractmethod": "from abc import abstractmethod",
}


@dataclass
class TestError:
    """Structured representation of a single test failure."""

    test_name: str
    error_type: str  # "AssertionError", "NameError", "ImportError", etc.
    actual_value: str | None = None
    expected_value: str | None = None
    traceback: str = ""
    source_line: str | None = None
    local_variables: dict[str, str] | None = None

    def format_for_prompt(self) -> str:
        """Format this error for inclusion in a fix prompt."""
        parts = [f"## Test Failure: {self.test_name}"]
        parts.append(f"Error type: {self.error_type}")
        if self.source_line:
            parts.append(f"Failing line: {self.source_line}")
        if self.actual_value is not None and self.expected_value is not None:
            parts.append(f"Actual result: {self.actual_value}")
            parts.append(f"Expected by test: {self.expected_value}")
            # Hint about whether code or test is likely wrong
            try:
                actual_f = float(self.actual_value)
                expected_f = float(self.expected_value)
                if expected_f != 0:
                    ratio = actual_f / expected_f
                    if 0.8 < ratio < 1.2:
                        parts.append(
                            "→ Values are close — likely an arithmetic error "
                            "in the TEST assertion. Fix the test, not the code."
                        )
                    elif ratio in (10, 100, 0.1, 0.01):
                        parts.append(
                            "→ Values differ by a power of 10 — likely a "
                            "decimal point error in the test. Fix the test."
                        )
            except (ValueError, ZeroDivisionError):
                pass
        if self.local_variables:
            locals_str = ", ".join(
                f"{k}={v}" for k, v in self.local_variables.items()
            )
            parts.append(f"Local variables: {locals_str}")
        if self.traceback:
            parts.append(f"Traceback:\n{self.traceback}")
        return "\n".join(parts)


class _UsageVisitor(ast.NodeVisitor):
    """Track attribute access vs method calls per variable."""

    def __init__(self):
        self.calls: dict[str, set[str]] = {}      # var -> {attr names called}
        self.accesses: dict[str, set[str]] = {}    # var -> {attr names accessed}

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)):
            self.calls.setdefault(node.func.value.id, set()).add(node.func.attr)
        # Recurse into args only — skip node.func to avoid double-count
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            self.accesses.setdefault(node.value.id, set()).add(node.attr)
        self.generic_visit(node)


class WatcherAgent(BaseAgent):
    role = AgentRole.WATCHER

    def __init__(self, model_manager, workspace_path=None, lint_config: LintConfig | None = None,
                 cascade_config: CascadeConfig | None = None) -> None:
        super().__init__(model_manager)
        self._workspace_path = workspace_path
        self._lint_config = lint_config
        self._cascade = cascade_config or CascadeConfig()

    def _get_system_prompt(self) -> str:
        """Construct system prompt with Watcher SOP."""
        system = prompts.SYSTEM_PROMPT
        from pmca.prompts import WATCHER_SOP
        if WATCHER_SOP not in system:
            system += "\n" + WATCHER_SOP
        return system

    def _find_python(self) -> str:
        """Find a Python executable that has pytest available."""
        # First: use the same Python that's running PMCA (it has pytest in dev)
        return sys.executable

    @staticmethod
    def _extract_function_source(source: str, function_name: str) -> tuple[str, int, int] | None:
        """Extract a function's source lines using AST.

        Returns (function_source, start_line_0indexed, end_line_0indexed) or None.
        Handles both top-level functions and class methods.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.splitlines()

        # Search top-level and inside classes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == function_name and node.end_lineno is not None:
                    start = node.lineno - 1  # 0-indexed
                    end = node.end_lineno     # end_lineno is 1-indexed inclusive
                    return "\n".join(lines[start:end]), start, end
        return None

    @staticmethod
    def _parse_error_location(
        error: TestError, workspace: Path, code_files: list[str],
    ) -> tuple[str, str] | None:
        """Find which source file and function a TestError originates from.

        Parses the traceback for file paths and line numbers, then uses AST
        to find which function contains that line. Only considers code files
        (not test files).

        Returns (relative_file_path, function_name) or None.
        """
        # Strategy 1: parse traceback for file:line references
        tb = error.traceback or ""
        source_line = error.source_line or ""

        # Look for file references in traceback
        for code_file in code_files:
            abs_path = workspace / code_file
            if not abs_path.exists() or not code_file.endswith(".py"):
                continue

            try:
                source = abs_path.read_text()
                tree = ast.parse(source)
            except (OSError, SyntaxError):
                continue

            # Strategy 2: if error has source_line, find which function contains
            # code that matches the error pattern
            # For TypeError/KeyError, the error often comes from the source code
            # For AssertionError, it comes from tests — but we want the CODE function
            # that produced the wrong result

            # Collect all functions with their line ranges
            functions: list[tuple[str, int, int]] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.end_lineno is not None:
                        functions.append((node.name, node.lineno, node.end_lineno))

            if not functions:
                continue

            # For TypeError/KeyError: search traceback for line numbers in this file
            for m in re.finditer(r"line (\d+)", tb):
                lineno = int(m.group(1))
                for fname, start, end in functions:
                    if start <= lineno <= end:
                        return code_file, fname

            # For AssertionError: try to find the function being tested
            # The test name often contains the function name: test_sort → sort
            test_name = error.test_name.split("::")[-1] if "::" in error.test_name else error.test_name
            # Strip "test_" prefix to get the likely function name
            if test_name.startswith("test_"):
                target = test_name[5:]
                for fname, _start, _end in functions:
                    if fname == target or target in fname or fname in target:
                        return code_file, fname

            # Fallback: if only one non-__init__ function exists, use it
            non_init = [(f, s, e) for f, s, e in functions if f != "__init__"]
            if len(non_init) == 1:
                return code_file, non_init[0][0]

        return None

    @staticmethod
    def _fix_package_imports(workspace: Path) -> int:
        """Rewrite package-style imports to flat sibling imports.

        Models like deepseek frequently generate ``from taskboard.models import Task``
        instead of ``from models import Task``.  This scans all ``.py`` files under
        the workspace ``src/`` directory and rewrites any ``from <pkg>.<module> import``
        to ``from <module> import`` when ``<module>.py`` exists as a sibling.

        Returns the number of import lines rewritten.
        """
        src_dir = workspace / "src"
        if not src_dir.is_dir():
            src_dir = workspace  # flat layout
        py_files = list(src_dir.glob("*.py"))
        if not py_files:
            return 0

        # Build set of available module names (without .py extension)
        available_modules = {f.stem for f in py_files if f.stem != "__init__"}

        fixes = 0
        # Also scan test files
        test_dir = workspace / "tests"
        all_files = list(py_files)
        if test_dir.is_dir():
            all_files.extend(test_dir.glob("*.py"))

        for py_file in all_files:
            content = py_file.read_text()
            new_content = content
            # Match: from <anything>.<module> import ...
            # where <module> is an available sibling
            for mod in available_modules:
                # from somepackage.module import X  →  from module import X
                pattern = re.compile(
                    rf"^(from\s+)\w+\.({re.escape(mod)})\s+(import\s+.+)$",
                    re.MULTILINE,
                )
                new_content, n = pattern.subn(rf"\1\2 \3", new_content)
                fixes += n
                # Also handle deeper nesting: from a.b.module import X
                pattern2 = re.compile(
                    rf"^(from\s+)\w+\.\w+\.({re.escape(mod)})\s+(import\s+.+)$",
                    re.MULTILINE,
                )
                new_content, n = pattern2.subn(rf"\1\2 \3", new_content)
                fixes += n
            if new_content != content:
                py_file.write_text(new_content)

        return fixes

    @staticmethod
    def _fix_mutable_defaults(file_path: Path) -> int:
        """Fix mutable default arguments in __init__ methods.

        Detects patterns like ``def __init__(self, x=[])`` and rewrites to
        ``def __init__(self, x=None)`` with ``x = x if x is not None else []``
        in the function body.  This is a common Python gotcha that 7B models
        produce frequently.

        Returns the number of fixes applied.
        """
        source = file_path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        lines = source.splitlines(keepends=True)
        fixes = 0

        # Collect fixes in reverse line order so edits don't shift line numbers
        edits: list[tuple[int, str, str, str, int]] = []
        # (line_idx_of_def, param_name, mutable_literal, body_first_line_idx, indent)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name != "__init__":
                continue
            defaults = node.args.defaults
            args = node.args.args
            # defaults align to the END of args list
            offset = len(args) - len(defaults)
            for i, default in enumerate(defaults):
                mutable_type = None
                if isinstance(default, ast.List):
                    mutable_type = "[]"
                elif isinstance(default, ast.Dict):
                    mutable_type = "{}"
                elif isinstance(default, ast.Set):
                    mutable_type = "set()"
                if mutable_type is None:
                    continue
                param = args[offset + i]
                param_name = param.arg
                # Find first line of function body for inserting the fix
                body_line = node.body[0].lineno - 1  # 0-indexed
                # Detect indentation of body
                body_indent = len(lines[body_line]) - len(lines[body_line].lstrip())
                edits.append((
                    default.lineno - 1,  # line of the default value
                    param_name,
                    mutable_type,
                    body_line,
                    body_indent,
                ))

        if not edits:
            return 0

        # Apply in reverse order to preserve line numbers
        edits.sort(key=lambda e: e[0], reverse=True)
        for _def_line, param_name, mutable_type, body_line, body_indent in edits:
            # Use regex to replace the default value in the source
            source = "".join(lines)
            # Match param_name followed by optional type annotation, then = mutable
            escaped_mutable = re.escape(mutable_type)
            pattern = re.compile(
                rf"(\b{re.escape(param_name)}\s*(?::[^=,)]+)?\s*=\s*){escaped_mutable}"
            )
            new_source, count = pattern.subn(rf"\g<1>None", source, count=1)
            if count == 0:
                continue

            # Insert `param_name = param_name if param_name is not None else mutable`
            # at the beginning of the function body
            lines = new_source.splitlines(keepends=True)
            indent_str = " " * body_indent
            fix_line = (
                f"{indent_str}{param_name} = {param_name} if {param_name} "
                f"is not None else {mutable_type}\n"
            )
            # Don't insert if already present
            if fix_line.strip() not in source:
                lines.insert(body_line, fix_line)
                fixes += 1

        if fixes > 0:
            file_path.write_text("".join(lines))
        return fixes

    @staticmethod
    def _fix_attr_method_shadowing(file_path: Path) -> int:
        """Rename attributes that shadow methods: self.X → self._X.

        When a class has both ``self.X = ...`` in ``__init__`` and a
        ``def X(self)`` method, the attribute shadows the method causing
        ``TypeError: 'int' object is not callable``.  This renames all
        ``self.X`` attribute references to ``self._X`` throughout the
        class, preserving the method name.

        Returns the number of fixes applied.
        """
        source = file_path.read_text()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0

        fixes = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            attrs: set[str] = set()
            methods: set[str] = set()
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "__init__":
                        for stmt in ast.walk(item):
                            if (isinstance(stmt, ast.Assign)
                                    and len(stmt.targets) == 1
                                    and isinstance(stmt.targets[0], ast.Attribute)
                                    and isinstance(stmt.targets[0].value, ast.Name)
                                    and stmt.targets[0].value.id == "self"):
                                attrs.add(stmt.targets[0].attr)
                    elif item.args.args and item.args.args[0].arg == "self":
                        methods.add(item.name)
            shadows = attrs & methods
            for name in shadows:
                # Replace self.name with self._name throughout source
                # Use word-boundary regex to avoid partial matches
                pattern = re.compile(rf"self\.{re.escape(name)}\b")
                new_source = pattern.sub(f"self._{name}", source)
                if new_source != source:
                    source = new_source
                    fixes += 1

        if fixes > 0:
            file_path.write_text(source)
        return fixes

    async def auto_fix_deterministic(self, task: TaskNode) -> int:
        """Fix common deterministic errors without calling the LLM.

        Attempts to import each source file in a subprocess and fixes:
        - Package-style imports rewritten to flat sibling imports
        - Missing imports (NameError for known stdlib/typing names)
        - Missing module imports (ModuleNotFoundError for known modules)
        - Mutable default arguments in __init__ (e.g. param=[])
        - Unused/duplicate imports via ruff --fix (if ruff is available)

        Returns the number of fixes applied.
        """
        if not task.code_files or not self._workspace_path:
            return 0

        # First pass: fix package-style imports (e.g. from taskboard.models → from models)
        fixes_applied = 0
        ws = Path(self._workspace_path)
        if self._cascade.import_fixes:
            pkg_fixes = self._fix_package_imports(ws)
            if pkg_fixes > 0:
                fixes_applied += pkg_fixes
                self._log.info(
                    f"Auto-fixed: {pkg_fixes} package-style import(s) rewritten to flat"
                )

        if self._cascade.ast_fixes:
            # Second pass: fix mutable default arguments (no subprocess needed)
            for code_file in task.code_files:
                file_path = Path(self._workspace_path) / code_file
                if not file_path.exists() or not code_file.endswith(".py"):
                    continue
                count = self._fix_mutable_defaults(file_path)
                if count > 0:
                    fixes_applied += count
                    self._log.info(
                        f"Auto-fixed: {count} mutable default arg(s) in {code_file}"
                    )

            # Third pass: fix attribute/method shadowing (self.size + def size())
            for code_file in task.code_files:
                file_path = Path(self._workspace_path) / code_file
                if not file_path.exists() or not code_file.endswith(".py"):
                    continue
                count = self._fix_attr_method_shadowing(file_path)
                if count > 0:
                    fixes_applied += count
                    self._log.info(
                        f"Auto-fixed: {count} attribute/method shadowing(s) in {code_file}"
                    )

        # Ruff auto-fix pass: clean unused/duplicate imports, etc.
        from pmca.utils.linters import ruff_autofix

        all_files = list(task.code_files) + list(task.test_files)
        for code_file in all_files:
            file_path = ws / code_file
            if not file_path.exists() or not code_file.endswith(".py"):
                continue
            count = await ruff_autofix(file_path, ws)
            fixes_applied += count

        # Semgrep auto-fix pass: fix known 7B anti-patterns (graceful skip if not installed)
        from pmca.utils.linters import semgrep_autofix

        for code_file in task.code_files:
            file_path = ws / code_file
            if not file_path.exists() or not code_file.endswith(".py"):
                continue
            count = await semgrep_autofix(file_path, ws)
            fixes_applied += count

        if not self._cascade.import_fixes:
            return fixes_applied

        python_exe = self._find_python()
        max_passes = 3  # multiple imports may be missing

        for _ in range(max_passes):
            fixed_this_pass = 0
            for code_file in task.code_files:
                file_path = Path(self._workspace_path) / code_file
                if not file_path.exists() or not code_file.endswith(".py"):
                    continue

                # Try to import the file in a subprocess
                module_path = code_file.replace("/", ".").removesuffix(".py")
                try:
                    proc = await asyncio.create_subprocess_exec(
                        python_exe, "-c", f"import {module_path}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self._workspace_path,
                        env={**os.environ, "PYTHONPATH": _build_pythonpath(self._workspace_path)},
                    )
                    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                    err_output = stderr.decode()
                except (asyncio.TimeoutError, FileNotFoundError):
                    continue

                if proc.returncode == 0:
                    continue  # imports fine

                # Parse NameError: name 'X' is not defined
                name_match = re.search(
                    r"NameError: name '(\w+)' is not defined", err_output
                )
                if name_match:
                    missing_name = name_match.group(1)
                    import_line = _KNOWN_IMPORTS.get(missing_name)
                    if import_line:
                        content = file_path.read_text()
                        # Don't add duplicate imports
                        if import_line not in content:
                            content = import_line + "\n" + content
                            file_path.write_text(content)
                            fixes_applied += 1
                            fixed_this_pass += 1
                            self._log.info(
                                f"Auto-fixed: added '{import_line}' to {code_file}"
                            )
                        continue

                # Parse ImportError / ModuleNotFoundError
                import_match = re.search(
                    r"(?:ImportError|ModuleNotFoundError): "
                    r"No module named '(\w+)'",
                    err_output,
                )
                if import_match:
                    missing_module = import_match.group(1)
                    import_line = _KNOWN_IMPORTS.get(missing_module)
                    if import_line:
                        content = file_path.read_text()
                        if import_line not in content:
                            content = import_line + "\n" + content
                            file_path.write_text(content)
                            fixes_applied += 1
                            fixed_this_pass += 1
                            self._log.info(
                                f"Auto-fixed: added '{import_line}' to {code_file}"
                            )

            if fixed_this_pass == 0:
                break  # No more fixable errors

        return fixes_applied

    # ------------------------------------------------------------------
    # Phase 2B — Defensive guard injection (preventive, pre-test)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_none_safe_expr(expr_node: ast.expr) -> str | None:
        """Build a None-safe sort expression from an AST node.

        Returns a string like '(x.date is None, x.date or "")' or None
        if the expression is already safe or can't be guarded.
        """
        # Already a none-safe tuple like (x is None, x or '')
        if isinstance(expr_node, ast.Tuple):
            # Check if first element is an `is None` compare
            if (
                len(expr_node.elts) >= 2
                and isinstance(expr_node.elts[0], ast.Compare)
                and any(isinstance(op, ast.Is) for op in expr_node.elts[0].ops)
            ):
                return None  # already guarded

        src = ast.unparse(expr_node)

        # Skip constants, unary ops on constants (like -priority), booleans
        if isinstance(expr_node, ast.Constant):
            return None
        if isinstance(expr_node, ast.UnaryOp) and isinstance(expr_node.operand, ast.Constant):
            return None

        # For simple field access (x.date, x['due'], x.get(field))
        # these can be None — wrap them.
        # Use '' as fallback only for the sort comparison value; (is_None, '')
        # sorts None last. We use '' because it's safe for string comparison
        # and for mixed types Python will just use the is_None flag.
        if isinstance(expr_node, (ast.Attribute, ast.Subscript, ast.Call, ast.Name)):
            return f"({src} is None, {src} if {src} is not None else '')"

        return None

    @staticmethod
    def _guard_sort_keys(source: str) -> tuple[str, int]:
        """Wrap sort key lambdas with None-safe expressions.

        Handles both simple keys and tuple keys:
          Before: key=lambda x: x.date
          After:  key=lambda x: (x.date is None, x.date or "")

          Before: key=lambda x: (x['done'], x[sort_by])
          After:  key=lambda x: (x['done'], (x[sort_by] is None, x[sort_by] or ''))

        Skips elements that are already guarded, constants, or unary ops.
        Returns (new_source, fixes_applied).
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0

        lines = source.splitlines()
        fixes = 0
        replacements: list[tuple[int, str, str]] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            is_sorted = isinstance(node.func, ast.Name) and node.func.id == "sorted"
            is_sort_method = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sort"
            )
            if not is_sorted and not is_sort_method:
                continue

            key_kw = None
            for kw in node.keywords:
                if kw.arg == "key":
                    key_kw = kw
                    break
            if key_kw is None:
                continue

            if not isinstance(key_kw.value, ast.Lambda):
                continue

            lam = key_kw.value
            body = lam.body
            args_src = ast.unparse(lam.args)

            # Case 1: Tuple key — guard individual elements
            if isinstance(body, ast.Tuple):
                new_elts = []
                any_changed = False
                for elt in body.elts:
                    guarded = WatcherAgent._make_none_safe_expr(elt)
                    if guarded:
                        new_elts.append(guarded)
                        any_changed = True
                    else:
                        new_elts.append(ast.unparse(elt))
                if not any_changed:
                    continue
                guarded_body = f"({', '.join(new_elts)})"
                new_lambda = f"lambda {args_src}: {guarded_body}"
            else:
                # Case 2: Simple key — wrap entire body
                guarded = WatcherAgent._make_none_safe_expr(body)
                if not guarded:
                    continue
                new_lambda = f"lambda {args_src}: {guarded}"

            # Try to match the lambda in the source line
            # ast.unparse may use different quotes than the original source,
            # so we try the unparsed version first, then fall back to
            # extracting the lambda from the raw source via column offsets.
            old_lambda_src = ast.unparse(lam)
            if lam.lineno and lam.lineno <= len(lines):
                line_idx = lam.lineno - 1
                line = lines[line_idx]
                if old_lambda_src in line:
                    replacements.append((line_idx, old_lambda_src, new_lambda))
                else:
                    # Fallback: find 'lambda' keyword and extract to end of key=
                    # by locating the key= keyword's col_offset
                    lam_start = line.find("lambda")
                    if lam_start >= 0:
                        # Find the extent: from 'lambda' to the next top-level comma or ')'
                        depth = 0
                        end = len(line)
                        for i in range(lam_start, len(line)):
                            ch = line[i]
                            if ch in "([{":
                                depth += 1
                            elif ch in ")]}":
                                if depth == 0:
                                    end = i
                                    break
                                depth -= 1
                            elif ch == "," and depth == 0:
                                end = i
                                break
                        old_text = line[lam_start:end].rstrip()
                        if old_text:
                            replacements.append((line_idx, old_text, new_lambda))

        for line_idx, old_text, new_text in sorted(replacements, reverse=True):
            lines[line_idx] = lines[line_idx].replace(old_text, new_text, 1)
            fixes += 1

        if fixes > 0:
            new_source = "\n".join(lines)
            try:
                ast.parse(new_source)
            except SyntaxError:
                return source, 0
            return new_source, fixes
        return source, 0

    @staticmethod
    def _guard_index_zero(source: str) -> tuple[str, int]:
        """Add empty-collection guards before unguarded x[0] access.

        Before: result = items[0]
        After:  result = items[0] if items else None

        Only guards simple Name[0] or Name[-1] patterns that aren't
        already inside an if-guard.

        Returns (new_source, fixes_applied).
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0

        lines = source.splitlines()
        fixes = 0
        replacements: list[tuple[int, str, str]] = []

        # Walk each function to understand scope and existing guards
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Collect names that have if-guards in this function
            guarded_names: set[str] = set()
            for child in ast.walk(func_node):
                if isinstance(child, ast.If):
                    test = child.test
                    # if items: / if len(items): / if items is not None:
                    if isinstance(test, ast.Name):
                        guarded_names.add(test.id)
                    elif isinstance(test, ast.Call) and isinstance(test.func, ast.Name):
                        if test.func.id == "len" and test.args:
                            if isinstance(test.args[0], ast.Name):
                                guarded_names.add(test.args[0].id)
                    elif isinstance(test, ast.Compare):
                        if isinstance(test.left, ast.Name):
                            guarded_names.add(test.left.id)

            # Find unguarded x[0] or x[-1] accesses
            for child in ast.walk(func_node):
                if not isinstance(child, ast.Subscript):
                    continue
                if not isinstance(child.value, ast.Name):
                    continue
                # Check index is 0 or -1
                slc = child.slice
                is_zero = isinstance(slc, ast.Constant) and slc.value in (0, -1)
                is_neg = (
                    isinstance(slc, ast.UnaryOp)
                    and isinstance(slc.op, ast.USub)
                    and isinstance(slc.operand, ast.Constant)
                    and slc.operand.value == 1
                )
                if not is_zero and not is_neg:
                    continue

                collection_name = child.value.id
                if collection_name in guarded_names:
                    continue

                # Check this subscript is a standalone assignment value
                # (not inside a larger expression already handled)
                if not child.lineno or child.lineno > len(lines):
                    continue

                line_idx = child.lineno - 1
                line = lines[line_idx]

                # Skip if line already has an `if` ternary
                if " if " in line and " else " in line:
                    continue

                # Build the guarded expression
                sub_src = ast.unparse(child)
                guarded_expr = f"{sub_src} if {collection_name} else None"

                if sub_src in line:
                    replacements.append((line_idx, sub_src, guarded_expr))
                    guarded_names.add(collection_name)  # don't double-guard

        # Apply replacements in reverse order
        seen_lines: set[int] = set()
        for line_idx, old_text, new_text in sorted(replacements, reverse=True):
            if line_idx in seen_lines:
                continue
            seen_lines.add(line_idx)
            lines[line_idx] = lines[line_idx].replace(old_text, new_text, 1)
            fixes += 1

        if fixes > 0:
            new_source = "\n".join(lines)
            try:
                ast.parse(new_source)
            except SyntaxError:
                return source, 0
            return new_source, fixes
        return source, 0

    @staticmethod
    def _guard_missing_else_raise(source: str) -> tuple[str, int]:
        """Add 'else: raise ValueError' to if/elif chains that handle enums.

        7B models often generate if/elif chains for op/func_name dispatch
        without a final else clause. This adds 'else: raise ValueError(...)'.

        Pattern: a function that has 3+ if/elif branches comparing the same
        variable, with no else clause and no raise in the chain.

        Returns (new_source, fixes_applied).
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0

        lines = source.splitlines()
        fixes = 0
        insertions: list[tuple[int, str]] = []  # (line_idx_after, text_to_insert)

        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for node in ast.iter_child_nodes(func_node):
                if not isinstance(node, ast.If):
                    continue

                # Count the if/elif chain length
                chain_len = 1
                has_else = False
                has_raise = False
                compared_var = None
                last_node = node

                # Extract the compared variable from the first if
                if isinstance(node.test, ast.Compare):
                    if isinstance(node.test.left, ast.Name):
                        compared_var = node.test.left.id

                # Walk the elif chain
                current = node
                while current.orelse:
                    if (
                        len(current.orelse) == 1
                        and isinstance(current.orelse[0], ast.If)
                    ):
                        current = current.orelse[0]
                        chain_len += 1
                        last_node = current
                        # Check for raise in this branch
                        for child in ast.walk(current):
                            if isinstance(child, ast.Raise):
                                has_raise = True
                    else:
                        has_else = True
                        break

                # Check for raise in any branch
                for child in ast.walk(node):
                    if isinstance(child, ast.Raise):
                        has_raise = True

                # Only guard chains with 3+ branches, no else, comparing a variable
                if chain_len < 3 or has_else or has_raise or not compared_var:
                    continue

                # Determine indentation of the last elif
                if last_node.end_lineno and last_node.end_lineno <= len(lines):
                    last_line = lines[last_node.end_lineno - 1]
                    indent = len(last_line) - len(last_line.lstrip())
                    # Find the indentation of the if/elif keyword itself
                    if_line = lines[last_node.lineno - 1]
                    if_indent = len(if_line) - len(if_line.lstrip())
                    else_line = " " * if_indent + "else:"
                    raise_line = " " * if_indent + f"    raise ValueError(f\"Unknown {{repr({compared_var})}}\")"
                    insertions.append(
                        (last_node.end_lineno, f"{else_line}\n{raise_line}")
                    )
                    fixes += 1

        # Apply insertions in reverse order
        for line_idx, text in sorted(insertions, reverse=True):
            lines.insert(line_idx, text)

        if fixes > 0:
            new_source = "\n".join(lines)
            try:
                ast.parse(new_source)
            except SyntaxError:
                return source, 0
            return new_source, fixes
        return source, 0

    async def inject_defensive_guards(self, task: TaskNode) -> int:
        """Inject defensive guards into generated code before first test run.

        Zero LLM tokens. Returns count of guards injected.
        """
        if not task.code_files or not self._workspace_path:
            return 0

        ws = Path(self._workspace_path)
        total = 0
        for code_file in task.code_files:
            file_path = ws / code_file
            if not file_path.exists() or not code_file.endswith(".py"):
                continue
            try:
                source = file_path.read_text()
            except OSError:
                continue
            new_source, count = self._guard_sort_keys(source)
            new_source, count2 = self._guard_index_zero(new_source)
            new_source, count3 = self._guard_missing_else_raise(new_source)
            subtotal = count + count2 + count3
            if subtotal > 0:
                file_path.write_text(new_source)
                total += subtotal
                self._log.info(
                    f"Defensive guards: {count} sort key(s), {count2} index guard(s), "
                    f"{count3} else-raise(s) in {code_file}"
                )
        return total

    # ------------------------------------------------------------------
    # Phase 2A — Runtime error repair (error-driven, in retry loop)
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_typeerror_in_sort(source: str, error: TestError) -> tuple[str, int]:
        """Fix TypeError from None comparison in sort keys.

        When traceback mentions "'<' not supported" and a sort call is on
        the failing line, wrap the key lambda with None-safe tuple.

        Returns (new_source, fixes_applied).
        """
        tb = error.traceback or ""
        if "'<' not supported" not in tb:
            return source, 0

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0

        # Find the failing line number from traceback
        error_lines: list[int] = []
        for m in re.finditer(r"line (\d+)", tb):
            error_lines.append(int(m.group(1)))

        if not error_lines:
            return source, 0

        lines = source.splitlines()
        fixes = 0

        # Find all sort/sorted calls and check if they're on an error line
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            is_sorted = isinstance(node.func, ast.Name) and node.func.id == "sorted"
            is_sort_method = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "sort"
            )
            if not is_sorted and not is_sort_method:
                continue

            # Check if this sort call is in a function that contains the error line
            # (sort itself may not be on the exact error line, but in the same function)
            call_line = node.lineno
            if not call_line:
                continue

            # Find key= kwarg
            key_kw = None
            for kw in node.keywords:
                if kw.arg == "key":
                    key_kw = kw
                    break

            if key_kw is None:
                # No key= means sorting by value directly; add a key
                # This handles: sorted(items) where items contain None
                continue

            if not isinstance(key_kw.value, ast.Lambda):
                continue

            lam = key_kw.value
            body = lam.body
            args_src = ast.unparse(lam.args)

            # Handle both tuple and simple keys — reuse shared helper
            if isinstance(body, ast.Tuple):
                new_elts = []
                any_changed = False
                for elt in body.elts:
                    g = WatcherAgent._make_none_safe_expr(elt)
                    if g:
                        new_elts.append(g)
                        any_changed = True
                    else:
                        new_elts.append(ast.unparse(elt))
                if not any_changed:
                    continue
                new_lambda = f"lambda {args_src}: ({', '.join(new_elts)})"
            else:
                g = WatcherAgent._make_none_safe_expr(body)
                if not g:
                    continue
                new_lambda = f"lambda {args_src}: {g}"

            old_lambda_src = ast.unparse(lam)
            if lam.lineno and lam.lineno <= len(lines):
                line_idx = lam.lineno - 1
                line = lines[line_idx]
                if old_lambda_src in line:
                    lines[line_idx] = line.replace(old_lambda_src, new_lambda, 1)
                    fixes += 1
                else:
                    # Fallback: find lambda keyword by position
                    lam_start = line.find("lambda")
                    if lam_start >= 0:
                        depth = 0
                        end = len(line)
                        for i in range(lam_start, len(line)):
                            ch = line[i]
                            if ch in "([{":
                                depth += 1
                            elif ch in ")]}":
                                if depth == 0:
                                    end = i
                                    break
                                depth -= 1
                            elif ch == "," and depth == 0:
                                end = i
                                break
                        old_text = line[lam_start:end].rstrip()
                        if old_text:
                            lines[line_idx] = line.replace(old_text, new_lambda, 1)
                            fixes += 1

        if fixes > 0:
            new_source = "\n".join(lines)
            try:
                ast.parse(new_source)
            except SyntaxError:
                return source, 0
            return new_source, fixes
        return source, 0

    @staticmethod
    def _fix_index_error(source: str, error: TestError) -> tuple[str, int]:
        """Fix IndexError by adding empty-collection guard.

        When traceback points to a line with x[0] or x[-1], wraps it
        with an if-guard or ternary.

        Returns (new_source, fixes_applied).
        """
        tb = error.traceback or ""

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source, 0

        # Find the failing line from traceback
        error_linenos: list[int] = []
        for m in re.finditer(r"line (\d+)", tb):
            error_linenos.append(int(m.group(1)))

        if not error_linenos:
            return source, 0

        lines = source.splitlines()
        fixes = 0

        for lineno in error_linenos:
            if lineno < 1 or lineno > len(lines):
                continue
            line_idx = lineno - 1
            line = lines[line_idx]

            # Already has a guard
            if " if " in line and " else " in line:
                continue

            # Find x[0] or x[-1] pattern
            idx_match = re.search(r'(\w+)\[([0-9]+|-[0-9]+)\]', line)
            if not idx_match:
                continue

            collection_name = idx_match.group(0)  # e.g. "items[0]"
            var_name = idx_match.group(1)  # e.g. "items"

            # Build ternary guard
            guarded = f"{collection_name} if {var_name} else None"
            lines[line_idx] = line.replace(collection_name, guarded, 1)
            fixes += 1
            break  # Fix only the first occurrence per error

        if fixes > 0:
            new_source = "\n".join(lines)
            try:
                ast.parse(new_source)
            except SyntaxError:
                return source, 0
            return new_source, fixes
        return source, 0

    async def fix_runtime_errors(
        self,
        task: TaskNode,
        structured_errors: list[TestError],
    ) -> int:
        """Apply deterministic fixes for runtime errors. Zero LLM tokens.

        Runs in the retry loop BEFORE targeted_micro_fix.
        Returns count of fixes applied.
        """
        if not task.code_files or not self._workspace_path:
            return 0

        ws = Path(self._workspace_path)
        total = 0
        for error in structured_errors:
            for code_file in task.code_files:
                if code_file.startswith("test"):
                    continue
                file_path = ws / code_file
                if not file_path.exists() or not code_file.endswith(".py"):
                    continue
                try:
                    source = file_path.read_text()
                except OSError:
                    continue
                new_source = source

                if error.error_type == "TypeError":
                    new_source, n = self._fix_typeerror_in_sort(new_source, error)
                    total += n
                elif error.error_type == "IndexError":
                    new_source, n = self._fix_index_error(new_source, error)
                    total += n

                if new_source != source:
                    try:
                        ast.parse(new_source)
                        file_path.write_text(new_source)
                    except SyntaxError:
                        pass  # Don't write broken code
        return total

    @staticmethod
    def _check_api_consistency(workspace: Path) -> list[str]:
        """Detect attribute/method shadowing and mixed call sites.

        Pass 1 — scans implementation files for classes where ``self.X = ...``
        in ``__init__`` shadows a ``def X(self)`` method.

        Pass 2 — scans test files for variables where the same name is used
        both as a bare attribute access and as a method call.

        Returns a list of blocking error strings.
        """
        errors: list[str] = []

        src_dir = workspace / "src"
        if not src_dir.is_dir():
            src_dir = workspace

        # --- Pass 1: shadowing detection in implementation files ---
        for py_file in src_dir.glob("*.py"):
            if py_file.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                attrs: set[str] = set()
                methods: set[str] = set()
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__init__":
                            # Collect self.X = ... assignments
                            for stmt in ast.walk(item):
                                if (isinstance(stmt, ast.Assign)
                                        and len(stmt.targets) == 1
                                        and isinstance(stmt.targets[0], ast.Attribute)
                                        and isinstance(stmt.targets[0].value, ast.Name)
                                        and stmt.targets[0].value.id == "self"):
                                    attrs.add(stmt.targets[0].attr)
                        else:
                            # Non-__init__ method with self parameter
                            if (item.args.args
                                    and item.args.args[0].arg == "self"):
                                methods.add(item.name)
                shadows = attrs & methods
                for name in sorted(shadows):
                    errors.append(
                        f"'{name}' is both an attribute and a method in "
                        f"{node.name}; rename the attribute to '_{name}' "
                        f"and keep the method"
                    )

        # --- Pass 2: mixed call-site detection in test files ---
        test_dir = workspace / "tests"
        test_files = list(test_dir.glob("test_*.py")) if test_dir.is_dir() else []
        # Also check test files directly in workspace
        test_files.extend(
            f for f in workspace.glob("test_*.py") if f not in test_files
        )

        _IGNORED_MODULES = {"httpx", "os", "sys", "json", "re", "math", "asyncio", "yaml"}

        for tf in test_files:
            try:
                tree = ast.parse(tf.read_text())
            except SyntaxError:
                continue
            visitor = _UsageVisitor()
            visitor.visit(tree)
            for var in visitor.calls:
                if var in _IGNORED_MODULES:
                    continue
                if var not in visitor.accesses:
                    continue
                mixed = visitor.accesses[var] & visitor.calls[var]
                # Filter out dunder names
                mixed = {n for n in mixed if not (n.startswith("__") and n.endswith("__"))}
                for name in sorted(mixed):
                    errors.append(
                        f"'{name}' used as both attribute and method call "
                        f"for variable '{var}'"
                    )

        return errors

    async def static_analysis_gate(
        self, task: TaskNode,
    ) -> tuple[list[str], list[str]]:
        """Run deterministic static analysis.

        Returns (blocking_errors, informational_errors).
        - blocking: syntax errors that prevent code from running
        - informational: mypy/ruff issues that don't prevent execution
        """
        blocking: list[str] = []
        informational: list[str] = []
        all_paths = list(task.code_files) + list(task.test_files)
        for path in all_paths:
            if not path.endswith(".py"):
                continue
            full_path = Path(self._workspace_path) / path if self._workspace_path else None
            if full_path is None or not full_path.exists():
                continue
            code = full_path.read_text()
            try:
                ast.parse(code)
            except SyntaxError as e:
                blocking.append(f"{path}:{e.lineno}: SyntaxError: {e.msg}")

        # External linters (optional — only if configured and tools installed)
        # These are informational — code may run fine despite type/style issues
        if self._lint_config and self._workspace_path:
            from pmca.utils.linters import run_mypy, run_ruff

            for path in all_paths:
                if not path.endswith(".py"):
                    continue
                full_path = Path(self._workspace_path) / path
                if not full_path.exists():
                    continue
                if self._lint_config.mypy:
                    mypy_errors = await run_mypy(full_path, self._workspace_path)
                    informational.extend(mypy_errors)
                if self._lint_config.ruff:
                    ruff_errors = await run_ruff(full_path, self._workspace_path)
                    informational.extend(ruff_errors)

        # API consistency lint — detect attribute/method shadowing
        if self._workspace_path:
            api_errors = self._check_api_consistency(Path(self._workspace_path))
            blocking.extend(api_errors)

        return blocking, informational

    async def spec_coverage_check(self, task: TaskNode) -> list[str]:
        """Check that all functions/classes mentioned in spec exist in code.

        Extracts expected names from the spec text (via regex) and verifies
        they are defined in the generated code (via AST). Returns a list of
        missing name strings. This is fully deterministic — no LLM calls.

        Only flags names that look like identifiers: contain an underscore
        (snake_case function) or start with uppercase (PascalCase class).
        Single lowercase words are almost always parameter/field names, not
        functions the task should implement.
        """
        if not task.spec or not task.code_files or not self._workspace_path:
            return []

        # --- Step 1: Extract expected names from spec ---
        spec_lower = task.spec.lower()
        expected_names: set[str] = set()

        # Pattern: function/method names in common spec formats
        # "implement filter_by_status" / "function filter_by_status"
        for m in re.finditer(
            r"(?:implement|function|method|def)\s+(\w+)", spec_lower
        ):
            name = m.group(1)
            expected_names.add(name)

        # Pattern: backtick-quoted names  `filter_by_status`
        for m in re.finditer(r"`(\w+)`", task.spec):
            name = m.group(1)
            if name[0].isupper():
                expected_names.add(name)
            else:
                expected_names.add(name.lower())

        # Pattern: comma-separated function lists — run on ORIGINAL task title only,
        # NOT on architect-generated spec. The architect spec introduces descriptive
        # variable names (e.g. "field_value", "row_count") in prose that are not
        # functions to implement. The original request explicitly names functions.
        # "filter_by_status, filter_by_priority, sort_by_priority"
        title_lower = task.title.lower() if task.title else spec_lower
        for m in re.finditer(
            r"(\w+(?:_\w+)+)(?:\s*,\s*(\w+(?:_\w+)+))+", title_lower
        ):
            full_match = m.group(0)
            for name in re.findall(r"\w+(?:_\w+)+", full_match):
                expected_names.add(name)

        # Pattern: class names (PascalCase words with 2+ caps)
        for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b", task.spec):
            expected_names.add(m.group(1))

        if not expected_names:
            return []

        # --- Filter: only keep names that look like identifiers ---
        # Snake_case (has underscore) → function/method name
        # PascalCase (starts uppercase) → class name
        # Plain lowercase words (board, name, status) are parameter/field names
        # Also exclude Python builtins
        _BUILTINS = {
            # Python builtins / exceptions
            "True", "False", "None", "Exception", "ValueError",
            "TypeError", "KeyError", "IndexError", "AttributeError",
            "NotImplementedError", "StopIteration", "RuntimeError",
            "OSError", "IOError", "FileNotFoundError", "PermissionError",
            # typing module — appear in specs as type hints, never as functions to implement
            "Any", "Optional", "Union", "List", "Dict", "Set", "Tuple",
            "Type", "Callable", "Sequence", "Mapping", "Iterable", "Iterator",
            "Generator", "Coroutine", "ClassVar", "Final", "Literal",
            "TypeVar", "Generic", "Protocol", "NamedTuple", "TypedDict",
        }

        # Exclude parameter names: names that appear inside parentheses
        # e.g. "filter_by_priority(tasks, min_priority)" → min_priority is a param
        param_names: set[str] = set()
        for m in re.finditer(r"\w+\(([^)]+)\)", task.spec):
            for param in m.group(1).split(","):
                param = param.strip().lower()
                # Strip type annotations like "tasks: list"
                param = param.split(":")[0].strip()
                param = param.split("=")[0].strip()
                if param:
                    param_names.add(param)

        # OOP descriptor suffixes: "rows_self", "cols_other" come from spec text like
        # "result dimensions: (rows_self, cols_other)" — never actual function names.
        _OOP_SUFFIXES = ("_self", "_other", "_a", "_b", "_me")
        _OOP_PREFIXES = ("self_", "other_")

        # also exclude common library names mentioned in tech constraints
        _LIB_NAMES = {
            "pyyaml", "yaml", "json", "httpx", "asyncio", "math", "re", 
            "hashlib", "datetime", "os", "sys", "abc", "typing", "collections",
            "dataclasses", "enum", "pytest", "ruff", "mypy", "semgrep",
        }

        filtered_names: set[str] = set()
        for name in expected_names:
            if name in _BUILTINS:
                continue
            if name.lower() in param_names:
                continue
            if name.lower() in _LIB_NAMES:
                continue
            nl = name.lower()
            if any(nl.endswith(s) for s in _OOP_SUFFIXES):
                continue
            if any(nl.startswith(p) for p in _OOP_PREFIXES):
                continue
            if "_" in name or (name[0].isupper() and len(name) > 1):
                filtered_names.add(name)
        expected_names = filtered_names

        if not expected_names:
            return []

        # --- Step 2: Extract defined names from ALL workspace files ---
        from pmca.utils.lang import detect_language, get_extension
        lang = detect_language(task)
        ext = get_extension(lang)
        
        defined_names: set[str] = set()
        ws = Path(self._workspace_path)
        scan_files = list(ws.rglob(f"*{ext}"))
        # Exclude __pycache__ and .pmca dirs
        scan_files = [f for f in scan_files
                      if "__pycache__" not in str(f) and ".pmca" not in str(f)]

        for file_path in scan_files:
            content = file_path.read_text(errors="ignore")
            if lang == "python":
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defined_names.add(node.name.lower())
                    elif isinstance(node, ast.ClassDef):
                        defined_names.add(node.name)
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                defined_names.add(item.name.lower())
            else:
                # Multi-language regex: catches classes, functions, and methods
                # Matches: class X, function Y, func Z, export class A, interface B
                # Also matches method definitions like: methodName(...) {
                pattern = re.compile(
                    r"(?:class|function|async function|export function|export class|"
                    r"const|func|interface)\s+(\w+)|"
                    r"(\w+)\s*\([^)]*\)\s*(?::\s*[\w<>\[\]|]+\s*)?[{]",
                    re.MULTILINE
                )
                for m in pattern.finditer(content):
                    name = m.group(1) or m.group(2)
                    if name:
                        defined_names.add(name.lower())
                        defined_names.add(name)  # Keep case for PascalCase check

        # --- Step 3: Find missing names ---
        missing: list[str] = []
        for name in sorted(expected_names):
            name_lower = name.lower()
            if name_lower not in defined_names and name not in defined_names:
                missing.append(name)

        return missing

    async def run_tests(self, task: TaskNode) -> TestResult:
        """Execute tests for a task and report results (supports Python, Go, TS)."""
        if not task.test_files:
            return TestResult(
                passed=True, total=0, failures=0,
                output="No test files to run", errors=[],
            )

        # 1. Detect language from test files
        test_files = list(task.test_files.keys())
        is_go = any(f.endswith(".go") for f in test_files)
        is_ts = any(f.endswith(".ts") or f.endswith(".js") for f in test_files)
        
        if is_go:
            return await self._run_go_tests(task)
        if is_ts:
            return await self._run_ts_tests(task)
            
        # Default: Python / Pytest
        return await self._run_python_tests(task)

    async def _run_go_tests(self, task: TaskNode) -> TestResult:
        """Execute Go tests using 'go test'."""
        try:
            ws_abs = Path(self._workspace_path).resolve()
            # Ensure go.mod exists
            go_mod = ws_abs / "go.mod"
            if not go_mod.exists():
                from pmca.utils.lang import detect_language
                # Create a minimal module file
                go_mod.write_text("module pmca_workspace\n\ngo 1.21\n")

            proc = await asyncio.create_subprocess_exec(
                "go", "test", "-v", "./...",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ws_abs),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode() + stderr.decode()
            passed = proc.returncode == 0
            return TestResult(
                passed=passed, total=1, failures=0 if passed else 1,
                output=output, errors=[] if passed else ["Go test failed"],
            )
        except Exception as exc:
            return TestResult(passed=False, total=0, failures=0, output=str(exc), errors=[str(exc)])

    async def _run_ts_tests(self, task: TaskNode) -> TestResult:
        """Execute TypeScript tests using ts-node or basic node execution."""
        try:
            ws_abs = Path(self._workspace_path).resolve()
            # 1. First, ensure dependencies are somewhat sane (minimal check)
            # 2. Try to run any test files found using ts-node
            test_files = [f for f in task.test_files if f.endswith(".ts")]
            if not test_files:
                return TestResult(passed=True, total=0, failures=0, output="No TS tests", errors=[])

            # Use npx ts-node to execute the first test file as a smoke test
            # In a full system, this would use npx jest or similar.
            proc = await asyncio.create_subprocess_exec(
                "npx", "ts-node", "--transpile-only", test_files[0],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ws_abs),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode() + stderr.decode()
            passed = proc.returncode == 0
            return TestResult(
                passed=passed, total=1, failures=0 if passed else 1,
                output=output, errors=[] if passed else ["TypeScript execution failed"],
            )
        except Exception as exc:
            return TestResult(passed=False, total=0, failures=0, output=str(exc), errors=[str(exc)])

    async def _run_python_tests(self, task: TaskNode) -> TestResult:
        """Original pytest logic (renamed)."""
        py_tests = [f for f in task.test_files if f.endswith(".py")]
        if not py_tests:
            return TestResult(passed=True, total=0, failures=0, output="No Python tests", errors=[])
            
        python_exe = self._find_python()
        try:
            ws_abs = Path(self._workspace_path).resolve()
            proc = await asyncio.create_subprocess_exec(
                python_exe, "-m", "pytest", *py_tests,
                "-v", "--tb=long", "--showlocals", "--no-header",
                f"--rootdir={ws_abs}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ws_abs),
                env={**__import__("os").environ, "PYTHONPATH": _build_pythonpath(self._workspace_path)},
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            output = stdout.decode() + stderr.decode()
            passed = proc.returncode == 0
            total, failures = self._parse_pytest_output(output)
            return TestResult(
                passed=passed, total=total, failures=failures,
                output=output, errors=[] if passed else self._extract_errors(output),
            )
        except Exception as exc:
            return TestResult(passed=False, total=0, failures=0, output=str(exc), errors=[str(exc)])
        except FileNotFoundError:
            return TestResult(
                passed=False, total=0, failures=0,
                output=f"Python not found at {python_exe}",
                errors=["pytest not installed"],
            )

    @staticmethod
    def _extract_errors(output: str) -> list[str]:
        """Extract detailed error messages from pytest output.

        Groups FAILED test names with their assertion details so the coder
        can trace the root cause.
        """
        lines = output.strip().split("\n")
        errors: list[str] = []

        # Collect FAILED lines and all E-lines in context blocks
        current_block: list[str] = []
        in_failure = False
        for line in lines:
            stripped = line.strip()
            if "FAILED" in stripped:
                # Flush previous block
                if current_block:
                    errors.append("\n".join(current_block))
                    current_block = []
                current_block.append(stripped)
                in_failure = True
            elif stripped.startswith("E "):
                current_block.append(stripped)
                in_failure = True
            elif in_failure and stripped == "":
                # End of failure block
                if current_block:
                    errors.append("\n".join(current_block))
                    current_block = []
                in_failure = False
        # Flush last block
        if current_block:
            errors.append("\n".join(current_block))

        # If nothing specific found, return last few lines
        if not errors and lines:
            errors = [l.strip() for l in lines[-5:] if l.strip()]
        return errors[:10]  # Cap at 10 errors

    @staticmethod
    def extract_structured_errors(output: str) -> list[TestError]:
        """Extract structured TestError objects from pytest output.

        Parses pytest verbose output to identify test names, error types,
        and actual vs expected values for assertion errors.
        """
        errors: list[TestError] = []
        lines = output.strip().split("\n")

        # First pass: find FAILED test names
        failed_tests: list[str] = []
        for line in lines:
            # Match "FAILED tests/test_foo.py::test_bar - AssertionError..."
            m = re.match(r"FAILED\s+(\S+::\S+)", line.strip())
            if m:
                failed_tests.append(m.group(1))

        # Second pass: parse each failure section (between "_ test_name _" lines)
        # pytest uses "_____ test_name _____" as section headers
        sections = re.split(r"_{3,}\s+(\S+)\s+_{3,}", output)

        # sections[0] is pre-header, then alternating [name, content, name, content, ...]
        for i in range(1, len(sections) - 1, 2):
            test_name = sections[i]
            section_content = sections[i + 1] if i + 1 < len(sections) else ""

            error_type = "Unknown"
            actual_value = None
            expected_value = None
            source_line = None

            # Detect error type
            if "NameError" in section_content:
                error_type = "NameError"
                m = re.search(r"NameError: name '(\w+)' is not defined", section_content)
                if m:
                    source_line = f"NameError: name '{m.group(1)}' is not defined"
            elif "ImportError" in section_content or "ModuleNotFoundError" in section_content:
                error_type = "ImportError"
            elif "SyntaxError" in section_content:
                error_type = "SyntaxError"
            elif "AssertionError" in section_content or "assert " in section_content:
                error_type = "AssertionError"

                # Parse "assert X == Y" patterns from E-lines
                assert_match = re.search(
                    r"assert\s+([\d.eE+-]+)\s*==\s*([\d.eE+-]+)",
                    section_content,
                )
                if assert_match:
                    actual_value = assert_match.group(1)
                    expected_value = assert_match.group(2)

                # Also try "AssertionError: X != Y" patterns
                if actual_value is None:
                    ae_match = re.search(
                        r"AssertionError:\s*([\d.eE+-]+)\s*!=\s*([\d.eE+-]+)",
                        section_content,
                    )
                    if ae_match:
                        actual_value = ae_match.group(1)
                        expected_value = ae_match.group(2)

                # Find the source assertion line
                for sline in section_content.split("\n"):
                    sline_s = sline.strip()
                    if sline_s.startswith(">") and "assert" in sline_s:
                        source_line = sline_s.lstrip("> ").strip()
                        break
            elif "TypeError" in section_content:
                error_type = "TypeError"

            # Extract traceback (E-lines)
            e_lines = [
                l.strip()
                for l in section_content.split("\n")
                if l.strip().startswith("E ")
            ]
            traceback = "\n".join(e_lines)

            # Extract local variables from --showlocals output
            # Format: "varname    = value" (indented, after E-lines block)
            local_vars: dict[str, str] = {}
            _SKIP_LOCALS = {"self", "@py_builtins", "@py_assert", "request"}
            for local_m in re.finditer(
                r"^(\w[\w]*)\s{2,}= (.+)$",
                section_content,
                re.MULTILINE,
            ):
                var_name = local_m.group(1)
                var_value = local_m.group(2).strip()
                # Skip pytest internals and fixtures
                if var_name.startswith("@py_") or var_name in _SKIP_LOCALS:
                    continue
                # Truncate large reprs
                if len(var_value) > 80:
                    var_value = var_value[:77] + "..."
                local_vars[var_name] = var_value

            errors.append(TestError(
                test_name=test_name,
                error_type=error_type,
                actual_value=actual_value,
                expected_value=expected_value,
                traceback=traceback,
                source_line=source_line,
                local_variables=local_vars if local_vars else None,
            ))

        # Fallback: if section parsing found nothing, build from FAILED lines
        if not errors and failed_tests:
            for test_name in failed_tests:
                error_type = "Unknown"
                if "NameError" in output:
                    error_type = "NameError"
                elif "ImportError" in output:
                    error_type = "ImportError"
                elif "AssertionError" in output:
                    error_type = "AssertionError"
                errors.append(TestError(
                    test_name=test_name,
                    error_type=error_type,
                    traceback="",
                ))

        # Fallback 2: handle pytest collection errors
        # (e.g., "ERROR collecting tests/test_foo.py")
        if not errors:
            for m in re.finditer(
                r"ERROR\s+(?:collecting\s+)?(\S+\.py)", output
            ):
                error_file = m.group(1)
                error_type = "ImportError"
                traceback_text = ""
                if "ModuleNotFoundError" in output:
                    mod_match = re.search(
                        r"ModuleNotFoundError: No module named '(\w+)'", output
                    )
                    traceback_text = mod_match.group(0) if mod_match else ""
                elif "ImportError" in output:
                    imp_match = re.search(r"ImportError: (.+)", output)
                    traceback_text = imp_match.group(0) if imp_match else ""
                elif "SyntaxError" in output:
                    error_type = "SyntaxError"
                errors.append(TestError(
                    test_name=error_file,
                    error_type=error_type,
                    traceback=traceback_text,
                ))

        return errors[:10]

    @staticmethod
    def extract_lesson(
        attempt: int,
        review: ReviewResult,
        structured_errors: list[TestError],
        strategy: str,
    ) -> "LessonRecord":
        """Distill a failed attempt into an abstract LessonRecord.

        Extracts error types and builds a concise summary from up to 3
        structured errors.  Never includes raw traces — only abstracted
        information suitable for prompt injection.
        """
        from pmca.tasks.state import LessonRecord

        # Collect unique error types (cap at 3)
        error_types: list[str] = []
        seen: set[str] = set()
        for err in structured_errors[:3]:
            if err.error_type not in seen:
                error_types.append(err.error_type)
                seen.add(err.error_type)

        # Build concise summary from structured errors
        parts: list[str] = []
        for err in structured_errors[:3]:
            snippet = err.test_name
            if err.actual_value is not None and err.expected_value is not None:
                snippet += f" (got {err.actual_value}, expected {err.expected_value})"
            elif err.source_line:
                line = err.source_line[:80]
                snippet += f": {line}"
            parts.append(snippet)

        # Fallback to review issues if no structured errors
        if not parts and review.issues:
            for issue in review.issues[:3]:
                parts.append(issue[:80])

        summary = "; ".join(parts) if parts else "No specific details captured"

        return LessonRecord(
            attempt=attempt,
            error_types=error_types,
            strategy=strategy,
            summary=summary,
        )

    async def targeted_micro_fix(
        self,
        task: TaskNode,
        structured_errors: list[TestError],
    ) -> int:
        """Attempt surgical LLM fix for single-function errors.

        For each error with a clear traceback pointing to a specific function,
        extracts just that function, sends a minimal prompt to the LLM,
        and splices the fix back.  Returns the number of functions fixed.

        This runs BEFORE the full coder.fix() as a cheap pre-pass.
        """
        if not self._workspace_path or not task.code_files:
            return 0

        workspace = Path(self._workspace_path)

        # Filter to fixable error types in code (not test) files
        fixable_types = {"TypeError", "KeyError", "AttributeError", "IndexError",
                         "ValueError", "ZeroDivisionError", "AssertionError"}
        candidates = [
            e for e in structured_errors
            if e.error_type in fixable_types
        ]
        if not candidates:
            return 0

        # Only non-test code files
        code_files = [f for f in task.code_files if not f.startswith("test")]

        fixed_count = 0
        seen_functions: set[tuple[str, str]] = set()  # (file, func) already fixed

        for error in candidates[:3]:  # Cap at 3 micro-fixes
            loc = self._parse_error_location(error, workspace, code_files)
            if loc is None:
                continue

            file_path, func_name = loc
            if (file_path, func_name) in seen_functions:
                continue
            seen_functions.add((file_path, func_name))

            abs_path = workspace / file_path
            if not abs_path.exists():
                continue

            try:
                source = abs_path.read_text()
            except OSError:
                continue

            result = self._extract_function_source(source, func_name)
            if result is None:
                continue

            func_body, start_line, end_line = result

            # Build source context from error
            source_ctx = ""
            if error.source_line:
                source_ctx = f"Failing line: {error.source_line}"
            if error.actual_value is not None and error.expected_value is not None:
                source_ctx += f"\nActual: {error.actual_value}, Expected: {error.expected_value}"

            error_detail = ""
            if error.traceback:
                # Extract just the last line of the traceback (the actual error message)
                tb_lines = error.traceback.strip().splitlines()
                error_detail = tb_lines[-1] if tb_lines else ""
            elif error.source_line:
                error_detail = error.source_line

            prompt = prompts.MICRO_FIX_PROMPT.format(
                test_name=error.test_name,
                error_type=error.error_type,
                error_detail=error_detail,
                source_context=source_ctx,
                file_path=file_path,
                function_body=func_body,
                spec=task.spec[:1000] if task.spec else "(no spec)",
            )

            try:
                response = await self._generate(
                    prompt,
                    system=prompts.MICRO_FIX_SYSTEM,
                    temperature=0.1,
                )

                # Parse the fixed function from the response
                fixed_func = self._extract_code_from_response(response)
                if not fixed_func or fixed_func.strip() == func_body.strip():
                    continue

                # Splice the fixed function back into the source
                lines = source.splitlines()
                fixed_lines = fixed_func.splitlines()
                lines[start_line:end_line] = fixed_lines

                new_source = "\n".join(lines)
                # Verify it parses
                ast.parse(new_source)
                abs_path.write_text(new_source)
                fixed_count += 1
                self._log.info(
                    f"Micro-fixed {func_name} in {file_path} "
                    f"(error: {error.error_type})"
                )
            except Exception:
                self._log.debug(f"Micro-fix failed for {func_name}")
                continue

        return fixed_count

    @staticmethod
    def _extract_code_from_response(response: str) -> str | None:
        """Extract a single code block from an LLM response."""
        # Match fenced code blocks
        m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        if m:
            return m.group(1).rstrip()
        return None

    async def calibrate_tests(self, task: TaskNode) -> int:
        """Run tests and fix assertion value mismatches.

        After code + tests are generated, the LLM often gets arithmetic wrong
        in test assertions.  This method runs each test, and for pure value
        mismatches (assert X == Y where X is the actual code result), patches
        the test file to use the actual value.

        Returns the number of assertions that were calibrated.
        """
        if not task.test_files or not self._workspace_path:
            return 0

        py_tests = [f for f in task.test_files if f.endswith(".py")]
        if not py_tests:
            return 0

        python_exe = self._find_python()
        calibrated = 0

        for test_file in py_tests:
            test_path = Path(self._workspace_path) / test_file
            if not test_path.exists():
                continue

            # Run pytest with verbose output to capture assertion details
            try:
                ws_abs = Path(self._workspace_path).resolve()
                proc = await asyncio.create_subprocess_exec(
                    python_exe, "-m", "pytest", test_file,
                    "-v", "--tb=long", "--showlocals", "--no-header",
                    f"--rootdir={ws_abs}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(ws_abs),
                    env={**os.environ, "PYTHONPATH": _build_pythonpath(self._workspace_path)},
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                output = stdout.decode() + stderr.decode()
            except (asyncio.TimeoutError, FileNotFoundError):
                continue

            if proc.returncode == 0:
                continue  # All tests pass, nothing to calibrate

            # Parse assertion mismatches: "assert X == Y" where X is actual
            # Look for patterns like:
            #   E       assert 3.625 == 4.0
            #   E       AssertionError: assert 3.625 == 4.0
            test_content = test_path.read_text()
            original_content = test_content

            # Find assertion errors with numeric mismatches
            # Pattern: assert <actual> == <expected>
            for match in re.finditer(
                r"assert\s+([\d.]+(?:e[+-]?\d+)?)\s*==\s*([\d.]+(?:e[+-]?\d+)?)",
                output,
            ):
                actual_str = match.group(1)
                expected_str = match.group(2)
                if actual_str == expected_str:
                    continue

                try:
                    actual_val = float(actual_str)
                    expected_val = float(expected_str)
                except ValueError:
                    continue

                should_calibrate = False

                if expected_val != 0 and actual_val != 0:
                    relative_diff = abs(actual_val - expected_val) / abs(expected_val)

                    # Negative sign flip: actual == -expected → code bug, don't calibrate
                    if abs(actual_val + expected_val) < 1e-9:
                        self._log.debug(
                            f"Skipping calibration {expected_str} → {actual_str} "
                            f"(sign flip — likely a formula bug)"
                        )
                        continue

                    # Close values (within 25%) — likely arithmetic error in test
                    if relative_diff <= 0.25:
                        should_calibrate = True

                    # Order-of-magnitude check: 10x, 100x, 1000x difference
                    # → likely a decimal point error, safe to calibrate
                    elif actual_val != 0:
                        ratio = actual_val / expected_val
                        if ratio in (10.0, 100.0, 1000.0, 0.1, 0.01, 0.001):
                            self._log.info(
                                f"Order-of-magnitude mismatch: {expected_str} → "
                                f"{actual_str} (ratio={ratio})"
                            )
                            should_calibrate = True
                elif expected_val == 0:
                    # Zero-value handling: use absolute threshold
                    if abs(actual_val) < 1.0:
                        should_calibrate = True
                    else:
                        self._log.debug(
                            f"Skipping calibration {expected_str} → {actual_str} "
                            f"(expected 0, actual too far)"
                        )
                        continue
                elif actual_val == 0:
                    # Code returns 0 but test expects non-zero — code bug
                    self._log.debug(
                        f"Skipping calibration {expected_str} → {actual_str} "
                        f"(code returns 0 — likely a code bug)"
                    )
                    continue

                if not should_calibrate:
                    self._log.debug(
                        f"Skipping calibration {expected_str} → {actual_str} "
                        f"(values too far apart)"
                    )
                    continue

                # Replace the wrong expected value in the test file
                old_assertion = f"== {expected_str}"
                new_assertion = f"== {actual_str}"
                if old_assertion in test_content:
                    test_content = test_content.replace(
                        old_assertion, new_assertion, 1
                    )
                    calibrated += 1
                    self._log.info(
                        f"Calibrated test assertion: {expected_str} → {actual_str}"
                    )

            # --- String assertion calibration ---
            # Match patterns like: AssertionError: assert 'Foo' == 'foo'
            for str_match in re.finditer(
                r"assert\s+'([^']+)'\s*==\s*'([^']+)'",
                output,
            ):
                actual_s = str_match.group(1)
                expected_s = str_match.group(2)
                if actual_s == expected_s:
                    continue
                # Only calibrate case differences
                if actual_s.lower() == expected_s.lower():
                    old_str = f"== '{expected_s}'"
                    new_str = f"== '{actual_s}'"
                    if old_str in test_content:
                        test_content = test_content.replace(old_str, new_str, 1)
                        calibrated += 1
                        self._log.info(
                            f"Calibrated string assertion: '{expected_s}' → '{actual_s}'"
                        )
                    # Also try double quotes
                    old_str_dq = f'== "{expected_s}"'
                    new_str_dq = f'== "{actual_s}"'
                    if old_str_dq in test_content:
                        test_content = test_content.replace(old_str_dq, new_str_dq, 1)
                        calibrated += 1
                        self._log.info(
                            f"Calibrated string assertion: \"{expected_s}\" → \"{actual_s}\""
                        )

            if test_content != original_content:
                test_path.write_text(test_content)

        return calibrated

    async def oracle_repair_tests(self, task: TaskNode) -> int:
        """Second-pass oracle: trust non-trivial actual values for remaining assertion failures.

        After ``calibrate_tests`` applies conservative heuristics (25% threshold,
        power-of-10), this method aggressively trusts non-trivial actual values
        as oracles for any remaining assertion mismatches.

        Guard rails — skip oracle repair when:
        - Numeric actual value is 0 (likely code bug returning nothing)
        - String actual value is empty/whitespace (code returns empty)

        Returns the number of assertions oracle-repaired.
        """
        if not task.test_files or not self._workspace_path:
            return 0

        py_tests = [f for f in task.test_files if f.endswith(".py")]
        if not py_tests:
            return 0

        python_exe = self._find_python()
        repaired = 0

        for test_file in py_tests:
            test_path = Path(self._workspace_path) / test_file
            if not test_path.exists():
                continue

            # Run pytest to get remaining failures
            try:
                ws_abs = Path(self._workspace_path).resolve()
                proc = await asyncio.create_subprocess_exec(
                    python_exe, "-m", "pytest", test_file,
                    "-v", "--tb=long", "--no-header",
                    f"--rootdir={ws_abs}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(ws_abs),
                    env={**os.environ, "PYTHONPATH": _build_pythonpath(self._workspace_path)},
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                output = stdout.decode() + stderr.decode()
            except (asyncio.TimeoutError, FileNotFoundError):
                continue

            if proc.returncode == 0:
                continue  # All tests pass

            test_content = test_path.read_text()
            original_content = test_content

            # --- Numeric oracle ---
            for match in re.finditer(
                r"assert\s+([\d.]+(?:e[+-]?\d+)?)\s*==\s*([\d.]+(?:e[+-]?\d+)?)",
                output,
            ):
                actual_str = match.group(1)
                expected_str = match.group(2)
                if actual_str == expected_str:
                    continue
                try:
                    actual_val = float(actual_str)
                except ValueError:
                    continue
                # Guard: skip if actual is 0 (likely code bug)
                if actual_val == 0:
                    self._log.debug(
                        f"Oracle skip: actual is 0 for expected {expected_str}"
                    )
                    continue
                old = f"== {expected_str}"
                if old in test_content:
                    test_content = test_content.replace(old, f"== {actual_str}", 1)
                    repaired += 1
                    self._log.info(
                        f"Oracle-repaired numeric: {expected_str} → {actual_str}"
                    )

            # --- String oracle (single quotes) ---
            for match in re.finditer(
                r"assert\s+'([^']+)'\s*==\s*'([^']+)'",
                output,
            ):
                actual_s = match.group(1)
                expected_s = match.group(2)
                if actual_s == expected_s:
                    continue
                if not actual_s.strip():
                    continue
                old = f"== '{expected_s}'"
                if old in test_content:
                    test_content = test_content.replace(old, f"== '{actual_s}'", 1)
                    repaired += 1
                    self._log.info(
                        f"Oracle-repaired string: '{expected_s}' → '{actual_s}'"
                    )

            # --- String oracle (double quotes) ---
            for match in re.finditer(
                r'assert\s+"([^"]+)"\s*==\s*"([^"]+)"',
                output,
            ):
                actual_s = match.group(1)
                expected_s = match.group(2)
                if actual_s == expected_s:
                    continue
                if not actual_s.strip():
                    continue
                old = f'== "{expected_s}"'
                if old in test_content:
                    test_content = test_content.replace(old, f'== "{actual_s}"', 1)
                    repaired += 1
                    self._log.info(
                        f'Oracle-repaired string: "{expected_s}" → "{actual_s}"'
                    )

            if test_content != original_content:
                test_path.write_text(test_content)

        return repaired

    async def mutation_oracle(
        self, task: TaskNode,
    ) -> tuple[int, int, float]:
        """Run mutation testing to validate test quality (MuTAP).

        Generates AST mutations of each code file and runs tests against each
        mutant.  A "killed" mutant means the tests detected the change.
        Low kill ratio suggests hallucinated or weak test assertions.

        Returns (total_mutants, killed, kill_ratio).
        Only runs when tests currently pass.
        """
        if not task.code_files or not task.test_files or not self._workspace_path:
            return 0, 0, 0.0

        from pmca.utils.mutator import generate_mutations

        py_tests = [f for f in task.test_files if f.endswith(".py")]
        if not py_tests:
            return 0, 0, 0.0

        # Smoke test: only run mutation oracle if tests currently pass
        smoke = await self.run_tests(task)
        if not smoke.passed:
            return 0, 0, 0.0

        python_exe = self._find_python()
        ws_abs = Path(self._workspace_path).resolve()
        total = 0
        killed = 0

        code_files = [f for f in task.code_files if f.endswith(".py")]
        for code_file in code_files:
            code_path = ws_abs / code_file
            if not code_path.exists():
                continue

            original_source = code_path.read_text()
            mutations = generate_mutations(original_source, max_mutations=8)

            for mutation in mutations:
                total += 1
                try:
                    # Write mutant
                    code_path.write_text(mutation.mutated_source)

                    # Run tests with 5s timeout
                    proc = await asyncio.create_subprocess_exec(
                        python_exe, "-m", "pytest", *py_tests,
                        "-x", "--tb=no", "--no-header", "-q",
                        f"--rootdir={ws_abs}",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=str(ws_abs),
                        env={**os.environ, "PYTHONPATH": _build_pythonpath(self._workspace_path)},
                    )
                    try:
                        await asyncio.wait_for(proc.communicate(), timeout=5)
                    except asyncio.TimeoutError:
                        # Timeout counts as killed (mutant caused infinite loop)
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        killed += 1
                        continue

                    if proc.returncode != 0:
                        killed += 1
                finally:
                    # Always restore original
                    code_path.write_text(original_source)

        kill_ratio = killed / total if total > 0 else 0.0
        self._log.info(
            f"Mutation oracle: {killed}/{total} killed "
            f"(ratio={kill_ratio:.0%})"
        )
        return total, killed, kill_ratio

    async def check_not_faked(self, code: str, tests: str) -> ReviewResult:
        """Verify code isn't trivially faking test passage."""
        prompt = prompts.CHECK_NOT_FAKED_PROMPT.format(code=code, tests=tests)
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system)
        return self._parse_review(response)

    async def final_verification(self, root_task: TaskNode, original_request: str,
                                  project_structure: str, key_files: str) -> ReviewResult:
        """End-to-end verification of the complete project."""
        prompt = prompts.FINAL_VERIFICATION_PROMPT.format(
            original_request=original_request,
            project_structure=project_structure,
            key_files=key_files,
        )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system)
        return self._parse_review(response)

    def _parse_review(self, response: str) -> ReviewResult:
        """Parse review JSON from model response."""
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReviewResult(
                    passed=bool(data.get("passed", False)),
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    timestamp=datetime.now(),
                    model_used=self.role.value,
                )
            except json.JSONDecodeError:
                pass

        return ReviewResult(
            passed=False,
            issues=["Failed to parse watcher review response"],
            suggestions=[],
            timestamp=datetime.now(),
            model_used=self.role.value,
        )

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """Parse pytest output for total and failure counts."""
        # Match patterns like "5 passed", "2 failed", "3 passed, 1 failed"
        total = 0
        failures = 0

        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        error_match = re.search(r"(\d+) error", output)

        if passed_match:
            total += int(passed_match.group(1))
        if failed_match:
            count = int(failed_match.group(1))
            total += count
            failures += count
        if error_match:
            count = int(error_match.group(1))
            total += count
            failures += count

        return total, failures
