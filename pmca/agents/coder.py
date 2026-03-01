"""Coder agent — code implementation from specifications."""

from __future__ import annotations

import hashlib
import json
import re
import warnings

from pmca.agents.base import BaseAgent
from pmca.models.config import AgentRole
from pmca.prompts import coder as prompts
from pmca.tasks.state import CodeFile
from pmca.tasks.tree import TaskNode

_TARGET_FILE_RE = re.compile(r"TARGET_FILE:\s*(\S+)")


class CoderAgent(BaseAgent):
    role = AgentRole.CODER

    def __init__(self, model_manager, project_mode: bool = False) -> None:
        super().__init__(model_manager)
        self._project_mode = project_mode
        # Track code hashes per task to detect duplicate fix attempts
        self._fix_hashes: dict[str, set[str]] = {}

    def _suggested_paths(self, task: TaskNode) -> tuple[str, str]:
        """Derive suggested file paths, preferring TARGET_FILE from spec."""
        m = _TARGET_FILE_RE.search(task.spec)
        if m:
            from pmca.utils.assembler import _normalize_target_path
            target = _normalize_target_path(m.group(1))
            # e.g. "src/models.py" → stem = "models"
            from pathlib import Path
            stem = Path(target).stem
            test_path = f"tests/test_{stem}.py"
            return target, test_path
        safe_name = self._derive_short_name(task.title)
        return f"src/{safe_name}.py", f"tests/test_{safe_name}.py"

    async def extract_spec_literals(self, spec: str) -> str:
        """Extract string literal groups from a spec using structured output.

        Returns a formatted text section listing each parameter and its valid
        values, ready to inject into the coding prompt.  Returns empty string
        if no literal groups are found.
        """
        prompt = prompts.EXTRACT_LITERALS_PROMPT.format(spec=spec)
        try:
            raw = await self._generate_structured(
                prompt,
                schema=prompts.EXTRACT_LITERALS_SCHEMA,
                temperature=0.0,
            )
            data = json.loads(raw)
            groups = data.get("groups", [])
            if not groups:
                return ""
            lines: list[str] = []
            for g in groups:
                param = g.get("parameter", "")
                values = g.get("values", [])
                if param and values:
                    quoted = ", ".join(f"'{v}'" for v in values)
                    lines.append(f"- `{param}` accepts: {quoted}")
            if not lines:
                return ""
            literals_text = "\n".join(lines)
            self._log.info(f"Extracted spec literals: {len(groups)} group(s)")
            return prompts.SPEC_LITERALS_SECTION.format(literals_text=literals_text)
        except Exception as exc:
            self._log.warning(f"Spec literal extraction failed (non-blocking): {exc}")
            return ""

    async def implement(self, task: TaskNode, context: str = "", difficulty: str = "complex") -> list[CodeFile]:
        """Generate code for a leaf task based on its spec."""
        suggested_path, suggested_test_path = self._suggested_paths(task)

        system = prompts.SYSTEM_PROMPT
        if self._project_mode:
            system += prompts.PROJECT_IMPORT_RULES

        # Phase 1: Extract string literals from original request (complex tasks only)
        # Use task.title (original request) not task.spec (architect's rewrite)
        # because the architect may paraphrase or restructure the literal values
        literals_section = ""
        if difficulty != "simple":
            literals_section = await self.extract_spec_literals(task.title)

        # Phase 2: Generate code with extracted literals injected
        if difficulty == "simple":
            prompt = prompts.IMPLEMENT_SIMPLE_PROMPT.format(
                spec=task.spec,
                context=context,
                suggested_path=suggested_path,
                suggested_test_path=suggested_test_path,
            )
        else:
            prompt = prompts.IMPLEMENT_PROMPT.format(
                spec=task.spec + literals_section,
                context=context,
                suggested_path=suggested_path,
                suggested_test_path=suggested_test_path,
            )
        response = await self._generate(prompt, system=system)
        return self._parse_code_blocks(response)

    async def generate_tests(self, task: TaskNode, context: str = "") -> list[CodeFile]:
        """Generate only test files from specification.

        Deprecated: Use TesterAgent.generate_tests() for 14B-quality tests.
        """
        warnings.warn(
            "CoderAgent.generate_tests() is deprecated. Use TesterAgent.generate_tests() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        suggested_path, suggested_test_path = self._suggested_paths(task)
        from pathlib import Path
        suggested_module = Path(suggested_path).stem
        prompt = prompts.GENERATE_TESTS_PROMPT.format(
            spec=task.spec,
            context=context,
            suggested_module=suggested_module,
            suggested_test_path=suggested_test_path,
        )
        response = await self._generate(prompt, system=prompts.SYSTEM_PROMPT)
        return self._parse_code_blocks(response)

    async def implement_with_tests(
        self, task: TaskNode, context: str = "", tests_content: str = "",
    ) -> list[CodeFile]:
        """Generate implementation that must pass provided tests."""
        suggested_path, _suggested_test_path = self._suggested_paths(task)
        system = prompts.SYSTEM_PROMPT
        if self._project_mode:
            system += prompts.PROJECT_IMPORT_RULES
        prompt = prompts.IMPLEMENT_WITH_TESTS_PROMPT.format(
            spec=task.spec,
            context=context,
            tests=tests_content,
            suggested_path=suggested_path,
        )
        response = await self._generate(prompt, system=system)
        return self._parse_code_blocks(response)

    async def implement_best_of_n(
        self, task: TaskNode, context: str, n: int, test_runner,
        tests_content: str = "",
    ) -> list[CodeFile]:
        """Generate N candidates, return the one passing the most tests."""
        from collections.abc import Callable

        candidates: list[list[CodeFile]] = []
        suggested_path, suggested_test_path = self._suggested_paths(task)
        for i in range(n):
            temp = 0.2 + i * 0.15  # 0.2, 0.35, 0.5, ...
            system = prompts.SYSTEM_PROMPT
            if self._project_mode:
                system += prompts.PROJECT_IMPORT_RULES

            if tests_content:
                prompt = prompts.IMPLEMENT_WITH_TESTS_PROMPT.format(
                    spec=task.spec,
                    context=context,
                    tests=tests_content,
                    suggested_path=suggested_path,
                )
            else:
                prompt = prompts.IMPLEMENT_PROMPT.format(
                    spec=task.spec,
                    context=context,
                    suggested_path=suggested_path,
                    suggested_test_path=suggested_test_path,
                )
            response = await self._generate(prompt, system=system, temperature=temp)
            files = self._parse_code_blocks(response)
            candidates.append(files)

        if not task.test_files and not tests_content:
            return candidates[0]  # No tests to score against

        best, best_score = candidates[0], -1
        for files in candidates:
            result = await test_runner(files)
            score = result.total - result.failures
            if score > best_score:
                best, best_score = files, score
            if result.failures == 0 and result.total > 0:
                break  # Perfect score, no need to try more

        return best

    async def fix(self, task: TaskNode, issues: list[str], file_manager=None, retry_num: int = 0) -> list[CodeFile]:
        """Fix code based on reviewer/watcher feedback.

        Uses alternating strategy: odd retries fix code, even retries fix tests.
        Tracks code hashes to detect and break out of duplicate fix loops.
        """
        # Build code blocks from actual files
        code_blocks_parts: list[str] = []
        all_files = task.code_files + task.test_files
        for path in all_files:
            content = None
            if file_manager:
                try:
                    content = file_manager.read_file(path)
                except FileNotFoundError:
                    pass
            if content:
                code_blocks_parts.append(f"```python\n# filepath: {path}\n{content}\n```")
            else:
                code_blocks_parts.append(f"```python\n# filepath: {path}\n# (file not readable)\n```")

        code_blocks_str = "\n\n".join(code_blocks_parts) if code_blocks_parts else f"```python\n{task.spec}\n```"

        # Check for duplicate code (hash-based dedup)
        code_hash = hashlib.md5(code_blocks_str.encode()).hexdigest()
        task_hashes = self._fix_hashes.setdefault(task.id, set())
        is_duplicate = code_hash in task_hashes
        task_hashes.add(code_hash)

        # Alternating strategy: odd retries fix code, even retries fix tests
        # On even retries (2, 4, ...) or when duplicate detected, try fixing tests instead
        fix_tests_mode = (retry_num % 2 == 0 and retry_num > 0) or is_duplicate

        issues_str = "\n".join(f"- {i}" for i in issues)

        if fix_tests_mode:
            self._log.info(
                f"Retry {retry_num}: using FIX_TESTS strategy"
                + (" (duplicate detected)" if is_duplicate else "")
            )
            prompt = prompts.FIX_TESTS_PROMPT.format(
                code_blocks=code_blocks_str,
                issues=issues_str,
                spec=task.spec,
            )
        else:
            prompt = prompts.FIX_PROMPT.format(
                code_blocks=code_blocks_str,
                issues=issues_str,
                spec=task.spec,
            )

        # Prepend dedup warning if duplicate detected
        if is_duplicate:
            prompt = prompts.DEDUP_PREFIX + prompt

        # Escalate temperature on retries to explore different solutions
        system = prompts.SYSTEM_PROMPT
        if self._project_mode:
            system += prompts.PROJECT_IMPORT_RULES
        # Narrow temperature band: 7B models produce brittle code above 0.5
        temp = min(0.2 + retry_num * 0.1, 0.5)
        response = await self._generate(prompt, system=system, temperature=temp)
        return self._parse_code_blocks(response)

    # _derive_short_name, _parse_code_blocks, _guess_extension inherited from BaseAgent
