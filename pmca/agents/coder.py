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

    def _get_system_prompt(self, task: TaskNode, role: AgentRole | None = None) -> str:
        """Construct system prompt, injecting language-specific skills."""
        from pmca.utils.lang import detect_language
        lang = detect_language(task)
        
        # VOCAL MANDATE: Make it impossible for the model to ignore the target language
        system = f"YOU ARE A {lang.upper()} EXPERT. YOU MUST WRITE CODE ONLY IN {lang.upper()}.\n\n"
        system += prompts.SYSTEM_PROMPT
        
        if lang == "typescript":
            from pmca.prompts import TYPESCRIPT_SKILLS
            if TYPESCRIPT_SKILLS not in system:
                system += "\n" + TYPESCRIPT_SKILLS
        elif lang == "go":
            from pmca.prompts import GO_SKILLS
            if GO_SKILLS not in system:
                system += "\n" + GO_SKILLS
        else:
            # Default Python skills for Qwen 3.5
            try:
                model_cfg = self._model._config.get_model(role or self.role)
                if model_cfg and "qwen3.5" in model_cfg.name.lower():
                    from pmca.prompts import QWEN_PYTHON_SKILLS
                    if QWEN_PYTHON_SKILLS not in system:
                        system += QWEN_PYTHON_SKILLS
                    if prompts.THINKING_PROMPT_PREFIX not in system:
                        system += "\n" + prompts.THINKING_PROMPT_PREFIX
            except Exception:
                pass

        if self._project_mode:
            system += prompts.PROJECT_IMPORT_RULES
        return system

    def _suggested_paths(self, task: TaskNode) -> tuple[str, str]:
        """Suggest file paths for implementation and tests, forcing language extension."""
        from pmca.utils.lang import detect_language, get_extension
        lang = detect_language(task)
        ext = get_extension(lang)

        m = _TARGET_FILE_RE.search(task.spec)
        if m:
            from pmca.utils.assembler import _normalize_target_path
            target = _normalize_target_path(m.group(1))
            from pathlib import Path
            p = Path(target)
            # Force extension if it doesn't match the language
            if p.suffix != ext:
                target = str(p.with_suffix(ext))

            stem = Path(target).stem
            test_path = f"tests/test_{stem}{ext}"
            return target, test_path

        safe_name = self._derive_short_name(task.title)
        return f"src/{safe_name}{ext}", f"tests/test_{safe_name}{ext}"


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

    async def implement(
        self, task: TaskNode, context: str = "", difficulty: str = "complex",
        role_override: AgentRole | None = None, think: bool | None = None,
        spec_literals: bool = True,
    ) -> list[CodeFile]:
        """Generate code for a leaf task based on its spec.

        Args:
            role_override: Use a different model role (e.g. CODER_REASONING
                          for reasoning-heavy tasks via adaptive routing).
            think: Control thinking mode for reasoning models.
        """
        suggested_path, suggested_test_path = self._suggested_paths(task)

        system = self._get_system_prompt(task, role_override)

        # Phase 1: Extract string literals from original request (complex tasks only)
        # Use task.title (original request) not task.spec (architect's rewrite)
        # because the architect may paraphrase or restructure the literal values
        literals_section = ""
        if difficulty != "simple" and spec_literals:
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
        response = await self._generate(prompt, system=system, role=role_override, think=think)
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
            suggested_path=suggested_path,
            suggested_test_path=suggested_test_path,
        )
        system = self._get_system_prompt(task)
        response = await self._generate(prompt, system=system)
        return self._parse_code_blocks(response)


    async def implement_with_tests(
        self, task: TaskNode, context: str = "", tests_content: str = "",
    ) -> list[CodeFile]:
        """Generate implementation that must pass provided tests."""
        suggested_path, _suggested_test_path = self._suggested_paths(task)
        system = self._get_system_prompt(task)
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
        cross_execution: bool = False,
    ) -> list[CodeFile]:
        """Generate N candidates, return the one passing the most tests.

        When *cross_execution* is True and n >= 2, applies ConVerTest-style
        cross-validation: runs the top candidate's code against every other
        candidate's tests to filter out hallucinated assertions, then re-scores.
        """
        candidates: list[list[CodeFile]] = []
        suggested_path, suggested_test_path = self._suggested_paths(task)
        for i in range(n):
            temp = 0.2 + i * 0.15  # 0.2, 0.35, 0.5, ...
            system = self._get_system_prompt(task)

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

        # Phase 1: Self-score each candidate
        scores: list[int] = []
        results_list = []
        best, best_score, best_idx = candidates[0], -1, 0
        for idx, files in enumerate(candidates):
            result = await test_runner(files)
            score = result.total - result.failures
            scores.append(score)
            results_list.append(result)
            if score > best_score:
                best, best_score, best_idx = files, score, idx
            if result.failures == 0 and result.total > 0:
                break  # Perfect score, no need to try more

        # Phase 2: Cross-execution gate (ConVerTest N×1 approach)
        if cross_execution and n >= 2 and best_score > 0 and best_score < results_list[best_idx].total:
            self._log.info(f"Cross-execution: validating top candidate (score {best_score}) against other candidates' tests")

            # Extract code-only and test-only files from each candidate
            def _split_files(files: list[CodeFile]) -> tuple[list[CodeFile], list[CodeFile]]:
                code = [f for f in files if not (f.path.startswith("test") or "/test_" in f.path)]
                tests = [f for f in files if f.path.startswith("test") or "/test_" in f.path]
                return code, tests

            best_code, _ = _split_files(best)

            filtered_tests_count = 0
            for idx, other_files in enumerate(candidates):
                if idx == best_idx:
                    continue
                _, other_tests = _split_files(other_files)
                if not other_tests:
                    continue

                # Run best code + other's tests
                hybrid = best_code + other_tests
                hybrid_result = await test_runner(hybrid)

                if hybrid_result.failures > 0:
                    filtered_tests_count += hybrid_result.failures
                    self._log.info(
                        f"Cross-execution: {hybrid_result.failures} test(s) from "
                        f"candidate {idx} failed on top code → likely hallucinated"
                    )

            if filtered_tests_count > 0:
                self._log.info(
                    f"Cross-execution filtered {filtered_tests_count} likely-hallucinated test(s) total"
                )

        return best

    async def fix(
        self,
        task: TaskNode,
        code_blocks_str: str,
        issues_str: str,
        lessons_str: str = "",
        memory_str: str = "",
        retry_num: int = 0,
        coder_role: AgentRole | None = None,
        think: bool | None = None,
        strategy: str = "FIX_CODE",
    ) -> list[CodeFile]:
        """Fix code based on test failures and reviewer feedback."""
        # Detect if we are stuck (same output)
        code_hash = hashlib.sha256(code_blocks_str.encode()).hexdigest()
        self._fix_hashes.setdefault(task.id, set())
        is_duplicate = code_hash in self._fix_hashes[task.id]
        self._fix_hashes[task.id].add(code_hash)

        if strategy == "FIX_TESTS":
            prompt = prompts.FIX_TESTS_PROMPT.format(
                spec=task.spec,
                code_blocks=code_blocks_str,
                issues=issues_str,
                lessons=lessons_str + memory_str,
            )
        else:
            prompt = prompts.FIX_PROMPT.format(
                code_blocks=code_blocks_str,
                lessons=lessons_str + memory_str,
                issues=issues_str,
                spec=task.spec,
            )

        # Prepend dedup warning if duplicate detected
        if is_duplicate:
            prompt = prompts.DEDUP_PREFIX + prompt

        # Escalate temperature on retries to explore different solutions
        system = self._get_system_prompt(task, coder_role)
        # Narrow temperature band: 7B models produce brittle code above 0.5
        temp = min(0.2 + retry_num * 0.1, 0.5)
        response = await self._generate(prompt, system=system, temperature=temp, role=coder_role, think=think)
        return self._parse_code_blocks(response)
