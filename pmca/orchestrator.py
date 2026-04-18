"""Main cascade orchestrator — manages the design→verify cascade cycle."""

from __future__ import annotations

import datetime
import heapq
import json as _json
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.panel import Panel
from rich.tree import Tree as RichTree

from pmca.agents.architect import ArchitectAgent
from pmca.agents.coder import CoderAgent
from pmca.agents.reviewer import ReviewerAgent
from pmca.agents.tester import TesterAgent
from pmca.agents.watcher import WatcherAgent
from pmca.api.events import CascadeEvent, EventType
from pmca.models.config import AgentRole, Config
from pmca.models.manager import ModelManager
from pmca.tasks.state import LessonRecord, ReviewResult, TaskStatus, TaskType, TestResult
from pmca.tasks.tree import TaskNode, TaskTree
from pmca.utils.assembler import FileAssembler
from pmca.utils.context import ContextManager
from pmca.utils.logger import get_console, get_logger
from pmca.workspace.file_manager import FileManager
from pmca.workspace.git_manager import GitManager

log = get_logger("orchestrator")
console = get_console()

# Regex for metadata parsing in subtask descriptions
_EXPORTS_RE = re.compile(r"EXPORTS:\s*(.+)")
_DEPENDS_RE = re.compile(r"DEPENDS_ON:\s*(.+)", re.IGNORECASE)

# Reviewer prompt fragments — kept as module constants so they're shared and
# don't clutter review_phase().
_TEST_STATUS_PASSING = (
    "\n## Test Status\n"
    "ALL TESTS PASS. The code has been tested and produces correct "
    "results. You are acting as a four-eyes sanity check — only reject "
    "if the code is clearly faked (hardcoded return values, stub "
    "implementations) or is missing entire functions from the spec. "
    "Do NOT reject working code for style, naming, or minor issues.\n"
)
_TEST_STATUS_FAILING = (
    "\n## Test Status\n"
    "Tests are FAILING with assertion errors. Focus your review "
    "on finding the bugs causing test failures.\n"
)


def _now() -> datetime.datetime:
    """Current timestamp — centralized to avoid `__import__('datetime')` at every call site."""
    return datetime.datetime.now()


def _missing_names_failure(missing_names: list) -> ReviewResult:
    """Build a failure ReviewResult for spec-coverage-gate overrides."""
    return ReviewResult(
        passed=False,
        issues=[f"Missing function/class from spec: {n}" for n in missing_names],
        suggestions=[
            f"Implement the '{n}' function as described in the specification"
            for n in missing_names
        ],
        timestamp=_now(),
        model_used="spec-coverage-gate",
    )


@dataclass
class _TriageContext:
    """Narrow context for triage diagnose+fix — one failing test."""
    code_file: str
    code_function: str
    test_file_path: str
    test_function: str


# --- Task profile keyword sets (used by Orchestrator._estimate_task_profile) ---

_DIFFICULTY_KEYWORDS: frozenset[str] = frozenset({
    "sort", "filter", "recursive", "tree", "graph", "regex", "priority",
})

_REASONING_STATE_KEYWORDS: frozenset[str] = frozenset({
    "status", "priority", "history", "overdue", "expire",
})

_REASONING_RETURN_FORMATS: frozenset[str] = frozenset({
    "-> dict", "-> list", "-> bool", "-> int",
    "returns list", "returns a list", "returns dict",
    "tuple", "mapping",
})

_REASONING_ERROR_KEYWORDS: frozenset[str] = frozenset({
    "raise valueerror", "raise keyerror", "raise typeerror",
    "empty", "invalid", "not found", "does not exist",
})

_REASONING_DEPENDENCY_KEYWORDS: frozenset[str] = frozenset({
    "depends on", "after calling", "updates the",
    "affects", "filters", "case-insensitive",
    "ascending", "descending",
})


class Orchestrator:
    """Controls the full cascade cycle: design → code → review → verify → integrate."""

    def __init__(
        self,
        config: Config,
        workspace_path: Path,
        event_callback: Callable[[CascadeEvent], None] | None = None,
    ) -> None:
        self._config = config
        self._workspace_path = workspace_path
        self._event_callback = event_callback
        self._apply_strategy_profiles(workspace_path)

        # Core state initialization
        self._task_file = workspace_path / ".pmca" / "tasks.json"
        self._snippet_store: dict[str, str] = {}
        self._gate_stats: dict[str, int] = defaultdict(int)

        # Project mode: enabled when max_depth > 1
        self._project_mode = config.cascade.max_depth > 1

        self._model_manager = ModelManager(config)
        self._file_manager = FileManager(workspace_path)
        self._git_manager = GitManager(workspace_path)
        self._task_tree = TaskTree()

        # RAG manager (optional — only if configured and deps installed)
        self._rag_manager = None
        if config.rag.enabled:
            try:
                from pmca.utils.rag import RAGManager
                self._rag_manager = RAGManager(config.rag)
            except Exception as exc:
                log.warning(f"RAG initialization failed: {exc}")

        self._context_manager = ContextManager(
            self._task_tree,
            project_mode=self._project_mode,
            rag_manager=self._rag_manager,
        )

        self._architect = ArchitectAgent(
            self._model_manager,
            max_children=config.cascade.max_children,
            project_mode=self._project_mode,
            quality_standards=config.cascade.quality_standards,
        )
        self._coder = CoderAgent(self._model_manager, project_mode=self._project_mode)
        self._reviewer = ReviewerAgent(self._model_manager)
        self._watcher = WatcherAgent(
            self._model_manager,
            workspace_path,
            lint_config=config.lint,
            cascade_config=config.cascade,
        )

        # Failure memory (optional — only if configured and deps installed)
        self._failure_memory = None
        if config.cascade.failure_memory:
            try:
                from pmca.utils.failure_memory import FailureMemoryManager
                mem_path = Path(workspace_path) / config.cascade.failure_memory_path
                self._failure_memory = FailureMemoryManager(persist_dir=str(mem_path))
                if not self._failure_memory.available:
                    self._failure_memory = None
            except Exception as exc:
                log.warning(f"Failure memory initialization failed: {exc}")

        # Tester agent: optional, uses 14B model for better test quality
        self._tester = None
        if AgentRole.TESTER in config.models:
            self._tester = TesterAgent(self._model_manager, project_mode=self._project_mode)

    def _apply_strategy_profiles(self, workspace_path: Path) -> None:
        """Apply Best Known Configurations (BKC) based on active models."""
        from pmca.models.profiles import get_profile_for_model

        # Check coder model as the primary driver of strategy
        coder_cfg = self._config.models.get(AgentRole.CODER)
        if not coder_cfg:
            self._think_architect_hint = False
            self._think_coder_hint = False
            return

        profile = get_profile_for_model(coder_cfg.name)
        if profile:
            log.info(f"Applying Strategy Profile: {profile.name} to orchestrator")
            # Override cascade defaults if not explicitly set in YAML
            cascade = self._config.cascade
            cascade.use_llm_reviewer = profile.use_llm_reviewer
            cascade.reviewer_bypass_on_pass = profile.reviewer_bypass_on_pass
            # Technique flags from profile
            cascade.import_fixes = profile.import_fixes
            cascade.ast_fixes = profile.ast_fixes
            cascade.test_calibration = profile.test_calibration
            cascade.micro_fix = profile.micro_fix
            cascade.lesson_injection = profile.lesson_injection
            cascade.spec_literals = profile.spec_literals
            cascade.test_triage = profile.test_triage
            cascade.runtime_fixes = profile.runtime_fixes
            cascade.defensive_guards = profile.defensive_guards
            # Set adaptive thinking hints based on profile
            self._think_architect_hint = profile.think_architect
            self._think_coder_hint = profile.think_coder

            if profile.best_of_n > 1:
                cascade.best_of_n = profile.best_of_n
        else:
            self._think_architect_hint = False
            self._think_coder_hint = False

    def _emit(self, event: CascadeEvent) -> None:
        """Send an event to the callback if one is registered."""
        if self._event_callback is not None:
            self._event_callback(event)

    async def run(self, user_request: str, think: bool | None = None) -> TaskNode:
        """Main entry point. Takes user request, returns completed task tree root."""
        self._think_override = think
        console.print(
            Panel(f"[bold]PMCA[/bold] — Processing: {user_request}", style="cyan")
        )

        # Initialize git
        if self._config.workspace.git_checkpoint:
            self._git_manager.init()

        # Auto-index RAG docs if configured
        if self._rag_manager and self._rag_manager.available and self._config.rag.docs_path:
            docs_path = Path(self._config.rag.docs_path).expanduser()
            if docs_path.is_dir():
                count = self._rag_manager.index_directory(docs_path)
                if count > 0:
                    log.info(f"Indexed {count} RAG chunks from {docs_path}")

        # Create root task
        root = self._task_tree.create_root(title=user_request)
        self._save_state()

        self._emit(CascadeEvent(
            event_type=EventType.CASCADE_START,
            task_title=user_request,
            task_id=root.id,
            message=f"Starting PMCA cascade for: {user_request}",
        ))

        # Run the cascade
        try:
            result = await self.cascade(root)
            if result.is_complete:
                console.print(
                    Panel("[bold green]Task completed successfully![/bold green]")
                )
                # Final verification pass
                await self._final_verification(result, user_request)
                self._emit(CascadeEvent(
                    event_type=EventType.CASCADE_COMPLETE,
                    task_title=user_request,
                    task_id=root.id,
                    message="Cascade completed successfully",
                ))
            else:
                console.print(
                    Panel(
                        f"[bold red]Task ended with status: {result.status.value}[/bold red]"
                    )
                )
                self._emit(CascadeEvent(
                    event_type=EventType.CASCADE_ERROR,
                    task_title=user_request,
                    task_id=root.id,
                    message=f"Cascade ended with status: {result.status.value}",
                ))
            # Collect filename normalization counts from agents
            if self._coder.filename_normalizations > 0:
                self._gate_stats["filename_norm"] += self._coder.filename_normalizations
            # Log gate telemetry summary
            if self._gate_stats:
                stats_str = ", ".join(
                    f"{k}={v}" for k, v in sorted(self._gate_stats.items())
                )
                log.info(f"Gate telemetry: {stats_str}")
            # Log LLM usage summary
            mm = self._model_manager
            if mm.total_llm_calls > 0:
                log.info(
                    f"LLM usage: {mm.total_llm_calls} calls, "
                    f"{mm.total_prompt_tokens} prompt + "
                    f"{mm.total_completion_tokens} completion tokens, "
                    f"{mm.total_llm_duration_ms:.0f}ms total"
                )
            return result
        except Exception as exc:
            self._emit(CascadeEvent(
                event_type=EventType.CASCADE_ERROR,
                task_title=user_request,
                task_id=root.id,
                message=f"Cascade error: {exc}",
                data={"error": str(exc)},
            ))
            raise
        finally:
            await self._model_manager.close()
            if self._rag_manager is not None:
                self._rag_manager.close()

    async def cascade(self, task: TaskNode) -> TaskNode:
        """Run the full design→verify cascade on a task."""
        log.info(f"Starting cascade for: {task.title} (depth={task.depth})")

        # Check max depth — in project mode, only the root (depth 0) decomposes
        # into modules.  All child tasks (depth >= 1) are implemented as leaves.
        # This prevents over-decomposition where models split modules into
        # individual functions, each generating fragmented code with broken imports.
        force_leaf = task.depth > self._config.cascade.max_depth
        if self._project_mode and task.depth >= 1:
            force_leaf = True

        if force_leaf:
            log.info(f"Treating '{task.title}' as leaf (depth={task.depth})")
            # Task may be PENDING (created by parent decomposition, never designed).
            # Transition to DESIGNING so code_phase can proceed.
            if task.status == TaskStatus.PENDING:
                task.transition(TaskStatus.DESIGNING)
            task = await self._code_leaf(task)
            return task

        # Phase 1: Design
        task = await self.design_phase(task)
        if task.is_failed:
            return task

        # If decomposed, cascade each child
        if task.status == TaskStatus.DECOMPOSED:
            children = self._task_tree.get_children(task.id)
            if self._project_mode:
                children = self._sort_by_dependencies(children)

            failed_children: list[str] = []
            for child in children:
                child_result = await self.cascade(child)
                if child_result.is_failed:
                    log.error(f"Child task '{child.title}' failed")
                    failed_children.append(child.title)
                    if not self._project_mode:
                        # In single-file mode, abort immediately
                        task.status = TaskStatus.FAILED
                        task.review_history.append(
                            ReviewResult(
                                passed=False,
                                issues=[f"Child task failed: {child.title}"],
                                suggestions=[],
                                timestamp=_now(),
                                model_used="orchestrator",
                            )
                        )
                        self._save_state()
                        return task
                    # In project mode, continue with remaining children

            if failed_children:
                if self._project_mode:
                    verified = [c for c in children if c.status == TaskStatus.VERIFIED]
                    log.warning(
                        f"{len(failed_children)} child(ren) failed: {failed_children}. "
                        f"{len(verified)} verified — attempting partial integration."
                    )
                    if not verified:
                        task.status = TaskStatus.FAILED
                        task.review_history.append(
                            ReviewResult(
                                passed=False,
                                issues=[f"All children failed: {failed_children}"],
                                suggestions=[],
                                timestamp=_now(),
                                model_used="orchestrator",
                            )
                        )
                        self._save_state()
                        return task

            # Phase 5: Integrate
            task = await self.integrate_phase(task)
        else:
            # Leaf task — code and verify
            task = await self._code_leaf(task)

        self._save_state()
        return task

    async def design_phase(self, task: TaskNode) -> TaskNode:
        """Architect generates spec. If decomposable, creates children."""
        log.info(f"[magenta]DESIGN[/magenta] phase for: {task.title}")
        task.transition(TaskStatus.DESIGNING)
        self._save_state()

        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="design",
            message=f"Designing: {task.title}",
        ))

        # Targeted Thinking Strategy: architect uses think=True only for complex tasks
        difficulty, reasoning_heavy, signals = self._estimate_task_profile(task)
        think_mode = reasoning_heavy or self._think_architect_hint
        if think_mode:
            log.info(f"Task '{task.title}' flagged reasoning_heavy or profile_hint → triggering architect thinking")

        # Generate spec
        context = self._context_manager.build_context(task)
        spec = await self._architect.generate_spec(task, context, think=think_mode)
        task.spec = spec
        log.info(f"Generated spec ({len(spec)} chars) for '{task.title}'")

        # Try to decompose
        subtasks = await self._architect.decompose(task, think=think_mode)

        if subtasks:
            log.info(f"Decomposing '{task.title}' into {len(subtasks)} subtasks")
            for st in subtasks:
                child = self._task_tree.add_child(
                    parent_id=task.id,
                    title=st["title"],
                    task_type=st.get("type", TaskType.FUNCTION),
                )
                child.spec = st.get("description", "")
                log.info(f"  Created subtask: {child.title}")

            task.transition(TaskStatus.DECOMPOSED)

            self._emit(CascadeEvent(
                event_type=EventType.TASK_DECOMPOSED,
                task_title=task.title,
                task_id=task.id,
                phase="design",
                message=f"Decomposed into {len(subtasks)} subtasks",
                data={"subtasks": [st["title"] for st in subtasks]},
            ))

            # Review each child spec against parent
            # Skip in project mode — the decomposition prompt already produces
            # well-structured specs and the 7B reviewer is unreliable here
            if not self._project_mode:
                await self._review_child_specs(task, think=think_mode)
        else:
            # Leaf — move directly to coding
            log.info(f"Task '{task.title}' is a leaf, proceeding to code")

        self._emit(CascadeEvent(
            event_type=EventType.PHASE_COMPLETE,
            task_title=task.title,
            task_id=task.id,
            phase="design",
            message=f"Design phase complete for: {task.title}",
        ))

        self._save_state()
        return task

    async def _review_child_specs(self, parent: TaskNode, think: bool | None = None) -> None:
        """Review each child's spec against parent spec."""
        children = self._task_tree.get_children(parent.id)
        for child in children:
            review = await self._reviewer.verify_spec(child.spec, parent.spec)
            child.review_history.append(review)

            if not review.passed:
                log.warning(f"Child spec review failed for '{child.title}'")
                # Retry with feedback
                for retry in range(self._config.cascade.max_retries):
                    child.retry_count += 1
                    refined = await self._architect.refine_spec(child, review, think=think)
                    child.spec = refined
                    review = await self._reviewer.verify_spec(child.spec, parent.spec)
                    child.review_history.append(review)
                    if review.passed:
                        log.info(f"Child spec passed after {retry + 1} retries")
                        break
                else:
                    log.error(
                        f"Child spec '{child.title}' failed after "
                        f"{self._config.cascade.max_retries} retries"
                    )

    async def _code_leaf(self, task: TaskNode) -> TaskNode:
        """Code and verify a leaf task.

        Pipeline:
          1. code_phase         — LLM generates code
          2. pre_review_gates   — deterministic repairs + analysis (zero tokens)
          3. review_phase       — LLM review + retry loop
          4. post_verify_steps  — edge cases, interface extraction (on success)
        """
        _, reasoning_heavy, _ = self._estimate_task_profile(task)
        think_mode = reasoning_heavy or self._think_coder_hint

        task = await self.code_phase(task, think=think_mode)
        if task.is_failed:
            return task

        await self._run_pre_review_gates(task)
        task = await self.review_phase(task)

        if task.is_complete:
            await self._run_post_verify_steps(task)
        return task

    async def _run_pre_review_gates(self, task: TaskNode) -> None:
        """Deterministic repair + analysis gates before review. Stores lint/spec state on task."""
        await self._run_auto_fix(task)
        await self._run_defensive_guards(task)
        await self._run_static_analysis_gate(task)
        await self._run_spec_coverage_gate(task)
        await self._run_test_calibration_gates(task)
        await self._run_mutation_oracle_gate(task)

    async def _run_auto_fix(self, task: TaskNode) -> None:
        auto_fixes = await self._watcher.auto_fix_deterministic(task)
        if auto_fixes > 0:
            log.info(f"Auto-fixed {auto_fixes} deterministic error(s)")
            self._gate_stats["auto_fix"] += auto_fixes
            if self._project_mode:
                self._refresh_snippets(task)

    async def _run_defensive_guards(self, task: TaskNode) -> None:
        """Preventive AST guards — disabled by default (empirically harmful on 7B)."""
        if not self._config.cascade.defensive_guards:
            return
        guards = await self._watcher.inject_defensive_guards(task)
        if guards > 0:
            log.info(f"Injected {guards} defensive guard(s)")
            self._gate_stats["defensive_guards"] = self._gate_stats.get("defensive_guards", 0) + guards

    async def _run_static_analysis_gate(self, task: TaskNode) -> None:
        """Syntax + API-consistency check. Stores lint issues on task for review_phase."""
        blocking_errors, lint_info = await self._watcher.static_analysis_gate(task)
        if blocking_errors:
            log.warning(f"Syntax errors found: {blocking_errors}")
            self._gate_stats["syntax_errors"] += len(blocking_errors)
            iface_errors = [e for e in blocking_errors if "attribute" in e and "method" in e]
            if iface_errors:
                self._gate_stats["interface_inconsistency"] += len(iface_errors)
        if lint_info:
            log.info(f"Linter issues ({len(lint_info)}): {lint_info}")
            self._gate_stats["lint_issues"] += len(lint_info)
        task._lint_issues = lint_info

    async def _run_spec_coverage_gate(self, task: TaskNode) -> None:
        """Verify all spec names are defined. Stores missing names on task for review_phase."""
        missing_names = await self._watcher.spec_coverage_check(task)
        if missing_names:
            log.warning(
                f"Spec-coverage gap: {len(missing_names)} name(s) from spec "
                f"not found in code: {missing_names}"
            )
            self._gate_stats["spec_coverage_gaps"] += len(missing_names)
            task._missing_spec_names = missing_names

    async def _run_test_calibration_gates(self, task: TaskNode) -> None:
        """Conservative calibration + aggressive oracle repair of test assertions."""
        if not self._config.cascade.test_calibration:
            return
        calibrated = await self._watcher.calibrate_tests(task)
        if calibrated > 0:
            log.info(f"Calibrated {calibrated} test assertion(s)")
            self._gate_stats["calibrations"] += calibrated

        oracle_repaired = await self._watcher.oracle_repair_tests(task)
        if oracle_repaired > 0:
            log.info(f"Oracle-repaired {oracle_repaired} test assertion(s)")
            self._gate_stats["oracle_repairs"] += oracle_repaired

    async def _run_mutation_oracle_gate(self, task: TaskNode) -> None:
        """Validate test quality via AST mutations."""
        if not self._config.cascade.mutation_oracle:
            return
        total, killed, kill_ratio = await self._watcher.mutation_oracle(task)
        if total == 0:
            return
        self._gate_stats["mutation_total"] = self._gate_stats.get("mutation_total", 0) + total
        self._gate_stats["mutation_killed"] = self._gate_stats.get("mutation_killed", 0) + killed
        if kill_ratio < 0.5:
            warning = (
                f"Mutation oracle: only {killed}/{total} mutants killed "
                f"({kill_ratio:.0%}). Tests may contain hallucinated assertions."
            )
            log.warning(warning)
            task._mutation_oracle_warning = warning

    async def _run_post_verify_steps(self, task: TaskNode) -> None:
        """Post-VERIFIED actions: edge case tests + interface extraction."""
        await self._generate_edge_case_tests(task)
        if self._project_mode:
            self._extract_and_attach_interface(task)

    async def _generate_edge_case_tests(self, task: TaskNode) -> None:
        """Informational edge-case tests. Never blocks verification — failures logged only."""
        if self._tester is None:
            return
        try:
            code_content = self._gather_code(task)
            tests_content = self._gather_tests(task)
            if not (code_content and tests_content):
                return
            edge_files = await self._tester.generate_edge_cases(task, code_content, tests_content)
            if not edge_files:
                return
            for cf in edge_files:
                self._file_manager.write_file(cf.path, cf.content)
            log.info(
                f"Generated {len(edge_files)} edge case test file(s) "
                f"for '{task.title}' (informational)"
            )
        except Exception as exc:
            log.warning(f"Edge case generation failed (non-blocking): {exc}")

    def _refresh_snippets(self, task: TaskNode) -> None:
        """Re-read code files from disk into the snippet store and task node."""
        for path in task.code_files:
            try:
                content = self._file_manager.read_file(path)
                task.code_files[path] = content
                if self._project_mode:
                    key = f"{task.id}:{path}"
                    self._snippet_store[key] = content
            except FileNotFoundError:
                pass

    def _extract_and_attach_interface(self, task: TaskNode) -> None:
        """Extract AST interface from code files and append to task spec."""
        interfaces: list[str] = []
        for path, content in task.code_files.items():
            iface = ArchitectAgent.extract_interface_from_code(content, path)
            if iface:
                interfaces.append(f"# {path}\n{iface}")
        if interfaces:
            task.spec += "\n[INTERFACE]\n" + "\n".join(interfaces)
            log.info(f"Attached interface to '{task.title}'")

    @staticmethod
    def _estimate_task_profile(task: TaskNode) -> tuple[str, bool, int]:
        """Estimate task difficulty and reasoning needs from spec (zero LLM cost).

        Returns (difficulty, reasoning_heavy, reasoning_signals):
          - difficulty: "simple" or "complex" (2+ indicators)
          - reasoning_heavy: True when 3+ reasoning signals warrant a stronger model
          - reasoning_signals: raw signal count (exposed for logging)
        """
        # Use title if spec is not yet available (first design pass).
        spec = (task.spec or task.title).lower()

        difficulty_count = Orchestrator._count_difficulty_indicators(spec)
        difficulty = "complex" if difficulty_count >= 2 else "simple"

        reasoning_signals = Orchestrator._count_reasoning_signals(spec)
        reasoning_heavy = reasoning_signals >= 3
        return difficulty, reasoning_heavy, reasoning_signals

    @staticmethod
    def _count_difficulty_indicators(spec: str) -> int:
        """Count indicators that suggest the task is algorithmically complex."""
        count = 0
        if spec.count("def ") + spec.count("class ") > 2:
            count += 1
        if "depends_on:" in spec and "none" not in spec.split("depends_on:")[1][:20]:
            count += 1
        if any(kw in spec for kw in _DIFFICULTY_KEYWORDS):
            count += 1
        if len(spec) > 500:
            count += 1
        return count

    @staticmethod
    def _count_reasoning_signals(spec: str) -> int:
        """Count signals suggesting the task benefits from a reasoning model."""
        signals = 0
        method_count = max(spec.count("def "), len(re.findall(r"->\s*\w+", spec)))
        if method_count >= 5:
            signals += 1
        if any(kw in spec for kw in _REASONING_STATE_KEYWORDS):
            signals += 1
        if sum(1 for kw in _REASONING_RETURN_FORMATS if kw in spec) >= 3:
            signals += 1
        if sum(1 for kw in _REASONING_ERROR_KEYWORDS if kw in spec) >= 2:
            signals += 1
        if any(kw in spec for kw in _REASONING_DEPENDENCY_KEYWORDS):
            signals += 1
        return signals

    async def code_phase(self, task: TaskNode, think: bool | None = None) -> TaskNode:
        """Coder implements a leaf-level task.

        Pipeline:
          1. transition state → CODING, emit PHASE_START
          2. select coder role (adaptive routing for reasoning-heavy tasks)
          3. optionally generate+review tests first (test-first mode)
          4. generate code (best-of-N | test-first | plain)
          5. persist files, emit CODE_GENERATED
        """
        log.info(f"[green]CODE[/green] phase for: {task.title}")
        self._begin_code_phase(task)

        context = self._context_manager.build_context(task)
        difficulty, coder_role = self._select_coder_role(task, think)

        tests_content = ""
        if self._config.cascade.test_first:
            tests_content, context = await self._generate_and_review_tests(task, context)

        code_files = await self._generate_code_files(
            task, context, difficulty, coder_role, think, tests_content,
        )
        self._persist_code_files(task, code_files)
        return task

    def _begin_code_phase(self, task: TaskNode) -> None:
        """Transition task into CODING and emit the phase-start event."""
        if task.status not in (TaskStatus.DESIGNING, TaskStatus.REVIEWING):
            task.transition(TaskStatus.CODING)
        else:
            task.status = TaskStatus.CODING
            task.updated_at = _now()

        self._save_state()
        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="code",
            message=f"Coding: {task.title}",
        ))

    def _select_coder_role(
        self, task: TaskNode, think: bool | None,
    ) -> tuple[str, AgentRole | None]:
        """Pick coder role via adaptive routing and stash overrides on task."""
        difficulty, reasoning_heavy, reasoning_signals = self._estimate_task_profile(task)
        coder_role: AgentRole | None = None
        if reasoning_heavy and AgentRole.CODER_REASONING in self._config.models:
            coder_role = AgentRole.CODER_REASONING
            log.info(
                f"Task difficulty: {difficulty}, reasoning_heavy=True "
                f"({reasoning_signals} signals) → routing to coder_reasoning"
            )
        else:
            log.info(
                f"Task difficulty: {difficulty}, "
                f"reasoning_signals={reasoning_signals} → using default coder"
            )
        task._coder_role_override = coder_role
        task._think_override = think
        return difficulty, coder_role

    async def _generate_and_review_tests(
        self, task: TaskNode, context: str,
    ) -> tuple[str, str]:
        """Generate tests, review quality up to N times, persist, return (tests_content, context).

        Test-quality review is skipped in project mode: 7B reviewers are unreliable on
        small leaf modules and tests are validated by actual execution later.
        """
        log.info("Test-first mode: generating tests from specification")
        test_files = await self._call_tester_or_coder_for_tests(task, context)
        tests_content = "\n\n".join(cf.content for cf in test_files)

        if not self._project_mode:
            tests_content, context, test_files = await self._review_test_quality(
                task, context, test_files, tests_content,
            )

        for cf in test_files:
            self._file_manager.write_file(cf.path, cf.content)
            task.test_files[cf.path] = cf.content
        log.info(f"Generated {len(test_files)} test file(s)")
        return tests_content, context

    async def _call_tester_or_coder_for_tests(self, task: TaskNode, context: str):
        """Prefer dedicated tester agent, fall back to coder."""
        if self._tester is not None:
            return await self._tester.generate_tests(task, context)
        return await self._coder.generate_tests(task, context)

    async def _review_test_quality(
        self, task: TaskNode, context: str, test_files, tests_content: str,
    ) -> tuple[str, str, list]:
        """Up to 3 review+regenerate cycles on generated tests. Returns updated tuple."""
        max_test_attempts = 3
        for test_attempt in range(max_test_attempts):
            test_review = await self._reviewer.verify_tests(
                tests_content, task.spec, context,
            )
            if test_review.passed:
                log.info(f"Test quality review passed (attempt {test_attempt + 1})")
                return tests_content, context, test_files

            log.warning(
                f"Test quality review failed (attempt {test_attempt + 1}/"
                f"{max_test_attempts}): {test_review.issues}"
            )
            issue_feedback = "\n".join(f"- {i}" for i in test_review.issues)
            context = (
                context
                + "\n\n## Test Review Feedback (fix these issues)\n"
                + issue_feedback
            )
            test_files = await self._call_tester_or_coder_for_tests(task, context)
            tests_content = "\n\n".join(cf.content for cf in test_files)

        log.warning("Test review failed after all attempts, using last tests")
        return tests_content, context, test_files

    async def _generate_code_files(
        self,
        task: TaskNode,
        context: str,
        difficulty: str,
        coder_role: AgentRole | None,
        think: bool | None,
        tests_content: str,
    ) -> list:
        """Dispatch to best-of-N, test-first, or plain implementation."""
        best_of_n = self._config.cascade.best_of_n
        if best_of_n > 1:
            return await self._generate_best_of_n(task, context, best_of_n, tests_content)
        if self._config.cascade.test_first and tests_content:
            return await self._coder.implement_with_tests(task, context, tests_content)
        return await self._coder.implement(
            task, context,
            difficulty=difficulty,
            role_override=coder_role,
            think=think,
            spec_literals=self._config.cascade.spec_literals,
        )

    async def _generate_best_of_n(
        self, task: TaskNode, context: str, best_of_n: int, tests_content: str,
    ) -> list:
        """Best-of-N sampling: generate candidates, test each, keep the winner."""
        log.info(f"Best-of-{best_of_n} sampling: generating {best_of_n} candidates")
        all_candidate_paths: set[str] = set()

        async def _test_runner(candidate_files):
            for cf in candidate_files:
                self._file_manager.write_file(cf.path, cf.content)
                all_candidate_paths.add(cf.path)
            old_code_files = list(task.code_files)
            task.code_files = [
                cf.path for cf in candidate_files
                if not (cf.path.startswith("test") or "/test_" in cf.path)
            ]
            result = await self._watcher.run_tests(task)
            task.code_files = old_code_files
            return result

        code_files = await self._coder.implement_best_of_n(
            task, context, best_of_n, _test_runner,
            tests_content=tests_content,
            cross_execution=self._config.cascade.cross_execution,
        )
        self._cleanup_losing_candidates(all_candidate_paths, {cf.path for cf in code_files})
        return code_files

    def _cleanup_losing_candidates(
        self, all_paths: set[str], winner_paths: set[str],
    ) -> None:
        """Delete orphan candidate files from losing best-of-N candidates."""
        for orphan in all_paths - winner_paths:
            try:
                (self._workspace_path / orphan).unlink(missing_ok=True)
            except OSError:
                pass

    def _persist_code_files(self, task: TaskNode, code_files: list) -> None:
        """Write files, attach to task, emit CODE_GENERATED, save state."""
        for cf in code_files:
            self._file_manager.write_file(cf.path, cf.content)
            if cf.path.startswith("test") or "/test_" in cf.path:
                task.test_files[cf.path] = cf.content
            else:
                task.code_files[cf.path] = cf.content
                if self._project_mode:
                    self._snippet_store[f"{task.id}:{cf.path}"] = cf.content

        log.info(
            f"Generated {len(code_files)} files for '{task.title}': "
            f"{[cf.path for cf in code_files]}"
        )
        self._emit(CascadeEvent(
            event_type=EventType.CODE_GENERATED,
            task_title=task.title,
            task_id=task.id,
            phase="code",
            message=f"Generated {len(code_files)} files",
            data={"files": [cf.path for cf in code_files]},
        ))
        self._save_state()

    async def review_phase(self, task: TaskNode) -> TaskNode:
        """Review and verify code for a leaf task.

        Flow: smoke test → if crash, skip review and fix directly
              → if tests pass, review → faking check → VERIFIED
              → if tests fail, review → combine issues → fix
        """
        log.info(f"[blue]REVIEW[/blue] phase for: {task.title}")
        self._begin_review_phase(task)

        code_content = self._gather_code(task)
        lesson_records: list[LessonRecord] = []
        max_retries = self._config.cascade.max_retries

        for attempt in range(max_retries + 1):
            smoke_result = await self._run_smoke_test(task)

            if smoke_result.passed:
                review = await self._review_passing_code(task, code_content)
                if review.passed and await self._verify_and_finalize(task, code_content, review):
                    return task
                review = self._augment_with_lint_issues(task, review)
                structured_errors: list = []
            else:
                structured_errors = self._watcher.extract_structured_errors(smoke_result.output)
                review = await self._build_failure_review(
                    task, code_content, smoke_result, structured_errors,
                )

            self._record_lesson(attempt, review, structured_errors, lesson_records)

            if attempt >= max_retries:
                break

            self._announce_retry(task, attempt, max_retries)
            if self._is_fresh_start_attempt(attempt):
                await self._regenerate_from_scratch(task)
            else:
                resolved_by_cheap_fixes = await self._apply_cheap_fixes(task, structured_errors)
                if resolved_by_cheap_fixes:
                    code_content = self._gather_code(task)
                    continue
                await self._invoke_coder_fix(task, code_content, review, lesson_records, attempt)

            code_content = self._gather_code(task)
            await self._post_fix_gates(task)

        await self._persist_failure_memory(task, lesson_records)
        return self._mark_task_failed(task, max_retries)

    # ------------------------------------------------------------------
    # review_phase helpers
    # ------------------------------------------------------------------

    def _begin_review_phase(self, task: TaskNode) -> None:
        task.status = TaskStatus.REVIEWING
        task.updated_at = _now()
        self._save_state()
        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message=f"Reviewing: {task.title}",
        ))

    async def _run_smoke_test(self, task: TaskNode) -> TestResult:
        """Run tests before LLM review. Emits a TEST_RESULT event."""
        smoke_result = await self._watcher.run_tests(task)
        self._emit(CascadeEvent(
            event_type=EventType.TEST_RESULT,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message="Smoke test passed" if smoke_result.passed else "Smoke test failed",
            data={"passed": smoke_result.passed, "smoke": True},
        ))
        return smoke_result

    async def _review_passing_code(self, task: TaskNode, code_content: str) -> ReviewResult:
        """Tests pass. Invoke reviewer (or bypass it) and enforce spec-coverage gate."""
        missing_names = getattr(task, "_missing_spec_names", [])
        spec_for_review = self._spec_with_missing_names(task.spec, missing_names)

        if self._should_bypass_reviewer(missing_names):
            log.info("Reviewer bypass — tests pass, no spec gaps, auto-approving")
            review = ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=_now(), model_used="bypass",
            )
        else:
            review = await self._reviewer.verify_code(
                code_content, spec_for_review, test_status=_TEST_STATUS_PASSING,
            )
        task.review_history.append(review)

        if review.passed and missing_names:
            log.warning(f"Overriding review pass — missing names: {missing_names}")
            review = _missing_names_failure(missing_names)
            task.review_history.append(review)

        self._emit(CascadeEvent(
            event_type=EventType.REVIEW_RESULT,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message="Review passed" if review.passed else f"Review failed: {', '.join(review.issues[:3])}",
            data={"passed": review.passed, "issues": review.issues},
        ))
        return review

    def _should_bypass_reviewer(self, missing_names: list) -> bool:
        c = self._config.cascade
        return (c.reviewer_bypass_on_pass or not c.use_llm_reviewer) and not missing_names

    @staticmethod
    def _spec_with_missing_names(spec: str, missing_names: list) -> str:
        if not missing_names:
            return spec
        return spec + (
            "\n\n## CRITICAL: Missing Functions\n"
            "The following functions/classes from the spec are NOT "
            "implemented in the code. This is a FAILURE:\n"
            + "\n".join(f"- {n}" for n in missing_names)
        )

    async def _verify_and_finalize(
        self, task: TaskNode, code_content: str, review: ReviewResult,
    ) -> bool:
        """Run faking check if applicable, then mark VERIFIED. Returns True if verified."""
        if not self._project_mode:
            test_content = self._gather_tests(task)
            if test_content:
                fake_check = await self._watcher.check_not_faked(code_content, test_content)
                task.review_history.append(fake_check)
                if not fake_check.passed:
                    log.warning("Code appears to be faked, retrying")
                    return False

        task.transition(TaskStatus.VERIFIED)
        if self._config.workspace.git_checkpoint:
            task.git_checkpoint = self._git_manager.checkpoint(task, "verified")
        log.info(f"Task '{task.title}' verified!")
        self._emit(CascadeEvent(
            event_type=EventType.TASK_COMPLETE,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message=f"Task verified: {task.title}",
        ))
        self._save_state()
        return True

    def _augment_with_lint_issues(self, task: TaskNode, review: ReviewResult) -> ReviewResult:
        """Append lint issues to a failed review so the coder fixes them alongside review issues."""
        if review.passed:
            return review
        lint_issues = getattr(task, "_lint_issues", [])
        if not lint_issues:
            return review
        return ReviewResult(
            passed=False,
            issues=review.issues + [f"[lint] {e}" for e in lint_issues],
            suggestions=review.suggestions,
            timestamp=review.timestamp,
            model_used=review.model_used,
        )

    async def _build_failure_review(
        self,
        task: TaskNode,
        code_content: str,
        smoke_result: TestResult,
        structured_errors: list,
    ) -> ReviewResult:
        """Tests failed. Build a review result from structured errors + optional LLM review."""
        is_crash = any(
            e.error_type in ("NameError", "ImportError", "SyntaxError", "TypeError")
            for e in structured_errors
        )
        if is_crash:
            log.warning(
                f"Code crashes on import/syntax — skipping review "
                f"({[e.error_type for e in structured_errors]})"
            )
            structured_issues = [e.format_for_prompt() for e in structured_errors]
            lint_issues = getattr(task, "_lint_issues", [])
            if lint_issues:
                structured_issues += [f"[lint] {e}" for e in lint_issues]
            return ReviewResult(
                passed=False,
                issues=structured_issues if structured_issues else smoke_result.errors,
                suggestions=["Fix the crash before anything else"],
                timestamp=_now(),
                model_used="watcher",
            )

        # Tests fail with assertion errors — still run review for spec compliance
        review = await self._reviewer.verify_code(
            code_content, task.spec, test_status=_TEST_STATUS_FAILING,
        )
        task.review_history.append(review)

        structured_issues = [e.format_for_prompt() for e in structured_errors]
        if not structured_issues and smoke_result.output:
            structured_issues = [f"Test output:\n{smoke_result.output[-1000:]}"]

        if self._tester is not None:
            tests_content = self._gather_tests(task)
            analysis = await self._tester.analyze_failure(
                task, smoke_result.output, code_content, tests_content,
            )
            log.info(
                f"Failure analysis: root_cause={analysis.root_cause}, "
                f"fix_target={analysis.suggested_fix_target}"
            )
            if analysis.specific_issues:
                structured_issues = analysis.specific_issues

        combined_issues = review.issues + structured_issues
        lint_issues = getattr(task, "_lint_issues", [])
        if lint_issues:
            combined_issues += [f"[lint] {e}" for e in lint_issues]

        log.warning(f"Tests failed: {[e.error_type for e in structured_errors]}")
        return ReviewResult(
            passed=False,
            issues=combined_issues,
            suggestions=review.suggestions + ["Fix the failing tests"],
            timestamp=_now(),
            model_used="watcher+reviewer",
        )

    def _record_lesson(
        self,
        attempt: int,
        review: ReviewResult,
        structured_errors: list,
        lesson_records: list[LessonRecord],
    ) -> None:
        if not self._config.cascade.lesson_injection:
            return
        strategy = "fix_tests" if (attempt > 0 and attempt % 2 == 0) else "fix_code"
        lesson = self._watcher.extract_lesson(
            attempt=attempt + 1,
            review=review,
            structured_errors=structured_errors,
            strategy=strategy,
        )
        lesson_records.append(lesson)
        self._gate_stats["lessons_injected"] += 1

    def _announce_retry(self, task: TaskNode, attempt: int, max_retries: int) -> None:
        log.info(f"Review failed, retry {attempt + 1}/{max_retries}")
        task.retry_count += 1
        self._emit(CascadeEvent(
            event_type=EventType.RETRY,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message=f"Retry {attempt + 1}/{max_retries}",
            data={"attempt": attempt + 1, "max_retries": max_retries},
        ))

    def _is_fresh_start_attempt(self, attempt: int) -> bool:
        return (
            attempt > 0
            and attempt % self._config.cascade.fresh_start_after == 0
        )

    async def _regenerate_from_scratch(self, task: TaskNode) -> None:
        log.info("Fresh start — regenerating from scratch")
        context = self._context_manager.build_context(task)
        fresh_role = getattr(task, "_coder_role_override", None)
        use_think = getattr(task, "_think_override", None)
        if self._config.cascade.test_first and task.test_files:
            tests_content = self._gather_tests(task)
            code_files = await self._coder.implement_with_tests(task, context, tests_content)
        else:
            code_files = await self._coder.implement(
                task, context, role_override=fresh_role, think=use_think,
            )

        new_code: dict = {}
        new_tests = dict(task.test_files)
        for cf in code_files:
            self._file_manager.write_file(cf.path, cf.content)
            if cf.path.startswith("test") or "/test_" in cf.path:
                new_tests[cf.path] = cf.content
            else:
                new_code[cf.path] = cf.content
        if new_code:
            task.code_files = new_code
        task.test_files = new_tests

    async def _apply_cheap_fixes(self, task: TaskNode, structured_errors: list) -> bool:
        """Try runtime fixes, micro-fix, and triage. Returns True if tests pass after any of them."""
        if not structured_errors:
            return False

        c = self._config.cascade
        if c.runtime_fixes:
            runtime_fixed = await self._watcher.fix_runtime_errors(task, structured_errors)
            if runtime_fixed > 0:
                log.info(f"Runtime-fixed {runtime_fixed} error(s)")
                self._gate_stats["runtime_fixes"] = self._gate_stats.get("runtime_fixes", 0) + runtime_fixed

        if c.micro_fix:
            micro_fixed = await self._watcher.targeted_micro_fix(task, structured_errors)
            if micro_fixed > 0:
                log.info(f"Micro-fixed {micro_fixed} function(s)")
                self._gate_stats["micro_fixes"] += micro_fixed
                if (await self._watcher.run_tests(task)).passed:
                    return True

        if c.test_triage:
            triage_fixed = await self._triage_failing_tests(task, structured_errors)
            if triage_fixed > 0:
                log.info(f"Triage fixed {triage_fixed} issue(s)")
                if (await self._watcher.run_tests(task)).passed:
                    return True

        return False

    async def _invoke_coder_fix(
        self,
        task: TaskNode,
        code_content: str,
        review: ReviewResult,
        lesson_records: list[LessonRecord],
        attempt: int,
    ) -> None:
        fix_role = getattr(task, "_coder_role_override", None)
        if fix_role and attempt >= 2:
            log.info("Routing fallback: switching from reasoning model to fast coder")
            fix_role = None
        use_think = getattr(task, "_think_override", None)

        fm_context = await self._build_failure_memory_context(review)

        fix_files = await self._coder.fix(
            task,
            code_content,
            "\n".join(review.issues),
            lessons_str="\n".join(lr.summary for lr in lesson_records),
            memory_str=fm_context,
            retry_num=attempt + 1,
            coder_role=fix_role,
            think=use_think,
        )
        for cf in fix_files:
            self._file_manager.write_file(cf.path, cf.content)

    async def _build_failure_memory_context(self, review: ReviewResult) -> str:
        if not self._failure_memory or not review.issues:
            return ""
        error_text = " ".join(review.issues[:3])
        similar = self._failure_memory.query_similar(error_text)
        patterns = self._failure_memory.query_patterns(error_text)
        parts = []
        if similar:
            parts.append("Past similar failures:\n" + "\n".join(f"- {s}" for s in similar))
        if patterns:
            parts.append("Known patterns:\n" + "\n".join(p.format_for_prompt() for p in patterns))
        return "\n".join(parts)

    async def _post_fix_gates(self, task: TaskNode) -> None:
        """Run all deterministic gates after a fix attempt: auto-fix, calibrate, oracle, mutation, spec, lint."""
        auto_fixes = await self._watcher.auto_fix_deterministic(task)
        if auto_fixes > 0:
            log.info(f"Auto-fixed {auto_fixes} deterministic error(s) after retry")
            self._gate_stats["auto_fix_retry"] += auto_fixes
            if self._project_mode:
                self._refresh_snippets(task)

        if self._config.cascade.test_calibration:
            recalibrated = await self._watcher.calibrate_tests(task)
            if recalibrated > 0:
                log.info(f"Re-calibrated {recalibrated} test assertion(s) after retry")
                self._gate_stats["calibrations_retry"] += recalibrated

            oracle_repaired = await self._watcher.oracle_repair_tests(task)
            if oracle_repaired > 0:
                log.info(f"Oracle-repaired {oracle_repaired} assertion(s) after retry")
                self._gate_stats["oracle_repairs"] += oracle_repaired

        if self._config.cascade.mutation_oracle:
            total, killed, kill_ratio = await self._watcher.mutation_oracle(task)
            if total > 0:
                self._gate_stats["mutation_total"] = self._gate_stats.get("mutation_total", 0) + total
                self._gate_stats["mutation_killed"] = self._gate_stats.get("mutation_killed", 0) + killed
                if kill_ratio < 0.5:
                    warning = (
                        f"Mutation oracle: only {killed}/{total} mutants killed "
                        f"({kill_ratio:.0%}). Tests may contain hallucinated assertions."
                    )
                    log.warning(warning)
                    task._mutation_oracle_warning = warning

        missing_names = await self._watcher.spec_coverage_check(task)
        task._missing_spec_names = missing_names
        log.info(f"Still missing after fix: {missing_names}" if missing_names
                 else "Spec-coverage gap resolved after fix")

        _, lint_info = await self._watcher.static_analysis_gate(task)
        task._lint_issues = lint_info
        log.info(f"Lint issues after fix ({len(lint_info)}): {lint_info}" if lint_info
                 else "Lint issues resolved after fix")

    async def _persist_failure_memory(
        self, task: TaskNode, lesson_records: list[LessonRecord],
    ) -> None:
        if not (self._failure_memory and lesson_records):
            return
        from pmca.utils.failure_memory import FailureEpisode
        for lesson in lesson_records:
            self._failure_memory.store_episode(FailureEpisode(
                task_spec_summary=task.spec[:200] if task.spec else "",
                error_signature=lesson.summary,
                error_types=lesson.error_types,
                fix_strategy=lesson.strategy,
                fix_description=lesson.summary,
                outcome="unresolved",
                task_title=task.title,
            ))
        self._failure_memory.distill_patterns()

    def _mark_task_failed(self, task: TaskNode, max_retries: int) -> TaskNode:
        log.error(f"Task '{task.title}' failed after {max_retries} retries")
        task.status = TaskStatus.FAILED
        task.updated_at = _now()
        self._emit(CascadeEvent(
            event_type=EventType.TASK_FAILED,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message=f"Task failed after {max_retries} retries: {task.title}",
        ))
        self._save_state()
        return task

    async def _triage_failing_tests(
        self, task: TaskNode, structured_errors: list,
    ) -> int:
        """Investigate each failing test with a focused mini-cascade.

        For each failing test:
          1. Locate code + test function sources (narrow context)
          2. Architect diagnoses: is the CODE wrong or the TEST wrong?
          3. Coder fixes the one thing that's wrong
          4. Watcher re-runs — if all tests pass, exit early

        Capped at 3 errors per call to limit LLM spend.
        Returns number of fixes applied.
        """
        if not structured_errors:
            return 0

        ws = Path(self._workspace_path)
        fixes_applied = 0

        for error in structured_errors[:3]:
            ctx = self._build_triage_context(task, error, ws)
            if ctx is None:
                continue

            verdict, diagnosis_text = await self._diagnose_triage(task, ctx, error)
            self._gate_stats["triage_investigations"] = (
                self._gate_stats.get("triage_investigations", 0) + 1
            )

            if verdict == "test_wrong":
                fixes_applied += await self._apply_triage_test_fix(task, ctx, diagnosis_text)
            else:
                fixes_applied += await self._apply_triage_code_fix(task, ctx, diagnosis_text)

            retest = await self._watcher.run_tests(task)
            if retest.passed:
                log.info(f"Triage resolved all failures after fixing {error.test_name}")
                return fixes_applied

        return fixes_applied

    def _build_triage_context(
        self, task: TaskNode, error, ws: Path,
    ) -> _TriageContext | None:
        """Locate code + test function sources for one failing test.

        Returns None if the file/function can't be pinned down, signalling
        the caller to skip this error.
        """
        code_files_list = (
            list(task.code_files.keys())
            if isinstance(task.code_files, dict)
            else list(task.code_files)
        )
        location = self._watcher._parse_error_location(error, ws, code_files_list)
        if not location:
            return None
        code_file, func_name = location

        code_path = ws / code_file
        if not code_path.exists():
            return None

        func_result = self._watcher._extract_function_source(code_path.read_text(), func_name)
        if not func_result:
            return None
        code_function = func_result[0]

        test_function, test_file_path = self._find_failing_test_function(task, error)
        if not test_function:
            return None

        return _TriageContext(
            code_file=code_file,
            code_function=code_function,
            test_file_path=test_file_path,
            test_function=test_function,
        )

    def _find_failing_test_function(
        self, task: TaskNode, error,
    ) -> tuple[str, str]:
        """Find the source of the failing test function in task.test_files."""
        test_name = (
            error.test_name.split("::")[-1] if "::" in error.test_name else error.test_name
        )
        test_files = task.test_files if isinstance(task.test_files, dict) else {}
        for tpath, tcontent in test_files.items():
            tf_result = self._watcher._extract_function_source(tcontent, test_name)
            if tf_result:
                return tf_result[0], tpath
        return "", ""

    async def _diagnose_triage(
        self, task: TaskNode, ctx: _TriageContext, error,
    ) -> tuple[str, str]:
        """Ask architect whether code or test is wrong. Returns (verdict, diagnosis_text)."""
        from pmca.prompts.coder import TRIAGE_DIAGNOSE_PROMPT

        error_text = (
            error.format_for_prompt() if hasattr(error, "format_for_prompt") else str(error)
        )
        diagnose_prompt = TRIAGE_DIAGNOSE_PROMPT.format(
            spec=task.spec or "",
            code_function=ctx.code_function,
            test_function=ctx.test_function,
            error=error_text[:500],
        )
        diagnosis_raw = await self._architect._generate(
            diagnose_prompt,
            role=AgentRole.ARCHITECT,
            think=self._think_architect_hint or None,
        )
        verdict, diagnosis_text = self._parse_diagnosis(diagnosis_raw)
        log.info(f"Triage [{error.test_name}]: verdict={verdict} ({diagnosis_text[:80]})")
        return verdict, diagnosis_text

    @staticmethod
    def _parse_diagnosis(raw: str) -> tuple[str, str]:
        """Extract verdict + human-readable diagnosis from architect JSON output.

        Defaults to ('code_wrong', raw) when JSON cannot be parsed.
        """
        try:
            json_match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            if not json_match:
                return "code_wrong", raw
            parsed = _json.loads(json_match.group())
            verdict = parsed.get("verdict", "code_wrong")
            reasoning = parsed.get("reasoning", "")
            correct_val = parsed.get("correct_value", "")
            return verdict, f"{reasoning} Correct value: {correct_val}"
        except (ValueError, KeyError):
            return "code_wrong", raw

    async def _apply_triage_test_fix(
        self, task: TaskNode, ctx: _TriageContext, diagnosis_text: str,
    ) -> int:
        """Let coder rewrite the failing test. Returns number of files updated."""
        from pmca.prompts.coder import TRIAGE_FIX_TEST_PROMPT

        fix_prompt = TRIAGE_FIX_TEST_PROMPT.format(
            spec=task.spec or "",
            code_function=ctx.code_function,
            test_function=ctx.test_function,
            diagnosis=diagnosis_text,
            filepath=ctx.test_file_path,
        )
        fix_response = await self._coder._generate(fix_prompt, role=AgentRole.CODER)
        fix_files = self._coder._parse_code_blocks(fix_response)
        test_files = task.test_files if isinstance(task.test_files, dict) else {}

        fixes = 0
        for cf in fix_files:
            if cf.path not in test_files:
                continue
            self._file_manager.write_file(cf.path, cf.content)
            task.test_files[cf.path] = cf.content
            fixes += 1
            self._gate_stats["triage_test_fixes"] = (
                self._gate_stats.get("triage_test_fixes", 0) + 1
            )
        return fixes

    async def _apply_triage_code_fix(
        self, task: TaskNode, ctx: _TriageContext, diagnosis_text: str,
    ) -> int:
        """Let coder rewrite the faulty code function. Returns number of files updated."""
        from pmca.prompts.coder import TRIAGE_FIX_CODE_PROMPT

        fix_prompt = TRIAGE_FIX_CODE_PROMPT.format(
            spec=task.spec or "",
            code_function=ctx.code_function,
            diagnosis=diagnosis_text,
            filepath=ctx.code_file,
        )
        fix_response = await self._coder._generate(fix_prompt, role=AgentRole.CODER)
        fix_files = self._coder._parse_code_blocks(fix_response)

        fixes = 0
        for cf in fix_files:
            self._file_manager.write_file(cf.path, cf.content)
            if isinstance(task.code_files, dict):
                task.code_files[cf.path] = cf.content
            fixes += 1
            self._gate_stats["triage_code_fixes"] = (
                self._gate_stats.get("triage_code_fixes", 0) + 1
            )
        return fixes

    async def integrate_phase(self, task: TaskNode) -> TaskNode:
        """Verify all children work together after they complete.

        Pipeline:
          1. readiness gate — all children must complete (project mode allows partial)
          2. transition to INTEGRATING, emit PHASE_START
          3. project-mode snippet assembly (if any snippets were stored)
          4. integration strategy:
               - project mode → deterministic collect + run integration tests
               - single-file mode → LLM integration review
        """
        log.info(f"[yellow]INTEGRATE[/yellow] phase for: {task.title}")

        if not self._children_ready_for_integration(task):
            return task

        self._begin_integrate_phase(task)
        if self._project_mode and self._snippet_store:
            self._assemble_snippets(task)

        children = self._task_tree.get_children(task.id)
        if self._project_mode:
            await self._integrate_project_mode(task, children)
        else:
            await self._integrate_single_file_mode(task, children)

        self._save_state()
        return task

    def _children_ready_for_integration(self, task: TaskNode) -> bool:
        """True if integration can proceed. Marks task FAILED + returns False otherwise."""
        if self._task_tree.all_children_complete(task.id):
            return True

        failed = self._task_tree.get_failed_children(task.id)
        if not self._project_mode:
            log.error(f"Cannot integrate — {len(failed)} children failed")
            task.status = TaskStatus.FAILED
            task.updated_at = _now()
            self._save_state()
            return False

        # Project mode: tolerate partial integration.
        verified = [
            c for c in self._task_tree.get_children(task.id)
            if c.status == TaskStatus.VERIFIED
        ]
        log.warning(
            f"Partial integration: {len(failed)} failed, {len(verified)} verified"
        )
        return True

    def _begin_integrate_phase(self, task: TaskNode) -> None:
        """Transition to INTEGRATING and emit PHASE_START."""
        task.status = TaskStatus.INTEGRATING
        task.updated_at = _now()
        self._save_state()
        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="integrate",
            message=f"Integrating: {task.title}",
        ))

    def _assemble_snippets(self, task: TaskNode) -> None:
        """Assemble per-child code snippets into final files (project mode only)."""
        assembler = FileAssembler(self._file_manager)
        assembled = assembler.assemble(task, self._task_tree, self._snippet_store)
        log.info(f"Assembled {len(assembled)} file(s) for '{task.title}'")

    async def _integrate_project_mode(
        self, task: TaskNode, children: list[TaskNode],
    ) -> None:
        """Deterministic integration: collect files, run tests, mark VERIFIED.

        Skips LLM review entirely — each child was independently verified,
        assembly is deterministic, and the 7B reviewer is too conservative
        with partial integration.
        """
        verified = [c for c in children if c.status == TaskStatus.VERIFIED]
        log.info(
            f"Project mode: skipping LLM integration review "
            f"({len(verified)}/{len(children)} children verified)"
        )
        self._collect_child_files(task, children)

        if task.test_files:
            test_result = await self._watcher.run_tests(task)
            if test_result.passed:
                log.info(f"Integration tests passed ({test_result.total} tests)")
            else:
                log.warning(
                    f"Integration tests: {test_result.failures}/{test_result.total} "
                    f"failed (non-blocking in project mode)"
                )

        task.transition(TaskStatus.VERIFIED)
        self._checkpoint_if_enabled(task)
        log.info(f"Integration complete for '{task.title}'")

    async def _integrate_single_file_mode(
        self, task: TaskNode, children: list[TaskNode],
    ) -> None:
        """LLM-based integration review. Marks task VERIFIED or FAILED."""
        children_summary = "\n\n".join(
            f"### {c.title}\nStatus: {c.status.value}\nSpec: {c.spec[:500]}\n"
            f"Files: {', '.join(c.code_files)}"
            for c in children
        )
        review = await self._reviewer.verify_integration(task, children_summary)
        task.review_history.append(review)

        if not review.passed:
            log.error(f"Integration failed for '{task.title}': {review.issues}")
            task.status = TaskStatus.FAILED
            task.updated_at = _now()
            return

        task.transition(TaskStatus.VERIFIED)
        self._checkpoint_if_enabled(task)
        self._collect_child_files(task, children)
        log.info(f"Integration verified for '{task.title}'")

    def _collect_child_files(
        self, task: TaskNode, children: list[TaskNode],
    ) -> None:
        """Merge child code+test files into parent task, preserving first-seen ownership."""
        seen: set[str] = set()
        for child in children:
            for f_path, f_content in child.code_files.items():
                if f_path not in seen:
                    task.code_files[f_path] = f_content
                    seen.add(f_path)
            for f_path, f_content in child.test_files.items():
                if f_path not in seen:
                    task.test_files[f_path] = f_content
                    seen.add(f_path)

    def _checkpoint_if_enabled(self, task: TaskNode) -> None:
        """Create a git checkpoint commit when workspace.git_checkpoint is on."""
        if self._config.workspace.git_checkpoint:
            task.git_checkpoint = self._git_manager.checkpoint(task, "integration verified")

    @staticmethod
    def _sort_by_dependencies(children: list[TaskNode]) -> list[TaskNode]:
        """Topological sort of children based on EXPORTS/DEPENDS_ON metadata.

        Uses Kahn's algorithm. Falls back to original order on cycles.
        """
        # Build mapping: export_name -> child index
        export_map: dict[str, int] = {}
        for idx, child in enumerate(children):
            m = _EXPORTS_RE.search(child.spec)
            if m:
                for name in m.group(1).split(","):
                    export_map[name.strip()] = idx

        # Build adjacency: child_idx -> set of child_idx it depends on
        in_degree: dict[int, int] = {i: 0 for i in range(len(children))}
        deps: dict[int, list[int]] = defaultdict(list)  # from -> list[to]

        for idx, child in enumerate(children):
            m = _DEPENDS_RE.search(child.spec)
            if m:
                dep_text = m.group(1).strip()
                if dep_text.upper() == "NONE":
                    continue
                for name in dep_text.split(","):
                    dep_idx = export_map.get(name.strip())
                    if dep_idx is not None and dep_idx != idx:
                        deps[dep_idx].append(idx)
                        in_degree[idx] += 1

        # Kahn's algorithm with heap for stable ordering
        queue = [i for i in range(len(children)) if in_degree[i] == 0]
        heapq.heapify(queue)
        sorted_indices: list[int] = []

        while queue:
            node = heapq.heappop(queue)
            sorted_indices.append(node)
            for neighbor in deps.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    heapq.heappush(queue, neighbor)

        if len(sorted_indices) != len(children):
            # Cycle detected — fall back to original order
            log.warning("Dependency cycle detected, using original child order")
            return children

        return [children[i] for i in sorted_indices]

    async def _final_verification(self, root: TaskNode, original_request: str) -> None:
        """Run final end-to-end verification."""
        log.info("Running final verification...")
        structure = self._file_manager.get_project_structure()
        all_files = self._file_manager.list_files()
        key_files_content = ""
        for f in all_files[:10]:  # Limit to first 10 files
            try:
                content = self._file_manager.read_file(f)
                key_files_content += f"\n### {f}\n```python\n{content}\n```\n"
            except FileNotFoundError:
                pass

        review = await self._watcher.final_verification(
            root, original_request, structure, key_files_content
        )
        root.review_history.append(review)

        if review.passed:
            console.print("[bold green]Final verification passed![/bold green]")
        else:
            console.print("[bold yellow]Final verification found issues:[/bold yellow]")
            for issue in review.issues:
                console.print(f"  - {issue}")

    def _gather_code(self, task: TaskNode) -> str:
        """Gather all code content for a task."""
        parts: list[str] = []
        for path, content in task.code_files.items():
            parts.append(f"# {path}\n{content}")
        return "\n\n".join(parts) if parts else ""

    def _gather_tests(self, task: TaskNode) -> str:
        """Gather all test content for a task."""
        parts: list[str] = []
        for path, content in task.test_files.items():
            parts.append(f"# {path}\n{content}")
        return "\n\n".join(parts) if parts else ""

    def _save_state(self) -> None:
        """Persist current task tree state."""
        self._task_tree.save(self._task_file)

    def print_tree(self) -> None:
        """Print the task tree to console."""
        root = self._task_tree.root
        if root is None:
            console.print("[dim]No tasks[/dim]")
            return

        tree = RichTree(f"[bold]{root.title}[/bold] ({root.status.value})")
        self._add_tree_children(tree, root)
        console.print(tree)

    def _add_tree_children(self, tree_node: RichTree, task: TaskNode) -> None:
        """Recursively add children to a Rich tree."""
        for child in self._task_tree.get_children(task.id):
            style = "green" if child.is_complete else "red" if child.is_failed else "dim"
            branch = tree_node.add(f"[{style}]{child.title} ({child.status.value})[/{style}]")
            self._add_tree_children(branch, child)

    @property
    def task_tree(self) -> TaskTree:
        return self._task_tree

    def get_generated_code(self) -> dict[str, str]:
        """Return all generated code files as {path: content} for API access."""
        result: dict[str, str] = {}
        root = self._task_tree.root
        if root is None:
            return result
        for node in self._task_tree.walk():
            for path, content in node.code_files.items():
                result[path] = content
        return result

    @classmethod
    def load_state(cls, config: Config, workspace_path: Path) -> Orchestrator:
        """Load an orchestrator from a saved state."""
        orch = cls(config, workspace_path)
        task_file = workspace_path / ".pmca" / "tasks.json"
        if task_file.exists():
            orch._task_tree = TaskTree.load(task_file)
            orch._context_manager = ContextManager(
                orch._task_tree,
                project_mode=orch._project_mode,
                rag_manager=orch._rag_manager,
            )
        return orch
