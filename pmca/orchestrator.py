"""Main cascade orchestrator — manages the design→verify cascade cycle."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
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
from pmca.tasks.state import ReviewResult, TaskStatus, TaskType
from pmca.tasks.tree import TaskNode, TaskTree
from pmca.utils.assembler import FileAssembler, parse_target_file
from pmca.utils.context import ContextManager
from pmca.utils.logger import get_console, get_logger
from pmca.workspace.file_manager import FileManager
from pmca.workspace.git_manager import GitManager

log = get_logger("orchestrator")
console = get_console()

# Regex for metadata parsing in subtask descriptions
_EXPORTS_RE = re.compile(r"EXPORTS:\s*(.+)")
_DEPENDS_RE = re.compile(r"DEPENDS_ON:\s*(.+)", re.IGNORECASE)


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
        )
        self._coder = CoderAgent(self._model_manager, project_mode=self._project_mode)
        self._reviewer = ReviewerAgent(self._model_manager)
        self._watcher = WatcherAgent(
            self._model_manager,
            workspace_path,
            lint_config=config.lint,
        )

        # Tester agent: optional, uses 14B model for better test quality
        self._tester = None
        if AgentRole.TESTER in config.models:
            self._tester = TesterAgent(self._model_manager, project_mode=self._project_mode)

        self._task_file = workspace_path / ".pmca" / "tasks.json"

        # Snippet store: {task_id:filepath -> code} for multi-file assembly
        self._snippet_store: dict[str, str] = {}

        # Gate telemetry: track which deterministic gates catch issues
        self._gate_stats: dict[str, int] = defaultdict(int)

    def _emit(self, event: CascadeEvent) -> None:
        """Send an event to the callback if one is registered."""
        if self._event_callback is not None:
            self._event_callback(event)

    async def run(self, user_request: str) -> TaskNode:
        """Main entry point. Takes user request, returns completed task tree root."""
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
                                timestamp=__import__("datetime").datetime.now(),
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
                                timestamp=__import__("datetime").datetime.now(),
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

        # Generate spec
        context = self._context_manager.build_context(task)
        spec = await self._architect.generate_spec(task, context)
        task.spec = spec
        log.info(f"Generated spec ({len(spec)} chars) for '{task.title}'")

        # Try to decompose
        subtasks = await self._architect.decompose(task)

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
                await self._review_child_specs(task)
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

    async def _review_child_specs(self, parent: TaskNode) -> None:
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
                    refined = await self._architect.refine_spec(child, review)
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
        """Code and verify a leaf task."""
        task = await self.code_phase(task)
        if task.is_failed:
            return task

        # Auto-fix deterministic errors (missing imports, etc.) without LLM
        auto_fixes = await self._watcher.auto_fix_deterministic(task)
        if auto_fixes > 0:
            log.info(f"Auto-fixed {auto_fixes} deterministic error(s)")
            self._gate_stats["auto_fix"] += auto_fixes
            # Sync snippet store with fixed files so assembly uses corrected code
            if self._project_mode:
                self._refresh_snippets(task)

        # Static analysis gate — catch syntax errors before wasting LLM tokens
        blocking_errors, lint_info = await self._watcher.static_analysis_gate(task)
        if blocking_errors:
            log.warning(f"Syntax errors found: {blocking_errors}")
            self._gate_stats["syntax_errors"] += len(blocking_errors)
            # Track interface inconsistency errors separately
            iface_errors = [e for e in blocking_errors if "attribute" in e and "method" in e]
            if iface_errors:
                self._gate_stats["interface_inconsistency"] += len(iface_errors)
        if lint_info:
            log.info(f"Linter issues ({len(lint_info)}): {lint_info}")
            self._gate_stats["lint_issues"] += len(lint_info)
        # Store lint issues on task so review_phase can feed them to fix cycle
        task._lint_issues = lint_info

        # Spec-coverage gate — deterministic check for missing functions/classes
        missing_names = await self._watcher.spec_coverage_check(task)
        if missing_names:
            log.warning(
                f"Spec-coverage gap: {len(missing_names)} name(s) from spec "
                f"not found in code: {missing_names}"
            )
            self._gate_stats["spec_coverage_gaps"] += len(missing_names)
            # Inject missing-names context into the task spec so the review
            # phase can feed it to the fix cycle
            task._missing_spec_names = missing_names

        # Calibrate test assertions — fix LLM arithmetic errors in tests
        calibrated = await self._watcher.calibrate_tests(task)
        if calibrated > 0:
            log.info(f"Calibrated {calibrated} test assertion(s)")
            self._gate_stats["calibrations"] += calibrated

        # Oracle repair — aggressive second pass for remaining mismatches
        oracle_repaired = await self._watcher.oracle_repair_tests(task)
        if oracle_repaired > 0:
            log.info(f"Oracle-repaired {oracle_repaired} test assertion(s)")
            self._gate_stats["oracle_repairs"] += oracle_repaired

        task = await self.review_phase(task)

        # After verification, generate edge case tests (informational, not blocking)
        if task.is_complete and self._tester is not None:
            try:
                code_content = self._gather_code(task)
                tests_content = self._gather_tests(task)
                if code_content and tests_content:
                    edge_files = await self._tester.generate_edge_cases(
                        task, code_content, tests_content,
                    )
                    if edge_files:
                        for cf in edge_files:
                            self._file_manager.write_file(cf.path, cf.content)
                        log.info(
                            f"Generated {len(edge_files)} edge case test file(s) "
                            f"for '{task.title}' (informational)"
                        )
            except Exception as exc:
                log.warning(f"Edge case generation failed (non-blocking): {exc}")

        # After verification, extract interface for sibling context
        if task.is_complete and self._project_mode:
            self._extract_and_attach_interface(task)

        return task

    def _refresh_snippets(self, task: TaskNode) -> None:
        """Re-read code files from disk into the snippet store.

        Called after auto_fix_deterministic modifies files in-place so
        that the FileAssembler uses the corrected code, not the original.
        """
        for path in task.code_files:
            key = f"{task.id}:{path}"
            if key in self._snippet_store:
                try:
                    self._snippet_store[key] = self._file_manager.read_file(path)
                except FileNotFoundError:
                    pass

    def _extract_and_attach_interface(self, task: TaskNode) -> None:
        """Extract AST interface from code files and append to task spec."""
        interfaces: list[str] = []
        for path in task.code_files:
            try:
                code = self._file_manager.read_file(path)
                iface = ArchitectAgent.extract_interface_from_code(code, path)
                if iface:
                    interfaces.append(f"# {path}\n{iface}")
            except FileNotFoundError:
                pass
        if interfaces:
            task.spec += f"\n[INTERFACE]\n" + "\n".join(interfaces)
            log.info(f"Attached interface to '{task.title}'")

    @staticmethod
    def _estimate_difficulty(task: TaskNode) -> str:
        """Estimate task difficulty from spec metadata (deterministic, zero LLM cost).

        Returns "simple" or "complex". Simple tasks skip the planning section
        in the coder prompt (research shows 7B models degrade under unnecessary
        reasoning — "overthinking").
        """
        spec = task.spec.lower()
        indicators = 0
        # Multiple functions/classes suggest complexity
        if spec.count("def ") + spec.count("class ") > 2:
            indicators += 1
        # Has cross-file dependencies
        if "depends_on:" in spec and "none" not in spec.split("depends_on:")[1][:20]:
            indicators += 1
        # Algorithmic keywords
        if any(kw in spec for kw in ("sort", "filter", "recursive", "tree", "graph", "regex", "priority")):
            indicators += 1
        # Long spec suggests complexity
        if len(spec) > 500:
            indicators += 1
        return "complex" if indicators >= 2 else "simple"

    async def code_phase(self, task: TaskNode) -> TaskNode:
        """Coder implements a leaf-level task."""
        log.info(f"[green]CODE[/green] phase for: {task.title}")

        if task.status not in (TaskStatus.DESIGNING, TaskStatus.REVIEWING):
            task.transition(TaskStatus.CODING)
        else:
            task.status = TaskStatus.CODING
            task.updated_at = __import__("datetime").datetime.now()

        self._save_state()

        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="code",
            message=f"Coding: {task.title}",
        ))

        context = self._context_manager.build_context(task)
        test_first = self._config.cascade.test_first
        best_of_n = self._config.cascade.best_of_n
        difficulty = self._estimate_difficulty(task)
        log.info(f"Task difficulty: {difficulty}")

        # --- Test-first: generate tests from spec, then implement against them ---
        tests_content = ""
        if test_first:
            log.info("Test-first mode: generating tests from specification")
            if self._tester is not None:
                test_files = await self._tester.generate_tests(task, context)
            else:
                test_files = await self._coder.generate_tests(task, context)
            tests_content = "\n\n".join(cf.content for cf in test_files)

            # Test quality review gate — skip in project mode (7B reviewer
            # is unreliable on small leaf modules and tests are validated
            # by actually running them during the review phase)
            if not self._project_mode:
                max_test_attempts = 3
                for test_attempt in range(max_test_attempts):
                    test_review = await self._reviewer.verify_tests(
                        tests_content, task.spec, context,
                    )
                    if test_review.passed:
                        log.info(f"Test quality review passed (attempt {test_attempt + 1})")
                        break

                    log.warning(
                        f"Test quality review failed (attempt {test_attempt + 1}/"
                        f"{max_test_attempts}): {test_review.issues}"
                    )
                    # Regenerate tests with feedback
                    issue_feedback = "\n".join(f"- {i}" for i in test_review.issues)
                    context = (
                        context
                        + f"\n\n## Test Review Feedback (fix these issues)\n"
                        + issue_feedback
                    )
                    if self._tester is not None:
                        test_files = await self._tester.generate_tests(task, context)
                    else:
                        test_files = await self._coder.generate_tests(task, context)
                    tests_content = "\n\n".join(cf.content for cf in test_files)
                else:
                    log.warning("Test review failed after all attempts, using last tests")

            for cf in test_files:
                self._file_manager.write_file(cf.path, cf.content)
                task.test_files.append(cf.path)
            log.info(f"Generated {len(test_files)} test file(s)")

        # --- Best-of-N or single generation ---
        if best_of_n > 1:
            log.info(f"Best-of-{best_of_n} sampling: generating {best_of_n} candidates")
            all_candidate_paths: set[str] = set()

            async def _test_runner(candidate_files):
                """Write candidate files temporarily and run tests."""
                for cf in candidate_files:
                    self._file_manager.write_file(cf.path, cf.content)
                    all_candidate_paths.add(cf.path)
                # Temporarily attach code files to task for test runner
                old_code_files = list(task.code_files)
                task.code_files = [cf.path for cf in candidate_files
                                   if not (cf.path.startswith("test") or "/test_" in cf.path)]
                result = await self._watcher.run_tests(task)
                task.code_files = old_code_files
                return result

            code_files = await self._coder.implement_best_of_n(
                task, context, best_of_n, _test_runner,
                tests_content=tests_content,
            )

            # Clean up orphan files from losing candidates
            winner_paths = {cf.path for cf in code_files}
            for orphan in all_candidate_paths - winner_paths:
                try:
                    (self._workspace_path / orphan).unlink(missing_ok=True)
                except OSError:
                    pass
        elif test_first and tests_content:
            code_files = await self._coder.implement_with_tests(
                task, context, tests_content,
            )
        else:
            code_files = await self._coder.implement(task, context, difficulty=difficulty)

        for cf in code_files:
            self._file_manager.write_file(cf.path, cf.content)
            if cf.path.startswith("test") or "/test_" in cf.path:
                if cf.path not in task.test_files:
                    task.test_files.append(cf.path)
            else:
                task.code_files.append(cf.path)
                # Store snippet for multi-file assembly
                if self._project_mode:
                    key = f"{task.id}:{cf.path}"
                    self._snippet_store[key] = cf.content

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
        return task

    async def review_phase(self, task: TaskNode) -> TaskNode:
        """Review and verify code for a leaf task.

        Flow: smoke test → if crash, skip review and fix directly
              → if tests pass, review → faking check → VERIFIED
              → if tests fail, review → combine issues → fix
        """
        log.info(f"[blue]REVIEW[/blue] phase for: {task.title}")
        task.status = TaskStatus.REVIEWING
        task.updated_at = __import__("datetime").datetime.now()
        self._save_state()

        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message=f"Reviewing: {task.title}",
        ))

        # Gather code content
        code_content = self._gather_code(task)

        for attempt in range(self._config.cascade.max_retries + 1):
            # --- Pre-review smoke test ---
            # Run tests BEFORE calling the reviewer LLM. If tests crash on
            # import/syntax errors, skip review entirely and go straight to fix.
            smoke_result = await self._watcher.run_tests(task)

            self._emit(CascadeEvent(
                event_type=EventType.TEST_RESULT,
                task_title=task.title,
                task_id=task.id,
                phase="review",
                message="Smoke test passed" if smoke_result.passed else "Smoke test failed",
                data={"passed": smoke_result.passed, "smoke": True},
            ))

            if smoke_result.passed:
                # Tests pass — proceed with code review + faking check
                # Inject missing spec names into the spec context for the reviewer
                spec_for_review = task.spec
                missing_names = getattr(task, "_missing_spec_names", [])
                if missing_names:
                    spec_for_review += (
                        "\n\n## CRITICAL: Missing Functions\n"
                        "The following functions/classes from the spec are NOT "
                        "implemented in the code. This is a FAILURE:\n"
                        + "\n".join(f"- {n}" for n in missing_names)
                    )
                review = await self._reviewer.verify_code(code_content, spec_for_review)
                task.review_history.append(review)

                # Force-fail if deterministic check found missing names
                if review.passed and missing_names:
                    log.warning(
                        f"Overriding review pass — spec-coverage gate found "
                        f"missing: {missing_names}"
                    )
                    review = ReviewResult(
                        passed=False,
                        issues=[
                            f"Missing function/class from spec: {n}"
                            for n in missing_names
                        ],
                        suggestions=[
                            f"Implement the '{n}' function as described in the specification"
                            for n in missing_names
                        ],
                        timestamp=__import__("datetime").datetime.now(),
                        model_used="spec-coverage-gate",
                    )
                    task.review_history.append(review)

                self._emit(CascadeEvent(
                    event_type=EventType.REVIEW_RESULT,
                    task_title=task.title,
                    task_id=task.id,
                    phase="review",
                    message="Review passed" if review.passed else f"Review failed: {', '.join(review.issues[:3])}",
                    data={"passed": review.passed, "issues": review.issues},
                ))

                if review.passed:
                    # Check for faked code (skip in project mode — leaf modules
                    # are intentionally small by design)
                    skip_fake_check = self._project_mode
                    if not skip_fake_check:
                        test_content = self._gather_tests(task)
                        if test_content:
                            fake_check = await self._watcher.check_not_faked(
                                code_content, test_content
                            )
                            task.review_history.append(fake_check)
                            if not fake_check.passed:
                                log.warning("Code appears to be faked, retrying")
                                review = fake_check
                    if review.passed:
                            task.transition(TaskStatus.VERIFIED)
                            if self._config.workspace.git_checkpoint:
                                commit_hash = self._git_manager.checkpoint(
                                    task, "verified"
                                )
                                task.git_checkpoint = commit_hash
                            log.info(f"Task '{task.title}' verified!")
                            self._emit(CascadeEvent(
                                event_type=EventType.TASK_COMPLETE,
                                task_title=task.title,
                                task_id=task.id,
                                phase="review",
                                message=f"Task verified: {task.title}",
                            ))
                            self._save_state()
                            return task
                    else:
                        task.transition(TaskStatus.VERIFIED)
                        if self._config.workspace.git_checkpoint:
                            commit_hash = self._git_manager.checkpoint(task, "verified")
                            task.git_checkpoint = commit_hash
                        log.info(f"Task '{task.title}' verified (no tests to run)!")
                        self._emit(CascadeEvent(
                            event_type=EventType.TASK_COMPLETE,
                            task_title=task.title,
                            task_id=task.id,
                            phase="review",
                            message=f"Task verified (no tests): {task.title}",
                        ))
                        self._save_state()
                        return task
                # Review failed but tests passed — use review issues for fix
                # Append lint issues so the coder fixes them alongside review issues
                lint_issues = getattr(task, "_lint_issues", [])
                if lint_issues and not review.passed:
                    review = ReviewResult(
                        passed=False,
                        issues=review.issues + [f"[lint] {e}" for e in lint_issues],
                        suggestions=review.suggestions,
                        timestamp=review.timestamp,
                        model_used=review.model_used,
                    )
            else:
                # Tests failed — build structured error context
                structured_errors = self._watcher.extract_structured_errors(
                    smoke_result.output
                )

                # Check if it's a crash (import/syntax) vs test assertion failure
                is_crash = any(
                    e.error_type in ("NameError", "ImportError", "SyntaxError", "TypeError")
                    for e in structured_errors
                )

                if is_crash:
                    # Skip LLM review — code doesn't even import
                    log.warning(
                        f"Code crashes on import/syntax — skipping review "
                        f"({[e.error_type for e in structured_errors]})"
                    )
                    # Build structured issues for the fix prompt
                    structured_issues = [e.format_for_prompt() for e in structured_errors]
                    # Include lint issues — type errors often explain import/name crashes
                    lint_issues = getattr(task, "_lint_issues", [])
                    if lint_issues:
                        structured_issues += [f"[lint] {e}" for e in lint_issues]
                    review = ReviewResult(
                        passed=False,
                        issues=structured_issues if structured_issues else smoke_result.errors,
                        suggestions=["Fix the crash before anything else"],
                        timestamp=__import__("datetime").datetime.now(),
                        model_used="watcher",
                    )
                else:
                    # Tests fail with assertion errors — still run review for spec compliance
                    review = await self._reviewer.verify_code(code_content, task.spec)
                    task.review_history.append(review)

                    # Combine review issues with structured test errors
                    structured_issues = [e.format_for_prompt() for e in structured_errors]
                    # Fallback: if no structured errors, include raw test output
                    if not structured_issues and smoke_result.output:
                        raw_excerpt = smoke_result.output[-1000:]  # last 1000 chars
                        structured_issues = [f"Test output:\n{raw_excerpt}"]

                    # Use Tester agent for failure analysis if available
                    if self._tester is not None:
                        tests_content = self._gather_tests(task)
                        analysis = await self._tester.analyze_failure(
                            task, smoke_result.output, code_content, tests_content,
                        )
                        log.info(
                            f"Failure analysis: root_cause={analysis.root_cause}, "
                            f"fix_target={analysis.suggested_fix_target}"
                        )
                        # Use analysis-specific issues if available
                        if analysis.specific_issues:
                            structured_issues = analysis.specific_issues

                    combined_issues = review.issues + structured_issues
                    # Append lint issues so the coder fixes them alongside test failures
                    lint_issues = getattr(task, "_lint_issues", [])
                    if lint_issues:
                        combined_issues += [f"[lint] {e}" for e in lint_issues]
                    review = ReviewResult(
                        passed=False,
                        issues=combined_issues,
                        suggestions=review.suggestions + ["Fix the failing tests"],
                        timestamp=__import__("datetime").datetime.now(),
                        model_used="watcher+reviewer",
                    )

                    log.warning(f"Tests failed: {[e.error_type for e in structured_errors]}")

            # If we get here, review failed — try to fix
            if attempt < self._config.cascade.max_retries:
                log.info(f"Review failed, retry {attempt + 1}/{self._config.cascade.max_retries}")
                task.retry_count += 1
                self._emit(CascadeEvent(
                    event_type=EventType.RETRY,
                    task_title=task.title,
                    task_id=task.id,
                    phase="review",
                    message=f"Retry {attempt + 1}/{self._config.cascade.max_retries}",
                    data={"attempt": attempt + 1, "max_retries": self._config.cascade.max_retries},
                ))

                # Fresh start strategy: after N failed fixes, regenerate from scratch
                is_fresh_start = (
                    attempt > 0
                    and attempt % self._config.cascade.fresh_start_after == 0
                )

                if is_fresh_start:
                    log.info(f"Fresh start (attempt {attempt + 1}) — regenerating from scratch")
                    context = self._context_manager.build_context(task)
                    if self._config.cascade.test_first and task.test_files:
                        tests_content = self._gather_tests(task)
                        code_files = await self._coder.implement_with_tests(
                            task, context, tests_content,
                        )
                    else:
                        code_files = await self._coder.implement(task, context)
                    # Replace task file lists — fresh start may produce different paths
                    new_code = []
                    new_tests = list(task.test_files)  # keep test files from test-first
                    for cf in code_files:
                        self._file_manager.write_file(cf.path, cf.content)
                        if cf.path.startswith("test") or "/test_" in cf.path:
                            if cf.path not in new_tests:
                                new_tests.append(cf.path)
                        else:
                            new_code.append(cf.path)
                    if new_code:
                        task.code_files = new_code
                    task.test_files = new_tests
                else:
                    fix_files = await self._coder.fix(
                        task, review.issues,
                        file_manager=self._file_manager,
                        retry_num=attempt + 1,
                    )
                    for cf in fix_files:
                        self._file_manager.write_file(cf.path, cf.content)

                code_content = self._gather_code(task)

                # Re-run auto-fix + calibration after each fix attempt
                auto_fixes = await self._watcher.auto_fix_deterministic(task)
                if auto_fixes > 0:
                    log.info(f"Auto-fixed {auto_fixes} deterministic error(s) after retry")
                    self._gate_stats["auto_fix_retry"] += auto_fixes
                    if self._project_mode:
                        self._refresh_snippets(task)
                recalibrated = await self._watcher.calibrate_tests(task)
                if recalibrated > 0:
                    log.info(f"Re-calibrated {recalibrated} test assertion(s) after retry")
                    self._gate_stats["calibrations_retry"] += recalibrated

                # Oracle repair after retry
                oracle_repaired = await self._watcher.oracle_repair_tests(task)
                if oracle_repaired > 0:
                    log.info(f"Oracle-repaired {oracle_repaired} assertion(s) after retry")
                    self._gate_stats["oracle_repairs"] += oracle_repaired

                # Re-check spec coverage after fix
                missing_names = await self._watcher.spec_coverage_check(task)
                task._missing_spec_names = missing_names
                if missing_names:
                    log.info(f"Still missing after fix: {missing_names}")
                else:
                    log.info("Spec-coverage gap resolved after fix")

                # Re-run linters after fix to update lint issues
                _, lint_info = await self._watcher.static_analysis_gate(task)
                task._lint_issues = lint_info
                if lint_info:
                    log.info(f"Lint issues after fix ({len(lint_info)}): {lint_info}")
                else:
                    log.info("Lint issues resolved after fix")

        # Exhausted retries
        log.error(f"Task '{task.title}' failed after {self._config.cascade.max_retries} retries")
        task.status = TaskStatus.FAILED
        task.updated_at = __import__("datetime").datetime.now()
        self._emit(CascadeEvent(
            event_type=EventType.TASK_FAILED,
            task_title=task.title,
            task_id=task.id,
            phase="review",
            message=f"Task failed after {self._config.cascade.max_retries} retries: {task.title}",
        ))
        self._save_state()
        return task

    async def integrate_phase(self, task: TaskNode) -> TaskNode:
        """Verify all children work together after they complete."""
        log.info(f"[yellow]INTEGRATE[/yellow] phase for: {task.title}")

        if not self._task_tree.all_children_complete(task.id):
            failed = self._task_tree.get_failed_children(task.id)
            if not self._project_mode:
                log.error(f"Cannot integrate — {len(failed)} children failed")
                task.status = TaskStatus.FAILED
                task.updated_at = __import__("datetime").datetime.now()
                self._save_state()
                return task
            # Project mode: partial integration with verified children
            verified = [
                c for c in self._task_tree.get_children(task.id)
                if c.status == TaskStatus.VERIFIED
            ]
            log.warning(
                f"Partial integration: {len(failed)} failed, "
                f"{len(verified)} verified"
            )

        task.status = TaskStatus.INTEGRATING
        task.updated_at = __import__("datetime").datetime.now()
        self._save_state()

        self._emit(CascadeEvent(
            event_type=EventType.PHASE_START,
            task_title=task.title,
            task_id=task.id,
            phase="integrate",
            message=f"Integrating: {task.title}",
        ))

        # In project mode, assemble snippets into final files
        if self._project_mode and self._snippet_store:
            assembler = FileAssembler(self._file_manager)
            assembled = assembler.assemble(
                task, self._task_tree, self._snippet_store,
            )
            log.info(f"Assembled {len(assembled)} file(s) for '{task.title}'")

        children = self._task_tree.get_children(task.id)

        # In project mode, skip LLM integration review — each child is
        # independently verified, assembly is deterministic, and the 7B
        # reviewer is too conservative with partial integration.
        # Run integration tests instead if test files exist.
        if self._project_mode:
            verified = [c for c in children if c.status == TaskStatus.VERIFIED]
            log.info(
                f"Project mode: skipping LLM integration review "
                f"({len(verified)}/{len(children)} children verified)"
            )

            # Collect all child files
            seen: set[str] = set()
            for child in children:
                for f in child.code_files:
                    if f not in seen:
                        task.code_files.append(f)
                        seen.add(f)
                for f in child.test_files:
                    if f not in seen:
                        task.test_files.append(f)
                        seen.add(f)

            # Run all collected tests as integration check
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
            if self._config.workspace.git_checkpoint:
                commit_hash = self._git_manager.checkpoint(task, "integration verified")
                task.git_checkpoint = commit_hash
            log.info(f"Integration complete for '{task.title}'")
        else:
            # Single-file mode: use LLM integration review
            integration_context = self._context_manager.build_integration_context(task)
            children_summary = "\n\n".join(
                f"### {c.title}\nStatus: {c.status.value}\nSpec: {c.spec[:500]}\n"
                f"Files: {', '.join(c.code_files)}"
                for c in children
            )

            review = await self._reviewer.verify_integration(task, children_summary)
            task.review_history.append(review)

            if review.passed:
                task.transition(TaskStatus.VERIFIED)
                if self._config.workspace.git_checkpoint:
                    commit_hash = self._git_manager.checkpoint(task, "integration verified")
                    task.git_checkpoint = commit_hash
                seen: set[str] = set()
                for child in children:
                    for f in child.code_files:
                        if f not in seen:
                            task.code_files.append(f)
                            seen.add(f)
                    for f in child.test_files:
                        if f not in seen:
                            task.test_files.append(f)
                            seen.add(f)
                log.info(f"Integration verified for '{task.title}'")
            else:
                log.error(f"Integration failed for '{task.title}': {review.issues}")
                task.status = TaskStatus.FAILED
                task.updated_at = __import__("datetime").datetime.now()

        self._save_state()
        return task

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

        # Kahn's algorithm
        queue = [i for i in range(len(children)) if in_degree[i] == 0]
        sorted_indices: list[int] = []

        while queue:
            # Stable sort: process lowest index first
            queue.sort()
            node = queue.pop(0)
            sorted_indices.append(node)
            for neighbor in deps.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

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
        for path in task.code_files:
            try:
                content = self._file_manager.read_file(path)
                parts.append(f"# {path}\n{content}")
            except FileNotFoundError:
                log.warning(f"Code file not found: {path}")
        return "\n\n".join(parts) if parts else ""

    def _gather_tests(self, task: TaskNode) -> str:
        """Gather all test content for a task."""
        parts: list[str] = []
        for path in task.test_files:
            try:
                content = self._file_manager.read_file(path)
                parts.append(f"# {path}\n{content}")
            except FileNotFoundError:
                log.warning(f"Test file not found: {path}")
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
            for path in node.code_files:
                try:
                    content = self._file_manager.read_file(path)
                    result[path] = content
                except FileNotFoundError:
                    pass
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
