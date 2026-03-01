"""Tests for PMCA v2 features: test-first, best-of-N, fresh start, static gate, SCoT."""

import ast
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmca.agents.coder import CoderAgent
from pmca.agents.tester import TesterAgent
from pmca.agents.watcher import WatcherAgent
from pmca.models.config import AgentRole, CascadeConfig, Config, ModelConfig
from pmca.models.manager import ModelManager
from pmca.agents.reviewer import ReviewerAgent
from pmca.orchestrator import Orchestrator
from pmca.prompts import coder as coder_prompts
from pmca.prompts import reviewer as reviewer_prompts
from pmca.prompts import tester as tester_prompts
from pmca.tasks.state import CodeFile, FailureAnalysis, ReviewResult, TaskStatus, TestResult
from pmca.tasks.tree import TaskNode
from pmca.utils.context import ContextManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return Config(
        models={
            AgentRole.ARCHITECT: ModelConfig(name="test:14b", temperature=0.3),
            AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.2),
            AgentRole.REVIEWER: ModelConfig(name="test:14b", temperature=0.1),
            AgentRole.WATCHER: ModelConfig(name="test:7b", temperature=0.1),
            AgentRole.TESTER: ModelConfig(name="test:14b", temperature=0.2),
        }
    )


@pytest.fixture
def mock_manager(config):
    manager = ModelManager(config)
    manager.generate = AsyncMock()
    manager.ensure_loaded = AsyncMock(return_value="test:7b")
    return manager


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Step 1: Config tests
# ---------------------------------------------------------------------------

class TestCascadeConfigDefaults:
    def test_default_values(self):
        cfg = CascadeConfig()
        assert cfg.best_of_n == 1
        assert cfg.fresh_start_after == 3
        assert cfg.test_first is False

    def test_from_yaml_parses_new_fields(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
cascade:
  max_depth: 2
  max_retries: 5
  best_of_n: 3
  fresh_start_after: 2
  test_first: true
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.cascade.best_of_n == 3
        assert cfg.cascade.fresh_start_after == 2
        assert cfg.cascade.test_first is True
        assert cfg.cascade.max_retries == 5

    def test_from_yaml_defaults_when_missing(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
cascade:
  max_depth: 2
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.cascade.best_of_n == 1
        assert cfg.cascade.fresh_start_after == 3
        assert cfg.cascade.test_first is False


# ---------------------------------------------------------------------------
# Step 2: Prompt tests
# ---------------------------------------------------------------------------

class TestNewPrompts:
    def test_generate_tests_prompt_exists(self):
        assert hasattr(coder_prompts, "GENERATE_TESTS_PROMPT")
        prompt = coder_prompts.GENERATE_TESTS_PROMPT
        assert "{spec}" in prompt
        assert "{context}" in prompt
        assert "{suggested_module}" in prompt
        assert "{suggested_test_path}" in prompt

    def test_implement_with_tests_prompt_exists(self):
        assert hasattr(coder_prompts, "IMPLEMENT_WITH_TESTS_PROMPT")
        prompt = coder_prompts.IMPLEMENT_WITH_TESTS_PROMPT
        assert "{spec}" in prompt
        assert "{tests}" in prompt
        assert "{context}" in prompt
        assert "{suggested_path}" in prompt

    def test_implement_prompt_has_icot_section(self):
        prompt = coder_prompts.IMPLEMENT_PROMPT
        assert "Intention (REQUIRED" in prompt
        assert "Specification:" in prompt
        assert "Idea:" in prompt

    def test_implement_simple_prompt_no_planning(self):
        prompt = coder_prompts.IMPLEMENT_SIMPLE_PROMPT
        assert "Intention" not in prompt
        assert "{spec}" in prompt
        assert "{suggested_path}" in prompt

    def test_implement_with_tests_prompt_has_icot_section(self):
        prompt = coder_prompts.IMPLEMENT_WITH_TESTS_PROMPT
        assert "Intention (REQUIRED" in prompt
        assert "Specification:" in prompt

    def test_generate_tests_prompt_format(self):
        """Ensure the prompt can be formatted without errors."""
        result = coder_prompts.GENERATE_TESTS_PROMPT.format(
            spec="Build a calculator",
            context="No context",
            suggested_module="calculator",
            suggested_test_path="tests/test_calculator.py",
        )
        assert "Build a calculator" in result
        assert "calculator" in result

    def test_implement_with_tests_prompt_format(self):
        result = coder_prompts.IMPLEMENT_WITH_TESTS_PROMPT.format(
            spec="Build a calculator",
            tests="def test_add(): assert add(1,2) == 3",
            context="No context",
            suggested_path="src/calculator.py",
        )
        assert "Build a calculator" in result
        assert "test_add" in result


# ---------------------------------------------------------------------------
# Step 2d: Coder agent method tests
# ---------------------------------------------------------------------------

class TestCoderNewMethods:
    @pytest.mark.asyncio
    async def test_generate_tests(self, mock_manager):
        mock_manager.generate.return_value = (
            "```python\n# filepath: tests/test_calc.py\n"
            "def test_add():\n    assert add(1, 2) == 3\n```"
        )
        coder = CoderAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function that adds two numbers"

        files = await coder.generate_tests(task, context="")
        assert len(files) == 1
        assert "test" in files[0].path
        assert "test_add" in files[0].content

    @pytest.mark.asyncio
    async def test_implement_with_tests(self, mock_manager):
        mock_manager.generate.return_value = (
            "```python\n# filepath: src/calc.py\n"
            "def add(a, b):\n    return a + b\n```"
        )
        coder = CoderAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"

        files = await coder.implement_with_tests(
            task, context="", tests_content="def test_add(): assert add(1,2)==3"
        )
        assert len(files) == 1
        assert "add" in files[0].content

    @pytest.mark.asyncio
    async def test_implement_with_tests_project_mode(self, mock_manager):
        mock_manager.generate.return_value = (
            "```python\n# filepath: src/calc.py\n"
            "def add(a, b):\n    return a + b\n```"
        )
        coder = CoderAgent(mock_manager, project_mode=True)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"

        files = await coder.implement_with_tests(task, context="", tests_content="tests")
        assert len(files) == 1
        # Check that PROJECT_IMPORT_RULES was included in the system prompt
        call_args = mock_manager.generate.call_args
        assert "Multi-File Project" in call_args.kwargs.get("system", "")

    @pytest.mark.asyncio
    async def test_implement_best_of_n_picks_best(self, mock_manager):
        """Best-of-N should pick the candidate with most passing tests."""
        responses = [
            "```python\n# filepath: src/calc.py\ndef add(a, b): return 0\n```",
            "```python\n# filepath: src/calc.py\ndef add(a, b): return a + b\n```",
            "```python\n# filepath: src/calc.py\ndef add(a, b): return a - b\n```",
        ]
        mock_manager.generate.side_effect = responses

        coder = CoderAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"
        task.test_files = ["tests/test_calc.py"]

        # Mock test runner: second candidate gets perfect score
        call_count = 0
        async def mock_runner(files):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return TestResult(passed=True, total=3, failures=0, output="", errors=[])
            return TestResult(passed=False, total=3, failures=2, output="", errors=["fail"])

        result = await coder.implement_best_of_n(task, "", 3, mock_runner)
        assert "a + b" in result[0].content
        # Should have stopped after finding perfect score (candidate 2)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_implement_best_of_n_no_tests_returns_first(self, mock_manager):
        """Without test files, best-of-N returns the first candidate."""
        mock_manager.generate.return_value = (
            "```python\n# filepath: src/calc.py\ndef add(a,b): return a+b\n```"
        )
        coder = CoderAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"
        # No test_files set

        async def mock_runner(files):
            raise AssertionError("Should not be called")

        result = await coder.implement_best_of_n(task, "", 3, mock_runner)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Step 5: Static analysis gate tests
# ---------------------------------------------------------------------------

class TestStaticAnalysisGate:
    @pytest.mark.asyncio
    async def test_valid_code_no_errors(self, mock_manager, workspace):
        watcher = WatcherAgent(mock_manager, workspace)
        (workspace / "src").mkdir()
        (workspace / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")

        task = TaskNode(title="calc")
        task.code_files = ["src/calc.py"]
        task.test_files = []

        blocking, informational = await watcher.static_analysis_gate(task)
        assert blocking == []
        assert informational == []

    @pytest.mark.asyncio
    async def test_syntax_error_detected(self, mock_manager, workspace):
        watcher = WatcherAgent(mock_manager, workspace)
        (workspace / "src").mkdir()
        (workspace / "src" / "bad.py").write_text("def add(a, b\n    return a + b\n")

        task = TaskNode(title="bad")
        task.code_files = ["src/bad.py"]
        task.test_files = []

        blocking, informational = await watcher.static_analysis_gate(task)
        assert len(blocking) == 1
        assert "SyntaxError" in blocking[0]
        assert "bad.py" in blocking[0]

    @pytest.mark.asyncio
    async def test_checks_both_code_and_test_files(self, mock_manager, workspace):
        watcher = WatcherAgent(mock_manager, workspace)
        (workspace / "src").mkdir()
        (workspace / "tests").mkdir()
        (workspace / "src" / "good.py").write_text("x = 1\n")
        (workspace / "tests" / "test_bad.py").write_text("def test(:\n    pass\n")

        task = TaskNode(title="mixed")
        task.code_files = ["src/good.py"]
        task.test_files = ["tests/test_bad.py"]

        blocking, informational = await watcher.static_analysis_gate(task)
        assert len(blocking) == 1
        assert "test_bad.py" in blocking[0]

    @pytest.mark.asyncio
    async def test_missing_file_skipped(self, mock_manager, workspace):
        watcher = WatcherAgent(mock_manager, workspace)

        task = TaskNode(title="missing")
        task.code_files = ["src/nonexistent.py"]
        task.test_files = []

        blocking, informational = await watcher.static_analysis_gate(task)
        assert blocking == []

    @pytest.mark.asyncio
    async def test_non_python_files_skipped(self, mock_manager, workspace):
        watcher = WatcherAgent(mock_manager, workspace)
        (workspace / "index.html").write_text("<html><body></body></html>")

        task = TaskNode(title="html")
        task.code_files = ["index.html"]
        task.test_files = []

        blocking, informational = await watcher.static_analysis_gate(task)
        assert blocking == []


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------

@pytest.fixture
def v2_config():
    """Config with all v2 features enabled."""
    return Config(
        models={
            AgentRole.ARCHITECT: ModelConfig(name="test:14b", temperature=0.3),
            AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.2),
            AgentRole.REVIEWER: ModelConfig(name="test:14b", temperature=0.1),
            AgentRole.WATCHER: ModelConfig(name="test:7b", temperature=0.1),
            AgentRole.TESTER: ModelConfig(name="test:14b", temperature=0.2),
        },
        cascade=CascadeConfig(
            max_depth=1,
            max_retries=5,
            best_of_n=1,  # Keep 1 for unit tests (best-of-N tested separately)
            fresh_start_after=3,
            test_first=True,
        ),
    )


@pytest.fixture
def v2_orchestrator(v2_config, workspace):
    orch = Orchestrator(v2_config, workspace)
    orch._model_manager.generate = AsyncMock()
    orch._model_manager.ensure_loaded = AsyncMock(return_value="test:7b")
    orch._model_manager.unload_current = AsyncMock()
    orch._model_manager.close = AsyncMock()
    orch._model_manager.is_ollama_running = AsyncMock(return_value=True)
    return orch


class TestTestFirstCodePhase:
    @pytest.mark.asyncio
    async def test_code_phase_generates_tests_first(self, v2_orchestrator):
        """When test_first=True, code_phase should call generate_tests then implement_with_tests."""
        orch = v2_orchestrator

        # Mock tester (used when configured) and coder methods
        orch._tester.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="def test_add(): assert add(1,2)==3"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        # Mock 14B reviewer approving the tests
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        result = await orch.code_phase(root)

        orch._tester.generate_tests.assert_called_once()
        orch._reviewer.verify_tests.assert_called_once()
        orch._coder.implement_with_tests.assert_called_once()
        assert "tests/test_calc.py" in result.test_files
        assert "src/calc.py" in result.code_files

    @pytest.mark.asyncio
    async def test_code_phase_disabled_test_first(self, config, workspace):
        """When test_first=False, code_phase uses normal implement()."""
        orch = Orchestrator(config, workspace)
        orch._model_manager.generate = AsyncMock()
        orch._model_manager.ensure_loaded = AsyncMock(return_value="test:7b")
        orch._model_manager.close = AsyncMock()

        orch._coder.implement = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
            CodeFile(path="tests/test_calc.py", content="def test(): pass"),
        ])

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)
        orch._coder.implement.assert_called_once()


class TestFreshStartStrategy:
    @pytest.mark.asyncio
    async def test_fresh_start_triggered_at_threshold(self, v2_orchestrator):
        """Fresh start should regenerate code after fresh_start_after attempts."""
        orch = v2_orchestrator

        # Set up task with existing files
        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.code_files = ["src/calc.py"]
        root.test_files = ["tests/test_calc.py"]

        # Write actual files so the watcher can find them
        (orch._workspace_path / "src").mkdir(parents=True)
        (orch._workspace_path / "tests").mkdir(parents=True)
        (orch._workspace_path / "src" / "calc.py").write_text("def add(a,b): return 0")
        (orch._workspace_path / "tests" / "test_calc.py").write_text(
            "from src.calc import add\ndef test_add(): assert add(1,2)==3"
        )

        # Mock: tests always fail, review always fails
        orch._watcher.run_tests = AsyncMock(return_value=TestResult(
            passed=False, total=1, failures=1, output="FAILED test_add\nE assert 0 == 3",
            errors=["AssertionError"],
        ))
        orch._reviewer.verify_code = AsyncMock(return_value=ReviewResult(
            passed=False, issues=["Wrong result"],
            suggestions=[], timestamp=__import__("datetime").datetime.now(), model_used="test",
        ))
        orch._coder.fix = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a,b): return 0"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a,b): return a+b"),
        ])
        orch._watcher.auto_fix_deterministic = AsyncMock(return_value=0)
        orch._watcher.calibrate_tests = AsyncMock(return_value=0)
        orch._watcher.spec_coverage_check = AsyncMock(return_value=[])
        orch._tester.analyze_failure = AsyncMock(return_value=FailureAnalysis(
            root_cause="code_bug",
            explanation="add() returns 0",
            suggested_fix_target="code",
            specific_issues=["add returns hardcoded 0"],
        ))

        root.status = TaskStatus.REVIEWING
        root.updated_at = __import__("datetime").datetime.now()
        await orch.review_phase(root)

        # fresh_start_after=3, so attempt index 3 triggers fresh start
        # implement_with_tests should have been called at least once
        assert orch._coder.implement_with_tests.call_count >= 1


class TestBestOfNCodePhase:
    @pytest.mark.asyncio
    async def test_best_of_n_used_when_configured(self, workspace):
        """When best_of_n > 1, code_phase should use implement_best_of_n."""
        cfg = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:14b", temperature=0.3),
                AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.2),
                AgentRole.REVIEWER: ModelConfig(name="test:14b", temperature=0.1),
                AgentRole.WATCHER: ModelConfig(name="test:7b", temperature=0.1),
            },
            cascade=CascadeConfig(
                max_depth=1,
                max_retries=3,
                best_of_n=3,
                test_first=False,
            ),
        )
        orch = Orchestrator(cfg, workspace)
        orch._model_manager.generate = AsyncMock()
        orch._model_manager.ensure_loaded = AsyncMock(return_value="test:7b")
        orch._model_manager.close = AsyncMock()

        orch._coder.implement_best_of_n = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a,b): return a+b"),
        ])

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)
        orch._coder.implement_best_of_n.assert_called_once()
        call_args = orch._coder.implement_best_of_n.call_args
        assert call_args[0][2] == 3  # n=3


class TestStaticGateInCodeLeaf:
    @pytest.mark.asyncio
    async def test_static_gate_called_in_code_leaf(self, v2_orchestrator):
        """_code_leaf should call static_analysis_gate."""
        orch = v2_orchestrator

        orch._tester.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="def test(): pass"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a,b): return a+b"),
        ])
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))
        orch._watcher.auto_fix_deterministic = AsyncMock(return_value=0)
        orch._watcher.static_analysis_gate = AsyncMock(return_value=([], []))
        orch._watcher.calibrate_tests = AsyncMock(return_value=0)
        orch._watcher.run_tests = AsyncMock(return_value=TestResult(
            passed=True, total=1, failures=0, output="1 passed", errors=[],
        ))
        orch._watcher.check_not_faked = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="test",
        ))
        orch._reviewer.verify_code = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="test",
        ))
        orch._tester.generate_edge_cases = AsyncMock(return_value=[])

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator"
        root.transition(TaskStatus.DESIGNING)

        await orch._code_leaf(root)
        orch._watcher.static_analysis_gate.assert_called_once()


# ---------------------------------------------------------------------------
# Test quality review gate (14B reviewer checks 7B tests)
# ---------------------------------------------------------------------------

class TestVerifyTestsPrompt:
    def test_prompt_exists(self):
        assert hasattr(reviewer_prompts, "VERIFY_TESTS_PROMPT")
        prompt = reviewer_prompts.VERIFY_TESTS_PROMPT
        assert "{spec}" in prompt
        assert "{tests}" in prompt
        assert "{context}" in prompt

    def test_prompt_checks_for_fakes(self):
        prompt = reviewer_prompts.VERIFY_TESTS_PROMPT
        assert "fake" in prompt.lower() or "trivial" in prompt.lower()
        assert "Missing imports" in prompt

    def test_prompt_format(self):
        result = reviewer_prompts.VERIFY_TESTS_PROMPT.format(
            spec="Build a calculator",
            tests="def test_add(): assert add(1,2) == 3",
            context="No context",
        )
        assert "Build a calculator" in result
        assert "test_add" in result


class TestReviewerVerifyTests:
    @pytest.mark.asyncio
    async def test_verify_tests_passes_good_tests(self, mock_manager):
        mock_manager.generate.return_value = '{"passed": true, "issues": [], "suggestions": []}'
        reviewer = ReviewerAgent(mock_manager)

        result = await reviewer.verify_tests(
            "def test_add(): assert add(1,2) == 3",
            "Add function",
        )
        assert result.passed is True
        assert result.issues == []

    @pytest.mark.asyncio
    async def test_verify_tests_rejects_bad_tests(self, mock_manager):
        mock_manager.generate.return_value = (
            '{"passed": false, "issues": ["test_add has no import for Board"], "suggestions": ["add import"]}'
        )
        reviewer = ReviewerAgent(mock_manager)

        result = await reviewer.verify_tests(
            "def test_add(): pass",
            "Add function",
        )
        assert result.passed is False
        assert len(result.issues) == 1


class TestTestReviewGateInCodePhase:
    @pytest.mark.asyncio
    async def test_review_gate_passes_on_first_try(self, v2_orchestrator):
        """When 14B reviewer approves tests, no retry needed."""
        orch = v2_orchestrator

        orch._tester.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="from calc import add\ndef test_add(): assert add(1,2)==3"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)
        orch._reviewer.verify_tests.assert_called_once()
        orch._tester.generate_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_gate_retries_on_failure(self, v2_orchestrator):
        """When 14B reviewer rejects tests, tester regenerates with feedback."""
        orch = v2_orchestrator

        orch._tester.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="def test_add(): pass"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        # First call rejects, second call accepts
        orch._reviewer.verify_tests = AsyncMock(side_effect=[
            ReviewResult(
                passed=False, issues=["test_add has no assertion"],
                suggestions=["Add real assertions"],
                timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
            ),
            ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
            ),
        ])

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)
        assert orch._tester.generate_tests.call_count == 2
        assert orch._reviewer.verify_tests.call_count == 2

    @pytest.mark.asyncio
    async def test_review_gate_uses_last_tests_after_max_attempts(self, v2_orchestrator):
        """After 3 failed reviews, use last tests anyway."""
        orch = v2_orchestrator

        orch._tester.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="def test_add(): pass"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        # Always reject
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=False, issues=["Still bad"],
            suggestions=[], timestamp=__import__("datetime").datetime.now(),
            model_used="reviewer",
        ))

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)
        # 1 initial generation + 3 retry regenerations = 4 total
        assert orch._tester.generate_tests.call_count == 4
        assert orch._reviewer.verify_tests.call_count == 3
        # Still wrote files and proceeded
        assert len(root.test_files) > 0


# ---------------------------------------------------------------------------
# Spec-Coverage Gate Tests
# ---------------------------------------------------------------------------

class TestSpecCoverageCheck:
    """Tests for the deterministic spec-coverage gate."""

    @pytest.fixture
    def watcher_with_workspace(self, tmp_path):
        mm = MagicMock()
        return WatcherAgent(mm, workspace_path=tmp_path), tmp_path

    def test_all_functions_present(self, watcher_with_workspace):
        """No missing names when code implements everything in spec."""
        watcher, ws = watcher_with_workspace
        code = "def filter_by_status(tasks): pass\ndef sort_by_priority(tasks): pass\n"
        (ws / "filters.py").write_text(code)

        task = TaskNode(title="filters", id="t1")
        task.spec = "Implement `filter_by_status` and `sort_by_priority`"
        task.code_files = ["filters.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert missing == []

    def test_missing_function_detected(self, watcher_with_workspace):
        """Detects function mentioned in spec but not in code."""
        watcher, ws = watcher_with_workspace
        code = "def sort_by_priority(tasks): pass\n"
        (ws / "filters.py").write_text(code)

        task = TaskNode(title="filters", id="t1")
        task.spec = "Implement `filter_by_status`, `filter_by_priority`, `sort_by_priority`"
        task.code_files = ["filters.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert "filter_by_status" in missing
        assert "filter_by_priority" in missing
        assert "sort_by_priority" not in missing

    def test_class_names_detected(self, watcher_with_workspace):
        """Detects PascalCase class names from spec."""
        watcher, ws = watcher_with_workspace
        code = "class TaskManager:\n    pass\n"
        (ws / "manager.py").write_text(code)

        task = TaskNode(title="manager", id="t1")
        task.spec = "Implement TaskManager and BoardService classes"
        task.code_files = ["manager.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert "TaskManager" not in missing
        assert "BoardService" in missing

    def test_no_spec_names_skips_check(self, watcher_with_workspace):
        """Returns empty list when no names can be extracted from spec."""
        watcher, ws = watcher_with_workspace
        code = "x = 1\n"
        (ws / "foo.py").write_text(code)

        task = TaskNode(title="foo", id="t1")
        task.spec = "Do something useful"
        task.code_files = ["foo.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert missing == []

    def test_comma_separated_functions_in_spec(self, watcher_with_workspace):
        """Extracts from comma-separated lists like 'filter_by_status, sort_by_date'."""
        watcher, ws = watcher_with_workspace
        code = "def sort_by_date(): pass\n"
        (ws / "utils.py").write_text(code)

        task = TaskNode(title="utils", id="t1")
        task.spec = "Functions: filter_by_status, sort_by_date, get_top_items"
        task.code_files = ["utils.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert "filter_by_status" in missing
        assert "get_top_items" in missing
        assert "sort_by_date" not in missing

    def test_syntax_error_in_code_skips_file(self, watcher_with_workspace):
        """Code with syntax errors is skipped, not crashed on."""
        watcher, ws = watcher_with_workspace
        (ws / "bad.py").write_text("def foo(\n")  # syntax error

        task = TaskNode(title="bad", id="t1")
        task.spec = "Implement `some_func`"
        task.code_files = ["bad.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert "some_func" in missing

    def test_parameter_names_excluded(self, watcher_with_workspace):
        """Parameter names inside function signatures are not flagged as missing."""
        watcher, ws = watcher_with_workspace
        code = "def filter_by_priority(tasks, min_priority): pass\n"
        (ws / "filters.py").write_text(code)

        task = TaskNode(title="filters", id="t1")
        task.spec = (
            "filter_by_priority(tasks, min_priority) returns tasks "
            "with priority >= min_priority"
        )
        task.code_files = ["filters.py"]

        import asyncio
        missing = asyncio.get_event_loop().run_until_complete(
            watcher.spec_coverage_check(task)
        )
        assert "min_priority" not in missing
        assert "filter_by_priority" not in missing


class TestFixMutableDefaults:
    """Tests for _fix_mutable_defaults auto-fixer."""

    def test_fixes_list_default(self, tmp_path):
        """Rewrites param=[] to param=None with body guard."""
        code = (
            "class Board:\n"
            "    def __init__(self, name, tasks=[]):\n"
            "        self.name = name\n"
            "        self.tasks = tasks\n"
        )
        f = tmp_path / "models.py"
        f.write_text(code)

        count = WatcherAgent._fix_mutable_defaults(f)
        assert count == 1
        fixed = f.read_text()
        assert "tasks=None" in fixed
        assert "tasks = tasks if tasks is not None else []" in fixed
        # Verify the fixed code actually works
        exec_globals: dict = {}
        exec(compile(fixed, "models.py", "exec"), exec_globals)
        Board = exec_globals["Board"]
        b1 = Board("A")
        b2 = Board("B")
        b1.tasks.append("x")
        assert b2.tasks == []  # no shared state

    def test_fixes_dict_default(self, tmp_path):
        """Rewrites param={} to param=None with body guard."""
        code = (
            "class Foo:\n"
            "    def __init__(self, data={}):\n"
            "        self.data = data\n"
        )
        f = tmp_path / "foo.py"
        f.write_text(code)

        count = WatcherAgent._fix_mutable_defaults(f)
        assert count == 1
        fixed = f.read_text()
        assert "data=None" in fixed
        assert "data = data if data is not None else {}" in fixed

    def test_no_fix_for_immutable_defaults(self, tmp_path):
        """Doesn't touch param=0 or param='hello'."""
        code = (
            "class Foo:\n"
            "    def __init__(self, x=0, y='hello'):\n"
            "        self.x = x\n"
            "        self.y = y\n"
        )
        f = tmp_path / "foo.py"
        f.write_text(code)

        count = WatcherAgent._fix_mutable_defaults(f)
        assert count == 0
        assert f.read_text() == code

    def test_fixes_with_type_annotation(self, tmp_path):
        """Handles param: list[Task] = []."""
        code = (
            "class Board:\n"
            "    def __init__(self, name: str, tasks: list = []):\n"
            "        self.name = name\n"
            "        self.tasks = tasks\n"
        )
        f = tmp_path / "models.py"
        f.write_text(code)

        count = WatcherAgent._fix_mutable_defaults(f)
        assert count == 1
        fixed = f.read_text()
        assert "= None" in fixed
        assert "tasks = tasks if tasks is not None else []" in fixed


class TestFixPackageImports:
    """Tests for _fix_package_imports: rewriting package-style to flat imports."""

    def test_rewrites_package_import(self, tmp_path):
        """from taskboard.models import Task → from models import Task."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "models.py").write_text("class Task: pass\n")
        (src / "filters.py").write_text(
            "from taskboard.models import Task\n\n"
            "def filter_by_status(tasks, status):\n"
            "    return [t for t in tasks if t.status == status]\n"
        )
        count = WatcherAgent._fix_package_imports(tmp_path)
        assert count == 1
        fixed = (src / "filters.py").read_text()
        assert "from models import Task" in fixed
        assert "taskboard" not in fixed

    def test_rewrites_test_imports(self, tmp_path):
        """Test files also get their package imports rewritten."""
        src = tmp_path / "src"
        src.mkdir()
        tests = tmp_path / "tests"
        tests.mkdir()
        (src / "models.py").write_text("class Task: pass\n")
        (tests / "test_models.py").write_text(
            "from taskboard.models import Task\n\n"
            "def test_task():\n"
            "    t = Task()\n"
        )
        count = WatcherAgent._fix_package_imports(tmp_path)
        assert count == 1
        fixed = (tests / "test_models.py").read_text()
        assert "from models import Task" in fixed

    def test_no_op_when_already_flat(self, tmp_path):
        """Does nothing when imports are already flat."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "models.py").write_text("class Task: pass\n")
        original = "from models import Task\n\ndef foo(): pass\n"
        (src / "filters.py").write_text(original)
        count = WatcherAgent._fix_package_imports(tmp_path)
        assert count == 0
        assert (src / "filters.py").read_text() == original

    def test_handles_deep_package_path(self, tmp_path):
        """from pkg.sub.models import Task → from models import Task."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "models.py").write_text("class Task: pass\n")
        (src / "service.py").write_text(
            "from my.pkg.models import Task, Board\n"
        )
        count = WatcherAgent._fix_package_imports(tmp_path)
        assert count == 1
        fixed = (src / "service.py").read_text()
        assert "from models import Task, Board" in fixed


class TestRuffAutofix:
    """Tests for ruff --fix integration in auto_fix_deterministic."""

    @pytest.mark.asyncio
    async def test_ruff_removes_unused_import(self, tmp_path):
        """ruff --fix removes unused imports from generated code."""
        code = (
            "import os\n"
            "from collections import Counter\n"
            "\n"
            "def add(a, b):\n"
            "    return a + b\n"
        )
        f = tmp_path / "calc.py"
        f.write_text(code)

        from pmca.utils.linters import ruff_autofix, is_ruff_available
        if not is_ruff_available():
            pytest.skip("ruff not installed")

        count = await ruff_autofix(f, tmp_path)
        assert count >= 1
        fixed = f.read_text()
        assert "import os" not in fixed
        assert "Counter" not in fixed
        assert "def add" in fixed

    @pytest.mark.asyncio
    async def test_ruff_removes_duplicate_import(self, tmp_path):
        """ruff --fix cleans up duplicate/redefined imports."""
        code = (
            "from models import Task, Board\n"
            "from models import Board, Task\n"
            "\n"
            "def foo(t: Task, b: Board):\n"
            "    pass\n"
        )
        f = tmp_path / "svc.py"
        f.write_text(code)

        from pmca.utils.linters import ruff_autofix, is_ruff_available
        if not is_ruff_available():
            pytest.skip("ruff not installed")

        count = await ruff_autofix(f, tmp_path)
        assert count >= 1
        fixed = f.read_text()
        # Should have only one import line for models
        import_lines = [l for l in fixed.splitlines() if l.startswith("from models")]
        assert len(import_lines) == 1

    @pytest.mark.asyncio
    async def test_ruff_no_fix_needed(self, tmp_path):
        """Returns 0 when code is already clean."""
        code = "def add(a, b):\n    return a + b\n"
        f = tmp_path / "clean.py"
        f.write_text(code)

        from pmca.utils.linters import ruff_autofix, is_ruff_available
        if not is_ruff_available():
            pytest.skip("ruff not installed")

        count = await ruff_autofix(f, tmp_path)
        assert count == 0


class TestSpecCoveragePrompt:
    """Tests for the updated VERIFY_TESTS_PROMPT completeness check."""

    def test_verify_tests_prompt_has_completeness_check(self):
        """VERIFY_TESTS_PROMPT should include spec-coverage check."""
        assert "Incomplete coverage" in reviewer_prompts.VERIFY_TESTS_PROMPT
        assert "ALL functions" in reviewer_prompts.VERIFY_TESTS_PROMPT

    def test_verify_tests_prompt_has_example(self):
        """Prompt should include an example of the completeness check."""
        assert "filter_by_status" in reviewer_prompts.VERIFY_TESTS_PROMPT
        assert "filter_by_priority" in reviewer_prompts.VERIFY_TESTS_PROMPT


class TestSpecCoverageIntegration:
    """Tests for spec-coverage gate integration in orchestrator."""

    @pytest.mark.asyncio
    async def test_spec_coverage_forces_review_failure(self, v2_orchestrator):
        """When spec-coverage finds missing functions, review is force-failed."""
        orch = v2_orchestrator
        orch._config.cascade.test_first = False
        orch._config.cascade.best_of_n = 1

        orch._coder.implement = AsyncMock(return_value=[
            CodeFile(path="src/filters.py", content="def sort_by_priority(): pass"),
        ])
        orch._watcher.check_not_faked = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="watcher",
        ))
        # Both smoke test and review pass
        orch._watcher.run_tests = AsyncMock(return_value=TestResult(
            passed=True, total=3, failures=0, output="3 passed", errors=[],
        ))
        orch._reviewer.verify_code = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))

        orch._coder.fix = AsyncMock(return_value=[
            CodeFile(path="src/filters.py", content="def sort_by_priority(): pass\ndef filter_by_status(): pass"),
        ])

        root = orch._task_tree.create_root("Build filters")
        root.spec = "Implement `filter_by_status` and `sort_by_priority`"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)

        # Simulate spec-coverage gate finding missing names (normally called in _code_leaf)
        root._missing_spec_names = ["filter_by_status"]

        await orch.review_phase(root)

        # Review should have been force-failed at least once
        has_spec_coverage_fail = any(
            "Missing function/class from spec" in (r.issues[0] if r.issues else "")
            for r in root.review_history
        )
        assert has_spec_coverage_fail

    @pytest.mark.asyncio
    async def test_spec_coverage_called_in_code_leaf(self, v2_orchestrator):
        """spec_coverage_check is called in _code_leaf."""
        orch = v2_orchestrator
        orch._config.cascade.test_first = False
        orch._config.cascade.best_of_n = 1

        orch._coder.implement = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        orch._watcher.run_tests = AsyncMock(return_value=TestResult(
            passed=True, total=1, failures=0, output="1 passed", errors=[],
        ))
        orch._reviewer.verify_code = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))
        orch._watcher.check_not_faked = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="watcher",
        ))
        orch._watcher.spec_coverage_check = AsyncMock(return_value=[])
        orch._tester.generate_edge_cases = AsyncMock(return_value=[])

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch._code_leaf(root)

        assert orch._watcher.spec_coverage_check.called


# ---------------------------------------------------------------------------
# Tester Agent Tests
# ---------------------------------------------------------------------------

class TestTesterPrompts:
    def test_system_prompt_exists(self):
        assert hasattr(tester_prompts, "SYSTEM_PROMPT")
        assert "Tester" in tester_prompts.SYSTEM_PROMPT

    def test_generate_tests_prompt_exists(self):
        prompt = tester_prompts.GENERATE_TESTS_PROMPT
        assert "{spec}" in prompt
        assert "{context}" in prompt
        assert "{suggested_module}" in prompt
        assert "{suggested_test_path}" in prompt

    def test_analyze_failure_prompt_exists(self):
        prompt = tester_prompts.ANALYZE_FAILURE_PROMPT
        assert "{spec}" in prompt
        assert "{code}" in prompt
        assert "{tests}" in prompt
        assert "{test_output}" in prompt

    def test_generate_edge_cases_prompt_exists(self):
        prompt = tester_prompts.GENERATE_EDGE_CASES_PROMPT
        assert "{spec}" in prompt
        assert "{code}" in prompt
        assert "{existing_tests}" in prompt

    def test_generate_tests_prompt_format(self):
        result = tester_prompts.GENERATE_TESTS_PROMPT.format(
            spec="Build a calculator",
            context="No context",
            suggested_module="calculator",
            suggested_test_path="tests/test_calculator.py",
        )
        assert "Build a calculator" in result
        assert "calculator" in result

    def test_analyze_failure_prompt_format(self):
        result = tester_prompts.ANALYZE_FAILURE_PROMPT.format(
            spec="Add function",
            code="def add(a, b): return a + b",
            tests="def test_add(): assert add(1,2)==3",
            test_output="FAILED",
        )
        assert "Add function" in result
        assert "FAILED" in result


class TestFailureAnalysisDataclass:
    def test_create(self):
        fa = FailureAnalysis(
            root_cause="code_bug",
            explanation="Function returns wrong value",
            suggested_fix_target="code",
            specific_issues=["add() returns 0 instead of sum"],
        )
        assert fa.root_cause == "code_bug"
        assert fa.suggested_fix_target == "code"
        assert len(fa.specific_issues) == 1

    def test_to_dict(self):
        fa = FailureAnalysis(
            root_cause="test_bug",
            explanation="Wrong expected value",
            suggested_fix_target="tests",
            specific_issues=["Expected 3 but should be 2"],
        )
        d = fa.to_dict()
        assert d["root_cause"] == "test_bug"
        assert d["suggested_fix_target"] == "tests"

    def test_from_dict(self):
        d = {
            "root_cause": "import_error",
            "explanation": "Missing import",
            "suggested_fix_target": "both",
            "specific_issues": ["No import for Board"],
        }
        fa = FailureAnalysis.from_dict(d)
        assert fa.root_cause == "import_error"
        assert fa.specific_issues == ["No import for Board"]

    def test_from_dict_defaults(self):
        fa = FailureAnalysis.from_dict({})
        assert fa.root_cause == "unknown"
        assert fa.suggested_fix_target == "code"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_generate_tests(self, mock_manager):
        mock_manager.generate.return_value = (
            "```python\n# filepath: tests/test_calc.py\n"
            "from calc import add\n"
            "def test_add():\n    assert add(1, 2) == 3\n"
            "def test_add_zero():\n    assert add(0, 0) == 0\n```"
        )
        tester = TesterAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function that adds two numbers"

        files = await tester.generate_tests(task, context="")
        assert len(files) == 1
        assert "test" in files[0].path
        assert "test_add" in files[0].content

    @pytest.mark.asyncio
    async def test_analyze_failure_code_bug(self, mock_manager):
        mock_manager.generate.return_value = (
            '{"root_cause": "code_bug", "explanation": "add() returns 0 instead of sum", '
            '"suggested_fix_target": "code", "specific_issues": ["add returns hardcoded 0"]}'
        )
        tester = TesterAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"

        analysis = await tester.analyze_failure(
            task, "FAILED: assert 0 == 3", "def add(a,b): return 0",
        )
        assert analysis.root_cause == "code_bug"
        assert analysis.suggested_fix_target == "code"
        assert len(analysis.specific_issues) == 1

    @pytest.mark.asyncio
    async def test_analyze_failure_test_bug(self, mock_manager):
        mock_manager.generate.return_value = (
            '{"root_cause": "test_bug", "explanation": "Test expects wrong value", '
            '"suggested_fix_target": "tests", "specific_issues": ["Expected 4 but should be 3"]}'
        )
        tester = TesterAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"

        analysis = await tester.analyze_failure(
            task, "FAILED: assert 3 == 4", "def add(a,b): return a+b",
        )
        assert analysis.root_cause == "test_bug"
        assert analysis.suggested_fix_target == "tests"

    @pytest.mark.asyncio
    async def test_analyze_failure_parse_error(self, mock_manager):
        mock_manager.generate.return_value = "This is not JSON at all"
        tester = TesterAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"

        analysis = await tester.analyze_failure(
            task, "FAILED", "def add(a,b): return 0",
        )
        assert analysis.root_cause == "unknown"
        assert "Failed to parse" in analysis.explanation

    @pytest.mark.asyncio
    async def test_generate_edge_cases(self, mock_manager):
        mock_manager.generate.return_value = (
            "```python\n# filepath: tests/test_calc_edge.py\n"
            "from calc import add\n"
            "def test_add_negative():\n    assert add(-1, -2) == -3\n```"
        )
        tester = TesterAgent(mock_manager)
        task = TaskNode(title="Build a calculator")
        task.spec = "Add function"

        files = await tester.generate_edge_cases(
            task, "def add(a,b): return a+b", "def test_add(): assert add(1,2)==3",
        )
        assert len(files) == 1
        assert "edge" in files[0].path
        assert "negative" in files[0].content


class TestTesterIntegration:
    @pytest.mark.asyncio
    async def test_tester_used_when_configured(self, v2_orchestrator):
        """When tester is configured, code_phase uses tester.generate_tests."""
        orch = v2_orchestrator

        # Verify tester is initialized
        assert orch._tester is not None

        orch._tester.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="def test_add(): assert add(1,2)==3"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)

        # Tester should have been called, not coder
        orch._tester.generate_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_without_tester(self, workspace):
        """When tester is not configured, falls back to coder.generate_tests."""
        cfg = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:14b", temperature=0.3),
                AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.2),
                AgentRole.REVIEWER: ModelConfig(name="test:14b", temperature=0.1),
                AgentRole.WATCHER: ModelConfig(name="test:7b", temperature=0.1),
                # No TESTER role
            },
            cascade=CascadeConfig(
                max_depth=1,
                max_retries=3,
                test_first=True,
            ),
        )
        orch = Orchestrator(cfg, workspace)
        orch._model_manager.generate = AsyncMock()
        orch._model_manager.ensure_loaded = AsyncMock(return_value="test:7b")
        orch._model_manager.close = AsyncMock()

        assert orch._tester is None

        orch._coder.generate_tests = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc.py", content="def test_add(): assert add(1,2)==3"),
        ])
        orch._coder.implement_with_tests = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a, b): return a + b"),
        ])
        orch._reviewer.verify_tests = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="reviewer",
        ))

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch.code_phase(root)

        # Coder should have been called as fallback
        orch._coder.generate_tests.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_analysis_used_in_review(self, v2_orchestrator):
        """When tests fail with assertion errors, tester analyzes failures."""
        orch = v2_orchestrator
        # Limit retries to avoid hitting fresh-start path
        orch._config.cascade.max_retries = 1

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.code_files = ["src/calc.py"]
        root.test_files = ["tests/test_calc.py"]

        (orch._workspace_path / "src").mkdir(parents=True)
        (orch._workspace_path / "tests").mkdir(parents=True)
        (orch._workspace_path / "src" / "calc.py").write_text("def add(a,b): return 0")
        (orch._workspace_path / "tests" / "test_calc.py").write_text(
            "from src.calc import add\ndef test_add(): assert add(1,2)==3"
        )

        # Tests fail with assertion error (not crash)
        orch._watcher.run_tests = AsyncMock(return_value=TestResult(
            passed=False, total=1, failures=1, output="FAILED test_add\nE assert 0 == 3",
            errors=["AssertionError"],
        ))
        orch._watcher.extract_structured_errors = MagicMock(return_value=[
            MagicMock(error_type="AssertionError", format_for_prompt=lambda: "assert 0 == 3"),
        ])
        orch._reviewer.verify_code = AsyncMock(return_value=ReviewResult(
            passed=False, issues=["Wrong result"],
            suggestions=[], timestamp=__import__("datetime").datetime.now(), model_used="test",
        ))
        orch._tester.analyze_failure = AsyncMock(return_value=FailureAnalysis(
            root_cause="code_bug",
            explanation="add() returns 0 instead of a+b",
            suggested_fix_target="code",
            specific_issues=["add() has hardcoded return 0"],
        ))
        orch._coder.fix = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a,b): return a+b"),
        ])
        orch._watcher.auto_fix_deterministic = AsyncMock(return_value=0)
        orch._watcher.calibrate_tests = AsyncMock(return_value=0)
        orch._watcher.spec_coverage_check = AsyncMock(return_value=[])

        root.status = TaskStatus.REVIEWING
        root.updated_at = __import__("datetime").datetime.now()
        await orch.review_phase(root)

        # Tester analyze_failure should have been called
        assert orch._tester.analyze_failure.called

    @pytest.mark.asyncio
    async def test_edge_cases_generated_after_verify(self, v2_orchestrator):
        """After task verification, edge case tests are generated."""
        orch = v2_orchestrator
        orch._config.cascade.test_first = False
        orch._config.cascade.best_of_n = 1

        orch._coder.implement = AsyncMock(return_value=[
            CodeFile(path="src/calc.py", content="def add(a,b): return a+b"),
            CodeFile(path="tests/test_calc.py", content="def test_add(): assert add(1,2)==3"),
        ])
        orch._watcher.auto_fix_deterministic = AsyncMock(return_value=0)
        orch._watcher.static_analysis_gate = AsyncMock(return_value=([], []))
        orch._watcher.spec_coverage_check = AsyncMock(return_value=[])
        orch._watcher.calibrate_tests = AsyncMock(return_value=0)
        orch._watcher.run_tests = AsyncMock(return_value=TestResult(
            passed=True, total=1, failures=0, output="1 passed", errors=[],
        ))
        orch._watcher.check_not_faked = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="test",
        ))
        orch._reviewer.verify_code = AsyncMock(return_value=ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=__import__("datetime").datetime.now(), model_used="test",
        ))
        orch._tester.generate_edge_cases = AsyncMock(return_value=[
            CodeFile(path="tests/test_calc_edge.py", content="def test_negative(): assert add(-1,-2)==-3"),
        ])

        root = orch._task_tree.create_root("Build a calculator")
        root.spec = "A calculator with add()"
        root.transition(TaskStatus.DESIGNING)

        await orch._code_leaf(root)

        assert root.status == TaskStatus.VERIFIED
        orch._tester.generate_edge_cases.assert_called_once()


class TestAgentRoleTester:
    def test_tester_in_agent_role_enum(self):
        assert AgentRole.TESTER == "tester"
        assert AgentRole("tester") == AgentRole.TESTER

    def test_tester_in_default_config(self):
        """Default config fallback includes TESTER role."""
        cfg = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:14b"),
                AgentRole.CODER: ModelConfig(name="test:7b"),
                AgentRole.REVIEWER: ModelConfig(name="test:14b"),
                AgentRole.WATCHER: ModelConfig(name="test:7b"),
                AgentRole.TESTER: ModelConfig(name="test:14b", temperature=0.2),
            },
        )
        assert AgentRole.TESTER in cfg.models
        assert cfg.models[AgentRole.TESTER].temperature == 0.2

    def test_config_without_tester_is_valid(self):
        """Config without TESTER is still valid (backward compat)."""
        cfg = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:14b"),
                AgentRole.CODER: ModelConfig(name="test:7b"),
                AgentRole.REVIEWER: ModelConfig(name="test:14b"),
                AgentRole.WATCHER: ModelConfig(name="test:7b"),
            },
        )
        assert AgentRole.TESTER not in cfg.models


# ---------------------------------------------------------------------------
# Feature: RAG Config Tests
# ---------------------------------------------------------------------------

class TestRAGConfig:
    def test_default_values(self):
        from pmca.models.config import RAGConfig
        cfg = RAGConfig()
        assert cfg.enabled is False
        assert cfg.docs_path == ""
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.n_results == 3
        assert cfg.persist_dir == "~/.pmca/rag"

    def test_config_has_rag_field(self):
        cfg = Config(
            models={AgentRole.CODER: ModelConfig(name="test:7b")},
        )
        assert hasattr(cfg, "rag")
        assert cfg.rag.enabled is False

    def test_from_yaml_parses_rag(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
rag:
  enabled: true
  docs_path: "./docs"
  embedding_model: "custom-model"
  n_results: 5
  persist_dir: "/tmp/rag"
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.rag.enabled is True
        assert cfg.rag.docs_path == "./docs"
        assert cfg.rag.embedding_model == "custom-model"
        assert cfg.rag.n_results == 5
        assert cfg.rag.persist_dir == "/tmp/rag"

    def test_from_yaml_defaults_when_missing(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.rag.enabled is False
        assert cfg.rag.docs_path == ""


# ---------------------------------------------------------------------------
# Feature: MCP Config Tests
# ---------------------------------------------------------------------------

class TestMCPConfig:
    def test_default_values(self):
        from pmca.models.config import MCPConfig
        cfg = MCPConfig()
        assert cfg.enabled is False
        assert cfg.server_name == "pmca"

    def test_config_has_mcp_field(self):
        cfg = Config(
            models={AgentRole.CODER: ModelConfig(name="test:7b")},
        )
        assert hasattr(cfg, "mcp")
        assert cfg.mcp.enabled is False

    def test_from_yaml_parses_mcp(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
mcp:
  enabled: true
  server_name: "my-pmca"
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.mcp.enabled is True
        assert cfg.mcp.server_name == "my-pmca"


# ---------------------------------------------------------------------------
# Feature: Lint Config Tests
# ---------------------------------------------------------------------------

class TestLintConfig:
    def test_default_values(self):
        from pmca.models.config import LintConfig
        cfg = LintConfig()
        assert cfg.mypy is False
        assert cfg.ruff is False

    def test_config_has_lint_field(self):
        cfg = Config(
            models={AgentRole.CODER: ModelConfig(name="test:7b")},
        )
        assert hasattr(cfg, "lint")
        assert cfg.lint.mypy is False
        assert cfg.lint.ruff is False

    def test_from_yaml_parses_lint(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
lint:
  mypy: true
  ruff: true
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.lint.mypy is True
        assert cfg.lint.ruff is True

    def test_from_yaml_defaults_when_missing(self, tmp_path):
        yaml_content = """\
models:
  coder:
    name: "test:7b"
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)
        cfg = Config.from_yaml(yaml_file)
        assert cfg.lint.mypy is False
        assert cfg.lint.ruff is False


# ---------------------------------------------------------------------------
# Feature: Linters Utility Tests
# ---------------------------------------------------------------------------

class TestLintersAvailability:
    def test_is_mypy_available_returns_bool(self):
        from pmca.utils.linters import is_mypy_available
        result = is_mypy_available()
        assert isinstance(result, bool)

    def test_is_ruff_available_returns_bool(self):
        from pmca.utils.linters import is_ruff_available
        result = is_ruff_available()
        assert isinstance(result, bool)


class TestRunMypy:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_installed(self, tmp_path):
        from pmca.utils.linters import run_mypy
        with patch("pmca.utils.linters.is_mypy_available", return_value=False):
            result = await run_mypy(tmp_path / "test.py", tmp_path)
            assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_valid_code(self, tmp_path):
        from pmca.utils.linters import run_mypy, is_mypy_available
        if not is_mypy_available():
            pytest.skip("mypy not installed")
        code_file = tmp_path / "good.py"
        code_file.write_text("x: int = 1\n")
        result = await run_mypy(code_file, tmp_path)
        assert result == []


class TestRunRuff:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_installed(self, tmp_path):
        from pmca.utils.linters import run_ruff
        with patch("pmca.utils.linters._find_tool", return_value=None):
            result = await run_ruff(tmp_path / "test.py", tmp_path)
            assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_valid_code(self, tmp_path):
        from pmca.utils.linters import run_ruff, is_ruff_available
        if not is_ruff_available():
            pytest.skip("ruff not installed")
        code_file = tmp_path / "good.py"
        code_file.write_text("x = 1\n")
        result = await run_ruff(code_file, tmp_path)
        assert result == []


class TestStaticAnalysisGateWithLinters:
    @pytest.mark.asyncio
    async def test_lint_config_none_skips_linters(self, mock_manager, workspace):
        """When lint_config is None, only ast.parse runs."""
        watcher = WatcherAgent(mock_manager, workspace, lint_config=None)
        (workspace / "src").mkdir()
        (workspace / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")

        task = TaskNode(title="calc")
        task.code_files = ["src/calc.py"]
        task.test_files = []

        blocking, informational = await watcher.static_analysis_gate(task)
        assert blocking == []
        assert informational == []

    @pytest.mark.asyncio
    async def test_lint_config_with_mypy_enabled(self, mock_manager, workspace):
        """When lint_config.mypy is True, mypy is called."""
        from pmca.models.config import LintConfig
        lint_config = LintConfig(mypy=True, ruff=False)
        watcher = WatcherAgent(mock_manager, workspace, lint_config=lint_config)
        (workspace / "src").mkdir()
        (workspace / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")

        task = TaskNode(title="calc")
        task.code_files = ["src/calc.py"]
        task.test_files = []

        with patch("pmca.utils.linters.run_mypy", new_callable=AsyncMock, return_value=[]) as mock_mypy:
            blocking, informational = await watcher.static_analysis_gate(task)
            mock_mypy.assert_called_once()
            assert blocking == []
            assert informational == []

    @pytest.mark.asyncio
    async def test_lint_config_with_ruff_enabled(self, mock_manager, workspace):
        """When lint_config.ruff is True, ruff is called."""
        from pmca.models.config import LintConfig
        lint_config = LintConfig(mypy=False, ruff=True)
        watcher = WatcherAgent(mock_manager, workspace, lint_config=lint_config)
        (workspace / "src").mkdir()
        (workspace / "src" / "calc.py").write_text("def add(a, b):\n    return a + b\n")

        task = TaskNode(title="calc")
        task.code_files = ["src/calc.py"]
        task.test_files = []

        with patch("pmca.utils.linters.run_ruff", new_callable=AsyncMock, return_value=[]) as mock_ruff:
            blocking, informational = await watcher.static_analysis_gate(task)
            mock_ruff.assert_called_once()
            assert blocking == []
            assert informational == []

    @pytest.mark.asyncio
    async def test_lint_errors_are_informational(self, mock_manager, workspace):
        """Linter errors go into informational, not blocking."""
        from pmca.models.config import LintConfig
        lint_config = LintConfig(mypy=True, ruff=True)
        watcher = WatcherAgent(mock_manager, workspace, lint_config=lint_config)
        (workspace / "bad.py").write_text("x: int = 'hello'\n")

        task = TaskNode(title="lint")
        task.code_files = ["bad.py"]
        task.test_files = []

        with patch("pmca.utils.linters.run_mypy", new_callable=AsyncMock, return_value=["bad.py:1: error: incompatible type"]):
            with patch("pmca.utils.linters.run_ruff", new_callable=AsyncMock, return_value=["bad.py:1:1: E741"]):
                blocking, informational = await watcher.static_analysis_gate(task)
                assert blocking == []  # No syntax errors
                assert len(informational) == 2
                assert any("incompatible type" in e for e in informational)
                assert any("E741" in e for e in informational)


# ---------------------------------------------------------------------------
# Feature: RAG Manager Tests
# ---------------------------------------------------------------------------

class TestRAGManagerChunking:
    def test_chunk_text_splits_by_paragraphs(self):
        from pmca.utils.rag import _chunk_text
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = _chunk_text(text, max_chars=100)
        assert len(chunks) >= 1
        assert "Paragraph one." in chunks[0]

    def test_chunk_text_respects_max_chars(self):
        from pmca.utils.rag import _chunk_text
        text = "A" * 500 + "\n\n" + "B" * 500
        chunks = _chunk_text(text, max_chars=600)
        assert len(chunks) == 2

    def test_chunk_text_handles_empty_text(self):
        from pmca.utils.rag import _chunk_text
        chunks = _chunk_text("")
        assert chunks == []

    def test_chunk_text_handles_single_large_paragraph(self):
        from pmca.utils.rag import _chunk_text
        text = "A" * 3000
        chunks = _chunk_text(text, max_chars=1000)
        assert len(chunks) >= 3


class TestRAGManagerInit:
    def test_init_without_chromadb(self):
        """RAGManager should handle missing chromadb gracefully."""
        from pmca.models.config import RAGConfig
        from pmca.utils.rag import RAGManager
        with patch.dict("sys.modules", {"chromadb": None}):
            config = RAGConfig(enabled=True)
            manager = RAGManager(config)
            assert not manager.available

    def test_query_returns_empty_when_unavailable(self):
        """Query returns empty list when RAG is not available."""
        from pmca.models.config import RAGConfig
        from pmca.utils.rag import RAGManager
        config = RAGConfig(enabled=True)
        manager = RAGManager.__new__(RAGManager)
        manager._config = config
        manager._client = None
        manager._embedding_fn = None
        manager._collection = None
        result = manager.query("test query")
        assert result == []

    def test_index_returns_zero_when_unavailable(self):
        """Index returns 0 when RAG is not available."""
        from pmca.models.config import RAGConfig
        from pmca.utils.rag import RAGManager
        config = RAGConfig(enabled=True)
        manager = RAGManager.__new__(RAGManager)
        manager._config = config
        manager._client = None
        manager._embedding_fn = None
        manager._collection = None
        result = manager.index_directory(Path("/nonexistent"))
        assert result == 0


# ---------------------------------------------------------------------------
# Feature: RAG in ContextManager Tests
# ---------------------------------------------------------------------------

class TestContextManagerWithRAG:
    def test_build_context_without_rag(self):
        """Without RAG, context builds normally."""
        from pmca.tasks.tree import TaskTree
        tree = TaskTree()
        root = tree.create_root("Test task")
        root.spec = "Build something"

        cm = ContextManager(tree)
        ctx = cm.build_context(root)
        assert "Build something" in ctx
        assert "Library Documentation" not in ctx

    def test_build_context_with_rag(self):
        """With RAG, retrieved chunks appear in context."""
        from pmca.tasks.tree import TaskTree
        tree = TaskTree()
        root = tree.create_root("Test task")
        root.spec = "Build a REST API"

        mock_rag = MagicMock()
        mock_rag.query.return_value = ["Flask route example: @app.route('/api')"]

        cm = ContextManager(tree, rag_manager=mock_rag)
        ctx = cm.build_context(root)
        assert "Library Documentation" in ctx
        assert "Flask route example" in ctx
        mock_rag.query.assert_called_once_with("Build a REST API")

    def test_build_context_rag_returns_empty(self):
        """When RAG returns no results, section is not added."""
        from pmca.tasks.tree import TaskTree
        tree = TaskTree()
        root = tree.create_root("Test task")
        root.spec = "Build something"

        mock_rag = MagicMock()
        mock_rag.query.return_value = []

        cm = ContextManager(tree, rag_manager=mock_rag)
        ctx = cm.build_context(root)
        assert "Library Documentation" not in ctx


# ---------------------------------------------------------------------------
# Feature: MCP Server Tests
# ---------------------------------------------------------------------------

class TestMCPServerCreation:
    def test_create_server_raises_without_mcp_package(self):
        """create_mcp_server raises ImportError without mcp package."""
        from pmca.models.config import MCPConfig
        config = Config(
            models={AgentRole.CODER: ModelConfig(name="test:7b")},
            mcp=MCPConfig(enabled=True),
        )
        with patch.dict("sys.modules", {"mcp": None, "mcp.server": None, "mcp.types": None}):
            with pytest.raises(ImportError, match="mcp"):
                from pmca.mcp.server import create_mcp_server
                create_mcp_server(config, Path("/tmp"))

    def test_create_server_with_mocked_mcp(self, workspace):
        """create_mcp_server returns a server when mcp is available."""
        mock_server_cls = MagicMock()
        mock_server_instance = MagicMock()
        mock_server_cls.return_value = mock_server_instance
        # Make the decorators return the function
        mock_server_instance.list_tools.return_value = lambda f: f
        mock_server_instance.call_tool.return_value = lambda f: f
        mock_server_instance.list_resources.return_value = lambda f: f
        mock_server_instance.read_resource.return_value = lambda f: f

        mock_mcp_module = MagicMock()
        mock_mcp_server_module = MagicMock()
        mock_mcp_server_module.Server = mock_server_cls
        mock_mcp_types_module = MagicMock()

        with patch.dict("sys.modules", {
            "mcp": mock_mcp_module,
            "mcp.server": mock_mcp_server_module,
            "mcp.types": mock_mcp_types_module,
        }):
            # Need to re-import to pick up mocked modules
            import importlib
            import pmca.mcp.server as mcp_mod
            importlib.reload(mcp_mod)

            config = Config(
                models={AgentRole.CODER: ModelConfig(name="test:7b")},
            )
            server = mcp_mod.create_mcp_server(config, workspace)
            assert server is mock_server_instance


# ---------------------------------------------------------------------------
# Feature: Orchestrator Wiring Tests
# ---------------------------------------------------------------------------

class TestOrchestratorLintWiring:
    def test_watcher_receives_lint_config(self, workspace):
        """Orchestrator passes lint config to WatcherAgent."""
        from pmca.models.config import LintConfig
        cfg = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:14b"),
                AgentRole.CODER: ModelConfig(name="test:7b"),
                AgentRole.REVIEWER: ModelConfig(name="test:14b"),
                AgentRole.WATCHER: ModelConfig(name="test:7b"),
            },
            lint=LintConfig(mypy=True, ruff=True),
        )
        orch = Orchestrator(cfg, workspace)
        assert orch._watcher._lint_config is not None
        assert orch._watcher._lint_config.mypy is True
        assert orch._watcher._lint_config.ruff is True

    def test_watcher_receives_default_lint_config(self, workspace):
        """Without explicit lint config, default (all False) is used."""
        cfg = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:14b"),
                AgentRole.CODER: ModelConfig(name="test:7b"),
                AgentRole.REVIEWER: ModelConfig(name="test:14b"),
                AgentRole.WATCHER: ModelConfig(name="test:7b"),
            },
        )
        orch = Orchestrator(cfg, workspace)
        assert orch._watcher._lint_config is not None
        assert orch._watcher._lint_config.mypy is False
        assert orch._watcher._lint_config.ruff is False


class TestOrchestratorRAGWiring:
    def test_rag_disabled_by_default(self, workspace):
        """RAG manager is None when not configured."""
        cfg = Config(
            models={
                AgentRole.CODER: ModelConfig(name="test:7b"),
                AgentRole.ARCHITECT: ModelConfig(name="test:14b"),
                AgentRole.REVIEWER: ModelConfig(name="test:14b"),
                AgentRole.WATCHER: ModelConfig(name="test:7b"),
            },
        )
        orch = Orchestrator(cfg, workspace)
        assert orch._rag_manager is None


# ---------------------------------------------------------------------------
# Feature: API Consistency Lint Tests
# ---------------------------------------------------------------------------

class TestCheckApiConsistency:
    """Tests for _check_api_consistency: attribute/method shadowing detection."""

    def test_shadowing_detected(self, tmp_path):
        """Detects self.size attribute AND def size() method in same class."""
        code = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self.size = 0\n"
            "\n"
            "    def size(self):\n"
            "        return self.size\n"
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "linked_list.py").write_text(code)

        errors = WatcherAgent._check_api_consistency(tmp_path)
        assert len(errors) == 1
        assert "size" in errors[0]
        assert "attribute" in errors[0]
        assert "method" in errors[0]
        assert "LinkedList" in errors[0]

    def test_no_false_positive_private_attr(self, tmp_path):
        """No error when using self._size private attr + def size() method."""
        code = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self._size = 0\n"
            "\n"
            "    def size(self):\n"
            "        return self._size\n"
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "linked_list.py").write_text(code)

        errors = WatcherAgent._check_api_consistency(tmp_path)
        assert errors == []

    def test_mixed_callsite_detected(self, tmp_path):
        """Detects ll.size (attribute) + ll.size() (method call) in test file."""
        impl = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self._size = 0\n"
            "    def size(self):\n"
            "        return self._size\n"
        )
        test_code = (
            "from linked_list import LinkedList\n"
            "def test_size():\n"
            "    ll = LinkedList()\n"
            "    assert ll.size == 0\n"     # bare attribute access
            "    assert ll.size() == 0\n"   # method call
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "linked_list.py").write_text(impl)
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_linked_list.py").write_text(test_code)

        errors = WatcherAgent._check_api_consistency(tmp_path)
        assert len(errors) >= 1
        assert any("size" in e and "ll" in e for e in errors)

    def test_clean_code_no_errors(self, tmp_path):
        """No errors for clean code with no shadowing."""
        code = (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._items = []\n"
            "\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n"
            "\n"
            "    def pop(self):\n"
            "        return self._items.pop()\n"
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "stack.py").write_text(code)

        errors = WatcherAgent._check_api_consistency(tmp_path)
        assert errors == []

    def test_shadowing_multiple_attrs(self, tmp_path):
        """Detects multiple shadowed names in one class."""
        code = (
            "class Foo:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "        self.length = 0\n"
            "    def count(self):\n"
            "        return self.count\n"
            "    def length(self):\n"
            "        return self.length\n"
        )
        src = tmp_path / "src"
        src.mkdir()
        (src / "foo.py").write_text(code)

        errors = WatcherAgent._check_api_consistency(tmp_path)
        assert len(errors) == 2

    def test_no_src_dir_uses_workspace(self, tmp_path):
        """When no src/ dir, scans workspace directly."""
        code = (
            "class Foo:\n"
            "    def __init__(self):\n"
            "        self.count = 0\n"
            "    def count(self):\n"
            "        return self.count\n"
        )
        (tmp_path / "foo.py").write_text(code)

        errors = WatcherAgent._check_api_consistency(tmp_path)
        assert len(errors) == 1
        assert "count" in errors[0]


# ---------------------------------------------------------------------------
# Feature: Oracle Repair Tests
# ---------------------------------------------------------------------------

class TestOracleRepairTests:
    """Tests for oracle_repair_tests: aggressive second-pass assertion patching."""

    @pytest.fixture
    def watcher_with_workspace(self, tmp_path):
        mm = MagicMock()
        return WatcherAgent(mm, workspace_path=tmp_path), tmp_path

    @pytest.mark.asyncio
    async def test_oracle_repair_numeric(self, watcher_with_workspace):
        """Patches numeric mismatches beyond calibration threshold."""
        watcher, ws = watcher_with_workspace
        (ws / "src").mkdir()
        (ws / "tests").mkdir()
        (ws / "src" / "calc.py").write_text(
            "def compute():\n    return 3.625\n"
        )
        test_code = (
            "from calc import compute\n"
            "def test_compute():\n"
            "    assert compute() == 5.0\n"
        )
        (ws / "tests" / "test_calc.py").write_text(test_code)

        task = TaskNode(title="calc")
        task.test_files = ["tests/test_calc.py"]

        repaired = await watcher.oracle_repair_tests(task)
        assert repaired == 1
        fixed = (ws / "tests" / "test_calc.py").read_text()
        assert "== 3.625" in fixed
        assert "== 5.0" not in fixed

    @pytest.mark.asyncio
    async def test_oracle_repair_skip_zero(self, watcher_with_workspace):
        """Skips oracle repair when actual value is 0 (likely code bug)."""
        watcher, ws = watcher_with_workspace
        (ws / "src").mkdir()
        (ws / "tests").mkdir()
        (ws / "src" / "calc.py").write_text(
            "def compute():\n    return 0\n"
        )
        test_code = (
            "from calc import compute\n"
            "def test_compute():\n"
            "    assert compute() == 42\n"
        )
        (ws / "tests" / "test_calc.py").write_text(test_code)

        task = TaskNode(title="calc")
        task.test_files = ["tests/test_calc.py"]

        repaired = await watcher.oracle_repair_tests(task)
        assert repaired == 0
        fixed = (ws / "tests" / "test_calc.py").read_text()
        assert "== 42" in fixed  # unchanged

    @pytest.mark.asyncio
    async def test_oracle_repair_string(self, watcher_with_workspace):
        """Patches string format mismatches."""
        watcher, ws = watcher_with_workspace
        (ws / "src").mkdir()
        (ws / "tests").mkdir()
        (ws / "src" / "fmt.py").write_text(
            "def greet(name):\n    return f'Hello, {name}!'\n"
        )
        test_code = (
            "from fmt import greet\n"
            "def test_greet():\n"
            "    assert greet('World') == 'Hi, World!'\n"
        )
        (ws / "tests" / "test_fmt.py").write_text(test_code)

        task = TaskNode(title="fmt")
        task.test_files = ["tests/test_fmt.py"]

        repaired = await watcher.oracle_repair_tests(task)
        assert repaired == 1
        fixed = (ws / "tests" / "test_fmt.py").read_text()
        assert "Hello, World!" in fixed
        assert "Hi, World!" not in fixed

    @pytest.mark.asyncio
    async def test_oracle_repair_no_test_files(self, watcher_with_workspace):
        """Returns 0 when no test files exist."""
        watcher, ws = watcher_with_workspace
        task = TaskNode(title="empty")
        task.test_files = []

        repaired = await watcher.oracle_repair_tests(task)
        assert repaired == 0

    @pytest.mark.asyncio
    async def test_oracle_repair_all_pass(self, watcher_with_workspace):
        """Returns 0 when all tests already pass."""
        watcher, ws = watcher_with_workspace
        (ws / "src").mkdir()
        (ws / "tests").mkdir()
        (ws / "src" / "calc.py").write_text(
            "def add(a, b):\n    return a + b\n"
        )
        test_code = (
            "from calc import add\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
        (ws / "tests" / "test_calc.py").write_text(test_code)

        task = TaskNode(title="calc")
        task.test_files = ["tests/test_calc.py"]

        repaired = await watcher.oracle_repair_tests(task)
        assert repaired == 0


# ---------------------------------------------------------------------------
# Feature: Metamorphic/Property Test Guidance Prompt Tests
# ---------------------------------------------------------------------------

class TestMetamorphicTestGuidance:
    """Tests for range/type assertion guidance in coder prompts."""

    def test_implement_prompt_has_range_guidance(self):
        prompt = coder_prompts.IMPLEMENT_PROMPT
        assert "range/type assertions" in prompt
        assert "isinstance(result, float)" in prompt

    def test_implement_simple_prompt_has_range_guidance(self):
        prompt = coder_prompts.IMPLEMENT_SIMPLE_PROMPT
        assert "range/type checks" in prompt

    def test_generate_tests_prompt_has_range_guidance(self):
        prompt = coder_prompts.GENERATE_TESTS_PROMPT
        assert "range/type checks" in prompt
        assert "isinstance(result, int)" in prompt


# ---------------------------------------------------------------------------
# Feature: API Lint Integration in Static Analysis Gate
# ---------------------------------------------------------------------------

class TestFixAttrMethodShadowing:
    """Tests for _fix_attr_method_shadowing auto-fixer."""

    def test_renames_shadowed_attr(self, tmp_path):
        """Renames self.size to self._size when def size() exists."""
        code = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self.size = 0\n"
            "\n"
            "    def append(self, value):\n"
            "        self.size += 1\n"
            "\n"
            "    def size(self):\n"
            "        return self.size\n"
        )
        f = tmp_path / "linked_list.py"
        f.write_text(code)

        count = WatcherAgent._fix_attr_method_shadowing(f)
        assert count == 1
        fixed = f.read_text()
        assert "self._size = 0" in fixed
        assert "self._size += 1" in fixed
        assert "return self._size" in fixed
        # Method name preserved
        assert "def size(self):" in fixed
        # No bare self.size left (only self._size)
        import re
        bare_refs = re.findall(r"self\.size\b", fixed)
        assert bare_refs == [], f"Unexpected bare self.size: {bare_refs}"

    def test_no_fix_when_no_shadowing(self, tmp_path):
        """No changes when using private attr convention."""
        code = (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._items = []\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n"
        )
        f = tmp_path / "stack.py"
        f.write_text(code)

        count = WatcherAgent._fix_attr_method_shadowing(f)
        assert count == 0
        assert f.read_text() == code

    def test_preserves_non_shadowed_attrs(self, tmp_path):
        """Only renames shadowed attrs, not all attrs."""
        code = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self.size = 0\n"
            "\n"
            "    def size(self):\n"
            "        return self.size\n"
        )
        f = tmp_path / "ll.py"
        f.write_text(code)

        WatcherAgent._fix_attr_method_shadowing(f)
        fixed = f.read_text()
        assert "self.head = None" in fixed  # unchanged
        assert "self._size" in fixed

    def test_syntax_error_returns_zero(self, tmp_path):
        """Returns 0 on syntax error, doesn't crash."""
        f = tmp_path / "bad.py"
        f.write_text("def foo(\n")
        count = WatcherAgent._fix_attr_method_shadowing(f)
        assert count == 0

    def test_fixed_code_executes(self, tmp_path):
        """Fixed code actually works — size() returns correct value."""
        code = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self.size = 0\n"
            "\n"
            "    def append(self, value):\n"
            "        self.size += 1\n"
            "\n"
            "    def size(self):\n"
            "        return self.size\n"
        )
        f = tmp_path / "ll.py"
        f.write_text(code)

        WatcherAgent._fix_attr_method_shadowing(f)
        fixed = f.read_text()

        exec_globals: dict = {}
        exec(compile(fixed, "ll.py", "exec"), exec_globals)
        ll = exec_globals["LinkedList"]()
        ll.append(1)
        ll.append(2)
        assert ll.size() == 2


class TestApiLintInStaticGate:
    """Tests that API consistency lint is integrated into static_analysis_gate."""

    @pytest.mark.asyncio
    async def test_gate_catches_shadowing(self, mock_manager, workspace):
        """static_analysis_gate returns blocking errors for shadowing."""
        code = (
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.size = 0\n"
            "    def size(self):\n"
            "        return self.size\n"
        )
        src = workspace / "src"
        src.mkdir()
        (src / "linked_list.py").write_text(code)

        watcher = WatcherAgent(mock_manager, workspace)
        task = TaskNode(title="ll")
        task.code_files = ["src/linked_list.py"]
        task.test_files = []

        blocking, _ = await watcher.static_analysis_gate(task)
        assert len(blocking) >= 1
        assert any("size" in e and "attribute" in e for e in blocking)

    @pytest.mark.asyncio
    async def test_gate_clean_code_no_api_errors(self, mock_manager, workspace):
        """static_analysis_gate returns no API errors for clean code."""
        code = (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._items = []\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n"
        )
        src = workspace / "src"
        src.mkdir()
        (src / "stack.py").write_text(code)

        watcher = WatcherAgent(mock_manager, workspace)
        task = TaskNode(title="stack")
        task.code_files = ["src/stack.py"]
        task.test_files = []

        blocking, _ = await watcher.static_analysis_gate(task)
        assert blocking == []
