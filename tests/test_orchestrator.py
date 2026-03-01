"""Tests for the orchestrator with mocked agents and model manager."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmca.models.config import AgentRole, Config, ModelConfig
from pmca.orchestrator import Orchestrator
from pmca.tasks.state import CodeFile, ReviewResult, TaskStatus, TaskType
from pmca.tasks.tree import TaskNode


@pytest.fixture
def config():
    return Config(
        models={
            AgentRole.ARCHITECT: ModelConfig(name="test:14b", temperature=0.3),
            AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.2),
            AgentRole.REVIEWER: ModelConfig(name="test:14b", temperature=0.1),
            AgentRole.WATCHER: ModelConfig(name="test:7b", temperature=0.1),
        }
    )


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def orchestrator(config, workspace):
    orch = Orchestrator(config, workspace)
    # Mock the model manager to avoid actual Ollama calls
    orch._model_manager.generate = AsyncMock()
    orch._model_manager.ensure_loaded = AsyncMock(return_value="test:14b")
    orch._model_manager.unload_current = AsyncMock()
    orch._model_manager.close = AsyncMock()
    orch._model_manager.is_ollama_running = AsyncMock(return_value=True)
    return orch


class TestOrchestrator:
    def test_create(self, orchestrator):
        assert orchestrator._config is not None
        assert orchestrator._task_tree is not None

    @pytest.mark.asyncio
    async def test_design_phase_leaf(self, orchestrator):
        """Test design phase for a leaf task (no decomposition)."""
        # Mock architect to return a spec and LEAF for decomposition
        orchestrator._architect.generate_spec = AsyncMock(return_value="### Purpose\nAdd numbers")
        orchestrator._architect.decompose = AsyncMock(return_value=[])

        root = orchestrator._task_tree.create_root("Add two numbers")
        result = await orchestrator.design_phase(root)

        assert result.spec == "### Purpose\nAdd numbers"
        assert result.status == TaskStatus.DESIGNING  # Stays in designing for leaf

    @pytest.mark.asyncio
    async def test_design_phase_decompose(self, orchestrator):
        """Test design phase with decomposition."""
        orchestrator._architect.generate_spec = AsyncMock(
            return_value="### Purpose\nBuild calculator"
        )
        orchestrator._architect.decompose = AsyncMock(
            return_value=[
                {"title": "Add", "type": TaskType.FUNCTION, "description": "Addition"},
                {"title": "Sub", "type": TaskType.FUNCTION, "description": "Subtraction"},
            ]
        )
        orchestrator._reviewer.verify_spec = AsyncMock(
            return_value=ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=datetime.now(), model_used="reviewer",
            )
        )

        root = orchestrator._task_tree.create_root("Calculator")
        result = await orchestrator.design_phase(root)

        assert result.status == TaskStatus.DECOMPOSED
        children = orchestrator._task_tree.get_children(root.id)
        assert len(children) == 2

    @pytest.mark.asyncio
    async def test_code_phase(self, orchestrator):
        """Test coding a leaf task."""
        orchestrator._coder.implement = AsyncMock(
            return_value=[
                CodeFile(path="src/add.py", content="def add(a, b): return a + b"),
                CodeFile(path="tests/test_add.py", content="def test_add(): assert add(1,2) == 3"),
            ]
        )

        root = orchestrator._task_tree.create_root("Add function")
        root.spec = "Implement addition"
        root.status = TaskStatus.DESIGNING

        result = await orchestrator.code_phase(root)
        assert len(result.code_files) == 1
        assert len(result.test_files) == 1
        assert orchestrator._file_manager.file_exists("src/add.py")

    @pytest.mark.asyncio
    async def test_review_phase_pass(self, orchestrator):
        """Test review phase when code passes."""
        orchestrator._reviewer.verify_code = AsyncMock(
            return_value=ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=datetime.now(), model_used="reviewer",
            )
        )
        orchestrator._watcher.run_tests = AsyncMock(
            return_value=MagicMock(passed=True)
        )
        orchestrator._watcher.check_not_faked = AsyncMock(
            return_value=ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=datetime.now(), model_used="watcher",
            )
        )

        root = orchestrator._task_tree.create_root("Test task")
        root.spec = "Test spec"
        root.status = TaskStatus.CODING
        root.code_files = ["src/test.py"]
        root.test_files = ["tests/test_test.py"]

        # Write the files so _gather_code works
        orchestrator._file_manager.write_file("src/test.py", "def foo(): return 1")
        orchestrator._file_manager.write_file("tests/test_test.py", "def test_foo(): pass")

        # Disable git checkpointing for this test
        orchestrator._config.workspace.git_checkpoint = False

        result = await orchestrator.review_phase(root)
        assert result.status == TaskStatus.VERIFIED

    @pytest.mark.asyncio
    async def test_review_phase_retry(self, orchestrator):
        """Test review phase retries on failure then passes."""
        fail_review = ReviewResult(
            passed=False, issues=["Bug found"], suggestions=["Fix it"],
            timestamp=datetime.now(), model_used="reviewer",
        )
        pass_review = ReviewResult(
            passed=True, issues=[], suggestions=[],
            timestamp=datetime.now(), model_used="reviewer",
        )

        orchestrator._reviewer.verify_code = AsyncMock(
            side_effect=[fail_review, pass_review]
        )
        orchestrator._coder.fix = AsyncMock(
            return_value=[CodeFile(path="src/test.py", content="def foo(): return 1")]
        )
        orchestrator._watcher.run_tests = AsyncMock(
            return_value=MagicMock(passed=True)
        )
        orchestrator._watcher.check_not_faked = AsyncMock(
            return_value=ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=datetime.now(), model_used="watcher",
            )
        )

        root = orchestrator._task_tree.create_root("Test task")
        root.spec = "Test spec"
        root.status = TaskStatus.CODING
        root.code_files = ["src/test.py"]
        root.test_files = ["tests/test_test.py"]

        orchestrator._file_manager.write_file("src/test.py", "def foo(): return 1")
        orchestrator._file_manager.write_file("tests/test_test.py", "def test_foo(): pass")
        orchestrator._config.workspace.git_checkpoint = False

        result = await orchestrator.review_phase(root)
        assert result.status == TaskStatus.VERIFIED
        assert orchestrator._coder.fix.call_count == 1

    @pytest.mark.asyncio
    async def test_integrate_phase(self, orchestrator):
        """Test integration phase with all children complete."""
        orchestrator._reviewer.verify_integration = AsyncMock(
            return_value=ReviewResult(
                passed=True, issues=[], suggestions=[],
                timestamp=datetime.now(), model_used="reviewer",
            )
        )

        root = orchestrator._task_tree.create_root("Calculator")
        root.status = TaskStatus.DECOMPOSED
        root.spec = "Build calculator"

        c1 = orchestrator._task_tree.add_child(root.id, "Add", TaskType.FUNCTION)
        c1.status = TaskStatus.VERIFIED
        c1.spec = "Addition"
        c1.code_files = ["src/add.py"]

        c2 = orchestrator._task_tree.add_child(root.id, "Sub", TaskType.FUNCTION)
        c2.status = TaskStatus.VERIFIED
        c2.spec = "Subtraction"
        c2.code_files = ["src/sub.py"]

        orchestrator._config.workspace.git_checkpoint = False
        result = await orchestrator.integrate_phase(root)
        assert result.status == TaskStatus.VERIFIED

    def test_print_tree(self, orchestrator, capsys):
        """Test tree printing doesn't crash."""
        root = orchestrator._task_tree.create_root("Root")
        orchestrator._task_tree.add_child(root.id, "C1", TaskType.FUNCTION)
        orchestrator.print_tree()  # Should not raise

    def test_load_state(self, config, workspace):
        """Test loading state from disk."""
        # Create and save state
        orch = Orchestrator(config, workspace)
        orch._task_tree.create_root("Test")
        orch._save_state()

        # Load it back
        loaded = Orchestrator.load_state(config, workspace)
        assert loaded.task_tree.root is not None
        assert loaded.task_tree.root.title == "Test"
