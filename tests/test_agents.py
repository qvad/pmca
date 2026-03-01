"""Tests for agent implementations with mocked model responses."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmca.agents.architect import ArchitectAgent
from pmca.agents.coder import CoderAgent
from pmca.agents.reviewer import ReviewerAgent
from pmca.agents.watcher import WatcherAgent
from pmca.models.config import AgentRole, Config, ModelConfig
from pmca.models.manager import ModelManager
from pmca.tasks.state import ReviewResult, TaskStatus, TaskType
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
def mock_manager(config):
    manager = ModelManager(config)
    manager.generate = AsyncMock()
    manager.ensure_loaded = AsyncMock(return_value="test:14b")
    return manager


class TestArchitectAgent:
    @pytest.mark.asyncio
    async def test_generate_spec(self, mock_manager):
        agent = ArchitectAgent(mock_manager)
        mock_manager.generate.return_value = "### Purpose\nA test spec"

        task = TaskNode(title="Build a calculator")
        spec = await agent.generate_spec(task, context="")
        assert "Purpose" in spec
        mock_manager.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_decompose_leaf(self, mock_manager):
        agent = ArchitectAgent(mock_manager)
        mock_manager.generate.return_value = "LEAF"

        task = TaskNode(title="Add two numbers", spec="Simple function")
        subtasks = await agent.decompose(task)
        assert subtasks == []

    @pytest.mark.asyncio
    async def test_decompose_with_subtasks(self, mock_manager):
        agent = ArchitectAgent(mock_manager)
        mock_manager.generate.return_value = '''
Here are the subtasks:
```json
[
  {"title": "Add function", "type": "function", "description": "Implement addition"},
  {"title": "Subtract function", "type": "function", "description": "Implement subtraction"}
]
```
'''
        task = TaskNode(title="Calculator", spec="Build a calculator")
        subtasks = await agent.decompose(task)
        assert len(subtasks) == 2
        assert subtasks[0]["title"] == "Add function"
        assert subtasks[1]["title"] == "Subtract function"

    @pytest.mark.asyncio
    async def test_decompose_max_children(self, mock_manager):
        agent = ArchitectAgent(mock_manager, max_children=2)
        mock_manager.generate.return_value = '''
```json
[
  {"title": "A", "type": "function", "description": "a"},
  {"title": "B", "type": "function", "description": "b"},
  {"title": "C", "type": "function", "description": "c"}
]
```
'''
        task = TaskNode(title="Big task", spec="Many subtasks")
        subtasks = await agent.decompose(task)
        assert len(subtasks) == 2

    @pytest.mark.asyncio
    async def test_refine_spec(self, mock_manager):
        agent = ArchitectAgent(mock_manager)
        mock_manager.generate.return_value = "### Purpose\nRefined spec"

        task = TaskNode(title="Test", spec="Original spec")
        feedback = ReviewResult(
            passed=False,
            issues=["Missing edge cases"],
            suggestions=["Add error handling"],
            timestamp=datetime.now(),
            model_used="reviewer",
        )
        refined = await agent.refine_spec(task, feedback)
        assert "Refined" in refined


class TestCoderAgent:
    @pytest.mark.asyncio
    async def test_implement(self, mock_manager):
        agent = CoderAgent(mock_manager)
        mock_manager.generate.return_value = '''
```python
# filepath: src/calculator.py
def add(a: int, b: int) -> int:
    return a + b
```

```python
# filepath: tests/test_calculator.py
def test_add():
    assert add(1, 2) == 3
```
'''
        task = TaskNode(title="Add function", spec="Implement addition")
        files = await agent.implement(task)
        assert len(files) == 2
        assert files[0].path == "src/calculator.py"
        assert "def add" in files[0].content

    @pytest.mark.asyncio
    async def test_implement_no_filepath(self, mock_manager):
        agent = CoderAgent(mock_manager)
        mock_manager.generate.return_value = '''
```python
def hello():
    print("hello")
```
'''
        task = TaskNode(title="Hello", spec="Print hello")
        files = await agent.implement(task)
        assert len(files) == 1
        assert files[0].path == "generated_0.py"

    @pytest.mark.asyncio
    async def test_fix(self, mock_manager):
        agent = CoderAgent(mock_manager)
        mock_manager.generate.return_value = '''
```python
# filepath: src/calculator.py
def add(a: int, b: int) -> int:
    if not isinstance(a, (int, float)):
        raise TypeError("a must be a number")
    return a + b
```
'''
        task = TaskNode(title="Add function", spec="Add with validation")
        files = await agent.fix(task, issues=["Missing type validation"])
        assert len(files) == 1
        assert "TypeError" in files[0].content


class TestReviewerAgent:
    @pytest.mark.asyncio
    async def test_verify_spec_pass(self, mock_manager):
        agent = ReviewerAgent(mock_manager)
        mock_manager.generate.return_value = '''
```json
{"passed": true, "issues": [], "suggestions": ["Consider edge cases"]}
```
'''
        result = await agent.verify_spec("child spec", "parent spec")
        assert result.passed
        assert result.issues == []

    @pytest.mark.asyncio
    async def test_verify_spec_fail(self, mock_manager):
        agent = ReviewerAgent(mock_manager)
        mock_manager.generate.return_value = '''
```json
{"passed": false, "issues": ["Missing error handling"], "suggestions": ["Add try/except"]}
```
'''
        result = await agent.verify_spec("child spec", "parent spec")
        assert not result.passed
        assert "Missing error handling" in result.issues

    @pytest.mark.asyncio
    async def test_verify_code(self, mock_manager):
        agent = ReviewerAgent(mock_manager)
        mock_manager.generate.return_value = '{"passed": true, "issues": [], "suggestions": []}'
        result = await agent.verify_code("def add(a, b): return a + b", "Add function")
        assert result.passed

    @pytest.mark.asyncio
    async def test_parse_failure(self, mock_manager):
        agent = ReviewerAgent(mock_manager)
        mock_manager.generate.return_value = "This is not JSON at all"
        result = await agent.verify_code("code", "spec")
        assert not result.passed
        assert "Failed to parse" in result.issues[0]


class TestWatcherAgent:
    @pytest.mark.asyncio
    async def test_check_not_faked_pass(self, mock_manager):
        agent = WatcherAgent(mock_manager)
        mock_manager.generate.return_value = '{"passed": true, "issues": [], "suggestions": []}'
        result = await agent.check_not_faked("real code", "real tests")
        assert result.passed

    @pytest.mark.asyncio
    async def test_check_not_faked_fail(self, mock_manager):
        agent = WatcherAgent(mock_manager)
        mock_manager.generate.return_value = '''
{"passed": false, "issues": ["Hard-coded return value"], "suggestions": ["Implement actual logic"]}
'''
        result = await agent.check_not_faked("return 42", "assert func() == 42")
        assert not result.passed

    @pytest.mark.asyncio
    async def test_run_tests_no_test_files(self, mock_manager):
        agent = WatcherAgent(mock_manager)
        task = TaskNode(title="No tests", test_files=[])
        result = await agent.run_tests(task)
        assert result.passed
        assert result.total == 0

    def test_parse_pytest_output(self, mock_manager):
        agent = WatcherAgent(mock_manager)
        output = "===== 3 passed, 1 failed in 0.5s ====="
        total, failures = agent._parse_pytest_output(output)
        assert total == 4
        assert failures == 1

    def test_parse_pytest_output_all_passed(self, mock_manager):
        agent = WatcherAgent(mock_manager)
        output = "===== 5 passed in 0.3s ====="
        total, failures = agent._parse_pytest_output(output)
        assert total == 5
        assert failures == 0
