"""Tests for multi-file project mode: interface extraction, dep sort, context."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from pmca.agents.architect import ArchitectAgent
from pmca.models.config import AgentRole, Config, ModelConfig
from pmca.models.manager import ModelManager
from pmca.orchestrator import Orchestrator
from pmca.tasks.state import ReviewResult, TaskStatus, TaskType
from pmca.tasks.tree import TaskNode, TaskTree
from pmca.agents.coder import CoderAgent
from pmca.utils.context import (
    ContextManager,
    _build_import_hint,
    _extract_exports,
    _extract_interface_section,
    _extract_metadata_lines,
    _extract_target_file,
)


# ---------------------------------------------------------------------------
# AST interface extraction
# ---------------------------------------------------------------------------


class TestExtractInterface:
    def test_simple_function(self):
        code = "def add(a: int, b: int) -> int:\n    return a + b\n"
        iface = ArchitectAgent.extract_interface_from_code(code)
        assert "def add(a: int, b: int) -> int: ..." in iface

    def test_class_with_methods(self):
        code = (
            "class Post:\n"
            "    def __init__(self, title: str) -> None:\n"
            "        self.title = title\n"
            "    def save(self) -> None:\n"
            "        pass\n"
        )
        iface = ArchitectAgent.extract_interface_from_code(code)
        assert "class Post:" in iface
        assert "def __init__(self, title: str) -> None: ..." in iface
        assert "def save(self) -> None: ..." in iface

    def test_syntax_error_returns_empty(self):
        assert ArchitectAgent.extract_interface_from_code("def (broken:") == ""

    def test_empty_class(self):
        code = "class Empty:\n    pass\n"
        iface = ArchitectAgent.extract_interface_from_code(code)
        assert "class Empty:" in iface
        assert "..." in iface

    def test_async_function(self):
        code = "async def fetch(url: str) -> str:\n    return ''\n"
        iface = ArchitectAgent.extract_interface_from_code(code)
        assert "async def fetch(url: str) -> str: ..." in iface

    def test_default_args(self):
        code = "def greet(name: str, greeting: str = 'Hello') -> str:\n    return f'{greeting} {name}'\n"
        iface = ArchitectAgent.extract_interface_from_code(code)
        assert "greeting: str='Hello'" in iface or "greeting: str = 'Hello'" in iface


# ---------------------------------------------------------------------------
# Context manager helpers
# ---------------------------------------------------------------------------


class TestContextHelpers:
    def test_extract_interface_section(self):
        spec = "Some spec text\n[INTERFACE]\nclass Foo:\n    def bar(self): ..."
        assert "class Foo:" in _extract_interface_section(spec)

    def test_extract_interface_section_missing(self):
        assert _extract_interface_section("no interface here") == ""

    def test_extract_metadata_lines(self):
        spec = (
            "TARGET_FILE: src/models.py\n"
            "EXPORTS: Post, Category\n"
            "DEPENDS_ON: NONE\n"
            "\nSome other text\n"
        )
        meta = _extract_metadata_lines(spec)
        assert "TARGET_FILE: src/models.py" in meta
        assert "EXPORTS: Post, Category" in meta
        assert "DEPENDS_ON: NONE" in meta
        assert "Some other text" not in meta


class TestContextManagerProjectMode:
    def test_sibling_with_interface(self):
        tree = TaskTree()
        root = tree.create_root("Project")
        root.status = TaskStatus.DECOMPOSED

        c1 = tree.add_child(root.id, "Model", TaskType.FUNCTION)
        c1.status = TaskStatus.VERIFIED
        c1.spec = "TARGET_FILE: models.py\nEXPORTS: Post\n[INTERFACE]\nclass Post:\n    def save(self): ..."

        c2 = tree.add_child(root.id, "View", TaskType.FUNCTION)
        c2.status = TaskStatus.PENDING
        c2.spec = "TARGET_FILE: views.py\nEXPORTS: list_posts\nDEPENDS_ON: Post"

        cm = ContextManager(tree, project_mode=True)
        ctx = cm.build_context(c2)

        # Completed sibling should show interface
        assert "class Post:" in ctx
        assert "[DONE]" in ctx

    def test_sibling_pending_shows_metadata(self):
        tree = TaskTree()
        root = tree.create_root("Project")
        root.status = TaskStatus.DECOMPOSED

        c1 = tree.add_child(root.id, "Model", TaskType.FUNCTION)
        c1.status = TaskStatus.PENDING
        c1.spec = "TARGET_FILE: models.py\nEXPORTS: Post\nDEPENDS_ON: NONE"

        c2 = tree.add_child(root.id, "View", TaskType.FUNCTION)
        c2.spec = "TARGET_FILE: views.py"

        cm = ContextManager(tree, project_mode=True)
        ctx = cm.build_context(c2)
        assert "TARGET_FILE: models.py" in ctx
        assert "EXPORTS: Post" in ctx


# ---------------------------------------------------------------------------
# Dependency sorting
# ---------------------------------------------------------------------------


class TestSortByDependencies:
    def test_simple_chain(self):
        """A depends on nothing, B depends on A's export."""
        tree = TaskTree()
        root = tree.create_root("Root")

        a = tree.add_child(root.id, "A", TaskType.FUNCTION)
        a.spec = "TARGET_FILE: a.py\nEXPORTS: Foo\nDEPENDS_ON: NONE"

        b = tree.add_child(root.id, "B", TaskType.FUNCTION)
        b.spec = "TARGET_FILE: b.py\nEXPORTS: Bar\nDEPENDS_ON: Foo"

        sorted_children = Orchestrator._sort_by_dependencies([a, b])
        assert sorted_children[0].title == "A"
        assert sorted_children[1].title == "B"

    def test_reverse_input_order(self):
        """Same as above but input order is reversed."""
        tree = TaskTree()
        root = tree.create_root("Root")

        a = tree.add_child(root.id, "A", TaskType.FUNCTION)
        a.spec = "TARGET_FILE: a.py\nEXPORTS: Foo\nDEPENDS_ON: NONE"

        b = tree.add_child(root.id, "B", TaskType.FUNCTION)
        b.spec = "TARGET_FILE: b.py\nEXPORTS: Bar\nDEPENDS_ON: Foo"

        sorted_children = Orchestrator._sort_by_dependencies([b, a])
        assert sorted_children[0].title == "A"
        assert sorted_children[1].title == "B"

    def test_no_metadata_preserves_order(self):
        tree = TaskTree()
        root = tree.create_root("Root")

        a = tree.add_child(root.id, "A", TaskType.FUNCTION)
        a.spec = "just a spec"

        b = tree.add_child(root.id, "B", TaskType.FUNCTION)
        b.spec = "another spec"

        sorted_children = Orchestrator._sort_by_dependencies([a, b])
        assert sorted_children[0].title == "A"
        assert sorted_children[1].title == "B"

    def test_cycle_falls_back(self):
        """Circular dependency should fall back to original order."""
        tree = TaskTree()
        root = tree.create_root("Root")

        a = tree.add_child(root.id, "A", TaskType.FUNCTION)
        a.spec = "EXPORTS: Foo\nDEPENDS_ON: Bar"

        b = tree.add_child(root.id, "B", TaskType.FUNCTION)
        b.spec = "EXPORTS: Bar\nDEPENDS_ON: Foo"

        sorted_children = Orchestrator._sort_by_dependencies([a, b])
        # Cycle → original order preserved
        assert sorted_children[0].title == "A"
        assert sorted_children[1].title == "B"

    def test_three_level_chain(self):
        """A -> B -> C dependency chain."""
        tree = TaskTree()
        root = tree.create_root("Root")

        a = tree.add_child(root.id, "A", TaskType.FUNCTION)
        a.spec = "EXPORTS: Base\nDEPENDS_ON: NONE"

        b = tree.add_child(root.id, "B", TaskType.FUNCTION)
        b.spec = "EXPORTS: Middle\nDEPENDS_ON: Base"

        c = tree.add_child(root.id, "C", TaskType.FUNCTION)
        c.spec = "EXPORTS: Top\nDEPENDS_ON: Middle"

        sorted_children = Orchestrator._sort_by_dependencies([c, b, a])
        titles = [ch.title for ch in sorted_children]
        assert titles == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# Architect project_mode flag
# ---------------------------------------------------------------------------


class TestArchitectProjectMode:
    @pytest.fixture
    def mock_manager(self):
        config = Config(
            models={
                AgentRole.ARCHITECT: ModelConfig(name="test:7b", temperature=0.3),
            }
        )
        manager = ModelManager(config)
        manager.generate = AsyncMock()
        manager.ensure_loaded = AsyncMock(return_value="test:7b")
        return manager

    @pytest.mark.asyncio
    async def test_decompose_uses_project_prompt_at_depth_0(self, mock_manager):
        agent = ArchitectAgent(mock_manager, project_mode=True)
        mock_manager.generate.return_value = "LEAF"
        task = TaskNode(title="Build pkg", spec="A package", depth=0)
        await agent.decompose(task)

        # generate(role, prompt, system=...) — prompt is second positional arg
        call_args = mock_manager.generate.call_args
        prompt = call_args[0][1]
        assert "TARGET_FILE" in prompt

    @pytest.mark.asyncio
    async def test_decompose_uses_standard_prompt_at_depth_1(self, mock_manager):
        """Project mode at depth > 0 falls back to standard prompt."""
        agent = ArchitectAgent(mock_manager, project_mode=True)
        mock_manager.generate.return_value = "LEAF"
        task = TaskNode(title="models module", spec="A module", depth=1)
        await agent.decompose(task)

        call_args = mock_manager.generate.call_args
        prompt = call_args[0][1]
        assert "TARGET_FILE" not in prompt

    @pytest.mark.asyncio
    async def test_decompose_uses_standard_prompt_no_project_mode(self, mock_manager):
        agent = ArchitectAgent(mock_manager, project_mode=False)
        mock_manager.generate.return_value = "LEAF"
        task = TaskNode(title="Add nums", spec="Simple function")
        await agent.decompose(task)

        call_args = mock_manager.generate.call_args
        prompt = call_args[0][1]
        assert "TARGET_FILE" not in prompt


# ---------------------------------------------------------------------------
# Import hint helpers
# ---------------------------------------------------------------------------


class TestImportHintHelpers:
    def test_extract_target_file(self):
        spec = "TARGET_FILE: src/models.py\nEXPORTS: Post"
        assert _extract_target_file(spec) == "src/models.py"

    def test_extract_target_file_missing(self):
        assert _extract_target_file("no target here") == ""

    def test_extract_exports(self):
        spec = "EXPORTS: Post, Category, Tag"
        exports = _extract_exports(spec)
        assert exports == ["Post", "Category", "Tag"]

    def test_extract_exports_missing(self):
        assert _extract_exports("no exports") == []

    def test_build_import_hint_with_exports(self):
        hint = _build_import_hint("src/models.py", ["Post", "Category"])
        assert hint == "  Import: from models import Post, Category\n"

    def test_build_import_hint_no_exports(self):
        hint = _build_import_hint("src/filters.py", [])
        assert hint == "  Module: filters\n"

    def test_build_import_hint_no_target(self):
        assert _build_import_hint("", ["Foo"]) == ""


class TestContextManagerImportHints:
    def test_sibling_done_shows_import_hint(self):
        tree = TaskTree()
        root = tree.create_root("Project")
        root.status = TaskStatus.DECOMPOSED

        c1 = tree.add_child(root.id, "Model", TaskType.FUNCTION)
        c1.status = TaskStatus.VERIFIED
        c1.spec = (
            "TARGET_FILE: src/models.py\n"
            "EXPORTS: Task\n"
            "DEPENDS_ON: NONE\n"
            "[INTERFACE]\n"
            "class Task:\n"
            "    def __init__(self, title: str): ..."
        )

        c2 = tree.add_child(root.id, "Filters", TaskType.FUNCTION)
        c2.spec = "TARGET_FILE: src/filters.py\nEXPORTS: filter_by_status\nDEPENDS_ON: Task"

        cm = ContextManager(tree, project_mode=True)
        ctx = cm.build_context(c2)

        assert "Import: from models import Task" in ctx
        assert "class Task:" in ctx
        assert "[DONE]" in ctx


# ---------------------------------------------------------------------------
# Coder project_mode flag
# ---------------------------------------------------------------------------


class TestCoderProjectMode:
    @pytest.fixture
    def mock_manager(self):
        config = Config(
            models={
                AgentRole.CODER: ModelConfig(name="test:7b", temperature=0.3),
            }
        )
        manager = ModelManager(config)
        manager.generate = AsyncMock()
        manager.ensure_loaded = AsyncMock(return_value="test:7b")
        return manager

    @pytest.mark.asyncio
    async def test_implement_uses_project_rules(self, mock_manager):
        coder = CoderAgent(mock_manager, project_mode=True)
        mock_manager.generate.return_value = (
            "```python\n# filepath: src/filters.py\n"
            "from models import Task\ndef filter_by_status(tasks, status):\n"
            "    return [t for t in tasks if t.status == status]\n```"
        )
        task = TaskNode(title="Implement filters", spec="Filter tasks")
        await coder.implement(task, context="")

        call_args = mock_manager.generate.call_args
        system = call_args[1].get("system", call_args[0][2] if len(call_args[0]) > 2 else "")
        assert "Multi-File Project Rules" in system
        assert "MUST import from them" in system

    @pytest.mark.asyncio
    async def test_implement_no_project_rules_by_default(self, mock_manager):
        coder = CoderAgent(mock_manager, project_mode=False)
        mock_manager.generate.return_value = (
            "```python\n# filepath: src/calc.py\ndef add(a, b):\n    return a + b\n```"
        )
        task = TaskNode(title="Calculator", spec="Add function")
        await coder.implement(task, context="")

        call_args = mock_manager.generate.call_args
        system = call_args[1].get("system", "")
        assert "Multi-File Project" not in system
