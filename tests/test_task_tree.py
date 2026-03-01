"""Tests for the task tree and state machine."""

import json
import tempfile
from pathlib import Path

import pytest

from pmca.tasks.state import (
    CodeFile,
    ReviewResult,
    TaskStatus,
    TaskType,
    TestResult,
    validate_transition,
)
from pmca.tasks.tree import TaskNode, TaskTree


class TestTaskStatus:
    def test_valid_transitions(self):
        assert validate_transition(TaskStatus.PENDING, TaskStatus.DESIGNING)
        assert validate_transition(TaskStatus.DESIGNING, TaskStatus.DECOMPOSED)
        assert validate_transition(TaskStatus.DESIGNING, TaskStatus.CODING)
        assert validate_transition(TaskStatus.CODING, TaskStatus.REVIEWING)
        assert validate_transition(TaskStatus.REVIEWING, TaskStatus.VERIFIED)
        assert validate_transition(TaskStatus.REVIEWING, TaskStatus.CODING)

    def test_invalid_transitions(self):
        assert not validate_transition(TaskStatus.PENDING, TaskStatus.CODING)
        assert not validate_transition(TaskStatus.VERIFIED, TaskStatus.PENDING)
        assert not validate_transition(TaskStatus.CODING, TaskStatus.DESIGNING)

    def test_failed_can_restart(self):
        assert validate_transition(TaskStatus.FAILED, TaskStatus.PENDING)

    def test_verified_is_terminal(self):
        for status in TaskStatus:
            assert not validate_transition(TaskStatus.VERIFIED, status)


class TestTaskNode:
    def test_create_default(self):
        node = TaskNode(title="Test task")
        assert node.title == "Test task"
        assert node.status == TaskStatus.PENDING
        assert node.depth == 0
        assert node.is_leaf
        assert not node.is_complete
        assert not node.is_failed

    def test_transition_valid(self):
        node = TaskNode(title="Test")
        node.transition(TaskStatus.DESIGNING)
        assert node.status == TaskStatus.DESIGNING

    def test_transition_invalid_raises(self):
        node = TaskNode(title="Test")
        with pytest.raises(ValueError, match="Invalid transition"):
            node.transition(TaskStatus.CODING)

    def test_serialization_roundtrip(self):
        node = TaskNode(
            title="Test task",
            spec="Some spec",
            code_files=["src/test.py"],
            task_type=TaskType.FUNCTION,
        )
        data = node.to_dict()
        restored = TaskNode.from_dict(data)
        assert restored.title == node.title
        assert restored.spec == node.spec
        assert restored.code_files == node.code_files
        assert restored.task_type == TaskType.FUNCTION

    def test_is_leaf(self):
        node = TaskNode(title="Leaf")
        assert node.is_leaf
        node.children.append("child-id")
        assert not node.is_leaf


class TestReviewResult:
    def test_serialization(self):
        from datetime import datetime
        review = ReviewResult(
            passed=True,
            issues=["issue1"],
            suggestions=["suggestion1"],
            timestamp=datetime(2025, 1, 1),
            model_used="test",
        )
        data = review.to_dict()
        restored = ReviewResult.from_dict(data)
        assert restored.passed == review.passed
        assert restored.issues == review.issues
        assert restored.model_used == review.model_used


class TestCodeFile:
    def test_serialization(self):
        cf = CodeFile(path="test.py", content="print('hello')")
        data = cf.to_dict()
        restored = CodeFile.from_dict(data)
        assert restored.path == cf.path
        assert restored.content == cf.content


class TestTestResult:
    def test_serialization(self):
        tr = TestResult(passed=True, total=5, failures=0, output="ok", errors=[])
        data = tr.to_dict()
        restored = TestResult.from_dict(data)
        assert restored.passed
        assert restored.total == 5


class TestTaskTree:
    def test_create_root(self):
        tree = TaskTree()
        root = tree.create_root("My project")
        assert root.title == "My project"
        assert root.depth == 0
        assert tree.root is root

    def test_add_child(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        child = tree.add_child(root.id, "Child", TaskType.MODULE)
        assert child.parent_id == root.id
        assert child.depth == 1
        assert child.id in root.children

    def test_get_children(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        c1 = tree.add_child(root.id, "C1", TaskType.FUNCTION)
        c2 = tree.add_child(root.id, "C2", TaskType.FUNCTION)
        children = tree.get_children(root.id)
        assert len(children) == 2
        assert c1 in children
        assert c2 in children

    def test_get_siblings(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        c1 = tree.add_child(root.id, "C1", TaskType.FUNCTION)
        c2 = tree.add_child(root.id, "C2", TaskType.FUNCTION)
        siblings = tree.get_siblings(c1.id)
        assert len(siblings) == 1
        assert c2 in siblings

    def test_all_children_complete(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        c1 = tree.add_child(root.id, "C1", TaskType.FUNCTION)
        c2 = tree.add_child(root.id, "C2", TaskType.FUNCTION)

        assert not tree.all_children_complete(root.id)

        c1.status = TaskStatus.VERIFIED
        assert not tree.all_children_complete(root.id)

        c2.status = TaskStatus.VERIFIED
        assert tree.all_children_complete(root.id)

    def test_walk(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        c1 = tree.add_child(root.id, "C1", TaskType.MODULE)
        c2 = tree.add_child(root.id, "C2", TaskType.MODULE)
        gc1 = tree.add_child(c1.id, "GC1", TaskType.FUNCTION)

        walked = tree.walk()
        assert len(walked) == 4
        assert walked[0] is root

    def test_summary(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        c1 = tree.add_child(root.id, "C1", TaskType.FUNCTION)
        c1.status = TaskStatus.VERIFIED
        summary = tree.summary()
        assert summary["pending"] == 1
        assert summary["verified"] == 1

    def test_persistence(self):
        tree = TaskTree()
        root = tree.create_root("Root")
        tree.add_child(root.id, "C1", TaskType.FUNCTION)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            tree.save(path)
            loaded = TaskTree.load(path)
            assert loaded.root is not None
            assert loaded.root.title == "Root"
            children = loaded.get_children(loaded.root.id)
            assert len(children) == 1
            assert children[0].title == "C1"
        finally:
            path.unlink()

    def test_get_node_missing_raises(self):
        tree = TaskTree()
        with pytest.raises(KeyError):
            tree.get_node("nonexistent")

    def test_add_child_missing_parent_raises(self):
        tree = TaskTree()
        with pytest.raises(KeyError):
            tree.add_child("nonexistent", "C1", TaskType.FUNCTION)
