"""Task tree data structure for hierarchical task management."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pmca.models.config import AgentRole
from pmca.tasks.state import (
    ReviewResult,
    TaskStatus,
    TaskType,
    validate_transition,
)


@dataclass
class TaskNode:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    depth: int = 0
    status: TaskStatus = TaskStatus.PENDING
    task_type: TaskType = TaskType.ARCHITECTURE
    title: str = ""
    spec: str = ""
    code_files: dict[str, str] = field(default_factory=dict)
    test_files: dict[str, str] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    review_history: list[ReviewResult] = field(default_factory=list)
    retry_count: int = 0
    git_checkpoint: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Transient cascade state (set by orchestrator, not persisted)
    _lint_issues: list[str] = field(default_factory=list, repr=False)
    _missing_spec_names: list[str] = field(default_factory=list, repr=False)
    _mutation_oracle_warning: str = field(default="", repr=False)
    _coder_role_override: AgentRole | None = field(default=None, repr=False)
    _think_override: bool | None = field(default=None, repr=False)

    def transition(self, new_status: TaskStatus) -> None:
        """Transition to a new status with validation."""
        if not validate_transition(self.status, new_status):
            raise ValueError(
                f"Invalid transition: {self.status.value} -> {new_status.value}"
            )
        self.status = new_status
        self.updated_at = datetime.now()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_complete(self) -> bool:
        return self.status == TaskStatus.VERIFIED

    @property
    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "status": self.status.value,
            "task_type": self.task_type.value,
            "title": self.title,
            "spec": self.spec,
            "code_files": self.code_files,
            "test_files": self.test_files,
            "children": self.children,
            "review_history": [r.to_dict() for r in self.review_history],
            "retry_count": self.retry_count,
            "git_checkpoint": self.git_checkpoint,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskNode:
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            depth=data.get("depth", 0),
            status=TaskStatus(data["status"]),
            task_type=TaskType(data.get("task_type", "architecture")),
            title=data.get("title", ""),
            spec=data.get("spec", ""),
            code_files=data.get("code_files", {}),
            test_files=data.get("test_files", {}),
            children=data.get("children", []),
            review_history=[
                ReviewResult.from_dict(r) for r in data.get("review_history", [])
            ],
            retry_count=data.get("retry_count", 0),
            git_checkpoint=data.get("git_checkpoint"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(),
        )


class TaskTree:
    """Manages a tree of TaskNodes with persistence."""

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}
        self._root_id: str | None = None

    @property
    def root(self) -> TaskNode | None:
        if self._root_id is None:
            return None
        return self._nodes.get(self._root_id)

    def create_root(self, title: str) -> TaskNode:
        """Create the root task node."""
        node = TaskNode(
            title=title,
            depth=0,
            task_type=TaskType.ARCHITECTURE,
        )
        self._nodes[node.id] = node
        self._root_id = node.id
        return node

    def add_child(self, parent_id: str, title: str, task_type: TaskType) -> TaskNode:
        """Add a child task to a parent node."""
        parent = self._nodes.get(parent_id)
        if parent is None:
            raise KeyError(f"Parent node {parent_id} not found")

        child = TaskNode(
            parent_id=parent_id,
            depth=parent.depth + 1,
            title=title,
            task_type=task_type,
        )
        self._nodes[child.id] = child
        parent.children.append(child.id)
        parent.updated_at = datetime.now()
        return child

    def get_node(self, node_id: str) -> TaskNode:
        """Get a node by ID."""
        node = self._nodes.get(node_id)
        if node is None:
            raise KeyError(f"Node {node_id} not found")
        return node

    def get_children(self, node_id: str) -> list[TaskNode]:
        """Get all children of a node."""
        node = self.get_node(node_id)
        return [self._nodes[cid] for cid in node.children if cid in self._nodes]

    def get_siblings(self, node_id: str) -> list[TaskNode]:
        """Get all siblings of a node (excluding itself)."""
        node = self.get_node(node_id)
        if node.parent_id is None:
            return []
        parent = self._nodes[node.parent_id]
        return [
            self._nodes[cid]
            for cid in parent.children
            if cid != node_id and cid in self._nodes
        ]

    def all_children_complete(self, node_id: str) -> bool:
        """Check if all children of a node are verified."""
        children = self.get_children(node_id)
        if not children:
            return True
        return all(c.is_complete for c in children)

    def get_failed_children(self, node_id: str) -> list[TaskNode]:
        """Get children that have failed."""
        return [c for c in self.get_children(node_id) if c.is_failed]

    def walk(self, node_id: str | None = None) -> list[TaskNode]:
        """Walk the tree depth-first from a starting node."""
        start_id = node_id or self._root_id
        if start_id is None:
            return []
        result: list[TaskNode] = []
        self._walk_recursive(start_id, result)
        return result

    def _walk_recursive(self, node_id: str, result: list[TaskNode]) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        result.append(node)
        for child_id in node.children:
            self._walk_recursive(child_id, result)

    def summary(self) -> dict[str, int]:
        """Count tasks by status."""
        counts: dict[str, int] = {}
        for node in self._nodes.values():
            key = node.status.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def save(self, path: Path) -> None:
        """Persist the task tree to a JSON file."""
        data = {
            "root_id": self._root_id,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> TaskTree:
        """Load a task tree from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        tree = cls()
        tree._root_id = data.get("root_id")
        tree._nodes = {
            nid: TaskNode.from_dict(ndata)
            for nid, ndata in data.get("nodes", {}).items()
        }
        return tree
