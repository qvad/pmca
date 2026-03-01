"""Tests for the file assembler — snippet merging and package init generation."""

import tempfile
from pathlib import Path

import pytest

from pmca.tasks.state import TaskStatus, TaskType
from pmca.tasks.tree import TaskTree
from pmca.utils.assembler import FileAssembler, parse_target_file
from pmca.workspace.file_manager import FileManager


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def file_manager(workspace):
    return FileManager(workspace)


@pytest.fixture
def assembler(file_manager):
    return FileAssembler(file_manager)


@pytest.fixture
def task_tree():
    tree = TaskTree()
    root = tree.create_root("Project")
    root.status = TaskStatus.DECOMPOSED
    root.spec = "Build a package"
    c1 = tree.add_child(root.id, "Post model", TaskType.FUNCTION)
    c1.status = TaskStatus.VERIFIED
    c2 = tree.add_child(root.id, "User model", TaskType.FUNCTION)
    c2.status = TaskStatus.VERIFIED
    return tree


class TestFileAssembler:
    def test_assemble_single_snippet_per_file(self, assembler, task_tree):
        root = task_tree.root
        children = task_tree.get_children(root.id)
        snippet_store = {
            f"{children[0].id}:src/models.py": "class Post:\n    pass\n",
            f"{children[1].id}:src/users.py": "class User:\n    pass\n",
        }
        paths = assembler.assemble(root, task_tree, snippet_store)
        assert sorted(paths) == ["src/models.py", "src/users.py"]

    def test_assemble_multiple_snippets_same_file(self, assembler, task_tree):
        root = task_tree.root
        children = task_tree.get_children(root.id)
        snippet_store = {
            f"{children[0].id}:src/models.py": (
                "from datetime import datetime\n\n"
                "class Post:\n    title: str = ''\n"
            ),
            f"{children[1].id}:src/models.py": (
                "from datetime import datetime\n"
                "from typing import Optional\n\n"
                "class Comment:\n    text: str = ''\n"
            ),
        }
        paths = assembler.assemble(root, task_tree, snippet_store)
        assert paths == ["src/models.py"]

        content = assembler._fm.read_file("src/models.py")
        # Imports should be deduplicated
        assert content.count("from datetime import datetime") == 1
        # Both classes should be present
        assert "class Post:" in content
        assert "class Comment:" in content

    def test_merge_deduplicates_imports(self, assembler):
        s1 = "import os\nfrom pathlib import Path\n\ndef foo():\n    pass\n"
        s2 = "import os\nimport sys\n\ndef bar():\n    pass\n"
        merged = assembler._merge_snippets([s1, s2], "test.py")
        # Each import should appear exactly once
        assert merged.count("import os") == 1
        assert "import sys" in merged
        assert "from pathlib import Path" in merged
        assert "def foo():" in merged
        assert "def bar():" in merged

    def test_merge_later_definition_wins(self, assembler):
        s1 = "def process():\n    return 1\n"
        s2 = "def process():\n    return 2\n"
        merged = assembler._merge_snippets([s1, s2], "test.py")
        # Later definition should win
        assert "return 2" in merged
        assert merged.count("def process()") == 1

    def test_ensure_package_init_files(self, assembler):
        paths = ["src/blog/models.py", "src/blog/views.py", "src/utils.py"]
        assembler.ensure_package_init_files(paths)
        assert assembler._fm.file_exists("src/__init__.py")
        assert assembler._fm.file_exists("src/blog/__init__.py")

    def test_ensure_init_no_duplicates(self, assembler):
        """Calling ensure_package_init_files twice should not error."""
        paths = ["src/pkg/mod.py"]
        assembler.ensure_package_init_files(paths)
        assembler.ensure_package_init_files(paths)  # idempotent
        assert assembler._fm.file_exists("src/__init__.py")
        assert assembler._fm.file_exists("src/pkg/__init__.py")

    def test_split_imports(self):
        code = "import os\nfrom sys import argv\n\ndef main():\n    pass\n"
        imports, body = FileAssembler._split_imports(code)
        assert imports == ["import os", "from sys import argv"]
        assert "def main():" in body

    def test_extract_definitions(self):
        body = "def foo():\n    return 1\n\nclass Bar:\n    x = 1\n"
        defs = FileAssembler._extract_definitions(body)
        assert "foo" in defs
        assert "Bar" in defs

    def test_extract_definitions_syntax_error(self):
        defs = FileAssembler._extract_definitions("def (broken:")
        assert "__unparseable__" in defs


class TestParseTargetFile:
    def test_basic(self):
        spec = "TARGET_FILE: src/models.py\nEXPORTS: Post\n"
        assert parse_target_file(spec) == "src/models.py"

    def test_normalizes_deep_path(self):
        """Package paths like src/blog/models.py are normalized to src/models.py."""
        spec = "TARGET_FILE: src/blog/models.py\nEXPORTS: Post\n"
        assert parse_target_file(spec) == "src/models.py"

    def test_normalizes_package_path(self):
        """src/taskboard/filters.py → src/filters.py."""
        spec = "TARGET_FILE: src/taskboard/filters.py\nEXPORTS: filter_by_status\n"
        assert parse_target_file(spec) == "src/filters.py"

    def test_missing(self):
        assert parse_target_file("no target here") is None

    def test_embedded(self):
        spec = "Some text\nTARGET_FILE: views.py\nmore text"
        assert parse_target_file(spec) == "views.py"
