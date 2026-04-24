"""Language detection, extension mapping, and test command resolution."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmca.tasks.tree import TaskNode

# Top 20 languages with detection patterns, extensions, and test commands
LANGUAGES: dict[str, dict] = {
    "python": {
        "patterns": [r"\b(python|py|pytest|pip|django|flask|fastapi)\b", r"\.py\b"],
        "ext": ".py",
        "test_ext": ".py",
        "test_cmd": ["python", "-m", "pytest"],
        "test_prefix": "test_",
    },
    "go": {
        "patterns": [r"\b(go|golang|goroutine|chan\b|func\s)", r"\.go\b"],
        "ext": ".go",
        "test_ext": "_test.go",
        "test_cmd": ["go", "test", "./..."],
        "test_prefix": "",
    },
    "typescript": {
        "patterns": [r"\b(typescript|ts|tsx|angular|nextjs|deno)\b", r"\.ts\b"],
        "ext": ".ts",
        "test_ext": ".test.ts",
        "test_cmd": ["npx", "jest"],
        "test_prefix": "",
    },
    "javascript": {
        "patterns": [r"\b(javascript|js|jsx|node|express|react)\b", r"\.js\b"],
        "ext": ".js",
        "test_ext": ".test.js",
        "test_cmd": ["npx", "jest"],
        "test_prefix": "",
    },
    "rust": {
        "patterns": [r"\b(rust|cargo|crate|fn\s|impl\s|struct\s.*\{)\b", r"\.rs\b"],
        "ext": ".rs",
        "test_ext": ".rs",
        "test_cmd": ["cargo", "test"],
        "test_prefix": "",
    },
    "java": {
        "patterns": [r"\b(java|jvm|maven|gradle|spring|junit)\b", r"\.java\b"],
        "ext": ".java",
        "test_ext": "Test.java",
        "test_cmd": ["mvn", "test"],
        "test_prefix": "",
    },
    "kotlin": {
        "patterns": [r"\b(kotlin|kt|ktor|coroutine)\b", r"\.kt\b"],
        "ext": ".kt",
        "test_ext": "Test.kt",
        "test_cmd": ["gradle", "test"],
        "test_prefix": "",
    },
    "csharp": {
        "patterns": [r"\b(c#|csharp|dotnet|\.net|asp\.net|nuget)\b", r"\.cs\b"],
        "ext": ".cs",
        "test_ext": "Tests.cs",
        "test_cmd": ["dotnet", "test"],
        "test_prefix": "",
    },
    "cpp": {
        "patterns": [r"\b(c\+\+|cpp|cmake|iostream|std::)\b", r"\.(cpp|cc|cxx|hpp)\b"],
        "ext": ".cpp",
        "test_ext": "_test.cpp",
        "test_cmd": ["ctest"],
        "test_prefix": "",
    },
    "c": {
        "patterns": [r"\b(c language|\.c file|#include\s*<stdio)\b", r"\.c\b"],
        "ext": ".c",
        "test_ext": "_test.c",
        "test_cmd": ["make", "test"],
        "test_prefix": "",
    },
    "ruby": {
        "patterns": [r"\b(ruby|rb|rails|rspec|gem|bundler)\b", r"\.rb\b"],
        "ext": ".rb",
        "test_ext": "_spec.rb",
        "test_cmd": ["rspec"],
        "test_prefix": "",
    },
    "php": {
        "patterns": [r"\b(php|laravel|composer|symfony|phpunit)\b", r"\.php\b"],
        "ext": ".php",
        "test_ext": "Test.php",
        "test_cmd": ["phpunit"],
        "test_prefix": "",
    },
    "swift": {
        "patterns": [r"\b(swift|swiftui|xcode|ios app|macos app)\b", r"\.swift\b"],
        "ext": ".swift",
        "test_ext": "Tests.swift",
        "test_cmd": ["swift", "test"],
        "test_prefix": "",
    },
    "scala": {
        "patterns": [r"\b(scala|sbt|akka|play framework)\b", r"\.scala\b"],
        "ext": ".scala",
        "test_ext": "Spec.scala",
        "test_cmd": ["sbt", "test"],
        "test_prefix": "",
    },
    "elixir": {
        "patterns": [r"\b(elixir|phoenix|mix|genserver|ecto)\b", r"\.ex\b"],
        "ext": ".ex",
        "test_ext": "_test.exs",
        "test_cmd": ["mix", "test"],
        "test_prefix": "",
    },
    "lua": {
        "patterns": [r"\b(lua|luajit|love2d)\b", r"\.lua\b"],
        "ext": ".lua",
        "test_ext": "_test.lua",
        "test_cmd": ["busted"],
        "test_prefix": "",
    },
    "r": {
        "patterns": [r"\b(r language|rstudio|tidyverse|ggplot)\b", r"\.R\b"],
        "ext": ".R",
        "test_ext": "_test.R",
        "test_cmd": ["Rscript", "-e", "testthat::test_dir('tests')"],
        "test_prefix": "",
    },
    "dart": {
        "patterns": [r"\b(dart|flutter|pub\.dev)\b", r"\.dart\b"],
        "ext": ".dart",
        "test_ext": "_test.dart",
        "test_cmd": ["dart", "test"],
        "test_prefix": "",
    },
    "shell": {
        "patterns": [r"\b(bash|shell|sh script|zsh)\b", r"\.(sh|bash)\b"],
        "ext": ".sh",
        "test_ext": "_test.sh",
        "test_cmd": ["bats"],
        "test_prefix": "",
    },
    "sql": {
        "patterns": [r"\b(sql|postgresql|mysql|sqlite|database query)\b", r"\.sql\b"],
        "ext": ".sql",
        "test_ext": ".sql",
        "test_cmd": [],
        "test_prefix": "",
    },
}


def detect_language(task: TaskNode) -> str:
    """Detect programming language from task title and spec."""
    text = (task.title + (task.spec or "")).lower()

    # Check each language's patterns (order matters — more specific first)
    for lang, info in LANGUAGES.items():
        for pattern in info["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                return lang

    return "python"  # default


def get_extension(lang: str) -> str:
    """Map language name to primary file extension."""
    info = LANGUAGES.get(lang)
    return info["ext"] if info else ".py"


def get_test_extension(lang: str) -> str:
    """Map language to test file extension/suffix."""
    info = LANGUAGES.get(lang)
    return info["test_ext"] if info else ".py"


def get_test_command(lang: str) -> list[str]:
    """Map language to test runner command."""
    info = LANGUAGES.get(lang)
    return info["test_cmd"] if info else ["python", "-m", "pytest"]


def is_test_file(path: str, lang: str = "python") -> bool:
    """Check if a file path looks like a test file for the given language."""
    info = LANGUAGES.get(lang)
    if not info:
        return path.startswith("test") or "/test_" in path

    test_ext = info["test_ext"]
    test_prefix = info["test_prefix"]

    name = path.split("/")[-1]

    # Python: test_*.py
    if test_prefix and name.startswith(test_prefix):
        return True
    # Go: *_test.go
    if test_ext and name.endswith(test_ext):
        return True
    # Java/C#: *Test.java, *Tests.cs
    if test_ext and name.endswith(test_ext):
        return True
    # Generic: test in path
    return "test" in name.lower()
