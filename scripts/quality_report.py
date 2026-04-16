#!/usr/bin/env python3
"""Code quality report — raw model vs PMCA cascade, static + LLM analysis.

Generates code from each model (raw), runs static analysis (ruff, mypy, radon),
AST structural checks, and produces a side-by-side quality comparison.

Usage:
    OLLAMA_HOST=http://localhost:11435 python scripts/quality_report.py
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11435")

SYSTEM = (
    "You implement Python classes from descriptions. "
    "Output ONLY the complete Python code. No markdown fences. No tests. No examples. No explanations."
)

# Two tasks: one medium, one complex
TASKS = {
    "linked_list": {
        "request": (
            "Create a LinkedList class (singly linked) with methods: "
            "append(value), prepend(value), delete(value) removes first occurrence "
            "(raise ValueError if not found), find(value) -> bool, size() -> int, "
            "to_list() -> list. The list should maintain insertion order."
        ),
        "checks": [
            ("has Node class", lambda c: "class Node" in c or "class _Node" in c),
            ("head pointer", lambda c: "self.head" in c or "self._head" in c),
            ("size counter or traversal", lambda c: "self._size" in c or "self.count" in c or "self._count" in c or "count" in c),
            ("delete raises ValueError", lambda c: "ValueError" in c),
            ("find returns bool", lambda c: "return True" in c and "return False" in c),
        ],
    },
    "todo_manager": {
        "request": (
            "Create a TodoManager class for managing todo tasks. "
            "Constructor takes no arguments. Methods: "
            "add(title: str, due_date: str = None, priority: int = 1, tags: list = None) -> int "
            "returns unique id starting from 1. priority must be 1-5 (raise ValueError). "
            "complete(task_id: int) -> bool marks done, False if not found. "
            "delete(task_id: int) -> bool removes, False if not found. "
            "get(task_id: int) -> dict with keys id,title,done,due_date,priority,tags. Raise KeyError if not found. "
            "list_tasks(status='all', sort_by='priority', tag=None) -> list[dict]. "
            "sort_by priority descending or due_date ascending (None last). "
            "search(query) case-insensitive title match. "
            "overdue(reference_date) pending tasks with due_date before reference_date. "
            "stats() -> dict with total,done,pending counts."
        ),
        "checks": [
            ("uses id counter (not len)", lambda c: "self.next_id" in c or "self._next_id" in c or "self._id_counter" in c or "self._counter" in c),
            ("tags default None not []", lambda c: "tags=None" in c.replace(" ", "") or "tags: list = None" in c),
            ("priority validation", lambda c: "ValueError" in c and "priority" in c.lower()),
            ("get raises KeyError", lambda c: "KeyError" in c),
            ("sort priority descending", lambda c: "reverse=True" in c),
            ("None date handling in sort", lambda c: ("is None" in c or "float(" in c or "datetime.max" in c) and "due_date" in c),
            ("overdue filters done=False", lambda c: bool(re.search(r"(not\s+\w+\[.done.\]|done.*==.*False|done.*is\s+False)", c))),
            ("delete returns bool", lambda c: "return False" in c and "def delete" in c),
            ("search case-insensitive", lambda c: ".lower()" in c and "def search" in c),
        ],
    },
}

MODELS = ["deepseek-coder-v2:16b", "codegemma:7b-instruct", "gemma4:e2b"]


@dataclass
class QualityScore:
    model: str
    task: str
    code: str
    lines: int = 0
    methods: int = 0
    classes: int = 0
    has_type_hints: bool = False
    type_hint_ratio: float = 0.0
    has_docstrings: bool = False
    docstring_ratio: float = 0.0
    mutable_defaults: list[str] = field(default_factory=list)
    shadows: list[str] = field(default_factory=list)
    bare_excepts: int = 0
    syntax_valid: bool = True
    syntax_error: str = ""
    spec_checks_passed: int = 0
    spec_checks_total: int = 0
    spec_failures: list[str] = field(default_factory=list)
    ruff_issues: int = 0
    ruff_details: list[str] = field(default_factory=list)
    mypy_issues: int = 0
    mypy_details: list[str] = field(default_factory=list)
    complexity_avg: float = 0.0
    complexity_max: int = 0


async def generate(model: str, request: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": request},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    async with httpx.AsyncClient(timeout=180) as c:
        resp = await c.post(f"{OLLAMA_URL}/api/chat", json=payload)
        resp.raise_for_status()
        code = resp.json()["message"]["content"]
        m = re.search(r"```python\s*\n(.*?)```", code, re.DOTALL)
        return m.group(1).strip() if m else code.strip()


def ast_analysis(code: str) -> dict:
    """AST-based structural analysis."""
    result = {
        "syntax_valid": True, "syntax_error": "",
        "lines": len(code.splitlines()),
        "methods": 0, "classes": 0,
        "type_hint_ratio": 0.0, "docstring_ratio": 0.0,
        "mutable_defaults": [], "shadows": [], "bare_excepts": 0,
    }

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["syntax_valid"] = False
        result["syntax_error"] = str(e)
        return result

    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    result["methods"] = len(funcs)
    result["classes"] = len(classes)

    # Type hints
    if funcs:
        typed = sum(1 for f in funcs if f.returns is not None or any(a.annotation for a in f.args.args if a.arg != "self"))
        result["type_hint_ratio"] = typed / len(funcs)

    # Docstrings
    if funcs:
        with_docs = sum(1 for f in funcs if ast.get_docstring(f))
        result["docstring_ratio"] = with_docs / len(funcs)

    # Mutable defaults
    for f in funcs:
        for d in f.args.defaults:
            if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                result["mutable_defaults"].append(f.name)

    # Shadows
    for cls in classes:
        attrs, methods = set(), set()
        for item in cls.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == "__init__":
                    for stmt in ast.walk(item):
                        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                                and isinstance(stmt.targets[0], ast.Attribute)
                                and isinstance(stmt.targets[0].value, ast.Name)
                                and stmt.targets[0].value.id == "self"):
                            attrs.add(stmt.targets[0].attr)
                elif item.args.args and item.args.args[0].arg == "self":
                    methods.add(item.name)
        for s in attrs & methods:
            result["shadows"].append(s)

    # Bare excepts
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            result["bare_excepts"] += 1

    return result


def run_ruff(code: str) -> tuple[int, list[str]]:
    """Run ruff check on code, return (issue_count, details)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", path, "--output-format=concise"],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip() and "Found" not in l]
        # Strip the temp path from each line
        cleaned = [re.sub(r"^.*\.py:", "", l) for l in lines]
        return len(cleaned), cleaned[:10]
    except Exception:
        return 0, []
    finally:
        os.unlink(path)


def run_mypy(code: str) -> tuple[int, list[str]]:
    """Run mypy on code, return (issue_count, details)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "mypy", path, "--ignore-missing-imports", "--no-error-summary"],
            capture_output=True, text=True, timeout=30,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines()
                 if l.strip() and "error:" in l]
        cleaned = [re.sub(r"^.*\.py:", "", l) for l in lines]
        return len(cleaned), cleaned[:10]
    except Exception:
        return 0, []
    finally:
        os.unlink(path)


def run_complexity(code: str) -> tuple[float, int]:
    """Run radon cyclomatic complexity, return (avg, max)."""
    try:
        from radon.complexity import cc_visit
        results = cc_visit(code)
        if not results:
            return 0.0, 0
        complexities = [r.complexity for r in results]
        return sum(complexities) / len(complexities), max(complexities)
    except Exception:
        return 0.0, 0


def check_spec(code: str, checks: list[tuple[str, callable]]) -> tuple[int, int, list[str]]:
    """Run spec-specific checks, return (passed, total, failures)."""
    passed = 0
    failures = []
    for name, check_fn in checks:
        try:
            if check_fn(code):
                passed += 1
            else:
                failures.append(name)
        except Exception:
            failures.append(f"{name} (check error)")
    return passed, len(checks), failures


async def analyze_model_task(model: str, task_name: str, task_info: dict) -> QualityScore:
    """Generate code and run full quality analysis."""
    code = await generate(model, task_info["request"])

    score = QualityScore(model=model, task=task_name, code=code)

    # AST analysis
    ast_result = ast_analysis(code)
    score.lines = ast_result["lines"]
    score.methods = ast_result["methods"]
    score.classes = ast_result["classes"]
    score.type_hint_ratio = ast_result["type_hint_ratio"]
    score.docstring_ratio = ast_result["docstring_ratio"]
    score.mutable_defaults = ast_result["mutable_defaults"]
    score.shadows = ast_result["shadows"]
    score.bare_excepts = ast_result["bare_excepts"]
    score.syntax_valid = ast_result["syntax_valid"]
    score.syntax_error = ast_result["syntax_error"]

    if not score.syntax_valid:
        return score

    # Spec checks
    passed, total, failures = check_spec(code, task_info["checks"])
    score.spec_checks_passed = passed
    score.spec_checks_total = total
    score.spec_failures = failures

    # Static analysis
    score.ruff_issues, score.ruff_details = run_ruff(code)
    score.mypy_issues, score.mypy_details = run_mypy(code)

    # Complexity
    score.complexity_avg, score.complexity_max = run_complexity(code)

    return score


def print_report(scores: list[QualityScore]) -> None:
    """Print formatted comparison report."""
    # Group by task
    by_task: dict[str, list[QualityScore]] = {}
    for s in scores:
        by_task.setdefault(s.task, []).append(s)

    for task_name, task_scores in by_task.items():
        print(f"\n{'#' * 70}")
        print(f"  {task_name.upper()}")
        print(f"{'#' * 70}")

        # Header
        models = [s.model.split(":")[0][:15] for s in task_scores]
        print(f"\n  {'Metric':<30}", end="")
        for m in models:
            print(f" {m:>15}", end="")
        print()
        print(f"  {'-' * (30 + 16 * len(models))}")

        # Metrics
        metrics = [
            ("Lines of code", lambda s: str(s.lines)),
            ("Methods", lambda s: str(s.methods)),
            ("Syntax valid", lambda s: "Y" if s.syntax_valid else f"N ({s.syntax_error[:20]})"),
            ("Type hint coverage", lambda s: f"{s.type_hint_ratio:.0%}"),
            ("Docstring coverage", lambda s: f"{s.docstring_ratio:.0%}"),
            ("Mutable defaults", lambda s: ", ".join(s.mutable_defaults) if s.mutable_defaults else "none"),
            ("Attr/method shadows", lambda s: ", ".join(s.shadows) if s.shadows else "none"),
            ("Bare excepts", lambda s: str(s.bare_excepts)),
            ("Spec checks", lambda s: f"{s.spec_checks_passed}/{s.spec_checks_total}"),
            ("Ruff issues", lambda s: str(s.ruff_issues)),
            ("Mypy errors", lambda s: str(s.mypy_issues)),
            ("Complexity (avg)", lambda s: f"{s.complexity_avg:.1f}"),
            ("Complexity (max)", lambda s: str(s.complexity_max)),
        ]

        for name, getter in metrics:
            print(f"  {name:<30}", end="")
            for s in task_scores:
                print(f" {getter(s):>15}", end="")
            print()

        # Spec failures
        print(f"\n  Spec failures:")
        for s in task_scores:
            if s.spec_failures:
                print(f"    {s.model.split(':')[0][:15]}: {', '.join(s.spec_failures)}")
            else:
                print(f"    {s.model.split(':')[0][:15]}: (all passed)")

        # Ruff details
        if any(s.ruff_issues > 0 for s in task_scores):
            print(f"\n  Top ruff issues:")
            for s in task_scores:
                if s.ruff_details:
                    print(f"    {s.model.split(':')[0][:15]}:")
                    for d in s.ruff_details[:3]:
                        print(f"      {d}")

    # Summary
    print(f"\n{'#' * 70}")
    print(f"  SUMMARY")
    print(f"{'#' * 70}")

    for s in scores:
        task_label = s.task[:12]
        model_label = s.model.split(":")[0][:15]
        quality = (
            s.spec_checks_passed / s.spec_checks_total * 40 if s.spec_checks_total else 0
        ) + (
            s.type_hint_ratio * 20
        ) + (
            (1 - min(s.ruff_issues, 10) / 10) * 15
        ) + (
            (1 - min(s.mypy_issues, 10) / 10) * 15
        ) + (
            10 if s.syntax_valid and not s.mutable_defaults and not s.shadows else 0
        )
        print(f"  {model_label:<15} {task_label:<12} quality={quality:.0f}/100  spec={s.spec_checks_passed}/{s.spec_checks_total}  ruff={s.ruff_issues}  mypy={s.mypy_issues}  hints={s.type_hint_ratio:.0%}")


async def main():
    print("Code Quality Report — Raw Model Output")
    print(f"Models: {', '.join(MODELS)}")
    print(f"Tasks: {', '.join(TASKS.keys())}")
    print(f"Ollama: {OLLAMA_URL}")
    print()

    scores = []
    for task_name, task_info in TASKS.items():
        for model in MODELS:
            print(f"  Generating {task_name} with {model}...", flush=True)
            try:
                score = await analyze_model_task(model, task_name, task_info)
                scores.append(score)
            except Exception as e:
                print(f"    ERROR: {e}")

    print_report(scores)

    # Save raw data
    output = []
    for s in scores:
        output.append({
            "model": s.model, "task": s.task, "lines": s.lines,
            "methods": s.methods, "syntax_valid": s.syntax_valid,
            "type_hints": s.type_hint_ratio, "docstrings": s.docstring_ratio,
            "mutable_defaults": s.mutable_defaults, "shadows": s.shadows,
            "spec_passed": s.spec_checks_passed, "spec_total": s.spec_checks_total,
            "spec_failures": s.spec_failures,
            "ruff": s.ruff_issues, "mypy": s.mypy_issues,
            "complexity_avg": s.complexity_avg, "complexity_max": s.complexity_max,
        })
    Path("results/current").mkdir(parents=True, exist_ok=True)
    with open("results/current/quality_report.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to results/current/quality_report.json")


if __name__ == "__main__":
    asyncio.run(main())
