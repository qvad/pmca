#!/usr/bin/env python3
"""Benchmark 6: Multi-file project — tests deep decomposition and assembly.

This benchmark exercises the new project mode features:
- Deep decomposition (max_depth=3)
- TARGET_FILE / EXPORTS / DEPENDS_ON metadata
- Dependency-ordered child processing
- Snippet store + file assembly
- Interface extraction and sibling context

Task: A Python package with 4 modules (~15 functions across files).
Each module has real cross-file dependencies. The 7B model must produce
correct imports, class references, and method calls across boundaries.

Usage:
    python benchmarks/bench6_multifile.py --run
    python benchmarks/bench6_multifile.py --validate /path/to/workspace
    python benchmarks/bench6_multifile.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Benchmark definition
# ---------------------------------------------------------------------------

TASK = (
    "Create a Python package called 'taskboard' with four modules.\n"
    "\n"
    "1. models.py — Two classes:\n"
    "   - Task class with fields: title (str), description (str), "
    "status (str, default 'todo'), priority (int, default 0). "
    "Methods: mark_done() sets status to 'done'; is_done() returns bool; "
    "to_dict() returns a dict with all fields.\n"
    "   - Board class with fields: name (str), tasks (list, default empty). "
    "Methods: add_task(task) appends to list; "
    "task_count() returns len of tasks; "
    "done_count() returns count of tasks where is_done() is True.\n"
    "\n"
    "2. filters.py — Three standalone functions that take a list of Task objects:\n"
    "   - filter_by_status(tasks, status) returns tasks matching that status.\n"
    "   - filter_by_priority(tasks, min_priority) returns tasks with "
    "priority >= min_priority.\n"
    "   - sort_by_priority(tasks) returns a NEW list sorted by priority descending "
    "(highest first).\n"
    "\n"
    "3. stats.py — Two standalone functions:\n"
    "   - completion_rate(board) takes a Board, returns fraction of done tasks "
    "as a float (0.0 if no tasks).\n"
    "   - priority_summary(board) takes a Board, returns a dict mapping each "
    "unique priority int to the count of tasks with that priority.\n"
    "\n"
    "4. service.py — Three functions that tie everything together:\n"
    "   - create_board(name) creates and returns a new Board.\n"
    "   - add_task(board, title, description, priority) creates a Task and adds "
    "it to the board, returns the Task.\n"
    "   - get_board_report(board) returns a dict with keys: 'name' (str), "
    "'total' (int), 'done' (int), 'completion_rate' (float), "
    "'by_priority' (dict from priority_summary).\n"
    "\n"
    "Include tests for each module."
)

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "project_7b.yaml"
CONFIG_PATH = DEFAULT_CONFIG  # May be overridden by --config CLI arg
DEFAULT_WORKSPACE = Path("/tmp/pmca-bench/bench6_multifile")

EXPECTED_FILES = {
    "models.py": ["Task", "Board", "mark_done", "is_done", "to_dict",
                   "add_task", "task_count", "done_count"],
    "filters.py": ["filter_by_status", "filter_by_priority", "sort_by_priority"],
    "stats.py": ["completion_rate", "priority_summary"],
    "service.py": ["create_board", "add_task", "get_board_report"],
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(workspace: Path) -> dict:
    """Run the full PMCA cascade and return results dict."""
    from pmca.models.config import Config
    from pmca.orchestrator import Orchestrator

    config = Config.from_yaml(CONFIG_PATH)

    workspace.mkdir(parents=True, exist_ok=True)
    console.print(Panel(
        f"[bold cyan]Benchmark 6: Multi-File Project[/bold cyan]\n\n"
        f"Task (first 200 chars): {TASK[:200]}...\n"
        f"Config: {CONFIG_PATH}\n"
        f"Workspace: {workspace}",
        title="Starting",
    ))

    orchestrator = Orchestrator(config, workspace)

    t0 = time.monotonic()
    root = asyncio.run(orchestrator.run(TASK))
    elapsed = time.monotonic() - t0

    console.print()
    orchestrator.print_tree()

    results = {
        "task": TASK,
        "config": str(CONFIG_PATH),
        "workspace": str(workspace),
        "status": root.status.value,
        "verified": root.is_complete,
        "elapsed_seconds": round(elapsed, 1),
        "task_summary": orchestrator.task_tree.summary(),
        "code_files": root.code_files,
        "test_files": root.test_files,
    }

    results_path = workspace / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nResults saved to {results_path}")

    return results


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def validate_workspace(workspace: Path) -> dict:
    """Validate the generated workspace against expectations."""
    checks: list[dict] = []

    src_files = list(workspace.rglob("*.py"))
    src_files = [f for f in src_files if "__pycache__" not in str(f)
                 and ".pmca" not in str(f)]
    non_test_src = [f for f in src_files if "test" not in f.name]
    test_src = [f for f in src_files if "test" in f.name]

    checks.append({
        "name": "Multiple source files (>= 4)",
        "passed": len(non_test_src) >= 4,
        "detail": f"{len(non_test_src)} source files: {[f.name for f in non_test_src]}",
    })

    checks.append({
        "name": "Test files generated",
        "passed": len(test_src) >= 1,
        "detail": f"{len(test_src)} test files: {[f.name for f in test_src]}",
    })

    all_code = ""
    for f in non_test_src:
        all_code += f.read_text(errors="replace")

    for filename, symbols in EXPECTED_FILES.items():
        matching = [f for f in non_test_src if f.name == filename]
        file_exists = len(matching) > 0
        checks.append({
            "name": f"File {filename} exists",
            "passed": file_exists,
            "detail": str(matching[0]) if matching else "NOT FOUND",
        })

        if file_exists:
            content = matching[0].read_text(errors="replace")
            for sym in symbols:
                found = sym in content
                checks.append({
                    "name": f"  {filename} :: '{sym}'",
                    "passed": found,
                    "detail": "found" if found else "MISSING",
                })

    # Key cross-file symbols exist somewhere in the code
    for sym in ["Task", "Board", "filter_by_status", "completion_rate",
                "create_board", "get_board_report"]:
        checks.append({
            "name": f"Symbol '{sym}' in assembled code",
            "passed": sym in all_code,
            "detail": "found" if sym in all_code else "MISSING",
        })

    init_files = list(workspace.rglob("__init__.py"))
    init_files = [f for f in init_files if "__pycache__" not in str(f)]
    checks.append({
        "name": "__init__.py file(s) created",
        "passed": len(init_files) >= 1,
        "detail": f"{len(init_files)} init files",
    })

    # Completeness: models.py has both Task and Board
    for f in non_test_src:
        content = f.read_text(errors="replace")
        if f.name == "models.py":
            has_both = "class Task" in content and "class Board" in content
            checks.append({
                "name": "models.py has both Task and Board classes",
                "passed": has_both,
                "detail": "complete" if has_both else "PARTIAL — possible overwrite",
            })

    # Run tests
    test_result = _run_tests(workspace)
    checks.append({
        "name": "Tests pass (pytest)",
        "passed": test_result["passed"],
        "detail": test_result["detail"],
    })

    # Task tree state
    tasks_json = workspace / ".pmca" / "tasks.json"
    if tasks_json.exists():
        with open(tasks_json) as f:
            state = json.load(f)
        node_count = len(state.get("nodes", {}))
        verified_count = sum(
            1 for n in state.get("nodes", {}).values()
            if n.get("status") == "verified"
        )
        checks.append({
            "name": f"Task tree: {node_count} nodes, {verified_count} verified",
            "passed": verified_count >= 2,
            "detail": f"depth breakdown: {_depth_summary(state)}",
        })
    else:
        checks.append({
            "name": "Task tree state exists",
            "passed": False,
            "detail": "tasks.json not found",
        })

    return {"checks": checks}


def _run_tests(workspace: Path) -> dict:
    """Run pytest on the workspace and return pass/fail summary."""
    import subprocess

    test_files = list(workspace.rglob("test_*.py"))
    test_files = [f for f in test_files if "__pycache__" not in str(f)]
    if not test_files:
        return {"passed": False, "detail": "No test files found"}

    try:
        import os
        pypath = str(workspace)
        src_dir = workspace / "src"
        if src_dir.is_dir():
            pypath += os.pathsep + str(src_dir)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(workspace), "-v",
             "--tb=short", "--no-header", "-q"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": pypath},
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0

        (workspace / "test_results.log").write_text(output)

        for line in output.splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                return {"passed": passed, "detail": line.strip()}

        return {"passed": passed, "detail": output[-200:] if output else "no output"}
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "pytest timed out (30s)"}
    except Exception as e:
        return {"passed": False, "detail": f"pytest error: {e}"}


def _depth_summary(state: dict) -> str:
    from collections import Counter
    depths = Counter()
    for node in state.get("nodes", {}).values():
        d = node.get("depth", 0)
        s = node.get("status", "?")
        depths[f"d{d}:{s}"] += 1
    return ", ".join(f"{k}={v}" for k, v in sorted(depths.items()))


# ---------------------------------------------------------------------------
# Manual probes
# ---------------------------------------------------------------------------

def run_probe(workspace: Path) -> list[dict]:
    """Run manual edge-case probes against generated code."""
    probes: list[dict] = []
    sys.path.insert(0, str(workspace))
    src_dir = workspace / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    try:
        # ---- Locate modules (try multiple import prefixes) ----
        task_cls = _try_import_attr(workspace, "models", "Task")
        board_cls = _try_import_attr(workspace, "models", "Board")
        filter_mod = _try_import_mod(workspace, "filters")
        stats_mod = _try_import_mod(workspace, "stats")
        svc_mod = _try_import_mod(workspace, "service")

        # ---- models.Task probes ----
        if task_cls:
            try:
                t = task_cls(title="Write tests", description="unit tests", priority=3)
                probes.append({"name": "Task() instantiation", "passed": True,
                               "detail": f"status={t.status!r}"})
            except Exception as e:
                probes.append({"name": "Task() instantiation", "passed": False,
                               "detail": str(e)})
                t = None

            if t:
                # is_done before mark_done
                try:
                    probes.append({"name": "Task.is_done() == False initially",
                                   "passed": t.is_done() is False,
                                   "detail": f"got {t.is_done()}"})
                except Exception as e:
                    probes.append({"name": "Task.is_done()", "passed": False,
                                   "detail": str(e)})

                # mark_done + is_done
                try:
                    t.mark_done()
                    probes.append({"name": "Task.mark_done() -> is_done()",
                                   "passed": t.is_done() is True,
                                   "detail": f"status={t.status!r}"})
                except Exception as e:
                    probes.append({"name": "Task.mark_done()", "passed": False,
                                   "detail": str(e)})

                # to_dict
                try:
                    d = t.to_dict()
                    ok = (isinstance(d, dict) and "title" in d
                          and d["title"] == "Write tests")
                    probes.append({"name": "Task.to_dict()",
                                   "passed": ok,
                                   "detail": f"keys={list(d.keys())}"})
                except Exception as e:
                    probes.append({"name": "Task.to_dict()", "passed": False,
                                   "detail": str(e)})
        else:
            probes.append({"name": "Task import", "passed": False,
                           "detail": "Could not import Task"})

        # ---- models.Board probes ----
        if board_cls and task_cls:
            try:
                b = board_cls(name="Sprint 1")
                t1 = task_cls(title="A", description="a", priority=1)
                t2 = task_cls(title="B", description="b", priority=2)
                t2.mark_done()
                b.add_task(t1)
                b.add_task(t2)

                probes.append({"name": "Board.task_count()",
                               "passed": b.task_count() == 2,
                               "detail": f"expected 2, got {b.task_count()}"})

                probes.append({"name": "Board.done_count()",
                               "passed": b.done_count() == 1,
                               "detail": f"expected 1, got {b.done_count()}"})
            except Exception as e:
                probes.append({"name": "Board operations", "passed": False,
                               "detail": str(e)})
        elif not board_cls:
            probes.append({"name": "Board import", "passed": False,
                           "detail": "Could not import Board"})

        # ---- filters probes ----
        if filter_mod and task_cls:
            tasks = [
                task_cls(title="Lo", description="", priority=1),
                task_cls(title="Hi", description="", priority=5),
                task_cls(title="Mid", description="", priority=3, status="done"),
            ]
            try:
                by_status = filter_mod.filter_by_status(tasks, "todo")
                probes.append({"name": "filter_by_status('todo')",
                               "passed": len(by_status) == 2,
                               "detail": f"expected 2, got {len(by_status)}"})
            except Exception as e:
                probes.append({"name": "filter_by_status()", "passed": False,
                               "detail": str(e)})

            try:
                by_prio = filter_mod.filter_by_priority(tasks, 3)
                probes.append({"name": "filter_by_priority(>=3)",
                               "passed": len(by_prio) == 2,
                               "detail": f"expected 2, got {len(by_prio)}"})
            except Exception as e:
                probes.append({"name": "filter_by_priority()", "passed": False,
                               "detail": str(e)})

            try:
                sorted_t = filter_mod.sort_by_priority(tasks)
                prios = [t.priority for t in sorted_t]
                probes.append({"name": "sort_by_priority() descending",
                               "passed": prios == [5, 3, 1],
                               "detail": f"got {prios}"})
                # Check it returns a new list
                probes.append({"name": "sort_by_priority() returns new list",
                               "passed": sorted_t is not tasks,
                               "detail": "same object" if sorted_t is tasks else "new list"})
            except Exception as e:
                probes.append({"name": "sort_by_priority()", "passed": False,
                               "detail": str(e)})
        elif not filter_mod:
            probes.append({"name": "filters import", "passed": False,
                           "detail": "Could not import filters"})

        # ---- stats probes ----
        if stats_mod and board_cls and task_cls:
            b = board_cls(name="Stats test")
            # Empty board
            try:
                rate = stats_mod.completion_rate(b)
                probes.append({"name": "completion_rate(empty board) == 0.0",
                               "passed": rate == 0.0,
                               "detail": f"got {rate}"})
            except Exception as e:
                probes.append({"name": "completion_rate(empty)", "passed": False,
                               "detail": str(e)})

            # Board with tasks
            t1 = task_cls(title="X", description="", priority=2)
            t2 = task_cls(title="Y", description="", priority=2)
            t3 = task_cls(title="Z", description="", priority=5)
            t1.mark_done()
            b.add_task(t1)
            b.add_task(t2)
            b.add_task(t3)

            try:
                rate = stats_mod.completion_rate(b)
                # 1 done out of 3 ~= 0.333
                ok = abs(rate - 1 / 3) < 0.01
                probes.append({"name": "completion_rate(1/3 done)",
                               "passed": ok,
                               "detail": f"expected ~0.333, got {rate}"})
            except Exception as e:
                probes.append({"name": "completion_rate()", "passed": False,
                               "detail": str(e)})

            try:
                ps = stats_mod.priority_summary(b)
                # priorities: 2 appears twice, 5 appears once
                ok = isinstance(ps, dict) and ps.get(2) == 2 and ps.get(5) == 1
                probes.append({"name": "priority_summary()",
                               "passed": ok,
                               "detail": f"got {ps}"})
            except Exception as e:
                probes.append({"name": "priority_summary()", "passed": False,
                               "detail": str(e)})
        elif not stats_mod:
            probes.append({"name": "stats import", "passed": False,
                           "detail": "Could not import stats"})

        # ---- service probes (ties everything together) ----
        if svc_mod and board_cls:
            try:
                b = svc_mod.create_board("Integration")
                probes.append({"name": "service.create_board()",
                               "passed": b is not None and hasattr(b, "name"),
                               "detail": f"type={type(b).__name__}"})
            except Exception as e:
                probes.append({"name": "service.create_board()", "passed": False,
                               "detail": str(e)})
                b = None

            if b:
                try:
                    t = svc_mod.add_task(b, "Deploy", "ship it", 5)
                    probes.append({"name": "service.add_task()",
                                   "passed": t is not None and b.task_count() == 1,
                                   "detail": f"count={b.task_count()}"})
                except Exception as e:
                    probes.append({"name": "service.add_task()", "passed": False,
                                   "detail": str(e)})

                try:
                    svc_mod.add_task(b, "Test", "write tests", 3)
                    report = svc_mod.get_board_report(b)
                    ok = (isinstance(report, dict)
                          and report.get("name") == "Integration"
                          and report.get("total") == 2
                          and report.get("done") == 0
                          and "completion_rate" in report
                          and "by_priority" in report)
                    probes.append({"name": "service.get_board_report()",
                                   "passed": ok,
                                   "detail": f"keys={list(report.keys()) if isinstance(report, dict) else 'N/A'}"})
                except Exception as e:
                    probes.append({"name": "service.get_board_report()",
                                   "passed": False, "detail": str(e)})
        elif not svc_mod:
            probes.append({"name": "service import", "passed": False,
                           "detail": "Could not import service"})

    finally:
        if str(workspace) in sys.path:
            sys.path.remove(str(workspace))
        src_str = str(workspace / "src")
        if src_str in sys.path:
            sys.path.remove(src_str)

    return probes


def _try_import_attr(workspace: Path, module: str, attr: str):
    """Try multiple import prefixes and return the attribute or None."""
    for prefix in ["", "src.", "taskboard.", "src.taskboard."]:
        try:
            mod = __import__(f"{prefix}{module}", fromlist=[attr])
            val = getattr(mod, attr, None)
            if val is not None:
                return val
        except (ImportError, ModuleNotFoundError):
            continue
    return None


def _try_import_mod(workspace: Path, module: str):
    """Try multiple import prefixes and return the module or None."""
    for prefix in ["", "src.", "taskboard.", "src.taskboard."]:
        try:
            mod = __import__(f"{prefix}{module}", fromlist=["__name__"])
            return mod
        except (ImportError, ModuleNotFoundError):
            continue
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict | None, validation: dict, probes: list[dict]) -> None:
    console.print()
    console.print(Panel("[bold]Benchmark 6: Multi-File Project — Results[/bold]",
                        style="cyan"))

    if results:
        table = Table(title="Run Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Status", results["status"])
        table.add_row("Verified", str(results["verified"]))
        table.add_row("Time", f"{results['elapsed_seconds']}s")
        table.add_row("Code files", str(len(results.get("code_files", []))))
        table.add_row("Test files", str(len(results.get("test_files", []))))
        console.print(table)

    vtable = Table(title="Validation Checks")
    vtable.add_column("Check", style="bold")
    vtable.add_column("Result")
    vtable.add_column("Detail")
    for c in validation.get("checks", []):
        style = "green" if c["passed"] else "red"
        mark = "PASS" if c["passed"] else "FAIL"
        vtable.add_row(c["name"], f"[{style}]{mark}[/{style}]", c.get("detail", ""))
    console.print(vtable)

    ptable = Table(title="Manual Probes")
    ptable.add_column("Probe", style="bold")
    ptable.add_column("Result")
    ptable.add_column("Detail")
    for p in probes:
        style = "green" if p["passed"] else "red"
        mark = "PASS" if p["passed"] else "FAIL"
        ptable.add_row(p["name"], f"[{style}]{mark}[/{style}]", p.get("detail", ""))
    console.print(ptable)

    v_pass = sum(1 for c in validation.get("checks", []) if c["passed"])
    v_total = len(validation.get("checks", []))
    p_pass = sum(1 for p in probes if p["passed"])
    p_total = len(probes)
    console.print(
        f"\n[bold]Validation:[/bold] {v_pass}/{v_total} passed  |  "
        f"[bold]Probes:[/bold] {p_pass}/{p_total} passed"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 6: Multi-file project decomposition",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true",
                       help="Run full PMCA cascade (requires Ollama)")
    group.add_argument("--validate", type=Path, metavar="WORKSPACE",
                       help="Validate an existing workspace")
    group.add_argument("--dry-run", action="store_true",
                       help="Print task and config, don't run")

    parser.add_argument("--workspace", "-w", type=Path, default=DEFAULT_WORKSPACE,
                        help=f"Workspace path (default: {DEFAULT_WORKSPACE})")
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help=f"Config YAML path (default: {DEFAULT_CONFIG})")

    args = parser.parse_args()

    global CONFIG_PATH
    if args.config:
        CONFIG_PATH = args.config.resolve()

    if args.dry_run:
        console.print(Panel(
            f"[bold]Task:[/bold]\n{TASK}\n\n"
            f"[bold]Config:[/bold] {CONFIG_PATH}\n"
            f"[bold]Workspace:[/bold] {args.workspace}\n"
            f"[bold]Expected files:[/bold] {list(EXPECTED_FILES.keys())}",
            title="Dry Run — Benchmark 6",
        ))
        return

    workspace = args.validate if args.validate else args.workspace

    results = None
    if args.run:
        results = run_benchmark(workspace)

    validation = validate_workspace(workspace)
    probes = run_probe(workspace)
    print_report(results, validation, probes)

    report = {
        "run": results,
        "validation": validation,
        "probes": probes,
    }
    report_path = workspace / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
