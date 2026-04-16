#!/usr/bin/env python3
"""Model Error Profiler — discovers failure patterns for any local model.

Runs a model on a subset of benchmark tasks (raw, no PMCA), collects
all errors, classifies them, and generates a repair profile config.

This is the foundation of the "model-agnostic" approach: instead of
hand-coding repairs for Qwen, we auto-detect which repairs each model needs.

Usage:
    # Profile a model on 10 tasks (quick, ~3 min)
    python scripts/profile_model.py --model qwen3.5:9b --tasks 10

    # Profile with full 37 tasks (~15 min)
    python scripts/profile_model.py --model deepseek-coder-v2:16b

    # Compare profiles
    python scripts/profile_model.py --compare results/profile_*.json
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

@dataclass
class ErrorInstance:
    """A single error observed during profiling."""
    task: str
    category: str        # "import", "shadow", "mutable_default", "name", "syntax", "type", "logic", "other"
    subcategory: str     # specific pattern within category
    raw_error: str       # first 200 chars of error output
    fixable: bool        # whether a deterministic repair could fix this


ERROR_CATEGORIES = {
    # Import errors
    "ModuleNotFoundError": ("import", "module_not_found"),
    "ImportError": ("import", "import_error"),
    # Name errors (missing imports, undefined vars)
    "NameError": ("name", "undefined_name"),
    # Syntax
    "SyntaxError": ("syntax", "syntax_error"),
    # Type errors (often from attribute/method shadowing or wrong signatures)
    "TypeError": ("type", "type_error"),
    # Attribute errors (often from wrong class structure)
    "AttributeError": ("attribute", "missing_attribute"),
    # Value errors (wrong logic)
    "ValueError": ("logic", "value_error"),
    # Key errors
    "KeyError": ("logic", "key_error"),
    # Index errors
    "IndexError": ("logic", "index_error"),
    # Assertion errors (wrong output)
    "AssertionError": ("logic", "wrong_output"),
}


def classify_error(stderr: str, code: str = "") -> tuple[str, str, bool]:
    """Classify an error into category, subcategory, and fixability.

    Returns (category, subcategory, fixable_deterministically).
    """
    # Check each known error type
    for error_type, (cat, subcat) in ERROR_CATEGORIES.items():
        if error_type in stderr:
            # Determine fixability
            if cat == "import":
                # Package-style import? (from pkg.module import X)
                if "No module named" in stderr:
                    m = re.search(r"No module named '(\w+\.\w+)'", stderr)
                    if m:
                        return "import", "package_style", True
                    # Simple missing module
                    return "import", "module_not_found", True
                return cat, subcat, True

            if cat == "name":
                # Known stdlib name?
                m = re.search(r"name '(\w+)' is not defined", stderr)
                if m:
                    name = m.group(1)
                    KNOWN = {"Optional", "List", "Dict", "Any", "Union", "Tuple",
                             "Counter", "defaultdict", "deque", "dataclass",
                             "Enum", "ABC", "abstractmethod", "re", "math", "json"}
                    if name in KNOWN:
                        return "name", "known_import", True
                    return "name", "undefined_name", False
                return cat, subcat, False

            if cat == "syntax":
                return cat, subcat, False  # Can't auto-fix syntax errors

            if cat == "type":
                # "not callable" often means attr/method shadowing
                if "not callable" in stderr:
                    return "type", "shadow_callable", True
                # "'<' not supported" often means None in sort
                if "'<' not supported" in stderr:
                    return "type", "none_in_sort", True
                return cat, subcat, False

            if cat == "attribute":
                return cat, subcat, False

            if cat == "logic":
                if error_type == "AssertionError":
                    # Check if it's a close numeric mismatch (calibratable)
                    m = re.search(r"assert ([\d.]+) == ([\d.]+)", stderr)
                    if m:
                        try:
                            actual = float(m.group(1))
                            expected = float(m.group(2))
                            if expected != 0 and abs(actual - expected) / abs(expected) < 0.25:
                                return "logic", "assertion_close_numeric", True
                        except ValueError:
                            pass
                    return "logic", "wrong_output", False
                return cat, subcat, False

    # Check for mutable default patterns in the code itself
    if code:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict)):
                            return "mutable_default", "list_or_dict", True
        except SyntaxError:
            pass

    # Check for attribute/method shadowing in code
    if code:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    attrs = set()
                    methods = set()
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name == "__init__":
                                for stmt in ast.walk(item):
                                    if (isinstance(stmt, ast.Assign)
                                            and len(stmt.targets) == 1
                                            and isinstance(stmt.targets[0], ast.Attribute)
                                            and isinstance(stmt.targets[0].value, ast.Name)
                                            and stmt.targets[0].value.id == "self"):
                                        attrs.add(stmt.targets[0].attr)
                            elif item.args.args and item.args.args[0].arg == "self":
                                methods.add(item.name)
                    if attrs & methods:
                        return "shadow", "attr_method", True
        except SyntaxError:
            pass

    return "other", "unknown", False


# ---------------------------------------------------------------------------
# Model completion (same as classeval adapter)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a Senior Python Engineer. Implement the class exactly as specified. "
    "Output ONLY the complete Python class with all methods implemented. "
    "Include necessary imports at the top. Do NOT include test code."
)


async def generate_code(model: str, request: str) -> str:
    """Generate code from a vague request using Ollama chat API."""
    import httpx

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_predict": 2048},
    }
    async with httpx.AsyncClient(timeout=180) as client:
        ollama_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        resp = await client.post(f"{ollama_url}/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


def extract_code(response: str) -> str:
    """Extract Python code from model response."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

def run_probes(code: str, task: dict, workspace: Path) -> list[ErrorInstance]:
    """Run probes against generated code and collect errors."""
    errors = []
    task_id = task["task_id"]

    # Write code to workspace
    # Determine module name from task_id
    code_path = workspace / "src" / f"{task_id}.py"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(code)

    pythonpath = f"{workspace}{os.pathsep}{workspace / 'src'}"

    for probe in task["probes"]:
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe["code"]],
                cwd=str(workspace),
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "PYTHONPATH": pythonpath},
            )
            if result.returncode != 0:
                stderr = (result.stderr or result.stdout)[-500:]
                category, subcategory, fixable = classify_error(stderr, code)
                errors.append(ErrorInstance(
                    task=f"{task_id}/{probe['name']}",
                    category=category,
                    subcategory=subcategory,
                    raw_error=stderr[:200],
                    fixable=fixable,
                ))
        except subprocess.TimeoutExpired:
            errors.append(ErrorInstance(
                task=f"{task_id}/{probe['name']}",
                category="other", subcategory="timeout",
                raw_error="TIMEOUT", fixable=False,
            ))
        except Exception as exc:
            errors.append(ErrorInstance(
                task=f"{task_id}/{probe['name']}",
                category="other", subcategory="exception",
                raw_error=str(exc)[:200], fixable=False,
            ))

    return errors


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def generate_profile(model: str, all_errors: list[ErrorInstance], total_probes: int) -> dict:
    """Generate a repair profile from collected errors."""
    # Count by category
    by_category = Counter(e.category for e in all_errors)
    by_subcategory = Counter(e.subcategory for e in all_errors)
    fixable_count = sum(1 for e in all_errors if e.fixable)

    # Determine which repairs to enable
    repairs = {
        "fix_package_imports": by_subcategory.get("package_style", 0) >= 2,
        "fix_known_imports": by_subcategory.get("known_import", 0) >= 2,
        "fix_mutable_defaults": by_subcategory.get("list_or_dict", 0) >= 1,
        "fix_attr_method_shadowing": (
            by_subcategory.get("attr_method", 0) >= 1
            or by_subcategory.get("shadow_callable", 0) >= 1
        ),
        "fix_none_in_sort": by_subcategory.get("none_in_sort", 0) >= 1,
        "calibrate_tests": by_subcategory.get("assertion_close_numeric", 0) >= 2,
        "oracle_repair": by_category.get("logic", 0) >= 5,
        "ruff_autofix": True,  # Always safe
    }

    # Strategy recommendations
    strategy = {
        "use_llm_reviewer": by_category.get("logic", 0) > total_probes * 0.3,
        "think_architect": True,  # Generally beneficial
        "think_coder": False,  # Generally harmful
        "reviewer_bypass_on_pass": True,  # Safe default
    }

    return {
        "model": model,
        "total_probes": total_probes,
        "total_errors": len(all_errors),
        "error_rate": len(all_errors) / total_probes if total_probes else 0,
        "fixable_errors": fixable_count,
        "fixable_rate": fixable_count / len(all_errors) if all_errors else 0,
        "by_category": dict(by_category.most_common()),
        "by_subcategory": dict(by_subcategory.most_common()),
        "recommended_repairs": repairs,
        "recommended_strategy": strategy,
        "top_errors": [
            {"task": e.task, "category": e.category, "subcategory": e.subcategory,
             "fixable": e.fixable, "error": e.raw_error[:100]}
            for e in all_errors[:20]
        ],
    }


# ---------------------------------------------------------------------------
# Compare profiles
# ---------------------------------------------------------------------------

def compare_profiles(profile_paths: list[str]) -> None:
    """Print side-by-side comparison of model profiles."""
    profiles = []
    for path in profile_paths:
        with open(path) as f:
            profiles.append(json.load(f))

    # Header
    models = [p["model"] for p in profiles]
    print(f"\n{'Category':<25}", end="")
    for m in models:
        print(f" {m:>20}", end="")
    print()
    print("-" * (25 + 21 * len(models)))

    # Error rates
    print(f"{'Error rate':<25}", end="")
    for p in profiles:
        print(f" {p['error_rate']:>19.1%}", end="")
    print()

    print(f"{'Fixable rate':<25}", end="")
    for p in profiles:
        print(f" {p['fixable_rate']:>19.1%}", end="")
    print()

    # By category
    all_cats = set()
    for p in profiles:
        all_cats.update(p["by_category"].keys())

    print()
    for cat in sorted(all_cats):
        print(f"  {cat:<23}", end="")
        for p in profiles:
            count = p["by_category"].get(cat, 0)
            print(f" {count:>20}", end="")
        print()

    # Repair recommendations
    print(f"\n{'Repair':<25}", end="")
    for m in models:
        print(f" {m:>20}", end="")
    print()
    print("-" * (25 + 21 * len(models)))

    all_repairs = set()
    for p in profiles:
        all_repairs.update(p["recommended_repairs"].keys())

    for repair in sorted(all_repairs):
        print(f"  {repair:<23}", end="")
        for p in profiles:
            enabled = p["recommended_repairs"].get(repair, False)
            symbol = "ON" if enabled else "—"
            print(f" {symbol:>20}", end="")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_profile(model: str, n_tasks: int | None = None) -> dict:
    """Profile a model's error patterns on the benchmark."""
    # Load benchmark
    bench_path = Path(__file__).resolve().parent.parent / "benchmark" / "pmca_bench.json"
    with open(bench_path) as f:
        tasks = json.load(f)

    if n_tasks:
        # Sample across tiers for representative coverage
        simple = [t for t in tasks if t["tier"] == "simple"][:max(2, n_tasks // 4)]
        medium = [t for t in tasks if t["tier"] == "medium"][:max(3, n_tasks // 3)]
        complex_ = [t for t in tasks if t["tier"] == "complex"][:max(3, n_tasks // 3)]
        tasks = simple + medium + complex_
        tasks = tasks[:n_tasks]

    total_probes = sum(len(t["probes"]) for t in tasks)
    print(f"Profiling {model} on {len(tasks)} tasks ({total_probes} probes)")
    print()

    all_errors: list[ErrorInstance] = []
    t0 = time.time()

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        n_probes = len(task["probes"])

        # Generate code
        try:
            response = await generate_code(model, task["request"])
            code = extract_code(response)
        except Exception as exc:
            print(f"  [{i+1}/{len(tasks)}] {task_id}: GENERATION ERROR - {exc}")
            # Count all probes as errors
            for probe in task["probes"]:
                all_errors.append(ErrorInstance(
                    task=f"{task_id}/{probe['name']}",
                    category="other", subcategory="generation_failed",
                    raw_error=str(exc)[:200], fixable=False,
                ))
            continue

        # Run probes
        with tempfile.TemporaryDirectory() as tmpdir:
            errors = run_probes(code, task, Path(tmpdir))

        passed = n_probes - len(errors)
        status = f"{passed}/{n_probes}" if errors else "PASS"
        err_cats = Counter(e.category for e in errors)
        cat_str = ", ".join(f"{c}:{n}" for c, n in err_cats.most_common(3)) if errors else ""

        print(f"  [{i+1}/{len(tasks)}] {task_id:<25} {status:>8}  {cat_str}")
        all_errors.extend(errors)

    elapsed = time.time() - t0
    probes_passed = total_probes - len(all_errors)

    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"  Probes: {probes_passed}/{total_probes} ({probes_passed/total_probes:.0%})")
    print(f"  Errors: {len(all_errors)} ({len([e for e in all_errors if e.fixable])} fixable)")
    print(f"  Time: {elapsed:.0f}s")
    print(f"{'='*60}")

    # Generate profile
    profile = generate_profile(model, all_errors, total_probes)
    profile["elapsed_s"] = elapsed
    profile["probes_passed"] = probes_passed

    # Save
    model_tag = model.replace(":", "_").replace("/", "_")
    output_path = f"results/profile_{model_tag}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nProfile saved to {output_path}")

    # Print repair recommendations
    print(f"\nRecommended repairs for {model}:")
    for repair, enabled in sorted(profile["recommended_repairs"].items()):
        symbol = "ON " if enabled else "OFF"
        print(f"  [{symbol}] {repair}")

    print("\nTop error patterns:")
    for subcat, count in Counter(e.subcategory for e in all_errors).most_common(5):
        fixable = sum(1 for e in all_errors if e.subcategory == subcat and e.fixable)
        print(f"  {subcat:<30} {count:>4} ({fixable} fixable)")

    return profile


def main():
    parser = argparse.ArgumentParser(description="Profile model error patterns")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--tasks", type=int, default=None, help="Number of tasks (default: all 37)")
    parser.add_argument("--compare", nargs="+", help="Compare saved profile JSON files")
    args = parser.parse_args()

    if args.compare:
        compare_profiles(args.compare)
        return

    asyncio.run(run_profile(args.model, args.tasks))


if __name__ == "__main__":
    main()
