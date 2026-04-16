#!/usr/bin/env python3
"""PMCA-Bench standalone evaluation runner.

Evaluates generated code against the 228 validation probes.
Works with ANY code generation system — just point it at a workspace
directory containing the generated Python modules.

Usage:
    # Evaluate all tasks in a workspace
    python benchmark/run.py --workspace ./output_dir

    # Evaluate a single task
    python benchmark/run.py --workspace ./output_dir --task calculator

    # Evaluate with verbose output (show failing probes)
    python benchmark/run.py --workspace ./output_dir -v

    # Output results as JSON
    python benchmark/run.py --workspace ./output_dir --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_benchmark() -> list[dict]:
    """Load the benchmark dataset."""
    bench_path = Path(__file__).parent / "pmca_bench.json"
    with open(bench_path) as f:
        return json.load(f)


def run_probe(probe_code: str, workspace: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute a single probe in the workspace directory.

    Returns (passed, error_output).
    """
    # Build PYTHONPATH: workspace root + workspace/src (common layout)
    ws = Path(workspace).resolve()
    pythonpath = str(ws)
    src_dir = ws / "src"
    if src_dir.is_dir():
        pythonpath = f"{ws}{os.pathsep}{src_dir}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            cwd=str(ws),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": pythonpath},
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout)[-500:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as exc:
        return False, str(exc)


def evaluate(
    workspace: str,
    tasks: list[dict] | None = None,
    task_filter: str | None = None,
    tier_filter: str | None = None,
    verbose: bool = False,
) -> dict:
    """Run all probes and return results."""
    if tasks is None:
        tasks = load_benchmark()

    if task_filter:
        tasks = [t for t in tasks if t["task_id"] == task_filter]
        if not tasks:
            print(f"Task '{task_filter}' not found in benchmark")
            sys.exit(1)

    if tier_filter:
        tasks = [t for t in tasks if t["tier"] == tier_filter]

    results = {
        "workspace": str(workspace),
        "tasks_total": len(tasks),
        "tasks_passed": 0,
        "probes_total": 0,
        "probes_passed": 0,
        "per_tier": {},
        "per_task": [],
    }

    for task in tasks:
        task_id = task["task_id"]
        tier = task["tier"]
        probes = task["probes"]

        task_result = {
            "task_id": task_id,
            "tier": tier,
            "n_probes": len(probes),
            "n_passed": 0,
            "passed": False,
            "failed_probes": [],
        }

        for probe in probes:
            passed, error = run_probe(probe["code"], workspace)
            results["probes_total"] += 1
            if passed:
                results["probes_passed"] += 1
                task_result["n_passed"] += 1
            else:
                task_result["failed_probes"].append({
                    "name": probe["name"],
                    "error": error[:200] if verbose else "",
                })

        task_result["passed"] = task_result["n_passed"] == len(probes)
        if task_result["passed"]:
            results["tasks_passed"] += 1

        # Per-tier tracking
        if tier not in results["per_tier"]:
            results["per_tier"][tier] = {"tasks": 0, "tasks_passed": 0, "probes": 0, "probes_passed": 0}
        results["per_tier"][tier]["tasks"] += 1
        results["per_tier"][tier]["probes"] += len(probes)
        results["per_tier"][tier]["probes_passed"] += task_result["n_passed"]
        if task_result["passed"]:
            results["per_tier"][tier]["tasks_passed"] += 1

        results["per_task"].append(task_result)

        # Print progress
        status = "PASS" if task_result["passed"] else f"FAIL ({task_result['n_passed']}/{len(probes)})"
        print(f"  [{tier:>7}] {task_id:<25} {status}")

        if verbose and task_result["failed_probes"]:
            for fp in task_result["failed_probes"]:
                print(f"           FAIL: {fp['name']}")
                if fp["error"]:
                    for line in fp["error"].strip().split("\n")[-2:]:
                        print(f"                 {line.strip()}")

    # Summary
    task_rate = results["tasks_passed"] / results["tasks_total"] if results["tasks_total"] else 0
    probe_rate = results["probes_passed"] / results["probes_total"] if results["probes_total"] else 0

    print(f"\n{'=' * 60}")
    print(f"  Task pass@1:   {task_rate:.1%} ({results['tasks_passed']}/{results['tasks_total']})")
    print(f"  Probe pass:    {probe_rate:.1%} ({results['probes_passed']}/{results['probes_total']})")
    for tier in ["simple", "medium", "complex"]:
        if tier in results["per_tier"]:
            t = results["per_tier"][tier]
            tp = t["tasks_passed"] / t["tasks"] if t["tasks"] else 0
            pp = t["probes_passed"] / t["probes"] if t["probes"] else 0
            print(f"    {tier:>7}:  {tp:.0%} tasks ({t['tasks_passed']}/{t['tasks']}), "
                  f"{pp:.0%} probes ({t['probes_passed']}/{t['probes']})")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PMCA-Bench: evaluate generated code against 228 validation probes"
    )
    parser.add_argument("--workspace", required=True, help="Directory containing generated Python modules")
    parser.add_argument("--task", default=None, help="Evaluate a single task")
    parser.add_argument("--tier", choices=["simple", "medium", "complex"], default=None, help="Filter by tier")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show failing probe details")
    parser.add_argument("--json", default=None, help="Save results as JSON")
    parser.add_argument("--timeout", type=int, default=10, help="Probe timeout in seconds")

    args = parser.parse_args()

    if not Path(args.workspace).is_dir():
        print(f"Workspace not found: {args.workspace}")
        sys.exit(1)

    results = evaluate(
        workspace=args.workspace,
        task_filter=args.task,
        tier_filter=args.tier,
        verbose=args.verbose,
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
