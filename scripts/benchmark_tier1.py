#!/usr/bin/env python3
"""Benchmark script to validate Tier 1 research features.

Runs a set of tasks through PMCA and reports:
- Task pass/fail
- Difficulty routing (simple vs complex)
- Telemetry (token counts, timing)
- Whether --showlocals produced useful local variable info
"""

import asyncio
import shutil
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmca.models.config import Config
from pmca.orchestrator import Orchestrator
from pmca.utils.logger import setup_logging

# Benchmark tasks: (name, request, expected_difficulty)
BENCHMARK_TASKS = [
    # Simple tasks — should route to IMPLEMENT_SIMPLE (no planning)
    (
        "calculator",
        "Create a Calculator class with add, subtract, multiply, divide methods. "
        "divide(a, 0) should raise ValueError.",
        "simple",
    ),
    (
        "stack",
        "Create a Stack class with push, pop, peek, is_empty, size methods. "
        "pop/peek on empty stack should raise IndexError.",
        "simple",
    ),
    # Complex tasks — should route to IMPLEMENT with ICoT planning
    (
        "text_analyzer",
        "Create a TextAnalyzer class with methods: word_count(text) returns int, "
        "char_frequency(text) returns dict of char->count (lowercase, skip spaces), "
        "most_common_word(text) returns the most frequent word (lowercase), "
        "sentence_count(text) returns number of sentences (split on .!?). "
        "All methods should handle empty string input gracefully.",
        "complex",
    ),
    (
        "task_manager",
        "Create a TaskManager class that manages tasks with priority. "
        "Methods: add_task(title: str, priority: int) -> int (returns task_id), "
        "complete_task(task_id: int) -> bool, "
        "get_pending() -> list[dict] sorted by priority (highest first), "
        "get_stats() -> dict with keys 'total', 'done', 'pending'. "
        "Priority is 1-5, raise ValueError for invalid priority. "
        "DEPENDS_ON: None",
        "complex",
    ),
]


async def run_benchmark(config_path: str) -> dict:
    """Run all benchmark tasks and collect results."""
    config = Config.from_yaml(Path(config_path))
    results = []

    for task_name, request, expected_difficulty in BENCHMARK_TASKS:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {task_name} (expected: {expected_difficulty})")
        print(f"{'='*60}")

        workspace = Path(f"./workspace/bench_{task_name}")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        orch = Orchestrator(config, workspace)

        t0 = time.monotonic()
        try:
            root = await orch.run(request)
            elapsed = time.monotonic() - t0

            # Collect telemetry
            mm = orch._model_manager
            result = {
                "task": task_name,
                "status": root.status.value,
                "passed": root.is_complete,
                "elapsed_s": round(elapsed, 1),
                "llm_calls": mm.total_llm_calls,
                "prompt_tokens": mm.total_prompt_tokens,
                "completion_tokens": mm.total_completion_tokens,
                "llm_duration_ms": round(mm.total_llm_duration_ms, 0),
                "retries": root.retry_count,
                "gate_stats": dict(orch._gate_stats),
            }
        except Exception as exc:
            elapsed = time.monotonic() - t0
            result = {
                "task": task_name,
                "status": "error",
                "passed": False,
                "elapsed_s": round(elapsed, 1),
                "error": str(exc),
            }

        results.append(result)
        print(f"\nRESULT: {result}")

    return {"tasks": results}


def print_summary(results: dict) -> None:
    """Print a summary table of benchmark results."""
    tasks = results["tasks"]
    total = len(tasks)
    passed = sum(1 for t in tasks if t.get("passed"))
    total_tokens = sum(t.get("prompt_tokens", 0) + t.get("completion_tokens", 0) for t in tasks)
    total_calls = sum(t.get("llm_calls", 0) for t in tasks)
    total_time = sum(t.get("elapsed_s", 0) for t in tasks)

    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"Tasks: {passed}/{total} passed")
    print(f"Total LLM calls: {total_calls}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total time: {total_time:.1f}s")
    print()

    # Per-task table
    print(f"{'Task':<20} {'Status':<12} {'Calls':<8} {'Tokens':<12} {'Time':<10} {'Retries':<8} {'Gates'}")
    print("-" * 90)
    for t in tasks:
        status = "PASS" if t.get("passed") else "FAIL"
        calls = t.get("llm_calls", "?")
        tokens = t.get("prompt_tokens", 0) + t.get("completion_tokens", 0)
        time_s = t.get("elapsed_s", "?")
        retries = t.get("retries", "?")
        gates = t.get("gate_stats", {})
        gate_str = ", ".join(f"{k}={v}" for k, v in gates.items()) if gates else "-"
        print(f"{t['task']:<20} {status:<12} {calls:<8} {tokens:<12,} {time_s:<10} {retries:<8} {gate_str}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/test_7b.yaml"
    setup_logging("INFO", "./benchmark.log")
    print(f"Using config: {config_path}")
    results = asyncio.run(run_benchmark(config_path))
    print_summary(results)
