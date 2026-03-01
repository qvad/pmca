#!/usr/bin/env python3
"""Raw model baseline benchmark — single-shot LLM, no agents or repair.

Sends each benchmark task as a single prompt directly to the model,
extracts the code block, writes it to disk, and runs the same validation
probes used by the full PMCA benchmark.

Usage:
    python scripts/benchmark_raw.py                                        # 7B
    python scripts/benchmark_raw.py --model qwen2.5-coder:14b-instruct-q4_K_M  # 14B
    python scripts/benchmark_raw.py --task calculator                      # single task
    python scripts/benchmark_raw.py --tier simple                          # tier
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse task definitions and probes from main benchmark
from benchmark import TASKS, BenchTask, Probe, run_probes

OLLAMA_HOST = "http://localhost:11434"

SYSTEM_PROMPT = (
    "You are an expert Python developer. Write clean, correct Python code. "
    "Return ONLY a single Python code block (```python ... ```) with the complete implementation. "
    "Include the class/function definition and any necessary imports. Do not include tests or example usage."
)


@dataclass
class RawResult:
    task_name: str
    tier: str
    model: str
    probes_total: int = 0
    probes_passed: int = 0
    probe_details: list[tuple[str, bool, str]] = field(default_factory=list)
    elapsed_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


def extract_code(response: str) -> str:
    """Extract the first Python code block from LLM response."""
    # Try ```python ... ``` first
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try generic ``` ... ```
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: return everything (might be raw code)
    return response.strip()


def derive_filename(task_name: str) -> str:
    """Derive the expected module filename from task name."""
    return f"{task_name}.py"


def generate_single_shot(
    model: str, request: str, context_window: int = 4096, temperature: float = 0.2
) -> tuple[str, int, int, float]:
    """Send a single prompt to Ollama and return (response, prompt_tok, completion_tok, elapsed_s)."""
    client = httpx.Client(base_url=OLLAMA_HOST, timeout=600.0)
    payload = {
        "model": model,
        "prompt": request,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": context_window,
        },
    }
    t0 = time.monotonic()
    resp = client.post("/api/generate", json=payload)
    elapsed = time.monotonic() - t0
    resp.raise_for_status()
    data = resp.json()
    client.close()
    return (
        data.get("response", ""),
        data.get("prompt_eval_count", 0),
        data.get("eval_count", 0),
        elapsed,
    )


def run_raw_task(
    model: str, task: BenchTask, context_window: int, temperature: float, python_exe: str
) -> RawResult:
    """Run a single task: one LLM call, extract code, write file, run probes."""
    workspace = Path(f"./workspace/raw_{task.name}")
    if workspace.exists():
        shutil.rmtree(workspace)
    src_dir = workspace / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    result = RawResult(task_name=task.name, tier=task.tier, model=model)

    try:
        response, p_tok, c_tok, elapsed = generate_single_shot(
            model, task.request, context_window, temperature
        )
        result.elapsed_s = round(elapsed, 1)
        result.prompt_tokens = p_tok
        result.completion_tokens = c_tok

        code = extract_code(response)
        filename = derive_filename(task.name)
        (src_dir / filename).write_text(code)

        # Run probes
        probe_results = run_probes(workspace, task.probes, python_exe)
        result.probes_total = len(probe_results)
        result.probes_passed = sum(1 for _, ok, _ in probe_results if ok)
        result.probe_details = probe_results

    except Exception as exc:
        result.error = str(exc)

    return result


def print_summary(results: list[RawResult], model: str) -> None:
    all_probes = sum(r.probes_total for r in results)
    passed_probes = sum(r.probes_passed for r in results)
    total_time = sum(r.elapsed_s for r in results)
    total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in results)

    print(f"\n{'='*70}")
    print(f"RAW MODEL BASELINE: {model}")
    print(f"{'='*70}")
    print(f"Probes:        {passed_probes}/{all_probes} ({passed_probes/max(all_probes,1)*100:.0f}%)")
    print(f"Total tokens:  {total_tokens:,}")
    print(f"Total time:    {total_time:.1f}s")

    print(f"\n{'Task':<16} {'Tier':<8} {'Probes':<10} {'Tokens':<10} {'Time':<8}")
    print("-" * 56)
    for r in results:
        probes = f"{r.probes_passed}/{r.probes_total}"
        tokens = r.prompt_tokens + r.completion_tokens
        print(f"{r.task_name:<16} {r.tier:<8} {probes:<10} {tokens:<10,} {r.elapsed_s:<8}")

    # Failed probes
    failed = [(r.task_name, name, err)
              for r in results for name, ok, err in r.probe_details if not ok]
    if failed:
        print(f"\nFailed probes ({len(failed)}):")
        for task_name, probe, err in failed:
            print(f"  {task_name}/{probe}: {err[:80]}")


def save_results(results: list[RawResult], model: str, path: str) -> None:
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "mode": "raw_single_shot",
        "results": [
            {
                "task": r.task_name,
                "tier": r.tier,
                "probes_passed": r.probes_passed,
                "probes_total": r.probes_total,
                "tokens": r.prompt_tokens + r.completion_tokens,
                "elapsed_s": r.elapsed_s,
                "error": r.error,
            }
            for r in results
        ],
    }
    Path(path).write_text(json.dumps(data, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Raw Model Baseline Benchmark")
    parser.add_argument("--model", default="qwen2.5-coder:7b-instruct-q4_K_M",
                        help="Ollama model name")
    parser.add_argument("--context-window", type=int, default=4096,
                        help="Context window size")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Generation temperature")
    parser.add_argument("--tier", choices=["simple", "medium", "complex"],
                        help="Only run tasks from this tier")
    parser.add_argument("--task", help="Only run a specific task by name")
    parser.add_argument("--output", "-o", default=None,
                        help="Save results JSON (default: raw_benchmark_{model_short}.json)")
    args = parser.parse_args()

    tasks = TASKS
    if args.tier:
        tasks = [t for t in tasks if t.tier == args.tier]
    if args.task:
        tasks = [t for t in tasks if t.name == args.task]

    if not tasks:
        print("No tasks match the filter")
        sys.exit(1)

    model = args.model
    model_short = model.split(":")[0].replace(".", "").replace("-", "_")
    python_exe = sys.executable

    print(f"Model:  {model}")
    print(f"Tasks:  {len(tasks)} ({', '.join(t.name for t in tasks)})")
    print(f"Mode:   Single-shot (no agents, no repair)\n")

    results = []
    for i, task in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {task.name} (tier={task.tier}) ... ", end="", flush=True)
        r = run_raw_task(model, task, args.context_window, args.temperature, python_exe)
        results.append(r)
        print(f"{r.probes_passed}/{r.probes_total} probes, {r.elapsed_s}s")
        for name, ok, err in r.probe_details:
            mark = "+" if ok else "X"
            print(f"  [{mark}] {name}" + (f" — {err}" if err else ""))

    print_summary(results, model)

    output = args.output or f"raw_benchmark_{model_short}.json"
    save_results(results, model, output)
    print(f"\nResults saved to {output}")


if __name__ == "__main__":
    main()
