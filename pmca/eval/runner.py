"""Benchmark runner — evaluates a Config against PMCA-Bench.

The single-purpose contract:
    BenchmarkRunner(config).run(benchmark_path) -> RunResult

A Config produces a score. Nothing else. No globals, no side effects on disk
beyond the optional results file. The tuner uses this as a black box.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pmca.models.config import Config
from pmca.orchestrator import Orchestrator


@dataclass(frozen=True)
class ProbeResult:
    """One probe execution outcome."""
    name: str
    passed: bool
    error: str = ""


@dataclass
class TaskResult:
    """One task's outcome — code generation + probe results."""
    task_id: str
    tier: str
    n_probes: int
    n_passed: int
    probes: list[ProbeResult] = field(default_factory=list)
    code: str = ""
    elapsed_s: float = 0.0
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.n_passed == self.n_probes and self.n_probes > 0

    @property
    def rate(self) -> float:
        return self.n_passed / self.n_probes if self.n_probes else 0.0


@dataclass
class RunResult:
    """Aggregate result of running a Config against a benchmark."""
    config_label: str
    benchmark_path: str
    n_tasks: int
    total_probes: int
    total_passed: int
    elapsed_s: float
    tasks: list[TaskResult] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.total_passed / self.total_probes if self.total_probes else 0.0

    @property
    def tasks_passed(self) -> int:
        return sum(1 for t in self.tasks if t.passed)

    def to_dict(self) -> dict:
        return {
            "config_label": self.config_label,
            "benchmark": self.benchmark_path,
            "n_tasks": self.n_tasks,
            "tasks_passed": self.tasks_passed,
            "total_probes": self.total_probes,
            "total_passed": self.total_passed,
            "rate": self.rate,
            "elapsed_s": self.elapsed_s,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "tier": t.tier,
                    "passed": t.passed,
                    "n_passed": t.n_passed,
                    "n_probes": t.n_probes,
                    "elapsed_s": t.elapsed_s,
                    "error": t.error,
                    "failed_probes": [p.name for p in t.probes if not p.passed],
                }
                for t in self.tasks
            ],
        }


def load_benchmark(path: str | Path) -> list[dict]:
    """Load benchmark tasks from a JSON file."""
    with open(path) as f:
        return json.load(f)


def run_probe(probe_code: str, workspace: Path, timeout: int = 15) -> ProbeResult:
    """Execute one probe in a workspace, return pass/fail with error excerpt."""
    name_marker = probe_code.split("\n", 1)[0][:80]
    pythonpath = f"{workspace}{os.pathsep}{workspace / 'src'}"

    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": pythonpath},
        )
        if result.returncode == 0:
            return ProbeResult(name=name_marker, passed=True)
        return ProbeResult(
            name=name_marker,
            passed=False,
            error=(result.stderr or result.stdout)[-300:],
        )
    except subprocess.TimeoutExpired:
        return ProbeResult(name=name_marker, passed=False, error="TIMEOUT")
    except Exception as exc:
        return ProbeResult(name=name_marker, passed=False, error=str(exc)[:300])


def run_task_probes(code: str, task: dict, workspace: Path) -> list[ProbeResult]:
    """Write code to workspace and run all probes for a task."""
    code_path = workspace / "src" / f"{task['task_id']}.py"
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(code)

    results = []
    for probe in task["probes"]:
        result = run_probe(probe["code"], workspace)
        results.append(ProbeResult(
            name=probe["name"],
            passed=result.passed,
            error=result.error,
        ))
    return results


class BenchmarkRunner:
    """Runs a Config against a benchmark file. Pure function, no globals."""

    def __init__(self, config: Config, label: str | None = None):
        self.config = config
        self.label = label or self._derive_label()

    def _derive_label(self) -> str:
        """Build a short label from the config's salient flags."""
        coder = self.config.models.get(next(iter(self.config.models)), None)
        model_name = coder.name if coder else "unknown"
        c = self.config.cascade
        flags = []
        if not c.use_llm_reviewer:
            flags.append("noreview")
        if c.reviewer_bypass_on_pass:
            flags.append("bypass")
        if not c.runtime_fixes:
            flags.append("nofix")
        if not c.import_fixes:
            flags.append("noimport")
        if c.best_of_n > 1:
            flags.append(f"bon{c.best_of_n}")
        flag_str = "+".join(flags) if flags else "default"
        return f"{model_name}|{flag_str}"

    async def run(
        self,
        benchmark_path: str | Path,
        task_filter: Iterable[str] | None = None,
        max_tasks: int | None = None,
    ) -> RunResult:
        """Run the configured cascade against every task in the benchmark."""
        tasks = load_benchmark(benchmark_path)
        if task_filter:
            wanted = set(task_filter)
            tasks = [t for t in tasks if t["task_id"] in wanted]
        if max_tasks:
            tasks = tasks[:max_tasks]

        total_probes = sum(len(t["probes"]) for t in tasks)
        total_passed = 0
        task_results: list[TaskResult] = []
        t0 = time.time()

        for task in tasks:
            result = await self._run_single_task(task)
            task_results.append(result)
            total_passed += result.n_passed

        return RunResult(
            config_label=self.label,
            benchmark_path=str(benchmark_path),
            n_tasks=len(tasks),
            total_probes=total_probes,
            total_passed=total_passed,
            elapsed_s=time.time() - t0,
            tasks=task_results,
        )

    async def _run_single_task(self, task: dict) -> TaskResult:
        """Run the cascade on one task and execute its probes."""
        task_id = task["task_id"]
        n_probes = len(task["probes"])
        t0 = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            (workspace / ".pmca").mkdir(parents=True, exist_ok=True)

            try:
                orch = Orchestrator(self.config, workspace)
                await orch.run(task["request"])
                code = self._extract_code_for_task(orch, task_id)
            except Exception as exc:
                return TaskResult(
                    task_id=task_id,
                    tier=task.get("tier", "unknown"),
                    n_probes=n_probes,
                    n_passed=0,
                    elapsed_s=time.time() - t0,
                    error=f"cascade_error: {exc}",
                )

            if not code:
                return TaskResult(
                    task_id=task_id,
                    tier=task.get("tier", "unknown"),
                    n_probes=n_probes,
                    n_passed=0,
                    elapsed_s=time.time() - t0,
                    error="no_code_generated",
                )

            probe_results = run_task_probes(code, task, workspace)
            n_passed = sum(1 for p in probe_results if p.passed)

            return TaskResult(
                task_id=task_id,
                tier=task.get("tier", "unknown"),
                n_probes=n_probes,
                n_passed=n_passed,
                probes=probe_results,
                code=code,
                elapsed_s=time.time() - t0,
            )

    @staticmethod
    def _extract_code_for_task(orch: Orchestrator, task_id: str) -> str:
        """Pull the generated code for a task from the orchestrator's task tree."""
        # The orchestrator stores code per node; find the one matching task_id
        # by file content (the request was about this task).
        for node in orch.task_tree.walk():
            for path, content in node.code_files.items():
                if content:
                    return content
        return ""
