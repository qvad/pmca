"""CLI entry point for the PMCA tuner.

Usage:
    python -m pmca.tuning.cli tune --model qwen3.5-coder --benchmark benchmark/pmca_bench_hard.json
    python -m pmca.tuning.cli evaluate --config config/qwen35_optimal.yaml --benchmark benchmark/pmca_bench.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pmca.eval.runner import BenchmarkRunner
from pmca.models.config import AgentRole, CascadeConfig, Config, ModelConfig
from pmca.tuning.parameters import PARAMETERS
from pmca.tuning.tuner import CoordinateDescentTuner, TuningStep


def _print_step(step: TuningStep) -> None:
    """Progress callback that prints each tuning step."""
    marker = "✓" if step.accepted else " "
    value_str = str(step.value)[:15]
    print(f"  {marker} {step.parameter:<25} = {value_str:<15} score={step.score:>5.0%}  ({step.elapsed_s:.0f}s)")


def _build_default_config(model: str) -> Config:
    """Build a baseline Config that uses the same model for every role."""
    return Config(
        models={
            AgentRole.ARCHITECT: ModelConfig(name=model, temperature=0.3),
            AgentRole.CODER: ModelConfig(name=model, temperature=0.2),
            AgentRole.REVIEWER: ModelConfig(name=model, temperature=0.1),
            AgentRole.WATCHER: ModelConfig(name=model, temperature=0.1),
        },
        cascade=CascadeConfig(
            max_depth=1,
            max_retries=2,
            best_of_n=1,
            fresh_start_after=99,
            use_llm_reviewer=True,
            reviewer_bypass_on_pass=False,
        ),
    )


async def cmd_tune(args) -> None:
    """Run coordinate descent tuning for a model."""
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = _build_default_config(args.model)

    parameters = PARAMETERS
    if args.params:
        wanted = set(args.params.split(","))
        parameters = [p for p in PARAMETERS if p.name in wanted]
        unknown = wanted - {p.name for p in PARAMETERS}
        if unknown:
            print(f"Unknown parameters: {unknown}", file=sys.stderr)
            sys.exit(1)

    calibration_tasks = args.calibration.split(",") if args.calibration else None

    print(f"\n{'='*60}")
    print(f"  TUNING — {args.model}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Parameters: {len(parameters)} ({', '.join(p.name for p in parameters)})")
    if calibration_tasks:
        print(f"  Calibration tasks: {len(calibration_tasks)}")
    print(f"  Max sweeps: {args.sweeps}")
    print(f"{'='*60}\n")

    tuner = CoordinateDescentTuner(
        base_config=config,
        benchmark_path=args.benchmark,
        parameters=parameters,
        max_sweeps=args.sweeps,
        calibration_tasks=calibration_tasks,
        progress_callback=_print_step,
    )

    print("Initial baseline run...")
    result = await tuner.tune()

    print(f"\n{'='*60}")
    print("  RESULT")
    print(f"{'='*60}")
    print(f"  Initial: {result.initial_score:.0%}")
    print(f"  Final:   {result.final_score:.0%}")
    print(f"  Δ:       {result.final_score - result.initial_score:+.0%}")
    print(f"  Sweeps:  {result.sweeps_completed}")
    print(f"  Runs:    {result.total_runs}")
    print(f"  Time:    {result.elapsed_s:.0f}s")
    print("\n  Best parameters:")
    for name, value in result.final_snapshot.items():
        if result.initial_snapshot.get(name) != value:
            print(f"    {name:<25} = {value}  (was {result.initial_snapshot.get(name)})")
    print(f"{'='*60}\n")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Saved tuning result to {args.output}")


async def cmd_evaluate(args) -> None:
    """Evaluate a single Config against a benchmark."""
    config = Config.from_yaml(Path(args.config))
    runner = BenchmarkRunner(config, label=args.label or "evaluation")

    print(f"\nEvaluating {runner.label} on {args.benchmark}\n")

    result = await runner.run(args.benchmark, max_tasks=args.max_tasks)

    print(f"\n{'='*60}")
    print(f"  RESULT — {runner.label}")
    print(f"{'='*60}")
    print(f"  Tasks: {result.tasks_passed}/{result.n_tasks}")
    print(f"  Probes: {result.total_passed}/{result.total_probes} ({result.rate:.0%})")
    print(f"  Time: {result.elapsed_s:.0f}s")
    print(f"{'='*60}\n")

    for t in result.tasks:
        status = "PASS" if t.passed else f"{t.n_passed}/{t.n_probes}"
        print(f"  {t.task_id:<30} {status:>10}  ({t.elapsed_s:.0f}s)")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nSaved to {args.output}")


def main():
    parser = argparse.ArgumentParser(prog="pmca-tune")
    sub = parser.add_subparsers(dest="command", required=True)

    tune = sub.add_parser("tune", help="Tune parameters for a model")
    tune.add_argument("--model", required=True, help="Model name (e.g. qwen3.5:9b)")
    tune.add_argument("--benchmark", required=True, help="Benchmark JSON path")
    tune.add_argument("--config", help="Starting Config YAML (optional)")
    tune.add_argument("--params", help="Comma-separated parameter names to tune (default: all)")
    tune.add_argument("--calibration", help="Comma-separated task IDs to use as calibration set")
    tune.add_argument("--sweeps", type=int, default=2, help="Max coordinate descent sweeps")
    tune.add_argument("--output", help="Save tuning result JSON")

    evaluate = sub.add_parser("evaluate", help="Run a Config against a benchmark")
    evaluate.add_argument("--config", required=True, help="Config YAML path")
    evaluate.add_argument("--benchmark", required=True, help="Benchmark JSON path")
    evaluate.add_argument("--label", help="Run label")
    evaluate.add_argument("--max-tasks", type=int, help="Limit task count")
    evaluate.add_argument("--output", help="Save run result JSON")

    args = parser.parse_args()

    if args.command == "tune":
        asyncio.run(cmd_tune(args))
    elif args.command == "evaluate":
        asyncio.run(cmd_evaluate(args))


if __name__ == "__main__":
    main()
