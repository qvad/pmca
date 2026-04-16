"""Coordinate-descent tuner for PMCA configurations.

Algorithm:
    1. Start from a baseline Config (default or user-supplied).
    2. Pick an ordering of parameters (most-impactful-first heuristic).
    3. For each parameter in order:
         a. Hold all other parameters fixed.
         b. Run the cascade for each candidate value.
         c. Pick the value with the highest score on the calibration set.
         d. Apply it to the working Config.
    4. Repeat the sweep until no parameter changes value (converged) or
       a max-sweeps budget is reached.
    5. Return the final Config and the score history.

Why coordinate descent and not grid search:
    - Grid search over 14 binary parameters = 16,384 runs. Infeasible.
    - Coordinate descent: 1 sweep = sum of |values| ≈ 30 runs. One full
      sweep is one day of compute even with a small calibration set.
    - Coordinate descent finds local optima but is robust to interactions
      because each parameter is evaluated against the *current best*, not
      the baseline. It converges in 1-2 sweeps for most starting points.

Why not Bayesian optimization:
    - Adds a dependency, complicates the code, and the search space is
      too small (mostly binary) for the BO model to add value.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pmca.eval.runner import BenchmarkRunner, RunResult
from pmca.models.config import Config
from pmca.tuning.parameters import PARAMETERS, Parameter, snapshot


@dataclass
class TuningStep:
    """One parameter probe during a sweep."""
    parameter: str
    value: Any
    score: float
    elapsed_s: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            "parameter": self.parameter,
            "value": self.value,
            "score": self.score,
            "elapsed_s": self.elapsed_s,
            "accepted": self.accepted,
        }


@dataclass
class TuningResult:
    """Output of a tuning run."""
    model: str
    benchmark: str
    initial_score: float
    final_score: float
    initial_snapshot: dict[str, Any]
    final_snapshot: dict[str, Any]
    steps: list[TuningStep] = field(default_factory=list)
    sweeps_completed: int = 0
    total_runs: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "benchmark": self.benchmark,
            "initial_score": self.initial_score,
            "final_score": self.final_score,
            "improvement": self.final_score - self.initial_score,
            "initial_snapshot": self.initial_snapshot,
            "final_snapshot": self.final_snapshot,
            "sweeps_completed": self.sweeps_completed,
            "total_runs": self.total_runs,
            "elapsed_s": self.elapsed_s,
            "steps": [s.to_dict() for s in self.steps],
        }


class CoordinateDescentTuner:
    """Tunes a Config one parameter at a time against a calibration benchmark."""

    def __init__(
        self,
        base_config: Config,
        benchmark_path: str | Path,
        parameters: list[Parameter] | None = None,
        max_sweeps: int = 2,
        calibration_tasks: list[str] | None = None,
        progress_callback: Callable[[TuningStep], None] | None = None,
    ):
        self.base_config = base_config
        self.benchmark_path = str(benchmark_path)
        self.parameters = parameters or PARAMETERS
        self.max_sweeps = max_sweeps
        self.calibration_tasks = calibration_tasks
        self.progress_callback = progress_callback

    async def tune(self) -> TuningResult:
        """Run coordinate descent to find the best parameter values."""
        t0 = time.time()
        working_config = copy.deepcopy(self.base_config)

        # Initial baseline
        initial_result = await self._evaluate(working_config)
        initial_score = initial_result.rate
        initial_snap = snapshot(working_config)

        steps: list[TuningStep] = []
        total_runs = 1
        current_score = initial_score
        sweep_no = 0

        for sweep_no in range(1, self.max_sweeps + 1):  # noqa: B007  (used after loop)
            sweep_changed = False

            for param in self.parameters:
                current_value = param.get(working_config)
                best_value = current_value
                best_score = current_score

                for candidate in param.values:
                    if candidate == current_value:
                        # Already evaluated as current_score
                        continue

                    # Try this value
                    trial_config = copy.deepcopy(working_config)
                    param.set(trial_config, candidate)

                    trial_t0 = time.time()
                    result = await self._evaluate(trial_config)
                    trial_elapsed = time.time() - trial_t0
                    total_runs += 1

                    accepted = result.rate > best_score
                    step = TuningStep(
                        parameter=param.name,
                        value=candidate,
                        score=result.rate,
                        elapsed_s=trial_elapsed,
                        accepted=accepted,
                    )
                    steps.append(step)
                    if self.progress_callback:
                        self.progress_callback(step)

                    if accepted:
                        best_value = candidate
                        best_score = result.rate

                # Lock in the best value for this parameter
                if best_value != current_value:
                    param.set(working_config, best_value)
                    current_score = best_score
                    sweep_changed = True

            if not sweep_changed:
                break

        return TuningResult(
            model=self._infer_model_name(working_config),
            benchmark=self.benchmark_path,
            initial_score=initial_score,
            final_score=current_score,
            initial_snapshot=initial_snap,
            final_snapshot=snapshot(working_config),
            steps=steps,
            sweeps_completed=sweep_no,
            total_runs=total_runs,
            elapsed_s=time.time() - t0,
        )

    async def _evaluate(self, config: Config) -> RunResult:
        """Run a single Config against the calibration benchmark."""
        runner = BenchmarkRunner(config)
        return await runner.run(
            self.benchmark_path,
            task_filter=self.calibration_tasks,
        )

    @staticmethod
    def _infer_model_name(config: Config) -> str:
        if not config.models:
            return "unknown"
        first_role = next(iter(config.models))
        return config.models[first_role].name
