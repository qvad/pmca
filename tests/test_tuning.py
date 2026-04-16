"""Tests for the tuning subsystem.

These tests are unit tests — they don't call any LLM. The Orchestrator
is mocked so we can verify the tuner's logic in isolation.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pmca.eval.runner import (
    BenchmarkRunner,
    ProbeResult,
    RunResult,
    TaskResult,
    run_probe,
)
from pmca.models.config import AgentRole, CascadeConfig, Config, ModelConfig
from pmca.tuning.parameters import (
    PARAMETERS,
    apply_snapshot,
    parameter_by_name,
    snapshot,
)
from pmca.tuning.tuner import CoordinateDescentTuner, TuningResult, TuningStep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_config(model: str = "test-model:7b") -> Config:
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
        ),
    )


# ---------------------------------------------------------------------------
# Parameter set
# ---------------------------------------------------------------------------


class TestParameters:
    def test_all_parameters_have_required_fields(self):
        for p in PARAMETERS:
            assert p.name
            assert p.description
            assert p.values
            assert p.setter
            assert p.getter

    def test_parameter_values_are_unique_per_parameter(self):
        for p in PARAMETERS:
            assert len(p.values) == len(set(map(repr, p.values))), \
                f"{p.name} has duplicate candidate values"

    def test_parameter_names_are_unique(self):
        names = [p.name for p in PARAMETERS]
        assert len(names) == len(set(names))

    def test_parameter_by_name(self):
        p = parameter_by_name("max_retries")
        assert p.name == "max_retries"

    def test_parameter_by_name_unknown_raises(self):
        with pytest.raises(KeyError):
            parameter_by_name("nonexistent_parameter")

    def test_setter_actually_sets(self):
        config = make_config()
        p = parameter_by_name("max_retries")
        p.set(config, 5)
        assert config.cascade.max_retries == 5

    def test_getter_returns_current_value(self):
        config = make_config()
        config.cascade.max_retries = 7
        p = parameter_by_name("max_retries")
        assert p.get(config) == 7

    def test_set_then_get_roundtrip(self):
        config = make_config()
        for p in PARAMETERS:
            for value in p.values:
                p.set(config, value)
                assert p.get(config) == value, \
                    f"{p.name} roundtrip failed for value {value}"

    def test_snapshot_returns_all_params(self):
        config = make_config()
        snap = snapshot(config)
        assert set(snap.keys()) == {p.name for p in PARAMETERS}

    def test_apply_snapshot_restores_values(self):
        config = make_config()
        target = {p.name: p.values[-1] for p in PARAMETERS}
        apply_snapshot(config, target)
        assert snapshot(config) == target

    def test_apply_snapshot_ignores_unknown_keys(self):
        config = make_config()
        original_snap = snapshot(config)
        apply_snapshot(config, {"nonexistent_key": "garbage"})
        assert snapshot(config) == original_snap


# ---------------------------------------------------------------------------
# Runner result classes
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_rate_zero_when_no_probes(self):
        r = RunResult(
            config_label="x",
            benchmark_path="x",
            n_tasks=0,
            total_probes=0,
            total_passed=0,
            elapsed_s=0,
        )
        assert r.rate == 0.0

    def test_rate_normal(self):
        r = RunResult(
            config_label="x",
            benchmark_path="x",
            n_tasks=10,
            total_probes=100,
            total_passed=75,
            elapsed_s=0,
        )
        assert r.rate == 0.75

    def test_tasks_passed_counts_full_pass(self):
        r = RunResult(
            config_label="x",
            benchmark_path="x",
            n_tasks=3,
            total_probes=10,
            total_passed=8,
            elapsed_s=0,
            tasks=[
                TaskResult(task_id="a", tier="simple", n_probes=3, n_passed=3),
                TaskResult(task_id="b", tier="simple", n_probes=4, n_passed=4),
                TaskResult(task_id="c", tier="simple", n_probes=3, n_passed=1),  # partial
            ],
        )
        assert r.tasks_passed == 2

    def test_to_dict_serializable(self):
        import json as _json
        r = RunResult(
            config_label="x", benchmark_path="x",
            n_tasks=1, total_probes=2, total_passed=1, elapsed_s=0.1,
            tasks=[TaskResult(task_id="a", tier="simple", n_probes=2, n_passed=1)],
        )
        _json.dumps(r.to_dict())  # must not raise


class TestTaskResult:
    def test_passed_requires_full_score(self):
        t = TaskResult(task_id="x", tier="simple", n_probes=5, n_passed=4)
        assert not t.passed

    def test_passed_true_when_all_pass(self):
        t = TaskResult(task_id="x", tier="simple", n_probes=5, n_passed=5)
        assert t.passed

    def test_passed_false_when_zero_probes(self):
        t = TaskResult(task_id="x", tier="simple", n_probes=0, n_passed=0)
        assert not t.passed

    def test_rate(self):
        t = TaskResult(task_id="x", tier="simple", n_probes=4, n_passed=3)
        assert t.rate == 0.75


# ---------------------------------------------------------------------------
# Probe runner (real subprocess execution)
# ---------------------------------------------------------------------------


class TestRunProbe:
    def test_passing_probe_returns_passed(self, tmp_path):
        result = run_probe("assert 1 + 1 == 2", tmp_path)
        assert result.passed
        assert result.error == ""

    def test_failing_probe_captures_error(self, tmp_path):
        result = run_probe("assert 1 + 1 == 3", tmp_path)
        assert not result.passed
        assert "AssertionError" in result.error

    def test_timeout(self, tmp_path):
        result = run_probe("import time; time.sleep(30)", tmp_path, timeout=1)
        assert not result.passed
        assert result.error == "TIMEOUT"

    def test_workspace_pythonpath(self, tmp_path):
        # Module in workspace/src should be importable
        src = tmp_path / "src"
        src.mkdir()
        (src / "mymod.py").write_text("VALUE = 42")
        result = run_probe("from mymod import VALUE; assert VALUE == 42", tmp_path)
        assert result.passed


# ---------------------------------------------------------------------------
# Tuner — coordinate descent logic with mocked evaluator
# ---------------------------------------------------------------------------


class FakeRunner:
    """Returns scores from a deterministic table keyed by parameter snapshot."""

    def __init__(self, score_table: dict[tuple, float]):
        self.score_table = score_table
        self.call_count = 0

    async def evaluate(self, config: Config) -> RunResult:
        self.call_count += 1
        snap = snapshot(config)
        key = tuple(sorted(snap.items(), key=lambda x: x[0]))
        rate = self.score_table.get(key, 0.5)
        return RunResult(
            config_label="fake",
            benchmark_path="fake",
            n_tasks=1,
            total_probes=10,
            total_passed=int(rate * 10),
            elapsed_s=0.01,
        )


def _patch_tuner_evaluator(tuner: CoordinateDescentTuner, fake: FakeRunner):
    """Replace tuner._evaluate with the fake runner's evaluate."""
    tuner._evaluate = fake.evaluate  # type: ignore[assignment]


class TestCoordinateDescentTuner:
    def test_tuner_runs_baseline_first(self):
        config = make_config()
        # Force every config to score 0.5
        fake = FakeRunner({})

        tuner = CoordinateDescentTuner(
            base_config=config,
            benchmark_path="fake.json",
            parameters=[parameter_by_name("max_retries")],
            max_sweeps=1,
        )
        _patch_tuner_evaluator(tuner, fake)
        result = asyncio.run(tuner.tune())

        assert fake.call_count >= 1
        assert result.initial_score == 0.5

    def test_tuner_picks_higher_scoring_value(self):
        config = make_config()
        config.cascade.max_retries = 2

        # max_retries=5 scores higher than 2
        def score_fn(snap_items):
            d = dict(snap_items)
            return 0.9 if d["max_retries"] == 5 else 0.5

        fake = FakeRunner({})
        async def evaluate(cfg):
            snap = snapshot(cfg)
            rate = score_fn(tuple(snap.items()))
            return RunResult(
                config_label="x", benchmark_path="x",
                n_tasks=1, total_probes=10,
                total_passed=int(rate * 10), elapsed_s=0,
            )

        tuner = CoordinateDescentTuner(
            base_config=config,
            benchmark_path="fake.json",
            parameters=[parameter_by_name("max_retries")],
            max_sweeps=1,
        )
        tuner._evaluate = evaluate  # type: ignore[assignment]
        result = asyncio.run(tuner.tune())

        assert result.final_snapshot["max_retries"] == 5
        assert result.final_score == 0.9

    def test_tuner_does_not_decrease_score(self):
        config = make_config()

        # Always returns lower score for any non-default value
        async def evaluate(cfg):
            snap = snapshot(cfg)
            initial = (snap["max_retries"] == 2)
            rate = 0.7 if initial else 0.3
            return RunResult(
                config_label="x", benchmark_path="x",
                n_tasks=1, total_probes=10,
                total_passed=int(rate * 10), elapsed_s=0,
            )

        config.cascade.max_retries = 2
        tuner = CoordinateDescentTuner(
            base_config=config,
            benchmark_path="fake.json",
            parameters=[parameter_by_name("max_retries")],
            max_sweeps=1,
        )
        tuner._evaluate = evaluate  # type: ignore[assignment]
        result = asyncio.run(tuner.tune())

        # The tuner should keep the original value because nothing beat it
        assert result.final_score == 0.7
        assert result.final_snapshot["max_retries"] == 2

    def test_tuner_records_all_steps(self):
        config = make_config()
        config.cascade.max_retries = 2

        async def evaluate(cfg):
            return RunResult(
                config_label="x", benchmark_path="x",
                n_tasks=1, total_probes=10, total_passed=5, elapsed_s=0,
            )

        param = parameter_by_name("max_retries")
        # max_retries has 5 values, current is 2 → 4 trial steps per sweep
        tuner = CoordinateDescentTuner(
            base_config=config,
            benchmark_path="fake.json",
            parameters=[param],
            max_sweeps=1,
        )
        tuner._evaluate = evaluate  # type: ignore[assignment]
        result = asyncio.run(tuner.tune())

        # All trial values except the current one should appear as steps
        trial_values = {step.value for step in result.steps}
        expected = set(param.values) - {2}
        assert trial_values == expected

    def test_tuner_stops_early_when_no_change(self):
        """If no parameter improves the score, the tuner should stop after sweep 1."""
        config = make_config()

        async def evaluate(cfg):
            return RunResult(
                config_label="x", benchmark_path="x",
                n_tasks=1, total_probes=10, total_passed=5, elapsed_s=0,
            )

        tuner = CoordinateDescentTuner(
            base_config=config,
            benchmark_path="fake.json",
            parameters=[parameter_by_name("max_retries"),
                        parameter_by_name("best_of_n")],
            max_sweeps=5,
        )
        tuner._evaluate = evaluate  # type: ignore[assignment]
        result = asyncio.run(tuner.tune())
        # Should not run all 5 sweeps if nothing changes
        assert result.sweeps_completed <= 1

    def test_progress_callback_invoked(self):
        config = make_config()
        steps_seen: list[TuningStep] = []

        def cb(step: TuningStep):
            steps_seen.append(step)

        async def evaluate(cfg):
            return RunResult(
                config_label="x", benchmark_path="x",
                n_tasks=1, total_probes=10, total_passed=5, elapsed_s=0,
            )

        tuner = CoordinateDescentTuner(
            base_config=config,
            benchmark_path="fake.json",
            parameters=[parameter_by_name("max_retries")],
            max_sweeps=1,
            progress_callback=cb,
        )
        tuner._evaluate = evaluate  # type: ignore[assignment]
        asyncio.run(tuner.tune())

        assert len(steps_seen) > 0


class TestTuningResult:
    def test_to_dict_serializable(self):
        import json as _json
        r = TuningResult(
            model="test",
            benchmark="bench.json",
            initial_score=0.5,
            final_score=0.8,
            initial_snapshot={"max_retries": 2},
            final_snapshot={"max_retries": 5},
        )
        _json.dumps(r.to_dict())  # must not raise

    def test_to_dict_includes_improvement(self):
        r = TuningResult(
            model="test",
            benchmark="bench.json",
            initial_score=0.4,
            final_score=0.7,
            initial_snapshot={},
            final_snapshot={},
        )
        d = r.to_dict()
        assert d["improvement"] == pytest.approx(0.3)
