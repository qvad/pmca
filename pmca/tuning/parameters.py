"""Tunable parameter space for PMCA per-model optimization.

A Parameter is one knob that the tuner can flip or vary. Each parameter
knows how to read/write itself on a Config object so the tuner doesn't
need to know which dataclass field to touch.

The set of parameters is intentionally small (~12). Adding more should
be a deliberate choice — every parameter doubles search cost.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pmca.models.config import AgentRole, Config


@dataclass
class Parameter:
    """A single tunable parameter on a Config object."""
    name: str
    description: str
    values: list[Any]              # discrete candidate values
    setter: Callable[[Config, Any], None]
    getter: Callable[[Config], Any]

    def set(self, config: Config, value: Any) -> None:
        self.setter(config, value)

    def get(self, config: Config) -> Any:
        return self.getter(config)


# ---------------------------------------------------------------------------
# Setters/getters — pure functions, easy to test
# ---------------------------------------------------------------------------

def _set_cascade(field_name: str):
    def _set(cfg: Config, value: Any) -> None:
        setattr(cfg.cascade, field_name, value)
    return _set


def _get_cascade(field_name: str):
    def _get(cfg: Config) -> Any:
        return getattr(cfg.cascade, field_name)
    return _get


def _set_coder_temperature(cfg: Config, value: float) -> None:
    if AgentRole.CODER in cfg.models:
        cfg.models[AgentRole.CODER].temperature = value


def _get_coder_temperature(cfg: Config) -> float:
    if AgentRole.CODER in cfg.models:
        return cfg.models[AgentRole.CODER].temperature
    return 0.2


def _set_architect_think(cfg: Config, value: bool) -> None:
    if AgentRole.ARCHITECT in cfg.models:
        cfg.models[AgentRole.ARCHITECT].think = value


def _get_architect_think(cfg: Config) -> bool:
    if AgentRole.ARCHITECT in cfg.models:
        return cfg.models[AgentRole.ARCHITECT].think or False
    return False


# ---------------------------------------------------------------------------
# The tunable parameter set
# ---------------------------------------------------------------------------

PARAMETERS: list[Parameter] = [
    # --- Cascade structure ---
    Parameter(
        name="max_retries",
        description="Retry budget per task before declaring failure",
        values=[0, 1, 2, 3, 5],
        setter=_set_cascade("max_retries"),
        getter=_get_cascade("max_retries"),
    ),
    Parameter(
        name="best_of_n",
        description="Generate N candidates, pick best (1 = disabled)",
        values=[1, 2, 3],
        setter=_set_cascade("best_of_n"),
        getter=_get_cascade("best_of_n"),
    ),
    Parameter(
        name="fresh_start_after",
        description="Regenerate from scratch after N failed fix attempts",
        values=[2, 3, 99],  # 99 = effectively disabled
        setter=_set_cascade("fresh_start_after"),
        getter=_get_cascade("fresh_start_after"),
    ),

    # --- Reviewer ---
    Parameter(
        name="use_llm_reviewer",
        description="Run LLM reviewer at all",
        values=[True, False],
        setter=_set_cascade("use_llm_reviewer"),
        getter=_get_cascade("use_llm_reviewer"),
    ),
    Parameter(
        name="reviewer_bypass_on_pass",
        description="Auto-approve when tests pass + spec coverage clean",
        values=[True, False],
        setter=_set_cascade("reviewer_bypass_on_pass"),
        getter=_get_cascade("reviewer_bypass_on_pass"),
    ),

    # --- Repair chain (split for finer ablation) ---
    Parameter(
        name="import_fixes",
        description="Package import rewriting + known imports injection",
        values=[True, False],
        setter=_set_cascade("import_fixes"),
        getter=_get_cascade("import_fixes"),
    ),
    Parameter(
        name="ast_fixes",
        description="Mutable defaults + attr/method shadow rename",
        values=[True, False],
        setter=_set_cascade("ast_fixes"),
        getter=_get_cascade("ast_fixes"),
    ),
    Parameter(
        name="test_calibration",
        description="calibrate_tests + oracle_repair",
        values=[True, False],
        setter=_set_cascade("test_calibration"),
        getter=_get_cascade("test_calibration"),
    ),
    Parameter(
        name="micro_fix",
        description="Targeted single-function LLM micro-fix",
        values=[True, False],
        setter=_set_cascade("micro_fix"),
        getter=_get_cascade("micro_fix"),
    ),
    Parameter(
        name="runtime_fixes",
        description="Error-driven AST fixes in retry loop",
        values=[True, False],
        setter=_set_cascade("runtime_fixes"),
        getter=_get_cascade("runtime_fixes"),
    ),

    # --- Generation behavior ---
    Parameter(
        name="lesson_injection",
        description="Inject prior failure summaries into fix prompts",
        values=[True, False],
        setter=_set_cascade("lesson_injection"),
        getter=_get_cascade("lesson_injection"),
    ),
    Parameter(
        name="spec_literals",
        description="Pre-extract string literal enums from spec",
        values=[True, False],
        setter=_set_cascade("spec_literals"),
        getter=_get_cascade("spec_literals"),
    ),

    # --- Per-role knobs ---
    Parameter(
        name="coder_temperature",
        description="Coder agent generation temperature",
        values=[0.1, 0.2, 0.3, 0.5],
        setter=_set_coder_temperature,
        getter=_get_coder_temperature,
    ),
    Parameter(
        name="architect_think",
        description="Architect uses thinking mode (Qwen 3.5 only)",
        values=[True, False],
        setter=_set_architect_think,
        getter=_get_architect_think,
    ),
]


def parameter_by_name(name: str) -> Parameter:
    for p in PARAMETERS:
        if p.name == name:
            return p
    raise KeyError(f"Unknown parameter: {name}")


def snapshot(config: Config) -> dict[str, Any]:
    """Return current values of all tunable parameters as a dict."""
    return {p.name: p.get(config) for p in PARAMETERS}


def apply_snapshot(config: Config, values: dict[str, Any]) -> None:
    """Apply a parameter snapshot to a Config in place."""
    for name, value in values.items():
        try:
            parameter_by_name(name).set(config, value)
        except KeyError:
            pass  # ignore stale parameter names
