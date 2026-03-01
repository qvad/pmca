# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PMCA (Portable Modular Coding Agent) — a fully local, hierarchical coding agent using Ollama-hosted LLMs (7B–16B). It implements a cascade: architect → decompose → code → deterministic repair → review → verify. Python ≥ 3.11, async/await throughout.

## Commands

```bash
# Install (dev)
pip install -e ".[dev,lint,rag,mcp]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_orchestrator.py -v

# Run a single test
pytest tests/test_v2_features.py::TestOracleRepair::test_oracle_fix -v

# Run with local variable inspection on failure
pytest --showlocals

# Lint (optional deps)
ruff check pmca/
mypy pmca/

# Run the agent
pmca run -c config/test_7b.yaml "Create a Calculator class"

# Run benchmarks
python scripts/benchmark.py --task calculator
python scripts/benchmark.py --tier simple   # simple|medium|complex
```

## Architecture

**Entry points:** `pmca/cli.py` (Click CLI), `pmca/api/server.py` (FastAPI, OpenAI-compatible), `pmca/mcp/server.py` (MCP stdio).

**Orchestrator** (`pmca/orchestrator.py`, ~1200 lines) drives the entire cascade. It manages a `TaskTree` (state machine in `pmca/tasks/`) and coordinates four agents:

- **ArchitectAgent** — generates specs, decides LEAF vs decompose into subtasks
- **CoderAgent** — implements code from specs, routes by difficulty (simple skips planning via `IMPLEMENT_SIMPLE_PROMPT`), supports best-of-N candidates and alternating fix strategies
- **ReviewerAgent** — verifies spec/code/test alignment
- **WatcherAgent** (~1350 lines, largest module) — runs pytest, runs lint, and owns the **deterministic repair chain** (zero LLM tokens): `auto_fix_deterministic()` → `static_analysis_gate()` → `spec_coverage_check()` → `calibrate_tests()` → `oracle_repair_tests()`

Prompt templates live in `pmca/prompts/` (one file per agent role). Model configuration and provider abstraction (Ollama, Groq, OpenAI, Liquid) are in `pmca/models/`.

**Multi-file project mode** activates when `max_depth > 1` in config. The decompose prompt produces one subtask per .py file with `TARGET_FILE`/`EXPORTS`/`DEPENDS_ON` metadata. Dependencies are topologically sorted (Kahn's algorithm), AST interface extraction gives sibling specs zero-token context, and `FileAssembler` (`pmca/utils/assembler.py`) merges snippets.

**Task state machine:** `PENDING → DESIGNING → DECOMPOSED → CODING → REVIEWING → INTEGRATING → VERIFIED`. Failed tasks can retry (`FAILED → PENDING`), review can send back to coding.

## Configuration

YAML files in `config/`. Key knobs:

- `cascade.max_depth`: 1 = single-file, 2+ = multi-file project mode
- `cascade.max_retries`: retry budget before FAILED
- `cascade.best_of_n`: generate N candidates, pick best (1 = disabled)
- `cascade.fresh_start_after`: regenerate from scratch after N failed fixes
- `models.<role>.name`: Ollama model tag per agent role
- `models.<role>.provider`: `ollama` (default), `groq`, `openai`, `liquid`

## Testing

Pytest with `asyncio_mode = "auto"` (configured in `pyproject.toml`). Key test files:

- `test_v2_features.py` — largest suite: best-of-N, fresh start, static analysis gate, oracle repair, difficulty routing
- `test_project_mode.py` — multi-file decomposition, topo sort, interface extraction, assembly
- `test_api.py` — FastAPI endpoints, SSE streaming
- `test_orchestrator.py` — cascade flow, design/code/review phases

All agents are mocked in unit tests via `pytest-mock`. Tests use `pytest-asyncio` for async test functions.

## Conventions

- All LLM and I/O operations are async (`async def`, `await`)
- Deterministic (zero-token) repairs always run before any LLM-based fix retry
- The watcher agent handles all test execution and repair logic — code fixes flow through it
- Prompt templates are string constants (not Jinja), formatted with `.format()` or f-strings
- Config is loaded into dataclasses (`pmca/models/config.py`) from YAML
