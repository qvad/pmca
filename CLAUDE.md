# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PMCA (Portable Modular Coding Agent) — a fully local, hierarchical coding agent using Ollama-hosted LLMs (7B–16B). It implements a cascade: architect → decompose → code → deterministic repair → review → verify. Python ≥ 3.11, async/await throughout.

## Commands

```bash
# Install for development (dev extras NOT included in [all])
pip install -e ".[all,dev]"

# Run all tests
pytest

# Run a single test file / single test
pytest tests/test_orchestrator.py -v
pytest tests/test_v2_features.py::TestOracleRepair::test_oracle_fix -v

# Lint
ruff check pmca/
mypy pmca/

# Run the agent
pmca run -c config/test_7b.yaml "Create a Calculator class"
pmca run -w ./my_project "Build a REST API client"

# Other CLI commands
pmca setup                                    # install Ollama + pull models
pmca status                                   # check task status
pmca resume                                   # resume interrupted work
pmca models                                   # list configured models
pmca serve --port 8000                        # OpenAI-compatible API server
pmca mcp --workspace ./workspace              # MCP stdio server

# Run benchmarks
python scripts/benchmark.py --task calculator
python scripts/benchmark.py --tier simple   # simple|medium|complex
python scripts/benchmark_raw.py              # raw model comparison (no cascade)
python scripts/benchmark_matrix.py           # cross-model × technique matrix
```

## Architecture

**Entry points:** `pmca/cli.py` (Click CLI), `pmca/api/server.py` (FastAPI, OpenAI-compatible), `pmca/mcp/server.py` (MCP stdio).

**Orchestrator** (`pmca/orchestrator.py`) drives the entire cascade. It manages a `TaskTree` and coordinates agents that all inherit from `BaseAgent` (`pmca/agents/base.py` — provides `_generate()`, `_generate_structured()`, `_parse_code_blocks()`):

- **ArchitectAgent** — generates specs, decides LEAF vs decompose into subtasks
- **CoderAgent** — implements code from specs, routes by difficulty (simple skips planning), supports best-of-N candidates and alternating fix strategies (odd=fix code, even=fix tests). Accepts `role_override` for routing to a stronger model and `think` for Ollama chain-of-thought. Calls `extract_spec_literals()` before implementation. Tracks `_fix_hashes` per task to skip duplicate fix attempts.
- **ReviewerAgent** — verifies spec/code/test alignment. `verify_tests()` is the quality gate in test-first mode (up to 3 regeneration loops).
- **WatcherAgent** (largest module) — runs pytest/lint and owns the **deterministic repair chain** (zero LLM tokens): `auto_fix_deterministic()` → `static_analysis_gate()` → `spec_coverage_check()` → `calibrate_tests()` → `oracle_repair_tests()`
- **TesterAgent** — separate test generation (used when `cascade.test_first: true`)

Prompt templates live in `pmca/prompts/` (one file per agent role, string constants formatted with `.format()`). Model configuration and provider abstraction (Ollama, Groq, OpenAI, Liquid) are in `pmca/models/`.

**Multi-file project mode** activates when `max_depth > 1` in config. The decompose prompt produces one subtask per .py file with `TARGET_FILE`/`EXPORTS`/`DEPENDS_ON` metadata. Dependencies are topologically sorted (Kahn's algorithm), AST interface extraction gives sibling specs zero-token context, and `FileAssembler` (`pmca/utils/assembler.py`) merges snippets.

**Task state machine** (`pmca/tasks/state.py`): `PENDING → DESIGNING → DECOMPOSED → CODING → REVIEWING → INTEGRATING → VERIFIED`. Failed tasks can retry (`FAILED → PENDING`), review can send back to coding. `LessonRecord` stores distilled lessons from failed fix attempts (session-only) for injection into subsequent fix prompts.

## Project Rules (MANDATORY)
- **NO BLIND VERIFICATION**: Code is not "verified" until `pytest` has physically executed and returned exit code 0. No profile or config is allowed to skip actual test execution.
- **DETERMINISTIC FIRST**: Zero-token deterministic repairs always run before any LLM-based fix retry. This is the core design principle.

## Configuration

YAML files in `config/`. Key knobs:

- `cascade.max_depth`: 1 = single-file, 2+ = multi-file project mode
- `cascade.max_retries`: retry budget before FAILED
- `cascade.best_of_n`: generate N candidates, pick best (1 = disabled)
- `cascade.fresh_start_after`: regenerate from scratch after N failed fixes
- `cascade.test_first`: run TesterAgent before CoderAgent (default false)
- `cascade.defensive_guards`: Phase 2B preventive AST guards (default false — harmful, avoid)
- `cascade.runtime_fixes`: Phase 2A error-driven AST fixes (default true)
- `cascade.mutation_oracle`: MuTAP mutation testing oracle for test quality (default false)
- `cascade.failure_memory`: ExpeRepair dual-memory; stores failure episodes and injects repair hints (default false)
- `cascade.cross_execution`: cross-test validation across candidates in best-of-N (default false)
- `lint.mypy` / `lint.ruff`: enable lint gates
- `models.<role>.name`: Ollama model tag per agent role
- `models.<role>.provider`: `ollama` (default), `groq`, `openai`, `liquid`

Provider API keys via env vars: `GROQ_API_KEY`, `OPENAI_API_KEY`, `LIQUID_API_KEY`.

## Testing

Pytest with `asyncio_mode = "auto"` (configured in `pyproject.toml`). No CI pipeline — tests run locally only.

All agents are mocked in unit tests via `pytest-mock` (`AsyncMock` for async methods). Common fixture pattern: create `Config` + `ModelManager` with `generate = AsyncMock()`, use `tempfile.TemporaryDirectory` for workspace.

## Conventions

- All LLM and I/O operations are async (`async def`, `await`)
- The watcher agent handles all test execution and repair logic — code fixes flow through it
- Config is loaded into dataclasses (`pmca/models/config.py`) from YAML
- Generated code and benchmark artifacts go to `workspace/` (gitignored)
- MCP server uses `asyncio.Lock` to serialize tool calls (orchestrator not thread-safe)
