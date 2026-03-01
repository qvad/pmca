# PMCA — Portable Modular Coding Agent

A fully local, hierarchical coding agent that uses Ollama-hosted LLMs (7B–16B) to decompose, implement, review, and verify code generation tasks. No cloud APIs required — runs entirely on your machine.

## Quick Start

```bash
pip install -e ".[all]"
pmca setup                          # install Ollama + pull models
pmca run "Create a Calculator class with add, subtract, multiply, divide"
```

## How It Works

PMCA breaks any programming task into pieces small enough for a local 7B model to solve reliably, then assembles and verifies the result. The core insight: local models can generate ~5–30 lines of correct code per call — the system's job is decomposition, verification, and assembly.

### Cascade Architecture

```
                         ┌──────────────┐
                         │   User Task  │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  Architect   │  generate spec, decide LEAF vs decompose
                         └──────┬───────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
              ┌─────▼────┐┌────▼─────┐┌────▼─────┐
              │ Subtask 1 ││ Subtask 2 ││ Subtask 3 │  recursive decomposition
              └─────┬────┘└────┬─────┘└────┬─────┘
                    │          │           │
                    └──────────┼───────────┘
                               │
                        ┌──────▼───────┐
                        │    Coder     │  implement each leaf
                        └──────┬───────┘
                               │
                    ┌──────────▼──────────┐
                    │  Deterministic      │  zero LLM tokens:
                    │  Repair Chain       │  auto-fix → static analysis →
                    │                     │  spec coverage → test calibration →
                    │                     │  oracle repair
                    └──────────┬──────────┘
                               │
                        ┌──────▼───────┐
                        │   Reviewer   │  verify spec/code/test alignment
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │   Watcher    │  run tests, retry loop with fixes
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │   Verified   │
                        └──────────────┘
```

### Four Agents

| Agent | Role | Key Capability |
|-------|------|----------------|
| **Architect** | Spec generation & decomposition | Decides LEAF vs subtasks; AST-based interface extraction for cross-file context |
| **Coder** | Implementation & fixing | Difficulty-based routing (simple tasks skip planning); best-of-N candidates; alternating fix strategy (odd=fix code, even=fix tests) |
| **Reviewer** | Verification | Checks spec/code/test alignment at each phase |
| **Watcher** | Test execution & deterministic repair | Largest agent — owns the entire repair chain (see below) |

### Deterministic Repair Chain

Before any LLM-based retry, PMCA runs a zero-token repair pipeline:

1. **auto_fix_deterministic** — package imports, mutable defaults, attribute/method shadowing, ruff auto-format, known imports injection
2. **static_analysis_gate** — `ast.parse` syntax check + API consistency lint (detects `self.size` shadowing `def size()`)
3. **spec_coverage_check** — regex + AST verify all spec names are defined in code
4. **calibrate_tests** — conservative: fix assertions within 25% or power-of-10 threshold
5. **oracle_repair_tests** — aggressive: trust non-zero/non-empty actual values as ground truth

### Multi-File Project Mode

When `max_depth > 1`, PMCA generates multi-file projects:
- Architect decomposes into one subtask per `.py` file with `TARGET_FILE` / `EXPORTS` / `DEPENDS_ON` metadata
- Dependencies are topologically sorted (Kahn's algorithm)
- AST interface extraction provides sibling modules with zero-token context
- `FileAssembler` merges all snippets into the final project

## Usage

### CLI

```bash
# Run a task
pmca run "Build a linked list with insert, delete, search"

# Use a specific config
pmca run -c config/test_7b.yaml "Implement FizzBuzz"

# Custom workspace
pmca run -w ./my_project "Create a REST API client"

# Check task status / resume interrupted work
pmca status
pmca resume

# List configured models and check availability
pmca models
```

### API Server (OpenAI-compatible)

```bash
pmca serve --port 8000
```

Exposes `/v1/chat/completions` (with SSE streaming), `/v1/models`, and `/health`. Compatible with OpenCode, aider, Continue, Cursor, or any OpenAI-compatible client — set base URL to `http://localhost:8000/v1` and model to `pmca`.

### MCP Server

```bash
pmca mcp --workspace ./workspace
```

Stdio transport for Claude Desktop or VS Code. Exposes `run_task`, `status`, and `resume` tools.

## Configuration

YAML files in `config/`. Example (`config/test_7b.yaml`):

```yaml
models:
  architect:
    name: "qwen2.5-coder:7b-instruct-q4_K_M"
    context_window: 4096
    temperature: 0.3
  coder:
    name: "qwen2.5-coder:7b-instruct-q4_K_M"
    context_window: 4096
    temperature: 0.2
  reviewer:
    name: "qwen2.5-coder:7b-instruct-q4_K_M"
    context_window: 4096
    temperature: 0.1
  watcher:
    name: "qwen2.5-coder:7b-instruct-q4_K_M"
    context_window: 4096
    temperature: 0.1

cascade:
  max_depth: 1        # 1 = single-file, 2+ = multi-file project mode
  max_retries: 3      # retry budget before marking FAILED
  max_children: 3     # max subtasks per decomposition

workspace:
  path: "./workspace"
  git_checkpoint: false
```

Included configs: `test_7b.yaml`, `test_14b.yaml`, `test_mixed.yaml`, `project_7b.yaml`, `project_mixed.yaml`, `project_deepseek.yaml`.

Supported model providers: **Ollama** (default, local), **Groq**, **OpenAI**, **Liquid** — set `provider` and `api_base` per role.

## Benchmark Results

All benchmarks use `qwen2.5-coder:7b-instruct-q4_K_M` on a single 16 GB GPU.

### Impact of Optimizations

Baseline = bare cascade (architect → coder → reviewer), no deterministic repair.
Optimized = full pipeline with all deterministic repair gates enabled.

```
                        Baseline (7B)          Optimized (7B)
                        ─────────────          ──────────────
Tasks passing           7 / 10                 8 / 10
External probes         52 / 56  (93%)         56 / 56  (100%)
Avg retries / task      ~1.5                   ~0.6
Total time (10 tasks)   ~200 s                 144.5 s
```

Optimizations that drove the improvement:

| Optimization | Effect |
|---|---|
| Attr/method shadowing auto-fix | linked_list: 4/6 → 6/6 probes |
| Oracle repair (2nd-pass assertions) | bank_account: 4/5 → 5/5 probes |
| Difficulty routing (skip planning for simple tasks) | Eliminated 7B "overthinking" regressions |
| Test calibration (25% / power-of-10 threshold) | Reduced retry cycles on numeric assertions |
| API consistency lint | Catches shadowing before tests run, saving full retry loops |

### Full Results (12 tasks)

```
Task             Tier      Result   Probes     LLM Calls  Time
────────────────────────────────────────────────────────────────
calculator       simple    PASS     6/6        7          84.4s
stack            simple    PASS     6/6        7          33.6s
counter          simple    PASS     4/4        6          23.7s
fizzbuzz         simple    PASS     3/3        6          24.9s
text_stats       medium    FAIL*    7/7        11         74.9s
bank_account     medium    FAIL*    4/5        11         93.1s
linked_list      medium    PASS     6/6        7          43.6s
task_board       complex   PASS     6/6        9          55.5s
matrix           complex   FAIL*    7/7        11         94.6s
lru_cache        complex   PASS     6/6        7          40.7s
data_pipeline    complex   FAIL*    10/10      7          237.5s
todo_manager     complex   FAIL*    10/10      7          122.9s
────────────────────────────────────────────────────────────────
Total                      7/12     75/76      96         929.4s
```

*\* Tasks marked FAIL exhaust retries on LLM-generated test assertions — 75 of 76 external probes still pass. The generated code is typically correct; the internal tests are wrong.*

## Installation

Requires Python >= 3.11.

```bash
# Full install
pip install -e ".[all]"

# Minimal (core only)
pip install -e .

# By feature
pip install -e ".[dev]"      # pytest, pytest-asyncio, pytest-mock
pip install -e ".[lint]"     # mypy, ruff
pip install -e ".[rag]"      # chromadb, sentence-transformers
pip install -e ".[mcp]"      # Model Context Protocol
```

## Key Design Decisions

- **Difficulty routing**: deterministic classifier (zero LLM cost) routes simple tasks to a planning-free prompt, avoiding 7B "overthinking" degradation
- **ICoT over SCoT**: replaced rigid SEQUENCE/BRANCH/LOOP chain-of-thought with Intention-based CoT (Specification + Idea) — simpler structure that 7B models follow reliably
- **Single-class LEAF rule**: decompose forces LEAF for any single-class spec; over-decomposition (splitting into per-method subtasks) consistently breaks code generation
- **Alternating fix strategy**: odd retries fix code, even retries fix tests — addresses the circular validation problem when the coder generates both
- **Oracle repair**: second-pass assertion fix that trusts actual non-trivial values as ground truth, catching mismatches beyond conservative calibration
- **Attribute/method shadowing auto-fix**: 7B models consistently generate `self.size = 0` alongside `def size(self)` — deterministic rename to `self._size` via regex

## License

MIT
