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

12 tasks across 3 tiers (simple / medium / complex), single 16 GB GPU. Each task is validated by deterministic probes — external tests that check the generated code without relying on LLM-generated assertions.

### Raw Model vs PMCA

"Raw" = single-shot prompt directly to the model, no agents or repair.
"PMCA" = full cascade pipeline (architect → coder → deterministic repair → reviewer → watcher).

```
                     Raw 7B       Raw 14B      PMCA + 7B
                     ──────       ───────      ─────────
Probes passing       62/76        75/76        71/76
Probe pass rate      82%          99%          93%
LLM calls            12           12           98
Total tokens         5,256        5,406        149,370
Wall time            30.6s        59.8s        280.4s
```

### Per-Task Breakdown

```
Task             Tier       Raw 7B   Raw 14B   PMCA+7B
─────────────────────────────────────────────────────────
calculator       simple      6/6      6/6       6/6
stack            simple      6/6      6/6       6/6
counter          simple      4/4      4/4       4/4
fizzbuzz         simple      3/3      3/3       3/3
text_stats       medium      7/7      7/7       6/7
bank_account     medium      5/5      5/5       4/5
linked_list      medium      4/6      6/6       6/6
task_board       complex     6/6      6/6       6/6
matrix           complex     7/7      7/7       7/7
lru_cache        complex     0/6      6/6       6/6
data_pipeline    complex     5/10     9/10      9/10
todo_manager     complex     9/10    10/10      8/10
─────────────────────────────────────────────────────────
Total                       62/76    75/76     71/76
```

### What the Numbers Show

**Raw 7B** fails on tasks that need correct imports (`lru_cache`: missing `OrderedDict`), attribute/method shadowing (`linked_list`: `self.size` vs `def size()`), and complex multi-step logic (`data_pipeline`). These are exactly the failure modes PMCA's deterministic repair chain targets.

**PMCA + 7B** recovers `lru_cache` (0→6 probes) and `linked_list` (4→6) through auto-fix and import injection, but loses probes on `text_stats`, `bank_account`, and `todo_manager` due to the circular validation problem — the LLM generates both code and tests, and sometimes the test assertions are wrong, causing retry loops that degrade code that was originally correct.

**Raw 14B** benefits from stronger reasoning and fewer mechanical errors (correct imports, no shadowing) but has no safety net — a single mistake means a failed probe with no retry. PMCA's value increases as task complexity grows and models get weaker.

### Where PMCA Helps Most

| Failure mode | Raw 7B | PMCA fix | Probes recovered |
|---|---|---|---|
| Missing imports (`OrderedDict`) | lru_cache: 0/6 | `KNOWN_IMPORTS` injection | +6 |
| Attr/method shadowing (`self.size` vs `def size()`) | linked_list: 4/6 | `_fix_attr_method_shadowing` | +2 |
| Complex pipeline logic | data_pipeline: 5/10 | Spec decomposition + repair chain | +4 |

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
