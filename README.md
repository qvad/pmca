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
"Phase 1" = PMCA + lesson records, semgrep autofix, cross-execution gate.
"Micro-fix" = Phase 1 + targeted LLM micro-fix for single-function errors.

```
                     Raw 7B       Raw 14B      PMCA + 7B    Micro-fix + 7B
                     ──────       ───────      ─────────    ──────────────
Probes passing       62/76        75/76        71/76        70/76
Probe pass rate      82%          99%          93%          92%
PMCA tasks passed    —            —            8/12         8/12
LLM calls            12           12           98           122
Total tokens         5,256        5,406        149,370      168,068
Wall time            30.6s        59.8s        280.4s       324.7s
```

### Per-Task Breakdown

```
Task             Tier       Raw 7B   Raw 14B   PMCA+7B   Micro-fix+7B
──────────────────────────────────────────────────────────────────────
calculator       simple      6/6      6/6       6/6       6/6
stack            simple      6/6      6/6       6/6       6/6
counter          simple      4/4      4/4       4/4       4/4
fizzbuzz         simple      3/3      3/3       3/3       3/3
text_stats       medium      7/7      7/7       6/7       7/7
bank_account     medium      5/5      5/5       4/5       4/5
linked_list      medium      4/6      6/6       6/6       6/6
task_board       complex     6/6      6/6       6/6       5/6
matrix           complex     7/7      7/7       7/7       7/7
lru_cache        complex     0/6      6/6       6/6       6/6
data_pipeline    complex     5/10     9/10      9/10      9/10
todo_manager     complex     9/10    10/10      8/10      7/10
──────────────────────────────────────────────────────────────────────
Total                       62/76    75/76     71/76     70/76
```

### Micro-Fix Results (2026-03-01)

Added targeted LLM micro-fix: when a test fails with a traceback pointing to a specific function, extract just that function via AST + error context, send a ~100-token prompt to the LLM for a surgical fix, and splice it back. Runs as a pre-pass before the full `coder.fix()`.

**Probe recovery**: text_stats recovered 1 probe (sentence_count: 6/7→7/7) via micro-fix. counter now passes internally (was exhausting retries on LLM test assertions).

**Remaining failures**: todo_manager (3 probes: None date handling, sort direction), data_pipeline (1 probe: empty data edge case), bank_account (1 probe: history format), task_board (1 probe: add_returns_id assertion). These are stochastic — the exact failing probes vary between runs as 7B output is nondeterministic.

**Cost tradeoff**: +24 LLM calls and +19K tokens vs baseline PMCA. Most of the increase is stochastic — lru_cache took 3 retries this run vs 0 previously. The micro-fix itself adds only 1-2 calls per task where it fires.

### Comprehensive Technique Inventory

Every technique in PMCA, categorized by type, with measured probe-level impact. "LLM cost" is per invocation. Techniques marked with measured probe deltas show the specific task where the effect was observed (Raw 7B → PMCA).

#### Deterministic Repair Chain (zero LLM tokens)

Runs before any LLM review and after each retry. Order matters — earlier fixes enable later ones.

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 1 | **Package import rewriting** | `from pkg.mod import X` → `from mod import X` when sibling exists | 7B generates fully-qualified imports in flat layouts → `ModuleNotFoundError` | Prevents import crashes on ~15% of tasks |
| 2 | **Mutable default fix** | `def f(x=[])` → `def f(x=None); x = x or []` | Shared-state bugs from mutable defaults | Prevents subtle bugs; hard to measure in probes |
| 3 | **Attr/method shadowing fix** | `self.size = 0` + `def size()` → renames to `self._size` via AST+regex | `TypeError: 'int' object is not callable` | **linked_list: 4/6 → 6/6 (+2 probes)** |
| 4 | **Ruff auto-fix** | `ruff check --fix` for unused/duplicate imports, style | Dangling imports, F401/F811 | Cleanup; reduces noise for LLM review |
| 5 | **Semgrep anti-pattern fix** | Custom rules: broad `except Exception` → specific types, `print()` removal | 7B anti-patterns that mask errors | Prevents silent failures |
| 6 | **Known imports injection** | Dictionary of ~50 stdlib imports, injects on `NameError` (3 passes) | 7B forgets `OrderedDict`, `Optional`, `Counter`, etc. | **lru_cache: 0/6 → 6/6 (+6 probes)** |

#### Static Analysis Gates (zero LLM tokens, blocking)

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 7 | **Syntax validation** | `ast.parse()` all .py files; block review on `SyntaxError` | Broken syntax wastes LLM review tokens | Prevents ~5% of wasted reviews |
| 8 | **API consistency lint** | AST visitor detects attr/method shadowing + mixed callsites in tests | Tests call `obj.X` and `obj.X()` inconsistently | Catches design errors before review |
| 9 | **Spec coverage check** | Regex+AST verify all spec names are defined in code | Coder omits required functions but tests pass on partial impl | Forces re-implementation of missing functions |

#### Test Calibration (zero LLM tokens)

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 10 | **Calibrate tests** (conservative) | Fix assertions within 25% relative error or power-of-10 difference | LLM arithmetic errors in test expectations | Prevents ~30% of false test failures |
| 11 | **Oracle repair** (aggressive) | Trust non-zero/non-empty actual values as ground truth | Assertion mismatches beyond 25% threshold | **bank_account: tests fixed without retry** |

#### Prompt Engineering (zero extra LLM cost)

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 12 | **ICoT planning** | Spec + Idea before code (replaced rigid SCoT) | 7B can't follow SEQUENCE/BRANCH/LOOP structure | Consistent improvement on complex tasks |
| 13 | **Difficulty routing** | Deterministic classifier routes simple→no-plan, complex→ICoT | 7B overthinks simple tasks, produces brittle code | Saves ~30% tokens on simple tier |
| 14 | **Simple task prompt** | Skip planning entirely for ≤2 functions, no cross-deps | Planning overhead degrades simple task quality | All simple tasks pass (19/19 probes) |
| 15 | **Metamorphic test guidance** | Prefer `isinstance(result, float)` over `assert result == 42.1234` | Fragile exact-literal assertions waste retry cycles | Reduces calibration needs |
| 16 | **String literal extraction** | JSON schema extracts enums from spec before coding (temp=0.0) | 7B hallucinates string values like status codes | Prevents string mismatch errors |
| 17 | **Spec literals injection** | "Use EXACTLY these strings" section in prompt | 7B ignores spec values, invents alternatives | data_pipeline filter values correct |

#### Fix Loop Strategies (LLM-based)

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 18 | **Alternating fix strategy** | Odd retries fix code, even retries fix tests | Circular validation: LLM fights its own wrong tests | Breaks fix loops on ~40% of retried tasks |
| 19 | **Hash-based dedup** | MD5 tracks code; if duplicate, force test-fix mode | Coder regenerates identical broken code | Forces exploration of new strategies |
| 20 | **Dedup warning prefix** | "Your previous fix was IDENTICAL — try fundamentally different approach" | Coder ignores implicit hints to change strategy | Explicit instruction works better for 7B |
| 21 | **Temperature escalation** | 0.2 → +0.1 per retry, capped at 0.5 | Local minima in solution space | Explores without losing coherence |
| 22 | **Lesson injection** | Last 3 `LessonRecord`s injected as "do NOT repeat these mistakes" | Coder repeats same failed fix pattern | **matrix: 3→0 retries, data_pipeline: 3→0** |
| 23 | **Fresh start** | After N failed fixes, regenerate from scratch (not fix) | LLM debugging decays exponentially after 3 attempts | Breaks stuck loops; last resort |
| 24 | **Targeted micro-fix** | Extract single function via AST + error, send ~100-token LLM prompt | Single-function bugs (wrong sort, missing guard) | **text_stats: 6/7→7/7 (+1 probe)** |

#### Sampling & Validation (LLM-based)

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 25 | **Best-of-N** | Generate N candidates at varying temps, pick best by test pass rate | High variance in 7B output | Reduces variance; configurable (default N=1) |
| 26 | **Cross-execution gate** | Run top candidate's code against other candidates' tests | Hallucinated test assertions pass only on wrong code | Filters ~15% of hallucinated tests |
| 27 | **Fake code detection** | LLM checks for hard-coded returns, stubs, pass statements | Coder generates trivial impl that passes tests by accident | Catches stub implementations |
| 28 | **Smoke test gate** | Run tests BEFORE LLM review; skip review if tests crash | Broken imports/syntax waste reviewer LLM tokens | Saves ~300 tokens per crash |

#### Multi-File Project Mode (max_depth > 1)

| # | Technique | What it does | Failure mode addressed | Measured impact |
|---|-----------|-------------|----------------------|-----------------|
| 29 | **One-subtask-per-file** | Decompose prompt produces TARGET_FILE/EXPORTS/DEPENDS_ON per file | Over-decomposition splits classes across files | Clean module boundaries |
| 30 | **Topological sort** | Kahn's algorithm orders files by dependency DAG | Dependent modules compiled before dependencies exist | Correct build order |
| 31 | **AST interface extraction** | Extract public signatures (no bodies) for zero-token sibling context | Sibling modules don't know what others export | Free cross-file context |
| 32 | **FileAssembler** | Set-based import dedup + AST definition dedup (later wins) | Duplicate imports/definitions when merging multi-file code | Clean merged output |
| 33 | **Single-class LEAF rule** | Force LEAF (don't decompose further) for single-class specs | Splitting a class into per-method subtasks breaks 7B generation | Prevents over-decomposition |

### Aggregate Effect on Probes

```
Task             Tier       Raw 7B   PMCA+7B   Delta   Key techniques responsible
──────────────────────────────────────────────────────────────────────────────────
calculator       simple      6/6      6/6        0     (already passes)
stack            simple      6/6      6/6        0     (already passes)
counter          simple      4/4      4/4        0     (already passes)
fizzbuzz         simple      3/3      3/3        0     (already passes)
text_stats       medium      7/7      7/7        0     micro-fix recovered (was 6/7)
bank_account     medium      5/5      4/5       -1     circular validation regression
linked_list      medium      4/6      6/6       +2     #3 attr/method shadowing fix
task_board       complex     6/6      5/6       -1     stochastic (varies per run)
matrix           complex     7/7      7/7        0     #22 lessons: 3→0 retries
lru_cache        complex     0/6      6/6       +6     #6 known imports injection
data_pipeline    complex     5/10     9/10      +4     #12 ICoT + repair chain
todo_manager     complex     9/10     7/10      -2     circular validation regression
──────────────────────────────────────────────────────────────────────────────────
Total                       62/76    70/76      +8     net: +12 recovered, -4 regressed
```

### What Helps, What Hurts

**Net positive (+12 probes recovered)**:
- Known imports injection: **+6** (lru_cache)
- Deterministic repair chain: **+4** (data_pipeline)
- Attr/method shadowing fix: **+2** (linked_list)

**Net negative (-4 probes regressed)**:
- Circular validation problem: **-3** (bank_account -1, todo_manager -2) — LLM generates both code AND tests; retry loop sometimes degrades correct code while trying to fix wrong tests
- Stochastic variance: **-1** (task_board) — 7B output is nondeterministic; same task produces different probes per run

**Net zero but efficiency gains**:
- Lesson injection: matrix/data_pipeline retries 3→0 (saved ~40K tokens across runs)
- Difficulty routing: simple tasks use ~30% fewer tokens
- Calibrate + oracle: prevents ~50% of false assertion failures

### The Circular Validation Problem

The biggest unsolved problem. When PMCA generates both code and tests:
1. Code is correct, but test assertion is wrong (`assert balance == 150.0` when correct answer is `50.0`)
2. Retry loop "fixes" code to match wrong test → code breaks
3. Alternating fix strategy + oracle repair mitigate but don't eliminate
4. 14B raw model avoids this entirely (no retry loop) but has no safety net for genuine bugs

This accounts for all 4 regressed probes vs raw 7B.

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

- **Verification-first architecture**: 13+ zero-token deterministic repairs run BEFORE any LLM review — this is the core design principle
- **Difficulty routing**: deterministic classifier (zero LLM cost) routes simple tasks to a planning-free prompt, avoiding 7B "overthinking" degradation
- **ICoT over SCoT**: replaced rigid SEQUENCE/BRANCH/LOOP chain-of-thought with Intention-based CoT (Specification + Idea) — simpler structure that 7B models follow reliably
- **Single-class LEAF rule**: decompose forces LEAF for any single-class spec; over-decomposition (splitting into per-method subtasks) consistently breaks code generation
- **Alternating fix strategy**: odd retries fix code, even retries fix tests — addresses the circular validation problem when the coder generates both
- **Oracle repair**: second-pass assertion fix that trusts actual non-trivial values as ground truth, catching mismatches beyond conservative calibration
- **Attribute/method shadowing auto-fix**: 7B models consistently generate `self.size = 0` alongside `def size(self)` — deterministic rename to `self._size` via regex
- **Targeted micro-fix**: extract single function via AST + error context, send ~100-token prompt for surgical fix before expensive full coder.fix()

## License

MIT
