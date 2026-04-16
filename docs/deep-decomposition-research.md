# Deep Decomposition for Local Code Generation

## The Core Idea

Local 7B-14B models can reliably generate **~5-30 lines of correct code** per generation. Anything beyond that — multi-method classes, cross-file dependencies, arithmetic in tests — becomes unreliable (~40-60% failure rate).

The hypothesis: **any programming task can be broken down until each leaf is solvable by a 7B model**. The system's job is decomposition, verification, and assembly — not generation of complex code.

This document captures research, experiments, and the current architecture.

---

## Current State (Mar 2026)

### Benchmark History

| Metric | Raw 7B | Baseline (Feb) | Phase 1 (Mar) | Micro-fix (Mar) |
|--------|--------|----------------|---------------|-----------------|
| Probes passing | 62/76 (82%) | 71/76 (93%) | 69/76 (91%) | 70/76 (92%) |
| PMCA tasks passed | — | 8/12 | 8/12 | 8/12 |
| LLM calls | 12 | 98 | 95 | 122 |
| Total tokens | 5,256 | 149,370 | 134,221 | 168,068 |
| Wall time | 30.6s | 280.4s | 250.1s | 324.7s |

Note: Single-run numbers. Probe count variance is ±5 per run due to 7B non-determinism.

### Matrix Benchmark (Mar 2026)

Cross-model comparison using `scripts/benchmark_matrix.py`. Each cell shows avg/76 probes (range across 2 runs).

```
Model × Combo Comparison (qwen7b, 12 tasks, 76 probes)
─────────────────────────────────────────────────────────────────
Combo             Avg Probes   Range     Calls   Tokens     Time
baseline          69.5/76      69-70     114     158K       287s
runtime_only      69.5/76      68-71     118     158K       291s
full_guards       65.5/76      58-73     118     168K       311s
─────────────────────────────────────────────────────────────────
```

**Conclusion**: Baseline and runtime_only are equivalent. Full guards introduce extreme variance (15-probe range) and lower average.

```
Cross-Model Comparison (baseline combo, 2 runs each)
──────────────────────────────────────────────────────────────────────
Model            Avg Probes   Range    Calls  Tokens    Time    Note
qwen7b           69.5/76      69-70    114    158K      287s    Best speed/quality
qwen14b          70.0/76      66-74    117    193K      864s    Best on complex tasks
deepseek16b      66.0/76      62-70    110    190K      346s    Weak on todo_manager
mixed            68.0/76      66-70    131    205K      545s    No benefit from mixed
llama8b          24.5/76      20-29    295    471K      1790s   Not a code model
──────────────────────────────────────────────────────────────────────
```

```
Per-Task Comparison (baseline, avg probes per model)
──────────────────────────────────────────────────────────────────────
Task              qwen7b  qwen14b  deepseek  llama8b  mixed
calculator         6/6     6/6      6/6       3/6      6/6
stack              6/6     6/6      6/6       6/6      6/6
counter            4/4     4/4      4/4       4/4      4/4
fizzbuzz           3/3     3/3      3/3       1/3      3/3
text_stats         6/7     3.5/7    6/7       0/7      6.5/7
bank_account       4/5     4/5      4.5/5     0/5      4/5
linked_list        6/6     6/6      6/6       3/6      5.5/6
task_board         6/6     6/6      5.5/6     1/6      6/6
matrix             7/7     7/7      7/7       3.5/7    7/7
lru_cache          6/6     6/6      6/6       3/6      6/6
data_pipeline      8.5/10  9.5/10   9/10      0/10     8.5/10
todo_manager       7/10    9/10     3/10      0/10     5.5/10
──────────────────────────────────────────────────────────────────────
Total             69.5    70.0     66.0      24.5     68.0
──────────────────────────────────────────────────────────────────────
```

Key per-task findings:
- **Simple tasks (19/19)**: All code models solve perfectly. Only llama8b struggles.
- **todo_manager**: qwen14b dominates (9/10). deepseek16b catastrophic (3/10). Complex multi-method class with edge cases.
- **data_pipeline**: qwen14b best (9.5/10), all others 8-9. The `empty_data` probe fails 80% of runs across all models.
- **text_stats**: High variance for all models (0-7 range). 7B sometimes generates static methods instead of instance methods.
- **bank_account/history**: Persistent failure (90% of runs). Format string `'deposit 100.0'` vs `'Deposited: $100.00'`.

### Most Common Failures (across all baseline runs)

| Probe | Failure rate | Root cause |
|-------|-------------|------------|
| `bank_account/history` | 90% | Format string mismatch |
| `todo_manager/list_by_priority` | 90% | Sort ascending instead of descending |
| `data_pipeline/empty_data` | 80% | IndexError/KeyError on empty list |
| `todo_manager/overdue` | 70% | None date comparison |
| `todo_manager/sort_by_due_date` | 70% | TypeError: None in sort |
| `text_stats/sentence_count` | 60% | Split logic incorrect |

### Implemented Techniques (cumulative)

**Baseline (Feb 2026)**: Full cascade + deterministic repair chain + ICoT + difficulty routing + oracle repair + attr/method shadowing fix + calibrate tests + known imports injection + alternating fix strategy + best-of-N + fresh start.

**Phase 1 (Mar 2026)**: + Lesson records (TraceCoder-style failure history in fix prompts) + semgrep autofix (7B anti-pattern rules) + cross-execution gate (ConVerTest for best-of-N).

**Micro-fix (Mar 2026)**: + Targeted LLM micro-fix (AST extraction of single function + surgical LLM fix before full coder.fix()).

**Phase 2 (Mar 2026)**: + Defensive guard injection (`inject_defensive_guards`: None-safe sort key wrapping, empty-collection `x[0]` guards, missing else-raise — pre-test, zero tokens) + Runtime error repair (`fix_runtime_errors`: TypeError-in-sort and IndexError fixes from tracebacks — in retry loop, zero tokens). **Disabled by default** — see evaluation below.

### Technique Evaluation (Phase 2 Research)

Each technique was tested with 2+ benchmark runs. Findings:

| Technique | Effect | Verdict | Detail |
|-----------|--------|---------|--------|
| **Deterministic repair chain** | +18 probes (62→~80 corrected) | **Keep (core)** | auto_fix + known_imports + shadowing + calibrate + oracle. Foundation of all quality gains. |
| **Lesson records** | -2 retries avg, -40K tokens | **Keep** | TraceCoder-style failure history in fix prompts. Saves tokens on matrix/data_pipeline. |
| **Micro-fix** | +1-2 probes stochastic | **Keep** | Surgical single-function LLM fix before full coder.fix(). Low cost, occasional wins on text_stats/todo_manager. |
| **Runtime error fixes (2A)** | ±0 probes avg | **Keep (harmless)** | Error-driven AST fixes in retry loop. Rarely fires (micro-fix usually handles it first). Zero cost when inactive. Config: `cascade.runtime_fixes: true`. |
| **Defensive guards (2B)** | -4 probes avg, ±15 variance | **Disabled** | Preventive AST transforms before first test. Causes semantic regressions: changes sort order, suppresses IndexError, injects ValueError. High variance (58-73 range vs 69-70 baseline). Config: `cascade.defensive_guards: false`. |
| **Guard: sort key wrapping** | Fires correctly on todo_manager | **Harmful net** | Wraps sort keys with None-safe tuples. Sometimes helps (73/76 best run) but sometimes breaks correct code (58/76 worst run). |
| **Guard: index zero** | Rarely fires | **Neutral** | Changes IndexError to return None — may break specs that expect exceptions. |
| **Guard: missing else-raise** | Fires on data_pipeline | **Harmful** | Injects ValueError where code expects implicit None return. |

**Why defensive guards fail**: They modify code that is *already correct* in most runs. The guards target rare failure modes (None in sorts, empty collections) but the semantic changes they introduce cause new failures more often than they prevent old ones. Error-driven repair (Phase 2A) is safer because it only touches code known to be broken.

### What Works

- **Deterministic repair chain**: 15+ zero-token fixes before LLM review
- **Known imports injection**: lru_cache 0/6 → 6/6 probes
- **Attr/method shadowing fix**: linked_list 4/6 → 6/6 probes
- **Lesson records**: matrix/data_pipeline retries 3→0 (saved ~40K tokens)
- **Micro-fix**: text_stats sentence_count recovered (6/7 → 7/7), todo_manager improvements
- **Oracle repair + calibration**: prevents ~50% of false assertion failures
- All simple tasks pass consistently (19/19 probes across all runs)
- **Model selection matters**: qwen14b solves todo_manager 9/10 where qwen7b gets 7/10

### What Still Fails

- **bank_account/history** (90% failure rate): Format string mismatch — model uses wrong format
- **todo_manager/list_by_priority** (90%): Sorts ascending instead of descending
- **data_pipeline/empty_data** (80%): IndexError/KeyError on empty list aggregation
- **todo_manager/sort_by_due_date** (70%): TypeError from None in sort comparison
- **text_stats/sentence_count** (60%): Split logic varies wildly across runs
- **Circular validation**: LLM generates wrong test assertions, retry loop degrades correct code
- **Non-determinism**: ±5 probe variance across runs (stochastic 7B output)

### Model Recommendations

| Use case | Model | Avg probes | Time | Note |
|----------|-------|-----------|------|------|
| Speed-optimized | qwen2.5-coder:7b | 69.5/76 | 287s | Best throughput, good quality |
| Quality-optimized | qwen2.5-coder:14b | 70.0/76 | 864s | Best on complex tasks (todo_manager 9/10) |
| Avoid | llama3.1:8b | 24.5/76 | 1790s | General-purpose, not a code model |
| Avoid | deepseek-coder-v2:16b | 66.0/76 | 346s | Worse than 7B on todo_manager |
| Avoid | mixed (deepseek+qwen) | 68.0/76 | 545s | No benefit from mixed arch/coder |

### Previous Benchmark Results

**10-task suite (Feb 2026)**: 56/56 probes (100%), 8/10 tasks, 68 LLM calls, 144.5s. This was before data_pipeline and todo_manager were added.

**Multi-file project mode (4 modules)**: Best run 14/15 tests, 9/11 probes. Average ~2/4 modules verified per run. 7B unreliable for multi-file cross-references.

---

## System Architecture

### Cascade Flow

```
run() → cascade(root)
         │
         ├─ force_leaf? (depth > max, or project_mode && depth >= 1)
         │   │
         │   yes → _code_leaf()                    no → design_phase()
         │          │                                    │
         │          ├─ auto_fix_deterministic()           ├─ architect.generate_spec()
         │          ├─ static_analysis_gate()             ├─ architect.decompose()
         │          ├─ spec_coverage_check()              │    LEAF → mark as leaf
         │          ├─ calibrate_tests()                  │    subtasks → create children
         │          ├─ oracle_repair_tests()              │
         │          │                                    ├─ review child specs (!project_mode)
         │          ▼                                    │
         │    review_phase()                             ▼
         │      │                                  cascade(child) for each child
         │      ├─ smoke test (run_tests)                │
         │      │    pass → review + faking check        ▼
         │      │    fail → extract errors           integrate_phase()
         │      │                                      ├─ assemble snippets (project_mode)
         │      ├─ retry loop (max_retries)            ├─ run integration tests
         │      │    ├─ coder.fix() or .implement()    └─ verify integration
         │      │    ├─ auto_fix_deterministic()
         │      │    ├─ calibrate_tests()
         │      │    ├─ oracle_repair_tests()
         │      │    └─ spec_coverage_check()
         │      │
         │      └─ VERIFIED or FAILED
```

### File Structure

```
pmca/
├── cli.py                  # Click CLI: run, status, resume, models, setup, serve, rag-index, mcp
├── orchestrator.py         # Cascade controller (1221 lines)
├── agents/
│   ├── architect.py        # Spec generation + decomposition (166 lines)
│   ├── coder.py            # Code implementation + fix (219 lines)
│   ├── reviewer.py         # LLM verification (85 lines)
│   ├── watcher.py          # Test execution + deterministic repair (1348 lines)
│   └── tester.py           # Test generation agent
├── prompts/
│   ├── architect.py        # DESIGN_SPEC, DECOMPOSE, DECOMPOSE_PROJECT prompts
│   ├── coder.py            # IMPLEMENT, IMPLEMENT_SIMPLE, FIX, FIX_TESTS prompts
│   ├── reviewer.py         # VERIFY_SPEC, VERIFY_CODE, VERIFY_TESTS prompts
│   └── watcher.py          # SMOKE_TEST, FINAL_VERIFY prompts
├── api/
│   └── server.py           # OpenAI-compatible FastAPI server
├── tasks/
│   └── state.py            # TaskNode, TaskState, TaskResult dataclasses
├── models/
│   └── manager.py          # Ollama ModelManager with telemetry
└── config/
    ├── test_7b.yaml        # Qwen 7B (primary benchmark config)
    ├── test_14b.yaml       # Qwen 14B
    └── test_mixed.yaml     # DeepSeek 16B architect + Qwen 7B coder
```

### Agents

**ArchitectAgent** (`pmca/agents/architect.py`)
- `generate_spec()` — structured spec: Purpose, Interface, Dependencies, Acceptance Criteria, Robustness
- `decompose()` — decides LEAF vs subtasks; single class always → LEAF; multi-file uses DECOMPOSE_PROJECT_PROMPT with TARGET_FILE/EXPORTS/DEPENDS_ON metadata
- `extract_interface_from_code()` — AST-based, zero LLM tokens, extracts public signatures for sibling context

**CoderAgent** (`pmca/agents/coder.py`)
- `implement()` — routes by difficulty: "simple" → IMPLEMENT_SIMPLE_PROMPT (skip planning), "complex" → IMPLEMENT_PROMPT (ICoT: Specification + Idea)
- `implement_best_of_n()` — N candidates at varying temperatures (0.2, 0.35, 0.5, ...), pick best by test pass count
- `implement_with_tests()` — generate code that must pass provided tests
- `fix()` — alternating strategy: odd retries fix code, even retries fix tests; hash-based dedup forces test-fix when duplicate code detected; temperature escalation: 0.2 + retry * 0.1

**ReviewerAgent** (`pmca/agents/reviewer.py`)
- `verify_spec()` — check child spec aligns with parent (skipped in project_mode)
- `verify_code()` — check implementation matches specification
- `verify_tests()` — check test quality (skipped in project_mode)
- `verify_integration()` — verify all children integrate (skipped in project_mode)

**WatcherAgent** (`pmca/agents/watcher.py`) — largest agent, handles all deterministic repair
- `run_tests()` — execute pytest with `--showlocals`, parse pass/fail counts
- `extract_structured_errors()` — parse pytest output into TestError objects with local_variables
- `static_analysis_gate()` — ast.parse syntax check + API consistency lint → (blocking, informational) errors
- `auto_fix_deterministic()` — multi-pass chain (see below)
- `calibrate_tests()` — conservative: fix assertions within 25% or power-of-10 threshold
- `oracle_repair_tests()` — aggressive: trust non-zero/non-empty actual values as oracles
- `check_not_faked()` — LLM call: verify code isn't stubbed
- `spec_coverage_check()` — regex + AST: verify all names from spec are defined in code
- `final_verification()` — end-to-end LLM check

### Deterministic Repair Chain

Runs before LLM review (zero tokens) and after each fix retry:

```
auto_fix_deterministic()
  ├─ _fix_package_imports()           # from pkg.module → from module
  ├─ _fix_mutable_defaults()          # def f(x=[]) → def f(x=None)
  ├─ _fix_attr_method_shadowing()     # self.size → self._size when def size() exists
  ├─ ruff --fix                       # auto-format
  └─ KNOWN_IMPORTS injection          # add missing stdlib/typing imports

static_analysis_gate()
  ├─ ast.parse()                      # syntax check
  └─ _check_api_consistency()         # detect attr/method shadowing + mixed callsites
       ├─ Pass 1: scan impl files     # self.X attr vs def X() method in same class
       └─ Pass 2: scan test files     # obj.X access vs obj.X() call (_UsageVisitor)

spec_coverage_check()
  ├─ extract names from spec          # regex: backticks, PascalCase, "implement X"
  └─ check defined in workspace       # AST scan for class/function definitions

calibrate_tests()                     # conservative assertion repair
  ├─ run pytest, parse failures
  ├─ numeric: fix if within 25% or power-of-10 (skip sign flips)
  └─ string: fix case differences only

oracle_repair_tests()                 # aggressive assertion repair (2nd pass)
  ├─ run pytest, parse remaining failures
  ├─ numeric: trust actual if actual != 0
  └─ string: trust actual if not empty/whitespace
```

### Gate Telemetry

`_gate_stats` (defaultdict(int)) auto-tracks all deterministic repairs:

| Counter | What it counts |
|---------|---------------|
| `auto_fix` | Deterministic fixes in first pass |
| `auto_fix_retry` | Deterministic fixes after retries |
| `syntax_errors` | Blocking errors from static analysis |
| `interface_inconsistency` | Attr/method shadowing detected by API lint |
| `lint_issues` | Informational lint warnings |
| `spec_coverage_gaps` | Missing names injected back into spec |
| `calibrations` | Test assertions fixed (conservative, 1st pass) |
| `calibrations_retry` | Test assertions fixed (conservative, retry) |
| `oracle_repairs` | Test assertions fixed (aggressive oracle) |

### Key Design Decisions

**Difficulty Routing** — `Orchestrator._estimate_difficulty()` is a deterministic classifier (zero LLM cost). Heuristics: function/class count > 2, cross-file deps, algorithmic keywords, spec length > 500 chars. Simple tasks skip the planning section in IMPLEMENT_PROMPT, which avoids 7B "overthinking" degradation.

**ICoT over SCoT** — Replaced SEQUENCE/BRANCH/LOOP structured chain-of-thought with simpler Intention-based CoT (Specification + Idea). SCoT's rigid structure confused 7B models; ICoT lets them state the idea in natural language before coding.

**Single-Class LEAF Rule** — The decompose prompt forces LEAF output for any single-class spec, even with many methods. Over-decomposition (splitting a class into per-method subtasks) consistently broke the code — each subtask generated a separate file with its own class definition.

**Alternating Fix Strategy** — Odd retries (1,3,5...) fix code, even retries (2,4...) fix tests. This addresses the circular validation problem: when the coder generates both code and tests, sometimes the test assertions are wrong, not the code.

**Oracle Repair** — After conservative calibration (25% threshold), a second pass aggressively trusts actual values. Guards: skip if actual is 0 (likely code returns nothing) or empty string (likely code bug). This catches mismatches beyond calibration's reach — e.g., `assert balance == 150.0` when actual is `50.0` (correct).

**Attr/Method Shadowing Auto-Fix** — 7B models consistently generate `self.size = 0` alongside `def size(self)`. Directive error messages ("rename the attribute") are ignored by 7B. The deterministic fix renames `self.size` → `self._size` via regex after AST detection. This is the pattern that fixed linked_list from 4/6 to 6/6 probes.

### Project Mode (Multi-File)

Enabled when `max_depth > 1`. At depth 0: decompose into file modules using DECOMPOSE_PROJECT_PROMPT (each subtask = one .py file with TARGET_FILE/EXPORTS/DEPENDS_ON metadata). At depth >= 1: force leaf, implement single file. Assembly uses FileAssembler with Kahn's topological sort on dependency metadata. LLM spec review, test review, and integration review are all skipped (7B too unreliable for these meta-tasks).

---

## Research Findings — Implementation Status

| # | Technique | Source | Status |
|---|-----------|--------|--------|
| 1 | Test-First Generation | TiCoder (ICSE 2025), AlphaCodium, AgentCoder | **Implemented** — `TesterAgent`, `test_first` config, `implement_with_tests()` |
| 2 | Best-of-N Sampling | Top Pass (2024), Scalable Best-of-N (2025) | **Implemented** — `implement_best_of_n()`, configurable `best_of_n` |
| 3 | Debugging Decay — Fresh Start | Nature Sci Reports 2025 | **Implemented** — `fresh_start_after` config |
| 4 | ICoT Prompting | ACM TOSEM 2024 (adapted) | **Implemented** — replaced SCoT |
| 5 | Static Analysis Gate | ICSE 2025 static analysis feedback | **Implemented** — full chain |
| 6 | Property-Based / Metamorphic Testing | FSE 2025 | **Prompt guidance only** — full Hypothesis not done |
| 7 | Sub-Module Composition | CodeChain (ICLR 2024) | **Skipped** — single-class LEAF rule works better |

### Additionally Implemented (not in original research plan)

| Technique | Source | Status |
|-----------|--------|--------|
| Lesson Records | TraceCoder (ICSE 2025) | **Implemented** — `LessonRecord`, `extract_lesson()`, injected into fix prompts |
| Semgrep Autofix | — | **Implemented** — custom YAML rules for 7B anti-patterns |
| Cross-Execution Gate | ConVerTest (2024) | **Implemented** — N×1 test validation for best-of-N |
| Targeted Micro-Fix | — | **Implemented** — AST function extraction + surgical LLM fix |

---

## Phase 2: Next Steps

### Analysis of Remaining Failures

The 6 remaining probe failures fall into 3 categories:

1. **None/null handling** (3 probes): 7B generates `sorted(items, key=lambda x: x['date'])` without handling `None` dates. Also `if date is None or date < ref` instead of `if date is not None and date < ref`.
2. **Empty collection guard** (1 probe): `self.data[0]` without `if not self.data: return default`.
3. **Format/logic** (2 probes): wrong format strings, off-by-one on ID counters.

Categories 1-2 are **systematic 7B anti-patterns** addressable with deterministic AST fixes. Category 3 is stochastic.

### Phase 2A: Deterministic Runtime Error Repair (zero tokens)

**Source**: CodeHalu (AAAI 2025), Preguss (2025), TraceFixer (2025)

Map common Python runtime errors to deterministic AST transformations. When a test fails with a traceback pointing to a specific line, apply a pattern-matched fix:

| Error Type | Pattern | Deterministic Fix |
|-----------|---------|-------------------|
| `TypeError: '<' not supported ... NoneType` | Sort/comparison with None | Wrap sort key: `key=lambda x: (x is None, x)` |
| `TypeError: ... NoneType has no attribute` | Attribute access on None | Inject `if obj is not None:` guard |
| `IndexError: list index out of range` | `data[0]` or `data[idx]` | Inject `if not data: return default` |
| `KeyError: 'X'` | `dict['X']` | Replace with `dict.get('X', default)` |
| `ZeroDivisionError` | `a / b` | Inject `if b == 0: raise ValueError(...)` |

This runs **before** the targeted micro-fix (which uses LLM) — a free pre-pass. Estimated impact: **+2-3 probes** (the None-handling and IndexError cases).

Implementation: `_fix_runtime_errors_deterministic(error: TestError, workspace: Path) -> int` in watcher.py. Parse traceback → find line → AST-match pattern → apply fix.

**Estimated effort**: Medium. **Impact**: High (directly addresses 4/6 remaining failures).

### Phase 2B: Defensive Guard Injection (zero tokens)

**Source**: Preguss (2025), NL2Contract (2025)

After code generation, before tests run: AST-walk all generated functions, find potential runtime error sites, inject guards proactively. Not error-driven — preventive.

Scan for:
- **Sort calls** with lambda keys → wrap key to handle None: `(v is None, v)`
- **`data[0]`** or **`data[-1]`** without preceding empty check → inject `if not data: return ...`
- **`raise KeyError`** or **`raise ValueError`** in methods whose spec says "return False/None if not found" → replace with return
- **Division** without zero guard → inject guard

This catches bugs before they ever manifest as test failures. Saves both LLM and test-execution budget.

**Estimated effort**: Medium. **Impact**: High (prevents bugs rather than fixing them post-hoc).

### Phase 2C: Type-Aware Edge Case Test Injection (zero tokens)

**Source**: EvalPlus (NeurIPS 2023/2025), LLM failure analysis (2025)

After the LLM generates tests, inject additional edge case assertions based on function signatures:

```
def f(items: list) → inject: f([]), f([None]), f([single_item])
def f(x: str)      → inject: f(""), f(" "), f(None)
def f(x: int)      → inject: f(0), f(-1), f(None)
def f(x: dict)     → inject: f({}), f({"key": None})
```

Extract types from AST (function annotations or `isinstance` checks in the body). Generate minimal test cases that exercise each edge case. Append to the test file.

These tests catch bugs early — if the code crashes on `f([])`, the deterministic repair chain has a clear error to work with.

**Estimated effort**: Medium. **Impact**: Medium (catches edge cases that LLM tests miss, but may also create false failures on intentionally non-handling code).

### Phase 2D: Lightweight Mutation Testing Gate (zero tokens)

**Source**: Meta Mutation-Guided Testing (FSE 2025), LLMLOOP (ICSME 2025)

After code passes all tests (just before VERIFIED), inject 5-10 AST mutations:
- Negate an `if` condition
- Swap `<` ↔ `<=`, `+` ↔ `-`
- Remove a `return` statement
- Change `True` ↔ `False`

Re-run tests for each mutant. If tests fail to catch >50% of mutants, flag the test suite as **weak** — don't declare VERIFIED yet. This prevents the "fake code" problem more reliably than the current LLM-based `check_not_faked`.

**Estimated effort**: Medium-high. **Impact**: Medium (improves test quality detection, reduces false VERIFIED).

### Phase 2E: Differential Testing for Best-of-N (zero tokens)

**Source**: Semantic Triangulation (2025), Ensemble Selection (2025)

Enhance `implement_best_of_n()`. Currently picks candidate by test pass count. Instead:
1. Extract function signatures from all N candidates
2. Generate 50 random type-compatible inputs
3. Run all candidates on those inputs
4. Pick the candidate whose outputs agree with the **majority** on the most inputs

This is fully deterministic after generation. Catches candidates that pass their own tests but disagree with the consensus.

**Estimated effort**: Medium. **Impact**: Medium (only when best_of_n > 1).

### Phase 2F: Grammar-Constrained Decoding (zero tokens)

**Source**: ICML 2025

Use Ollama's GBNF grammar support to guarantee syntactically valid Python output. Eliminates a class of syntax errors that currently consume retry budget.

For structured output (JSON specs, task decomposition), use JSON schema mode (already supported by Ollama). For code generation, define a grammar that ensures balanced delimiters and valid indentation.

**Estimated effort**: Low. **Impact**: Low-medium (syntax errors are already caught by `auto_fix_deterministic`).

### Phase 2G: Execution Trace Divergence (cheap LLM)

**Source**: TraceFixer (2025), Divergence-Driven Debugging (ICPC 2025)

When micro-fix fails: instrument the failing test with `sys.settrace()` to capture variable values at each line. Identify where `actual != expected` first occurs. Include this divergence point in the fix prompt:

> "At line 42, `balance` is -10 but should be 0. The preceding assignment `balance = balance - amount` did not check for overdraft."

This gives the 7B model much more targeted fix guidance than the current "here's the function + error" approach.

**Estimated effort**: High. **Impact**: Medium (improves micro-fix success rate on complex bugs).

### Priority Matrix

| ID | Technique | Tokens | Effort | Impact | Dependencies |
|----|-----------|--------|--------|--------|-------------|
| **2A** | Runtime error repair | 0 | Medium | **High** | None |
| **2B** | Defensive guard injection | 0 | Medium | **High** | None |
| **2C** | Edge case test injection | 0 | Medium | Medium | None |
| **2D** | Mutation testing gate | 0 | Medium-high | Medium | None |
| **2E** | Differential testing | 0 | Medium | Medium | best_of_n > 1 |
| **2F** | Grammar-constrained decoding | 0 | Low | Low-medium | Ollama GBNF support |
| **2G** | Execution trace divergence | ~100 | High | Medium | micro-fix |

**Recommended implementation order**: 2A → 2B → 2C → 2D (all zero-token, progressively harder). 2A+2B together should recover 3-4 of the remaining 6 probes.

---

## What's Left to Explore (unchanged)

### Infrastructure

- **Ollama Environment Optimization**: Flash attention, KV cache tuning, batch scheduling for best-of-N. Pure infrastructure, no code changes.
- **Speculative Decoding**: Pair 7B target with 0.5B draft model for 2x throughput (when Ollama supports it).

### Training-Time

- **LoRA Adapters**: Fine-tune on PMCA fix trajectories (need 500+ examples, Unsloth feasible on 16GB VRAM). Log all (prompt, code, error, fix) tuples.
- **Self-Correction RL** (CoCoS, 2025): Online RL that trains small models to maintain correct outputs while correcting incorrect ones. 35.8% improvement on MBPP with 1B models.
- **IFD Data Selection**: Use Instruction Following Difficulty scoring to select most informative training examples.

### Evaluation

- **ReasonFlux-Coder-7B** (CURE, NeurIPS 2025): RL-trained code+test co-evolution model. Drop-in evaluation candidate for Qwen-coder-7B.

### Low Priority / Skip

- **tree-sitter**: ast module handles Python fine.
- **LongCodeZip**: Contexts small enough at 4K.
- **Full PDB**: --showlocals sufficient.
- **Full OpenTelemetry**: JSON telemetry sufficient.
- **Full Hypothesis**: Prompt guidance covers 80% of value.

---

## Key Metrics

| Metric | Raw 7B | Baseline (Feb) | Current (Mar) | Target |
|--------|--------|----------------|---------------|--------|
| Probes (12 tasks) | 62/76 (82%) | 71/76 (93%) | **70/76 (92%)** | 74/76 (97%) |
| PMCA tasks | — | 8/12 | **8/12** | 10/12 |
| Multi-file project | — | ~50% verified | ~50% verified | 80%+ |
| Average retries | — | ~1.2 | **~0.8** | ≤0.5 |
| Benchmark time | 30.6s | 280.4s | **324.7s** | ≤200s |

## References

- AlphaCodium: Flow Engineering for Code (arXiv:2401.08500)
- CodeChain: Modular Self-Revision (ICLR 2024, arXiv:2310.08992)
- Blueprint2Code: Multi-Agent Pipeline (Frontiers AI 2025)
- AgentCoder: Multi-Agent with Test Designer (arXiv:2312.13010)
- CODESIM: Simulation-Driven Debugging (NAACL 2025, arXiv:2502.05664)
- TiCoder: Test-Driven Interactive Code Gen (ICSE 2025, arXiv:2404.10100)
- SCoT: Structured Chain-of-Thought (ACM TOSEM 2024, arXiv:2305.06599)
- Debugging Decay Index (Nature Sci Reports 2025)
- Property-Based Testing for LLM Code (FSE 2025, arXiv:2506.18315)
- Qwen2.5-Coder Technical Report (arXiv:2409.12186)
- ClassEval: Class-Level Code Gen (ICSE 2024)
- CodeHalu: Code Hallucination Taxonomy (AAAI 2025, arXiv:2405.00253)
- AST Hallucination Detection & Correction (FORGE 2026, arXiv:2601.19106)
- LLMLOOP: Iterative Feedback Loops (ICSME 2025)
- NL2Contract: Spec-to-Contract Extraction (arXiv:2510.12702)
- Preguss: Runtime Error-Guided Specification (arXiv:2512.24594)
- EvalPlus: Type-Aware Mutation Testing (NeurIPS 2023, arXiv:2305.01210)
- CURE: Co-Evolving Coder and Tester via RL (NeurIPS 2025, arXiv:2506.03136)
- TestART: Template-Based Test Repair (arXiv:2408.03095)
- Semantic Triangulation for Code (arXiv:2511.12288)
- TraceFixer: Execution Trace-Driven Repair (arXiv:2304.12743)
- Divergence-Driven Debugging (ICPC 2025)
- CoCoS: Self-Correcting Small Models (arXiv:2505.23060)
- Grammar-Constrained Decoding (ICML 2025)
- Meta Mutation-Guided Test Generation (FSE 2025, arXiv:2501.12862)
- Static Analysis as Feedback Loop (arXiv:2508.14419)
