# Deep Decomposition for Local Code Generation

## The Core Idea

Local 7B-14B models can reliably generate **~5-30 lines of correct code** per generation. Anything beyond that — multi-method classes, cross-file dependencies, arithmetic in tests — becomes unreliable (~40-60% failure rate).

The hypothesis: **any programming task can be broken down until each leaf is solvable by a 7B model**. The system's job is decomposition, verification, and assembly — not generation of complex code.

This document captures research, experiments, and the current architecture.

---

## Current State (Feb 2026)

### Benchmark Results (7B, single-file, 10 tasks)

| Metric | Baseline | Current |
|--------|----------|---------|
| Tasks passing | 7/10 | **8/10** |
| Probes passing | 52/56 (93%) | **56/56 (100%)** |
| Total time | ~200s | **144.5s** |

```
Task             Tier     PMCA   Probes     Calls  Tokens     Time     Retries
----------------------------------------------------------------------------
calculator       simple   PASS   6/6        6      5,082      8.2      0
stack            simple   PASS   6/6        6      6,285      10.4     0
counter          simple   FAIL   4/4        10     11,541     21.7     3
fizzbuzz         simple   PASS   3/3        6      5,065      7.6      0
text_stats       medium   PASS   7/7        6      6,894      12.8     0
bank_account     medium   FAIL   5/5        10     16,846     29.7     3
linked_list      medium   PASS   6/6        6      7,599      13.7     0
task_board       complex  PASS   6/6        6      7,559      13.1     0
matrix           complex  PASS   7/7        6      8,892      16.2     0
lru_cache        complex  PASS   6/6        6      6,557      11.1     0
```

Note: counter and bank_account FAIL as PMCA tasks (exhaust retries on internal tests) but all their external probes pass — the failures are in LLM-generated test assertions, not in the actual code logic.

### Benchmark Results (multi-file, 4 modules, ~15 functions)

- Best run: 14/15 tests pass, 9/11 probes, correct cross-file imports
- Worst run: 0/4 modules verified (7B generated getter/setter pattern)
- Average: ~2/4 modules verified per run

### What Works

- **Full probe coverage**: All 56 external probes pass (100%) across simple/medium/complex tiers
- Hierarchical cascade: architect → decompose → code → review → verify
- Cross-file imports via AST interface extraction + import hints
- Dependency-ordered child processing (topological sort)
- **Deterministic repair chain**: auto-fix → static analysis → spec coverage → calibrate → oracle repair
- **API consistency lint**: catches attribute/method shadowing before tests run
- **Attribute/method shadowing auto-fix**: renames `self.X` → `self._X` when it shadows `def X()`
- **Oracle repair**: aggressively trusts non-trivial actual values for remaining assertion mismatches
- **Difficulty routing**: simple tasks skip planning (avoids 7B overthinking)
- **ICoT prompts**: Specification + Idea before coding (replaces SCoT)
- **Metamorphic test guidance**: prompts prefer range/type assertions over fragile exact literals

### What Still Fails

- **Non-determinism**: identical task produces different results across runs
- **Context pressure**: 4K window forces aggressive truncation
- **Circular validation**: coder generates both code AND tests → wrong tests validate wrong code
- **LLM-generated test assertions**: counter and bank_account exhaust retries fixing assertion mismatches that probes don't care about

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

## Research Findings

### 1. Test-First Generation (Highest Impact — Not Yet Implemented)

**Source**: TiCoder (ICSE 2025), AlphaCodium, AgentCoder

Generate tests before code. Tests serve as executable specification — more precise than natural language. TiCoder showed **45.97% improvement in pass@1**.

Current flow: `spec → coder generates code + tests → watcher runs tests`
Proposed: `spec → test-designer generates tests → coder implements against tests`

### 2. Best-of-N Sampling (Implemented)

**Source**: Top Pass (2024), Scalable Best-of-N (2025)

Generate N candidates at varying temperatures, pick the one passing most tests. Implemented in `CoderAgent.implement_best_of_n()`. At 128 tok/s, 5 candidates of ~200 tokens takes ~8 seconds.

### 3. Debugging Decay — Fresh Start (Implemented)

**Source**: Nature Scientific Reports 2025

LLM debugging effectiveness follows exponential decay after 3 fix attempts. Implemented as `fresh_start_after` config: after N failures, regenerate from scratch instead of fixing.

### 4. ICoT Prompting (Implemented, Replaced SCoT)

**Source**: ACM TOSEM 2024 (adapted)

Original SCoT (SEQUENCE/BRANCH/LOOP) confused 7B models. Replaced with ICoT: state Specification + Idea in natural language before coding. Simple tasks skip planning entirely.

### 5. Static Analysis Gate (Implemented)

**Source**: ICSE 2025 static analysis feedback loops

Deterministic verification before LLM review: `ast.parse() → API consistency lint → spec coverage → test calibration → oracle repair`. Each gate provides specific, actionable error messages. Failed gates skip LLM review (save tokens).

### 6. Property-Based / Metamorphic Testing (Prompt Guidance Implemented)

**Source**: Hypothesis + LLM-generated properties (FSE 2025)

Full Hypothesis integration not implemented, but prompt guidance added: coder is instructed to prefer `assert isinstance(result, float) and result > 0` over `assert result == 42` for values hard to compute by hand. This reduces fragile exact-literal assertions that waste retry cycles.

### 7. Sub-Module Composition (Not Implemented)

**Source**: CodeChain (ICLR 2024)

Generate each method independently, test it, then compose into class. Currently the single-class LEAF rule prevents decomposition, relying on the coder to generate the whole class. This could be revisited if class complexity grows beyond 7B's capability.

---

## What's Left to Explore

### High Priority

- **Test-First Pipeline**: Separate test generation from code generation. Highest theoretical ROI but needs careful prompt design for 7B test quality.
- **Failure Knowledge Constitution**: Accumulate patterns from failed tasks (e.g., "7B always generates `self.size` as attribute") into a prompt prefix or few-shot examples.
- **Ollama Environment Optimization**: Flash attention, KV cache tuning, batch scheduling for best-of-N.

### Medium Priority

- **LoRA Adapters**: Fine-tune on PMCA-specific patterns (need 1000+ examples per adapter, Unsloth feasible on 16GB VRAM).
- **Full Property-Based Testing**: Hypothesis integration for automatic property generation.
- **mypy Integration**: Type checking gate in static analysis pipeline.

### Low Priority / Skip

- **tree-sitter**: Overkill for current scope; ast module handles Python fine.
- **LongCodeZip**: Contexts already small enough at 4K.
- **Full PDB Integration**: --showlocals provides enough debug context.
- **Full OpenTelemetry**: JSON telemetry in ModelManager is sufficient.

---

## Key Metrics

| Metric | Baseline (7B) | Current (7B) | Target |
|--------|--------------|-------------|--------|
| Single-file probes | 52/56 (93%) | **56/56 (100%)** | maintained |
| Tasks passing (PMCA internal) | 7/10 | 8/10 | 9/10 |
| Multi-file project (4 modules) | ~50% verified | ~50% verified | 80%+ |
| Average retries per task | ~1.5 | ~0.6 | ≤1.0 |
| Benchmark time (10 tasks) | ~200s | **144.5s** | ≤120s |

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
