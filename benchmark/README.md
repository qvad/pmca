# PMCA-Bench: A Vague-Request Class-Level Code Generation Benchmark

## What This Benchmark Tests

Most code generation benchmarks give the model a **precise specification** — a function signature, a docstring with examples, or a class skeleton with method stubs. The model's job is to fill in the blanks.

**PMCA-Bench tests something different.** The model receives a **vague natural language request** — the kind a developer would type into a chat window — and must produce a complete, working Python class with correct behavior. No skeleton. No method signatures. No type hints given. The model (or system) must infer the design, implement it, and get the edge cases right.

This is the gap in the benchmark landscape. HumanEval/MBPP test function completion. ClassEval tests skeleton filling. SWE-bench tests repo patching. **Nothing tests vague-request-to-working-class generation** — until now.

## Dataset

**37 tasks, 228 validation probes, 3 difficulty tiers.**

| Tier | Tasks | Probes | Characteristics |
|------|-------|--------|-----------------|
| Simple | 9 | 47 | Single class, 2-4 methods, straightforward logic |
| Medium | 11 | 64 | Multiple methods, state management, edge cases |
| Complex | 17 | 117 | 6-15 methods, cross-method dependencies, date handling, sorting, None edge cases |

### What Makes It Hard

- **Vague requests**: "Create a TodoManager class with add, delete, filtering by status, sorting by priority, and overdue detection" — no exact method signatures given
- **Implicit conventions**: history format, sort direction, None handling, exception types — all implied, not specified
- **Cross-method state**: methods depend on each other (add → complete → stats must be consistent)
- **Edge cases**: empty input, zero values, None dates in sort keys, invalid priorities

### What Makes It Fair

- **All probes written before any model was tested** — no benchmark hacking
- **Probes never shown to the model** — they test the external API, not internal structure
- **Deterministic grading** — each probe is a Python snippet that exits 0 (pass) or non-zero (fail)
- **No implementation bias** — probes don't check variable names, data structures, or algorithms, only behavior

## File Format

### pmca_bench.json

```json
[
  {
    "task_id": "calculator",
    "tier": "simple",
    "request": "Create a Calculator class with add, subtract, multiply, divide methods...",
    "probes": [
      {
        "name": "add",
        "code": "from calculator import Calculator; c = Calculator(); assert c.add(2, 3) == 5"
      },
      ...
    ]
  },
  ...
]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Unique task identifier (also the expected module name) |
| `tier` | string | `"simple"`, `"medium"`, or `"complex"` |
| `request` | string | The vague NL request — this is the ONLY input to the model/system |
| `probes` | array | Validation probes (never shown to the model) |
| `probes[].name` | string | Probe identifier within the task |
| `probes[].code` | string | Python code that must exit 0 to pass |

## How to Run

### Requirements

- Python 3.11+
- The code generation system must produce a Python module importable as `from {task_id} import {ClassName}`
- Module must be on `PYTHONPATH` or in the current directory

### Evaluation

For each task:

1. **Input**: Give the `request` string to your code generation system
2. **Output**: The system produces one or more `.py` files
3. **Validation**: Run each probe as a subprocess:

```python
import subprocess, sys

def run_probe(probe_code: str, workspace: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", probe_code],
        cwd=workspace,
        capture_output=True,
        timeout=10,
    )
    return result.returncode == 0
```

### Metrics

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Task pass@1** | tasks where ALL probes pass / total tasks | End-to-end success rate |
| **Probe pass rate** | total probes passed / total probes | Granular correctness |
| **Tier breakdown** | pass rates per tier | Difficulty scaling |

### Reporting

When reporting results, include:

```
Model: <name and size>
System: <raw model / with cascade / with repair chain / etc.>
Runs: <number of runs>
Task pass@1: X% (N/37) [mean ± std across runs]
Probe pass rate: X% (N/228)
  Simple:  X% (N/47)
  Medium:  X% (N/64)
  Complex: X% (N/117)
Time: Xs total (Xs/task avg)
```

## Standalone Runner

```bash
# Run with any system that produces Python files in a workspace directory
python benchmark/run.py --workspace ./output_dir

# Or evaluate an existing workspace
python benchmark/run.py --workspace ./output_dir --task calculator
```

## Task Catalog

### Simple Tier (9 tasks, 47 probes)

| Task | Methods | Key Challenge |
|------|---------|---------------|
| calculator | 4 | divide-by-zero exception |
| stack | 5 | LIFO ordering, empty-stack exceptions |
| word_count | 1 | Lowercase normalization, empty input |
| fizzbuzz | 1 | Modulo logic, string formatting |
| temperature_converter | 4 | Formula precision, absolute zero validation |
| string_utils | 4 | Palindrome with spaces, case-insensitive |
| queue_ds | 5 | FIFO ordering, empty-queue exceptions |
| shopping_cart | 5 | Float arithmetic, discount logic |
| trie | 4 | Tree structure, prefix search |

### Medium Tier (11 tasks, 64 probes)

| Task | Methods | Key Challenge |
|------|---------|---------------|
| text_stats | 4 | Sentence splitting ambiguity, tie-breaking |
| bank_account | 5 | Transfer between accounts, history format |
| linked_list | 6 | Node management, deletion edge cases |
| state_machine | 5 | Transition validation, history tracking |
| lru_cache_ttl | 5 | Time-based expiry, LRU eviction order |
| json_validator | 4 | Nested structure validation, type checking |
| event_emitter | 5 | Callback management, wildcard patterns |
| rate_limiter | 4 | Time-window tracking, sliding window |
| cipher | 4 | Encryption/decryption round-trip |
| csv_parser | 4 | Quoted fields, delimiter handling |
| expression_eval | 3 | Operator precedence, parentheses |

### Complex Tier (17 tasks, 117 probes)

| Task | Methods | Key Challenge |
|------|---------|---------------|
| task_board | 6 | Priority sorting, completion tracking |
| matrix | 7 | Multiplication dimensions, transpose |
| lru_cache | 6 | Eviction order after access, update behavior |
| data_pipeline | 10 | Method chaining, 7 operators, aggregation |
| todo_manager | 10 | Due dates, overdue detection, tag filtering, None sort |
| graph | 6 | BFS/DFS traversal, shortest path |
| database_table | 7 | WHERE/ORDER BY/JOIN operations |
| file_system | 6 | Path resolution, recursive operations |
| scheduler | 7 | Time slot conflicts, recurring events |
| inventory | 7 | Stock tracking, reorder alerts |
| markdown_parser | 5 | Nested formatting, code blocks |
| permission_system | 6 | Role hierarchy, permission inheritance |
| pub_sub | 6 | Topic routing, message history |
| cache_system | 7 | Multiple eviction policies, TTL |
| game_engine | 6 | Entity management, collision detection |
| calculator_advanced | 6 | Variables, functions, expression parsing |
| http_router | 6 | Path parameters, middleware chain |

## Comparison with Existing Benchmarks

| Benchmark | Tasks | Input Type | Output Level | External Probes |
|-----------|-------|-----------|-------------|-----------------|
| HumanEval | 164 | Docstring + signature | Function body | Yes |
| MBPP | 974 | NL description | Function | Yes |
| ClassEval | 100 | Class skeleton | Method bodies | Yes |
| SWE-bench | 300 | GitHub issue | Patch | Yes (repo tests) |
| **PMCA-Bench** | **37** | **Vague NL request** | **Full class + module** | **Yes (228 probes)** |

### Why PMCA-Bench Exists

Every existing benchmark provides the model with structural hints: a function signature (HumanEval), a class skeleton (ClassEval), or an existing codebase (SWE-bench). These test the model's ability to **fill in blanks**.

PMCA-Bench provides only a natural language description. The model must:
1. **Design** the class interface (method names, parameters, return types)
2. **Implement** every method with correct logic
3. **Handle edge cases** that are implied but not explicitly stated
4. **Produce importable code** with the right module/class names

This tests the full spec-to-code pipeline, not just code completion.

## Published Results

| System | Task pass@1 | Probe pass rate | Notes |
|--------|------------|-----------------|-------|
| Qwen 2.5 Coder 7B (raw) | — | 62/228 (82%) | Single-shot, no cascade |
| Qwen 2.5 Coder 7B + PMCA | — | 69.5/228 (91%) | Full cascade, 2-run avg |
| Qwen 2.5 Coder 14B + PMCA | — | 70/228 (92%) | Full cascade, 2-run avg |
| Qwen 3.5 9B + PMCA (optimized) | 37/37 (100%) | 228/228 (100%) | Hybrid Think + No-Reviewer |
| Llama 3.1 8B + PMCA | — | 24.5/228 (32%) | Not a code model |
| DeepSeek Coder V2 16B + PMCA | — | 66/228 (87%) | Fails on todo_manager |

## Citation

If you use this benchmark, please cite:

```
@misc{pmca-bench-2026,
  title={PMCA-Bench: A Vague-Request Class-Level Code Generation Benchmark},
  author={Sherstobitov, Dmitrii},
  year={2026},
  url={https://github.com/qvad/pmca}
}
```

## License

Same as the PMCA repository.
