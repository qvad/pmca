# Code Quality Status

Honest accounting of the codebase's structural debt.

## What's Clean

- **All ruff checks pass** (strict config: F, E, W, I, B, UP rule sets)
- **Type hints** on all public APIs (`pmca/eval/`, `pmca/tuning/`)
- **31 unit tests** covering the tuner subsystem with mocked evaluators
- **67 total tests pass**

Linting contract documented in `pyproject.toml`:
```toml
[tool.ruff.lint]
select = ["F", "E", "W", "I", "B", "UP"]
```

## What's Not Clean

Two god files contain most of the cascade logic and most of the complexity debt.

### `pmca/orchestrator.py` — 1,609 lines

| Method | Complexity | Rating |
|--------|-----------|--------|
| `review_phase` | **85** | **F** |
| `_code_leaf` | 29 | D |
| `code_phase` | 27 | D |
| `integrate_phase` | 26 | D |
| `_triage_failing_tests` | 25 | D |

Average complexity: **C (13.5)**.

### `pmca/agents/watcher.py` — 2,426 lines

| Method | Complexity | Rating |
|--------|-----------|--------|
| `spec_coverage_check` | 44 | F |
| `extract_structured_errors` | 39 | E |
| `_fix_typeerror_in_sort` | 35 | E |
| `auto_fix_deterministic` | 34 | E |
| `_guard_index_zero` | 33 | E |
| `_guard_sort_keys` | 31 | E |
| `_check_api_consistency` | 31 | E |
| `calibrate_tests` | 30 | D |

8 methods at D-F complexity. These are the repair chain — each method handles many AST patterns and edge cases.

## Why Not Refactored

The complexity is concentrated in methods with extensive conditional logic for AST pattern matching and error recovery. Refactoring `review_phase` (595 lines, complexity 85) would mean extracting ~8 helper methods while preserving the tangled state flow (task, attempt, code_content, structured_errors, review, lesson_records). Without integration tests covering the full cascade, a refactor risks silent behavior changes that would only surface during benchmark runs hours later.

**The pragmatic decision**: leave the complexity in place, document it, and refactor incrementally when specific methods need to change for a feature. Do not refactor for aesthetics.

## Refactor Priority (When Touched)

1. **`review_phase`** — split into `_run_smoke_test`, `_review_passing_code`, `_handle_test_failures`, `_retry_with_feedback`. Each becomes testable in isolation.
2. **`spec_coverage_check`** — the name extraction logic (regex → AST → filter) has 3 distinct responsibilities that can be separate functions.
3. **`auto_fix_deterministic`** — already calls separate `_fix_*` methods; main branch is a linear pipeline that reads like orchestration. Low refactor ROI.

## Linting

```bash
# Auto-fix safe issues
ruff check pmca/ scripts/ --fix

# Strict check (must pass before commit)
ruff check pmca/ scripts/

# Type check (informational)
mypy pmca/ --ignore-missing-imports
```
