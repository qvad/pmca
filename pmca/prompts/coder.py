"""Prompt templates for the Coder agent."""

SYSTEM_PROMPT = """\
You are the Coder agent in a hierarchical coding system.
Your role is to implement code based on design specifications.

Rules:
- Write clean, well-structured code
- Follow the specification exactly — do not add unrequested features
- CRITICAL: Use EXACT names from the specification for functions, classes, methods,
  parameters, dict keys, and return value keys. Do NOT rename them to be "more descriptive".
  Example: if spec says return dict with keys 'total' and 'done', use exactly those keys,
  NOT 'total_tasks' or 'done_count'
- CRITICAL: When the spec lists specific string values (operator names, function names,
  mode names), use those EXACT strings in your code — copy them character-for-character.
  Do NOT substitute synonyms, symbols, or longer/shorter versions.
  Example: if spec says op is one of 'eq', 'gt', 'lt' — use 'eq', NOT '==' or 'equal';
  use 'gt', NOT '>' or 'greater_than'.
  Example: if spec says func_name is one of 'upper', 'lower' — use 'upper', NOT 'uppercase'
- Use type hints for all Python function signatures
- CRITICAL: Do NOT use `from typing import Number` — it does not exist. Use `float | int` or `int` instead.
- CRITICAL: Every code block MUST start with a `# filepath: <path>` comment on the first line
- CRITICAL: File/module name must match the primary class name from the USER REQUEST (lowercase). E.g., if user asks for a `Graph` class, name the file `src/graph.py` — NOT `graph_undirected.py` or `graph_impl.py`. Class name must also match exactly.
- Also generate corresponding test files with `# filepath: tests/test_<name>.py`
- Do NOT fake implementations — every function must contain real logic
- Handle edge cases only if the spec explicitly requires them
- Do NOT add runtime isinstance() type checks unless explicitly requested — rely on type hints instead
- Tests should use simple values (integers, plain strings) unless the spec requires otherwise
- CRITICAL: Before writing test assertions, MANUALLY COMPUTE the expected result step by step
  Example: word_count("hello world") → split → ["hello", "world"] → len=2 → assert == 2
  Example: filter(items, min=4) with values [3,5,2] → 3>=4? No, 5>=4? Yes, 2>=4? No → [item_with_5]
- Do NOT guess expected test values — trace through your code logic to compute them
- CRITICAL: Test removal/completion consistency — if your test calls a remove/delete/complete
  method with a specific ID, trace WHICH EXACT item is removed. After removal, assert that the
  REMAINING items exist, NOT the removed one.
  Example: add("A",id=1), add("B",id=2), complete(id=1) → A is removed → assert remaining is B, NOT A
- If you cannot implement something, say so honestly
"""

IMPLEMENT_PROMPT = """\
Implement the following specification.

## Specification
{spec}

## Context
{context}

## Intention (REQUIRED — do this BEFORE writing code)
1. Specification: State the inputs, outputs, and constraints in one sentence each
2. Idea: State the core algorithm in 1-2 sentences (data structures, strategy)

Then implement.

## Output Format
Output EVERY file as a fenced code block. The FIRST line inside each block MUST be a filepath comment:

```python
# filepath: src/calculator.py
def add(a, b):
    return a + b
```

```python
# filepath: tests/test_calculator.py
from calculator import add
def test_add():
    assert add(1, 2) == 3
```

For non-Python files use the appropriate language tag:
```html
# filepath: index.html
<!DOCTYPE html>
...
```

## Suggested file paths
- Implementation: {suggested_path}
- Tests: {suggested_test_path}
You may choose better names if appropriate, but keep them short.

## Requirements
- Follow the specification's interface exactly
- Include all edge case handling from the acceptance criteria
- Tests should cover the main functionality and edge cases
- CRITICAL: Tests must import using the MODULE NAME only, NOT the directory path
  Example: `from calculator import add` (CORRECT) — NOT `from src.calculator import add` (WRONG)
- For computed values hard to verify by hand (floats, string formatting, counts),
  prefer range/type assertions: `assert isinstance(result, float) and result > 0`
  Use exact `==` only for trivially computable values (e.g., 1+2==3, len("ab")==2)
- Do NOT use placeholder/stub implementations
"""

IMPLEMENT_SIMPLE_PROMPT = """\
Implement the following specification.

## Specification
{spec}

## Context
{context}

## Output Format
Output EVERY file as a fenced code block. The FIRST line inside each block MUST be a filepath comment:

```python
# filepath: src/calculator.py
def add(a, b):
    return a + b
```

```python
# filepath: tests/test_calculator.py
from calculator import add
def test_add():
    assert add(1, 2) == 3
```

## Suggested file paths
- Implementation: {suggested_path}
- Tests: {suggested_test_path}
You may choose better names if appropriate, but keep them short.

## Requirements
- Follow the specification's interface exactly
- Include all edge case handling from the acceptance criteria
- Tests should cover the main functionality and edge cases
- CRITICAL: Tests must import using the MODULE NAME only, NOT the directory path
  Example: `from calculator import add` (CORRECT) — NOT `from src.calculator import add` (WRONG)
- For hard-to-compute expected test values, prefer range/type checks over exact literals
- Do NOT use placeholder/stub implementations
"""

GENERATE_TESTS_PROMPT = """\
Generate test cases for the following specification. Do NOT implement the code — only write tests.

## Specification
{spec}

## Context
{context}

## Rules
- Write pytest test functions that verify the specification's behavior
- Use simple, concrete values (integers, plain strings)
- MANUALLY COMPUTE expected values step by step before writing assertions
  Example: filter(items, min=4) with values [3,5,2]: 3>=4? No, 5>=4? Yes, 2>=4? No → [item_with_5]
- Cover: normal cases, edge cases (empty input, zero, boundary values)
- Import from the implementation file using: from {suggested_module} import ...
- CRITICAL: Import ALL classes and functions you use in tests — do NOT use bare names
  If a test needs Board or Task from another module, add the import at the top of the test file
  Check the Context section for available sibling modules and their exports
- When expected values require complex computation, use range/type checks instead:
  `assert isinstance(result, int) and result > 0` — reserve exact `==` for trivial values
- Each test function should test ONE behavior

## Output Format
```python
# filepath: {suggested_test_path}
<test code here>
```
"""

IMPLEMENT_WITH_TESTS_PROMPT = """\
Implement code that passes the following tests.

## Specification
{spec}

## Tests (your code MUST pass these)
```python
{tests}
```

## Context
{context}

## Intention (REQUIRED — do this BEFORE writing code)
1. Specification: State the inputs, outputs, and constraints in one sentence each
2. Idea: State the core algorithm in 1-2 sentences (data structures, strategy)

Then implement.

## Output Format
Output ONLY the implementation file (NOT the test file — tests are already provided):

```python
# filepath: {suggested_path}
<implementation code>
```

## Requirements
- Your code MUST pass all the provided tests
- Follow the specification's interface exactly
- Do NOT modify the tests — write code that satisfies them
"""

FIX_PROMPT = """\
Fix the following code based on test failures and reviewer feedback.

## Current Code
{code_blocks}
{lessons}
## Issues / Test Failures
{issues}

## Original Specification
{spec}

## Debugging Instructions
1. READ the "Actual result" vs "Expected by test" values in the issues above
2. For each failure, TRACE through the code with the test's EXACT input values step by step
3. Compare your trace result with the "Actual result" — if they match, the CODE is correct and the TEST is wrong
4. Compare your trace result with the "Expected by test" — if they match, the TEST is correct and the CODE is wrong
5. Fix the root cause. Make sure your fix handles ALL cases in the spec.

Common bugs to check:
- Missing edge case handling: empty input, zero values, single-element lists
- Off-by-one errors in counting or indexing
- Wrong sort order (ascending vs descending, alphabetical vs numeric)
- Missing import statements for classes defined in sibling modules
- WRONG TEST ASSERTIONS: If "Actual result" and "Expected by test" are close (within 20%), the test is likely wrong — recompute the expected value by tracing code step by step
- Items marked [lint] are type/style errors from mypy/ruff — fix these too

## Output Format
Output ALL corrected files as fenced code blocks, each with `# filepath: <path>` on the first line.
Address every issue listed above. Do NOT introduce new issues.
Keep the same file paths as the current code.
"""

FIX_TESTS_PROMPT = """\
The implementation code appears CORRECT based on the specification, but the TEST ASSERTIONS contain wrong expected values.

## Current Code
{code_blocks}
{lessons}
## Test Failures (values the code actually produces vs what the tests expect)
{issues}

## Original Specification
{spec}

## Instructions
The implementation logic is correct. The TEST ASSERTIONS have wrong expected values.
For each failing assertion:
1. Read the implementation code carefully
2. Trace through the code step by step with the test's input values
3. Compute the CORRECT expected result by hand — check each comparison individually:
   Example: filter(items, min=4) with values [3,5,2]:
     - 3 >= 4? NO (3 is less than 4, exclude)
     - 5 >= 4? YES (include)
     - 2 >= 4? NO (exclude)
     → result = [item_with_value_5] (only 1 item, NOT 2)
4. Update the test assertion to use the correct expected value

Do NOT modify the implementation code — only fix the test files.
Do NOT guess values — compute each one step by step.

## Output Format
Output ALL corrected files as fenced code blocks, each with `# filepath: <path>` on the first line.
Keep the same file paths as the current code.
Output both source and test files (even if source is unchanged).
"""

PROJECT_IMPORT_RULES = """\

## Multi-File Project Rules
This task is part of a MULTI-FILE Python project. Other modules already exist.

CRITICAL RULES:
- Sibling modules marked [DONE] in the Context section ALREADY EXIST — you MUST import from them
- Do NOT redefine classes or functions that are already defined in sibling modules
- Use the exact import statement shown in the "Import:" hint for each [DONE] sibling
- Only define NEW classes/functions that are specific to THIS module
- Your test file should also import from sibling modules as needed
"""

# ---------------------------------------------------------------------------
# Test triage — per-failure investigation cascade
# ---------------------------------------------------------------------------

TRIAGE_DIAGNOSE_PROMPT = """\
A test is failing. Your ONLY job is to determine: is the CODE wrong, or is the TEST wrong?

## Original Specification
{spec}

## Code Under Test
```python
{code_function}
```

## Failing Test
```python
{test_function}
```

## Error
{error}

## Instructions
1. Read the specification carefully — it is the source of truth
2. Trace through the CODE with the test's input values, step by step
3. Determine what the code ACTUALLY returns
4. Compare with what the test EXPECTS
5. Compare with what the SPECIFICATION says the answer should be

## Output (JSON only)
```json
{{
  "verdict": "code_wrong" or "test_wrong",
  "reasoning": "one sentence explaining why",
  "correct_value": "what the correct return value should be according to the spec",
  "wrong_value": "what the wrong side currently produces"
}}
```
"""

TRIAGE_FIX_CODE_PROMPT = """\
Fix this ONE function. The test is correct, the code is wrong.

## Specification
{spec}

## Current Function (WRONG)
```python
{code_function}
```

## What it should do
{diagnosis}

## Output
Output ONLY the corrected function as a fenced code block with `# filepath: {filepath}` on the first line.
Include the full function, not just the changed lines.
"""

TRIAGE_FIX_TEST_PROMPT = """\
Fix this ONE test. The code is correct, the test assertion is wrong.

## Specification
{spec}

## Code (CORRECT — do not change)
```python
{code_function}
```

## Current Test (WRONG assertion)
```python
{test_function}
```

## What the correct value should be
{diagnosis}

## Output
Output ONLY the corrected test function as a fenced code block with `# filepath: {filepath}` on the first line.
Include the full test function, not just the changed lines.
"""

DEDUP_PREFIX = """\
IMPORTANT: Your previous fix attempt produced IDENTICAL code to a prior attempt.
You MUST try a fundamentally different approach this time.
Consider: different algorithm, different data structure, or fixing the tests instead of the code.

"""

LESSONS_SECTION = """\

## Lessons from Previous Attempts
The following summarizes what went wrong in earlier fix attempts. Do NOT repeat these mistakes:
{lessons_text}
"""

FAILURE_MEMORY_SECTION = """\

## Similar Past Failures
{memory_text}
"""

# ---------------------------------------------------------------------------
# Two-phase spec-literal extraction
# ---------------------------------------------------------------------------

EXTRACT_LITERALS_PROMPT = """\
Read the specification below and extract ALL groups of string literal values \
that the code must accept as parameter values.

Look for patterns like:
- "op is one of 'eq', 'ne', 'gt'" → parameter "op", values ["eq", "ne", "gt"]
- "func_name is one of 'upper', 'lower'" → parameter "func_name", values ["upper", "lower"]
- "status is 'all', 'done', or 'pending'" → parameter "status", values ["all", "done", "pending"]

Only extract groups of enumerated string values. Do NOT extract:
- Class or function names (DataPipeline, filter, sort)
- Type names (str, int, list, dict)
- Format patterns (YYYY-MM-DD)
- Single standalone values that are not part of a group

## Specification
{spec}
"""

EXTRACT_LITERALS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string"},
                    "values": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["parameter", "values"],
            },
        },
    },
    "required": ["groups"],
}

SPEC_LITERALS_SECTION = """\

## String Literal Values (use EXACTLY these strings in your code)
{literals_text}\
"""

THINKING_PROMPT_PREFIX = """\
## Step-by-Step Reasoning (Mental Tracing)
Before implementing, I will:
1.  **Analyze the Interface:** List every method and its exact signature from the spec.
2.  **Logic Trace:** For each method, I will trace the execution path for both common and edge cases (empty input, single element, zero).
3.  **Test Verification:** For every test case I generate, I will manually compute the expected result step-by-step. I will NOT guess values.
4.  **Consistency Check:** Ensure all string literals and class/method names match the specification EXACTLY.

My reasoning starts here:
"""
