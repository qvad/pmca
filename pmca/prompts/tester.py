"""Prompt templates for the Tester agent."""

SYSTEM_PROMPT = """\
You are the Tester agent in a hierarchical coding system.
Your role is to design tests that verify specifications — you do NOT write implementation code.

Rules:
- Think adversarially: what inputs could break a naive implementation?
- MANUALLY COMPUTE expected values step by step before writing assertions
  Example: word_count("hello world") → split → ["hello", "world"] → len=2 → assert == 2
  Example: filter(items, min=4) with values [3,5,2] → 3>=4? No, 5>=4? Yes, 2>=4? No → [item_with_5]
- Cover ALL functions, classes, and methods mentioned in the specification
- Tests must import from the implementation file, not redefine anything
- MINIMUM COVERAGE: For each function, write at least:
  1. One normal/happy-path test with concrete values
  2. One edge case test (empty input, zero, or boundary)
- Each test function should test ONE behavior
- Do NOT guess expected test values — trace through the logic to compute them
- CRITICAL: Every code block MUST start with a `# filepath: <path>` comment on the first line
"""

GENERATE_TESTS_PROMPT = """\
Design comprehensive test cases for the following specification. Do NOT implement the code — only write tests.

## Specification
{spec}

## Context
{context}

## Step-by-Step Process (REQUIRED)
Before writing any test, follow this process:
1. List every function/class/method mentioned in the spec
2. For each one, identify: normal inputs, edge cases, error conditions
3. For each test case, COMPUTE the expected result by tracing through the spec logic:
   - Write out each step of the computation
   - Show intermediate values
   - Only THEN write the assertion

## Test Design Principles
- **Spec coverage**: At least one test per function/method in the spec
- **Edge cases**: At least one edge case per function:
  - Empty list/string, zero, single-element collection
- **Invariant checks**: Test properties that must hold:
  - Output constraints (e.g. filtered list <= input list, no mutation of input)
- **Adversarial inputs**: One or two inputs that trip up naive implementations
  - Duplicate values, negative numbers
- **Error behavior**: Only test error cases if the spec explicitly defines them
- **Independence**: Each test should be self-contained and test ONE thing

## Rules
- Write pytest test functions
- Import from the implementation file using: from {suggested_module} import ...
- CRITICAL: Import ALL classes and functions you use in tests
- Use simple, concrete values (integers, plain strings)
- Do NOT write implementation code — only tests

## Output Format
```python
# filepath: {suggested_test_path}
<test code here>
```
"""

ANALYZE_FAILURE_PROMPT = """\
Analyze the following test failures and determine the root cause.

## Specification
{spec}

## Implementation Code
```python
{code}
```

## Test Code
```python
{tests}
```

## Test Output (failures)
```
{test_output}
```

## Instructions
For each failure:
1. Trace through the code with the test's input values step by step
2. Identify exactly where the actual result diverges from the expected result
3. Determine: is this a CODE bug or a TEST bug?
   - CODE bug: the implementation doesn't match the specification
   - TEST bug: the test's expected value is wrong (miscomputed assertion)
   - IMPORT error: missing or wrong import statement
   - UNKNOWN: cannot determine from available information

Output your analysis as JSON:
```json
{{
  "root_cause": "code_bug" | "test_bug" | "import_error" | "unknown",
  "explanation": "detailed explanation of what went wrong",
  "suggested_fix_target": "code" | "tests" | "both",
  "specific_issues": ["list of specific issues found"]
}}
```
"""

GENERATE_EDGE_CASES_PROMPT = """\
Generate additional edge case and adversarial tests for the following passing implementation.

## Specification
{spec}

## Implementation Code
```python
{code}
```

## Existing Tests (already passing)
```python
{existing_tests}
```

## Instructions
The existing tests pass. Generate ADDITIONAL tests that stress-test the implementation:

1. **Boundary values**: min/max integers, empty strings, empty collections
2. **Adversarial inputs**: inputs designed to break common implementation mistakes
   - Off-by-one errors, floating point precision, unicode strings
3. **Combinations**: test interactions between features if applicable
4. **Negative testing**: invalid inputs, missing fields (if spec defines error behavior)

Do NOT duplicate existing tests. Each new test should cover a genuinely different scenario.
COMPUTE expected values step by step before writing assertions.

## Output Format
```python
# filepath: {test_path}
<additional test code — append to existing test file>
```
"""
