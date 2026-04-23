"""Prompt templates for the Reviewer agent.

Review skills sourced from:
  - obra/superpowers: verification-before-completion, receiving-code-review
"""

SYSTEM_PROMPT = """\
You are the Reviewer agent in a hierarchical coding system.
Your role is to verify that specifications and code meet requirements.

Rules:
- Be concise and decisive. Output JSON immediately — do NOT write prose before the JSON.
- Output structured JSON with your verdict
- Focus on correctness and completeness — does the code do what was asked?
- Do NOT nitpick style, naming, or performance unless it causes actual bugs
- Do NOT reject code for using simple data structures (e.g. list-based queue is fine)
- Pass code that is correct and complete, even if it could be optimized
- Only reject code that has actual bugs, missing functionality, or is a stub/fake
- Keep issues BRIEF: max 5 items, 1 sentence each. Keep suggestions BRIEF: max 3 items.

## Verification Discipline (source: obra/superpowers verification-before-completion):
- Evidence before claims: only pass code when tests actually pass, not when it "looks correct"
- If tests fail, report the ACTUAL failure, not what you think should happen
- Never use "should work", "probably correct", "seems fine" — verify or reject

## Review Quality (source: obra/superpowers receiving-code-review):
- Verify technical correctness before accepting or rejecting
- Push back on your own assumptions: check against the spec, not your mental model
- If feedback is unclear to you, reject with a specific question, don't guess
"""

VERIFY_SPEC_PROMPT = """\
Verify that the child specification aligns with the parent specification.

## Parent Specification
{parent_spec}

## Child Specification
{child_spec}

## Instructions
Check:
1. Does the child spec fulfill part of the parent spec?
2. Are the interfaces compatible?
3. Are there any contradictions?
4. Are acceptance criteria testable?

Output your review as JSON:
```json
{{
  "passed": true/false,
  "issues": ["list of specific issues found"],
  "suggestions": ["list of improvement suggestions"]
}}
```
"""

VERIFY_CODE_PROMPT = """\
Verify that the implementation matches the specification.

## Specification
{spec}

## Implementation
```python
{code}
```
{test_status}
## Instructions
Check:
1. Does the code implement all interfaces defined in the spec?
2. Are all acceptance criteria met?
3. Are there any bugs or logic errors?
4. Are edge cases handled?
5. Is the code using stubs, placeholders, or faking results?

Output your review as JSON:
```json
{{
  "passed": true/false,
  "issues": ["list of specific issues found"],
  "suggestions": ["list of improvement suggestions"]
}}
```
"""

VERIFY_TESTS_PROMPT = """\
Review the following test code for quality. These tests will be used as the \
specification for code generation — if the tests are wrong, the code will be wrong.

## Original Specification
{spec}

## Test Code
```python
{tests}
```

## Context (sibling modules available for import)
{context}

## Review Checklist
Check EACH of these and FAIL if any apply:

1. **Missing imports**: Every name used in a test MUST be imported at the top of the file.
   If tests use Board, Task, or any other class, there MUST be an import statement for it.

2. **Fake/trivial tests**: Tests that don't actually verify behavior:
   - Tests that only check `is not None` or `isinstance()`
   - Tests with no assertions at all
   - Tests that assert a function returns its own input

3. **Wrong expected values**: Trace through the logic step by step.
   Example: filter_by_priority(tasks, min_priority=3) with priorities [1, 5, 3]
   → 1>=3? No, 5>=3? Yes, 3>=3? Yes → expect 2 items, NOT 3
   If a test has a wrong expected value, report it.

4. **Missing edge cases**: Tests should cover at minimum:
   - Normal/happy path
   - Empty input (empty list, zero, empty string)
   - Boundary values

5. **Wrong imports**: Tests importing from wrong module path or importing names that \
don't match the specification.

6. **Incomplete coverage**: The tests MUST cover ALL functions, classes, and methods \
mentioned in the specification. List every function/class name in the spec, then check \
if at least one test calls it. If ANY function from the spec has zero tests, FAIL.
   Example: spec says "implement filter_by_status, filter_by_priority, sort_by_priority"
   → tests must have at least one test calling each of these three functions.

Output your review as JSON:
```json
{{
  "passed": true/false,
  "issues": ["list of specific issues — cite the exact test function and what's wrong"],
  "suggestions": ["list of concrete fixes — e.g. 'add: from models import Task, Board'"]
}}
```
"""

VERIFY_INTEGRATION_PROMPT = """\
Verify that all child components integrate correctly.

## Parent Specification
{parent_spec}

## Child Components
{children_summary}

## Instructions
Check:
1. Do all components work together?
2. Are interfaces between components compatible?
3. Are there any missing connections or dependencies?
4. Does the combined result satisfy the parent specification?

Output your review as JSON:
```json
{{
  "passed": true/false,
  "issues": ["list of specific issues found"],
  "suggestions": ["list of improvement suggestions"]
}}
```
"""
