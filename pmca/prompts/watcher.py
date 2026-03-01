"""Prompt templates for the Watcher agent."""

SYSTEM_PROMPT = """\
You are the Watcher agent in a hierarchical coding system.
Your role is to execute tests and verify code integrity.

Rules:
- Run tests and report results accurately
- Check for fake/stub implementations that trivially pass tests
- Verify that code actually does what it claims
- Be suspicious of implementations that seem too simple
- Output structured JSON with your findings
"""

CHECK_NOT_FAKED_PROMPT = """\
Analyze the following code and tests to determine if the implementation is genuine.

## Implementation
```python
{code}
```

## Tests
```python
{tests}
```

## Instructions
Check for:
1. Hard-coded return values that match test expectations
2. Functions that simply return the input or a constant
3. Logic that only works for the specific test cases
4. Missing error handling that should exist per the spec
5. Empty function bodies or pass statements

Output your review as JSON:
```json
{{
  "passed": true/false,
  "issues": ["list of concerns about fake/stub code"],
  "suggestions": ["list of improvements needed"]
}}
```
"""

FINAL_VERIFICATION_PROMPT = """\
Perform a final end-to-end verification of the completed project.

## Original Request
{original_request}

## Project Structure
{project_structure}

## Key Files
{key_files}

## Instructions
Verify:
1. Does the project fulfill the original request?
2. Are all components integrated correctly?
3. Do the tests provide adequate coverage?
4. Are there any obvious issues or missing pieces?

Output your review as JSON:
```json
{{
  "passed": true/false,
  "issues": ["list of issues found"],
  "suggestions": ["list of improvements"]
}}
```
"""
