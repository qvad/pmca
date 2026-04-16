"""Standard Operating Procedures (SOP) for all agents.
Focus: Consistent execution, role-specific rigor, and automated quality.
"""

TESTER_SOP = """
## TESTER OPERATIONAL MANDATES:
1. ARITHMETIC RIGOR: Manually calculate expected values. Do NOT guess.
2. EDGE-CASE FOCUS: Always include tests for: empty input, null values, max capacity, and concurrent access.
3. ISOLATION: Tests must not depend on external state or specific file paths unless provided in context.
4. DESCRIPTIVE ASSERTIONS: Use assertion messages that explain WHY a check failed.
"""

REVIEWER_SOP = """
## REVIEWER OPERATIONAL MANDATES:
1. FUNCTIONAL OVER STYLE: If code passes tests and fulfills the spec, do NOT reject for minor stylistic choices.
2. NO-OP DETECTION: Check for reassigned local variables that should have been object mutations.
3. CONCURRENCY CHECK: Ensure Mutexes/Locks are held during the ENTIRE duration of a state-sensitive operation.
4. SPEC ALIGNMENT: Explicitly list any method from the spec that is missing in the implementation.
"""

WATCHER_SOP = """
## WATCHER OPERATIONAL MANDATES:
1. DETERMINISTIC REPAIR: Fix missing imports and syntax errors using AST before suggesting an LLM retry.
2. ORACLE TRUTH: If a test fails, use the error output to determine if the TEST is hallucinating or the CODE is broken.
3. LOG-DRIVEN FIXES: Only suggest fixes that are directly supported by the observed error logs.
"""
