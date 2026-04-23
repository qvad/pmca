"""Modern Python skills — runtime-version-aware coding conventions.

Injected into all Python code generation prompts to prevent common
LLM mistakes caused by training on pre-3.9/3.10 code.
"""

PYTHON_MODERN_SKILLS = """
## Python Version Compatibility (CRITICAL — read before writing ANY code):

1. TYPE HINTS — use builtins, NOT typing module:
   WRONG: from typing import List, Dict, Tuple, Set, Optional
   RIGHT: use list, dict, tuple, set directly in annotations
   RIGHT: from typing import Any, Callable  (these are still valid)
   RIGHT: def foo(x: list[dict[str, int]]) -> list[str]:
   RIGHT: def bar(x: str | None = None) -> dict:
   WRONG: from typing import list  (lowercase list was NEVER in typing)

2. UNION TYPES — use | syntax, NOT Union/Optional:
   WRONG: from typing import Union, Optional
   WRONG: x: Optional[str] = None
   RIGHT: x: str | None = None
   RIGHT: x: int | float = 0

3. DATACLASSES — prefer over raw dicts for domain objects:
   from dataclasses import dataclass, field

4. WALRUS OPERATOR — use for assignment in conditions:
   if (n := len(items)) > 0:

5. MATCH STATEMENT — available for pattern matching:
   match command:
       case "quit": ...

6. F-STRINGS — always use f-strings, never .format() or %:
   RIGHT: f"Hello {name}"
   WRONG: "Hello {}".format(name)

7. PATHLIB — prefer over os.path:
   from pathlib import Path
   RIGHT: Path("dir") / "file.py"
   WRONG: os.path.join("dir", "file.py")

8. EXCEPTION GROUPS — use ExceptionGroup for multiple errors (3.11+)

9. NO __future__ imports needed — annotations are native

10. GUARD CLAUSE PATTERN:
    def process(data):
        if not data:
            return []
        # main logic here

## Implementation Discipline:
- NEVER leave placeholder comments like "# TODO: implement" or "# add logic here"
- ALWAYS fully implement every function — no stubs, no pass statements in production code
- Do EXACTLY what the spec asks — no more, no less. Don't rename, restructure, or "improve" beyond the request
- Don't add demo/example code at the bottom of modules — if needed, put it under `if __name__ == "__main__":`

## Test Quality:
- Tests must verify ACTUAL BEHAVIOR, not just that functions exist
- Each test should call the function with real inputs and assert real outputs
- Make tests deterministic — use fixed dates, not datetime.now() in assertions
- Test one behavior per test function
"""
