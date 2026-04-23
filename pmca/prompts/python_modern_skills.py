"""Python coding skills — sourced from open-source skill repositories.

Sources:
  - wshobson/agents: python-code-style, python-type-safety, python-error-handling
  - wshobson/agents: python-design-patterns, python-project-structure
  - obra/superpowers: systematic-debugging (fix context)

Injected into coder system prompts for all Python tasks.
"""

# Injected into every Python code generation call
PYTHON_MODERN_SKILLS = """
## Python Code Style (source: wshobson/agents)
- Use `ruff` conventions: snake_case for files/functions/variables, PascalCase for classes, SCREAMING_SNAKE_CASE for constants
- Import order: stdlib → third-party → local. Use absolute imports only.
- Line length: 120 characters max
- Clarity over brevity in naming

## Type Safety (source: wshobson/agents)
- Annotate ALL public function signatures with types
- Use modern syntax: `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- Use `X | None` not `Optional[X]`, `int | str` not `Union[int, str]`
- NEVER import list, dict, tuple, set, type from typing — they are builtins since Python 3.9
- Only import from typing: Any, Callable, ClassVar, Final, Literal, Protocol, TypeVar, Generic
- Use Protocols for structural typing instead of ABC when possible

## Error Handling (source: wshobson/agents)
- Fail fast: validate inputs at public API boundaries BEFORE expensive operations
- Use specific exceptions: ValueError for bad args, KeyError for missing keys, TypeError for wrong types
- Include descriptive messages: `raise ValueError(f"priority must be 1-5, got {priority}")`
- Preserve exception chains: `raise NewError(...) from original_error`
- For batch processing: track successes and failures separately, don't abort on first error

## Design Patterns (source: wshobson/agents)
- KISS: choose the simplest approach that works. No premature abstraction.
- Single Responsibility: each class/function does ONE thing
- Composition over inheritance: prefer has-a over is-a
- Rule of Three: don't abstract until you see the pattern three times
- Functions: max ~20 lines. If longer, extract helpers.
- Use @dataclass for value objects, regular classes for behavior

## Project Structure (source: wshobson/agents)
- One concept per file: one class or a group of related functions
- Flat directory structure: avoid deep nesting
- Test files mirror source structure: src/models.py → tests/test_models.py

## Implementation Discipline
- NEVER leave placeholder stubs: no `pass`, no `# TODO`, no `...` in production code
- ALWAYS fully implement every function
- Do EXACTLY what the spec asks — no scope creep, no "improvements" beyond the request
- Put demo/example code under `if __name__ == "__main__":` — never at module level
"""
