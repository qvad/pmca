"""Architect skills — sourced from open-source skill repositories.

Sources:
  - wshobson/agents: python-design-patterns, python-project-structure
  - obra/superpowers: writing-plans
"""

ARCHITECT_SKILLS = """
## ARCHITECTURAL MANDATES:
1. INTERFACE-FIRST: Define the exact public API (methods, parameters, return types) before describing internal logic.
2. ATOMIC DECOMPOSITION: Break complex tasks into subtasks that are "implementation-ready" (typically < 200 lines of code).
3. DEPENDENCY AWARENESS: Explicitly state which subtasks depend on others. Use the `DEPENDS_ON: task_id` syntax.
4. ERROR CONTRACTS: Define exactly what exceptions/errors each method must raise for specific failure cases.
5. NO HALLUCINATED LIBS: Only specify standard library components or libraries explicitly requested by the user.
6. DATA FLOW: Describe how data moves through the system. Avoid "magic" state; prefer explicit passing of arguments.

## Design Principles (source: wshobson/agents python-design-patterns):
- KISS: choose the simplest pattern that works. No metaclasses, no descriptors unless explicitly needed.
- Single Responsibility: each module/class does ONE thing.
- Composition over Inheritance: prefer has-a over is-a.
- Rule of Three: don't abstract until the pattern appears three times.
- Use @dataclass for data containers, regular classes for behavior.
- Use dict only for truly dynamic/unknown-key data.

## Structure (source: wshobson/agents python-project-structure):
- One concept per file. Flat directory structure.
- Test files mirror source: src/models.py → tests/test_models.py.
"""
