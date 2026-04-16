"""Architectural Design Skills for local models.
Focus: Precise decomposition, interface stability, and implementability.
"""

ARCHITECT_SKILLS = """
## ARCHITECTURAL MANDATES:
1. INTERFACE-FIRST: Define the exact public API (methods, parameters, return types) before describing internal logic.
2. ATOMIC DECOMPOSITION: Break complex tasks into subtasks that are "implementation-ready" (typically < 200 lines of code).
3. DEPENDENCY AWARENESS: Explicitly state which subtasks depend on others. Use the `DEPENDS_ON: task_id` syntax.
4. ERROR CONTRACTS: Define exactly what exceptions/errors each method must raise for specific failure cases.
5. NO HALLUCINATED LIBS: Only specify standard library components or libraries explicitly requested by the user.
6. DATA FLOW: Describe how data moves through the system. Avoid "magic" state; prefer explicit passing of arguments.
"""
