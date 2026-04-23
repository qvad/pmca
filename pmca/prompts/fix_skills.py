"""Error-driven fix skills — sourced from open-source skill repositories.

Sources:
  - obra/superpowers: systematic-debugging
  - wshobson/agents: python-testing-patterns
  - Custom: domain-specific regex/SQL/counting rules

Injected into coder fix prompts based on detected error patterns.
"""

# The core debugging discipline (from obra/superpowers systematic-debugging)
SYSTEMATIC_DEBUGGING = """
## Systematic Debugging (source: obra/superpowers)

BEFORE attempting ANY fix:
1. READ the error message completely — line numbers, types, values
2. TRACE the data flow backward — where does the bad value originate?
3. FORM a hypothesis: "I think X is the root cause because Y"
4. Make the SMALLEST possible change to test the hypothesis
5. ONE variable at a time — don't fix multiple things at once

Red flags — STOP if you catch yourself:
- "Just try changing X and see if it works"
- Adding multiple changes at once
- Not understanding WHY the error happens
"""

# Domain-specific fix rules activated by error patterns
FIX_SKILLS: dict[str, str] = {

    "regex": """
## Regex Fix Rules:
- When counting SQL subqueries, match `(\\s*SELECT` not just `(` — parentheses in VALUES(), function calls are NOT subqueries
- To count subquery depth: iterate chars tracking depth, but only increment when `(` is followed by SELECT keyword
- Always use word boundaries for SQL keywords: r'\\bJOIN\\b' not 'JOIN'
- Be careful with re.findall on overlapping patterns
""",

    "assignment": """
## SQL Assignment vs Comparison:
- In SET clauses (after SET, before WHERE): `=` is assignment — do NOT count as operator
- In WHERE/ON/HAVING clauses: `=` is comparison — DO count it
- Split query at SET...WHERE boundary to separate contexts
- Pattern: r'\\bSET\\b(.+?)\\bWHERE\\b' captures the assignment zone
""",

    "counting": """
## Counting Fix Rules:
- table_count: extract identifiers after FROM, JOIN, INTO, UPDATE. Use set() for dedup. Handle aliases (FROM users u → table is "users")
- join_count: count r'\\bJOIN\\b' — LEFT JOIN, INNER JOIN all contain JOIN
- Don't count SQL keywords as function calls. Exclude: SELECT, FROM, WHERE, JOIN, ON, SET, INTO, VALUES, ORDER, GROUP, HAVING, LIMIT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, AS, AND, OR, NOT, IN, EXISTS, BETWEEN, LIKE, IS, NULL, CASE, WHEN, THEN, ELSE, END, WITH, DISTINCT, UNION, INTERSECT, EXCEPT
""",

    "assertion": """
## Assertion Fix Rules (source: wshobson/agents python-testing-patterns):
- When assert X == Y fails: trace the code to determine if X (actual) or Y (expected) is wrong
- If the CODE produces the right answer, fix the TEST value
- If the TEST has the right expected value, fix the CODE logic
- Each test should verify ONE behavior — don't combine multiple assertions that test different things
- Use descriptive test names: test_<unit>_<scenario>_<expected>
""",

    "import": """
## Import Fix Rules:
- Python 3.9+: use builtin list, dict, set, tuple — NOT from typing
- Python 3.10+: use X | None — NOT Optional[X]
- If getting "cannot import name 'list' from 'typing'" → remove list/dict/set/tuple from typing imports, use builtins directly
- Only import from typing: Any, Callable, ClassVar, Final, Literal, Protocol, TypeVar, Generic
""",

    "dataclass": """
## Dataclass Fix Rules:
- frozen=True prevents mutation — only use for immutable value objects
- Mutable defaults: use field(default_factory=list) not default=[]
- set fields: field(default_factory=set) not default=set()
- dict fields: field(default_factory=dict) not default={}
""",
}


def get_fix_skills(error_output: str, code_content: str = "") -> str:
    """Select relevant fix skills based on error patterns.

    Always includes the systematic debugging discipline.
    Adds domain-specific skills based on keywords in error output and code.
    """
    skills: list[str] = [SYSTEMATIC_DEBUGGING]
    error_lower = error_output.lower()
    code_lower = code_content.lower()

    # Regex / pattern matching issues
    if any(kw in error_lower for kw in ("subquery", "depth", "paren", "regex", "re.findall", "re.search")):
        skills.append(FIX_SKILLS["regex"])
    if any(kw in code_lower for kw in ("re.findall", "re.search", "re.match", "re.sub")):
        skills.append(FIX_SKILLS["regex"])

    # Assignment vs comparison
    if any(kw in error_lower for kw in ("expression_complexity", "operator", "assignment")):
        skills.append(FIX_SKILLS["assignment"])
    if "SET " in code_content and "WHERE" in code_content:
        skills.append(FIX_SKILLS["assignment"])

    # Counting issues
    if any(kw in error_lower for kw in ("table_count", "join_count", "count")):
        skills.append(FIX_SKILLS["counting"])

    # Assertion mismatches
    if "assert" in error_lower:
        skills.append(FIX_SKILLS["assertion"])

    # Import issues
    if any(kw in error_lower for kw in ("import", "modulenotfounderror", "cannot import")):
        skills.append(FIX_SKILLS["import"])

    # Dataclass issues
    if any(kw in error_lower for kw in ("dataclass", "frozen", "default_factory", "unhashable")):
        skills.append(FIX_SKILLS["dataclass"])

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    return "\n## Fix Skills (domain-specific guidance):\n" + "\n".join(unique)
