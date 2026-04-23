"""Error-driven fix skills — injected into coder fix prompts based on failure type.

When the watcher detects specific error patterns, the relevant skill block
is appended to the fix prompt so the model has domain knowledge for the repair.
"""

# Maps error keywords → skill text to inject
FIX_SKILLS: dict[str, str] = {

    "regex": """
## Regex Debugging Rules:
- When counting SQL subqueries, match `(SELECT` not just `(` — parentheses in VALUES(), function calls, etc. are NOT subqueries
- To count subquery depth: iterate through characters tracking depth, but only increment when `(` is followed by SELECT keyword
- r'\\(\\s*SELECT' is the pattern for subquery detection, not bare parentheses
- When matching SQL keywords, always use word boundaries: r'\\bJOIN\\b' not just 'JOIN'
- Be careful with re.findall on overlapping patterns — use non-overlapping groups
""",

    "assignment": """
## SQL Assignment vs Comparison:
- In SET clauses, `=` is assignment, NOT a comparison operator — don't count it as expression complexity
- Pattern: after SET keyword and before WHERE, all `=` are assignments
- In WHERE/ON/HAVING clauses, `=` is comparison — DO count it
- Split the query into segments: SET...WHERE boundary separates assignment from comparison context
""",

    "counting": """
## Counting Rules for SQL Analysis:
- table_count: extract identifiers after FROM, JOIN, INTO, UPDATE keywords. Use set() for deduplication. Handle aliases (FROM users u → table is "users", not "u")
- join_count: count r'\\bJOIN\\b' occurrences (LEFT JOIN, INNER JOIN, RIGHT JOIN all contain JOIN)
- Don't count SQL keywords (SELECT, FROM, WHERE, etc.) as function calls when counting expression_complexity
- SQL keywords to exclude from function call counting: SELECT, FROM, WHERE, JOIN, ON, SET, INTO, VALUES, ORDER, GROUP, HAVING, LIMIT, INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, AS, AND, OR, NOT, IN, EXISTS, BETWEEN, LIKE, IS, NULL, CASE, WHEN, THEN, ELSE, END, DISTINCT, ALL, ANY, UNION, INTERSECT, EXCEPT, WITH
""",

    "assertion": """
## Fixing Assertion Mismatches:
- When assert X == Y fails, check if X (actual) is correct by tracing the code logic manually
- If the CODE produces the right answer, fix the TEST assertion value
- If the TEST has the right expected value, fix the CODE logic
- Common mistakes: off-by-one in counting, boundary conditions (< vs <=), forgetting to handle edge cases like empty input
""",

    "import": """
## Import Debugging:
- Python 3.9+: use builtin list, dict, set, tuple — NOT from typing
- Python 3.10+: use X | None — NOT Optional[X]
- If getting "cannot import name 'list' from 'typing'" → remove list/dict/set/tuple from typing imports
- Circular imports: use TYPE_CHECKING guard or import inside function
""",

    "dataclass": """
## Dataclass Rules:
- frozen=True makes instances hashable but prevents mutation — use frozen only for value objects
- set fields need field(default_factory=set) not default=set()
- dict fields need field(default_factory=dict) not default={}
- For mutable defaults always use field(default_factory=...)
""",
}


def get_fix_skills(error_output: str, code_content: str = "") -> str:
    """Select relevant fix skills based on error output and code content.

    Returns a combined skill block to append to the fix prompt.
    """
    skills: list[str] = []
    error_lower = error_output.lower()
    code_lower = code_content.lower()

    # Regex-related: parentheses counting, pattern matching issues
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
    seen = set()
    unique: list[str] = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    if not unique:
        return ""

    return "\n## Fix Skills (domain-specific guidance):\n" + "\n".join(unique)
