import re
from dataclasses import dataclass, field


@dataclass
class QueryProfile:
    statement_type: str
    clauses_used: set
    join_count: int
    subquery_depth: int
    table_count: int
    expression_complexity: int


@dataclass
class StatsReport:
    total_queries: int
    statement_distribution: dict
    clause_frequency: dict
    avg_join_count: float
    avg_subquery_depth: float
    max_subquery_depth: int
    complexity_histogram: dict


class QueryStats:
    SQL_KEYWORDS = {
        'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP',
        'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'WITH',
        'UNION', 'WINDOW', 'RETURNING', 'JOIN', 'LEFT', 'RIGHT', 'INNER',
        'OUTER', 'CROSS', 'NATURAL', 'ON', 'SET', 'INTO', 'VALUES', 'AS',
        'AND', 'OR', 'NOT', 'IN', 'BETWEEN', 'LIKE', 'IS', 'NULL', 'TRUE',
        'FALSE', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'ALL',
        'ANY', 'EXISTS', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END', 'ASC', 'DESC',
        'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'CONSTRAINT', 'INDEX',
        'TABLE', 'VIEW', 'SCHEMA', 'DATABASE', 'COLUMN', 'DEFAULT', 'CHECK',
        'UNIQUE', 'CASCADE', 'RESTRICT', 'NO', 'ACTION', 'SET', 'NULL',
        'DEFERRABLE', 'INITIALLY', 'IMMEDIATE', 'DEFERRED', 'TRIGGER',
        'BEFORE', 'AFTER', 'INSTEAD', 'OF', 'EACH', 'ROW', 'STATEMENT',
        'EXECUTE', 'FUNCTION', 'PROCEDURE', 'LANGUAGE', 'PLPGSQL', 'SQL',
        'RETURNS', 'VOID', 'TABLE', 'RECURSIVE', 'MATERIALIZED', 'TEMPORARY',
        'TEMP', 'UNLOGGED', 'IF', 'EXISTS', 'NOT', 'EXISTS', 'ONLY',
        'ILIKE', 'SIMILAR', 'TO', 'ARRAY', 'ROW', 'RANGE', 'MULTIRANGE',
        'JSON', 'JSONB', 'XML', 'UUID', 'INET', 'CIDR', 'MACADDR', 'BIT',
        'VARBIT', 'BYTEA', 'TEXT', 'CHAR', 'VARCHAR', 'INT', 'INTEGER',
        'BIGINT', 'SMALLINT', 'SERIAL', 'BIGSERIAL', 'DECIMAL', 'NUMERIC',
        'REAL', 'DOUBLE', 'PRECISION', 'FLOAT', 'BOOLEAN', 'BOOL', 'DATE',
        'TIME', 'TIMESTAMP', 'INTERVAL', 'TIMETZ', 'TIMESTAMPTZ',
    }

    @staticmethod
    def parse(query: str) -> QueryProfile:
        upper_query = query.upper()

        # 1. statement_type: first keyword
        statement_type = QueryStats._extract_statement_type(upper_query)

        # 2. clauses_used
        clauses_used = QueryStats._extract_clauses(upper_query)

        # 3. join_count
        join_count = len(re.findall(r'\bJOIN\b', upper_query))

        # 4. subquery_depth: count nesting of (SELECT ...) patterns
        subquery_depth = QueryStats._compute_subquery_depth(query)

        # 5. table_count: distinct identifiers after FROM/JOIN/INTO/UPDATE
        table_count = QueryStats._count_tables(upper_query)

        # 6. expression_complexity: count comparison operators and function calls
        #    in WHERE/ON/HAVING clauses ONLY, NOT in SET clauses
        expression_complexity = QueryStats._compute_expression_complexity(query)

        return QueryProfile(
            statement_type=statement_type,
            clauses_used=clauses_used,
            join_count=join_count,
            subquery_depth=subquery_depth,
            table_count=table_count,
            expression_complexity=expression_complexity,
        )

    @staticmethod
    def _extract_statement_type(upper_query: str) -> str:
        # Handle WITH ... SELECT (CTE) — statement type is still SELECT
        match = re.match(r'\s*(WITH\b.*)?\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b', upper_query)
        if match:
            return match.group(2)
        return 'SELECT'  # fallback

    @staticmethod
    def _extract_clauses(upper_query: str) -> set:
        clauses: set[str] = set()

        if re.search(r'\bJOIN\b', upper_query):
            clauses.add('JOIN')
        if re.search(r'\bWHERE\b', upper_query):
            clauses.add('WHERE')
        if re.search(r'\bGROUP\s+BY\b', upper_query):
            clauses.add('GROUP BY')
        if re.search(r'\bHAVING\b', upper_query):
            clauses.add('HAVING')
        if re.search(r'\bORDER\s+BY\b', upper_query):
            clauses.add('ORDER BY')
        if re.search(r'\bLIMIT\b', upper_query):
            clauses.add('LIMIT')
        if re.search(r'\bWITH\b', upper_query):
            clauses.add('CTE')
        if re.search(r'\bUNION\b', upper_query):
            clauses.add('UNION')
        if re.search(r'\bWINDOW\b', upper_query):
            clauses.add('WINDOW')
        if re.search(r'\bRETURNING\b', upper_query):
            clauses.add('RETURNING')

        return clauses

    @staticmethod
    def _compute_subquery_depth(query: str) -> int:
        """Count max nesting depth of (SELECT ...) patterns.
        Use r'\(\s*SELECT' to detect subquery start."""
        # We scan for '(' followed by optional whitespace and SELECT keyword
        # Track nesting depth by counting how many (SELECT patterns are nested
        upper = query.upper()
        max_depth = 0
        current_depth = 0

        # Find all positions of (SELECT pattern
        pattern = re.compile(r'\(\s*SELECT\b')
        for match in pattern.finditer(upper):
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth

        # But we need to track nesting properly. The above counts total (SELECT
        # occurrences but doesn't account for closing parens.
        # Better approach: track open parens that precede SELECT, and decrement
        # when we see closing parens that end a subquery.
        #
        # Simplified: count the maximum nesting by tracking (SELECT opens and ) closes.
        # We need to match each (SELECT with its corresponding ).
        #
        # Robust approach: use a stack-based parser for parentheses, but only
        # increment depth when an open paren is immediately followed by SELECT.

        depth = 0
        max_depth = 0
        i = 0
        while i < len(upper):
            if upper[i] == '(':
                # Check if this paren is followed by SELECT (with optional whitespace)
                rest = upper[i + 1:]
                if re.match(r'\s*SELECT\b', rest):
                    depth += 1
                    if depth > max_depth:
                        max_depth = depth
            elif upper[i] == ')':
                if depth > 0:
                    depth -= 1
            i += 1

        return max_depth

    @staticmethod
    def _count_tables(upper_query: str) -> int:
        """Extract distinct table references after FROM/JOIN/INTO/UPDATE keywords."""
        tables: set[str] = set()

        # Pattern: after FROM, JOIN, INTO, UPDATE, capture the identifier
        # Handle: FROM table1, FROM table1 alias, JOIN table2 alias, etc.
        # Also handle: UPDATE table SET ...
        # Also handle: INSERT INTO table (...)

        # FROM clause: FROM table_name [alias]
        for match in re.finditer(r'\bFROM\s+(\w+)', upper_query):
            tables.add(match.group(1))

        # JOIN clause: JOIN table_name [alias]
        for match in re.finditer(r'\bJOIN\s+(\w+)', upper_query):
            tables.add(match.group(1))

        # INTO clause: INSERT INTO table_name
        for match in re.finditer(r'\bINTO\s+(\w+)', upper_query):
            tables.add(match.group(1))

        # UPDATE clause: UPDATE table_name SET ...
        for match in re.finditer(r'\bUPDATE\s+(\w+)', upper_query):
            tables.add(match.group(1))

        return len(tables)

    @staticmethod
    def _compute_expression_complexity(query: str) -> int:
        """Count comparison operators and function calls in WHERE/ON/HAVING clauses ONLY.
        Do NOT count = in SET clauses. Count function calls (word + open paren)
        excluding SQL keywords."""
        upper = query.upper()
        complexity = 0

        # Extract WHERE, ON, HAVING clause content
        clause_contents = QueryStats._extract_clause_contents(upper)

        for content in clause_contents:
            # Count comparison operators: =, <, >, !=, AND, OR
            # Be careful: != is two chars, so check != before =
            # Also avoid counting = inside strings (simplified: we don't parse strings)

            # Count != first
            complexity += len(re.findall(r'!=', content))

            # Count = but not != (already counted) and not ==
            # Use negative lookbehind/lookahead to exclude != and ==
            complexity += len(re.findall(r'(?<!=)=(?!=)', content))

            # Count < and >
            complexity += len(re.findall(r'<', content))
            complexity += len(re.findall(r'>', content))

            # Count AND and OR as keywords
            complexity += len(re.findall(r'\bAND\b', content))
            complexity += len(re.findall(r'\bOR\b', content))

            # Count function calls: word followed by (, excluding SQL keywords
            for match in re.finditer(r'\b(\w+)\s*\(', content):
                func_name = match.group(1)
                if func_name not in QueryStats.SQL_KEYWORDS:
                    complexity += 1

        return complexity

    @staticmethod
    def _extract_clause_contents(upper_query: str) -> list[str]:
        """Extract the content of WHERE, ON, HAVING clauses.
        Returns a list of strings, one per clause found."""
        contents: list[str] = []

        # WHERE clause: from WHERE to next major clause or end
        where_match = re.search(r'\bWHERE\b(.+?)(?=\b(GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|UNION|WINDOW|RETURNING)\b|$)', upper_query, re.DOTALL)
        if where_match:
            contents.append(where_match.group(1))

        # ON clause: from ON to next major clause or end
        on_matches = re.finditer(r'\bON\b(.+?)(?=\b(WHERE|GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|UNION|WINDOW|RETURNING|JOIN)\b|$)', upper_query, re.DOTALL)
        for match in on_matches:
            contents.append(match.group(1))

        # HAVING clause: from HAVING to next major clause or end
        having_match = re.search(r'\bHAVING\b(.+?)(?=\b(ORDER\s+BY|LIMIT|UNION|WINDOW|RETURNING)\b|$)', upper_query, re.DOTALL)
        if having_match:
            contents.append(having_match.group(1))

        return contents

    @staticmethod
    def aggregate(profiles: list[QueryProfile]) -> StatsReport:
        """Aggregate multiple QueryProfiles into a StatsReport."""
        if not profiles:
            return StatsReport(
                total_queries=0,
                statement_distribution={},
                clause_frequency={},
                avg_join_count=0.0,
                avg_subquery_depth=0.0,
                max_subquery_depth=0,
                complexity_histogram={'simple': 0, 'medium': 0, 'complex': 0, 'extreme': 0},
            )

        total = len(profiles)

        # statement_distribution
        statement_dist: dict[str, int] = {}
        for p in profiles:
            st = p.statement_type
            statement_dist[st] = statement_dist.get(st, 0) + 1

        # clause_frequency
        clause_freq: dict[str, int] = {}
        for p in profiles:
            for clause in p.clauses_used:
                clause_freq[clause] = clause_freq.get(clause, 0) + 1

        # avg_join_count
        total_joins = sum(p.join_count for p in profiles)
        avg_joins = total_joins / total

        # avg_subquery_depth
        total_depth = sum(p.subquery_depth for p in profiles)
        avg_depth = total_depth / total

        # max_subquery_depth
        max_depth = max(p.subquery_depth for p in profiles)

        # complexity_histogram
        histogram: dict[str, int] = {'simple': 0, 'medium': 0, 'complex': 0, 'extreme': 0}
        for p in profiles:
            c = p.expression_complexity
            if c <= 2:
                histogram['simple'] += 1
            elif c <= 5:
                histogram['medium'] += 1
            elif c <= 10:
                histogram['complex'] += 1
            else:
                histogram['extreme'] += 1

        return StatsReport(
            total_queries=total,
            statement_distribution=statement_dist,
            clause_frequency=clause_freq,
            avg_join_count=avg_joins,
            avg_subquery_depth=avg_depth,
            max_subquery_depth=max_depth,
            complexity_histogram=histogram,
        )

    @staticmethod
    def format_report(report: StatsReport) -> str:
        """Format a StatsReport as a human-readable string."""
        lines: list[str] = []
        lines.append(f"Total Queries: {report.total_queries}")
        lines.append("")

        lines.append("Statement Distribution:")
        for stype, count in sorted(report.statement_distribution.items()):
            lines.append(f"  {stype}: {count}")
        lines.append("")

        lines.append("Clause Frequency:")
        for clause, count in sorted(report.clause_frequency.items()):
            lines.append(f"  {clause}: {count}")
        lines.append("")

        lines.append(f"Average Join Count: {report.avg_join_count:.2f}")
        lines.append(f"Average Subquery Depth: {report.avg_subquery_depth:.2f}")
        lines.append(f"Max Subquery Depth: {report.max_subquery_depth}")
        lines.append("")

        lines.append("Complexity Histogram:")
        for bucket, count in report.complexity_histogram.items():
            lines.append(f"  {bucket}: {count}")

        return '\n'.join(lines)