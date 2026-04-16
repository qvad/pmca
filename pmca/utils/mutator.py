"""AST-based mutation engine for test quality validation (MuTAP)."""

from __future__ import annotations

import ast
import copy
import enum
import random
from collections.abc import Callable
from dataclasses import dataclass


class MutationType(str, enum.Enum):
    ARITH_OP = "arith_op"
    CMP_OP = "cmp_op"
    RETURN_VALUE = "return_value"
    CONSTANT = "constant"
    NEGATE_COND = "negate_cond"


@dataclass
class Mutation:
    mutation_type: MutationType
    line: int
    description: str
    mutated_source: str


# Operator swap tables
_ARITH_SWAPS: dict[type, type] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}

_CMP_SWAPS: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
}


@dataclass
class _MutationSite:
    """A candidate location for a mutation."""
    mutation_type: MutationType
    line: int
    description: str
    apply: Callable[[ast.Module], ast.Module]


class _MutationCollector(ast.NodeVisitor):
    """Walk an AST and collect candidate mutation sites."""

    def __init__(self, source_lines: list[str]) -> None:
        self.sites: list[_MutationSite] = []
        self._source_lines = source_lines

    def visit_BinOp(self, node: ast.BinOp) -> None:
        op_type = type(node.op)
        if op_type in _ARITH_SWAPS:
            new_op_type = _ARITH_SWAPS[op_type]
            line = node.lineno

            def _apply(tree: ast.Module, _node=node, _new=new_op_type) -> ast.Module:
                t = copy.deepcopy(tree)
                for n in ast.walk(t):
                    if (isinstance(n, ast.BinOp)
                            and n.lineno == _node.lineno
                            and n.col_offset == _node.col_offset
                            and type(n.op) is type(_node.op)):
                        n.op = _new()
                        break
                return t

            self.sites.append(_MutationSite(
                mutation_type=MutationType.ARITH_OP,
                line=line,
                description=f"L{line}: {op_type.__name__} → {new_op_type.__name__}",
                apply=_apply,
            ))
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for i, op in enumerate(node.ops):
            op_type = type(op)
            if op_type in _CMP_SWAPS:
                new_op_type = _CMP_SWAPS[op_type]
                line = node.lineno
                col = node.col_offset
                idx = i

                def _apply(tree: ast.Module, _line=line, _col=col, _idx=idx,
                           _old=op_type, _new=new_op_type) -> ast.Module:
                    t = copy.deepcopy(tree)
                    for n in ast.walk(t):
                        if (isinstance(n, ast.Compare)
                                and n.lineno == _line
                                and n.col_offset == _col):
                            if _idx < len(n.ops) and type(n.ops[_idx]) is _old:
                                n.ops[_idx] = _new()
                            break
                    return t

                self.sites.append(_MutationSite(
                    mutation_type=MutationType.CMP_OP,
                    line=line,
                    description=f"L{line}: {op_type.__name__} → {new_op_type.__name__}",
                    apply=_apply,
                ))
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            line = node.lineno

            def _apply(tree: ast.Module, _line=line) -> ast.Module:
                t = copy.deepcopy(tree)
                for n in ast.walk(t):
                    if isinstance(n, ast.Return) and n.lineno == _line and n.value is not None:
                        n.value = ast.Constant(value=None)
                        break
                return t

            self.sites.append(_MutationSite(
                mutation_type=MutationType.RETURN_VALUE,
                line=line,
                description=f"L{line}: return → return None",
                apply=_apply,
            ))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            line = node.lineno
            old_val = node.value
            new_val = old_val + 1 if old_val != 0 else 1

            def _apply(tree: ast.Module, _line=line, _old=old_val, _new=new_val) -> ast.Module:
                t = copy.deepcopy(tree)
                for n in ast.walk(t):
                    if (isinstance(n, ast.Constant)
                            and n.lineno == _line
                            and n.value == _old
                            and isinstance(n.value, (int, float))
                            and not isinstance(n.value, bool)):
                        n.value = _new
                        break
                return t

            self.sites.append(_MutationSite(
                mutation_type=MutationType.CONSTANT,
                line=line,
                description=f"L{line}: {old_val} → {new_val}",
                apply=_apply,
            ))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        line = node.lineno

        def _apply(tree: ast.Module, _line=line) -> ast.Module:
            t = copy.deepcopy(tree)
            for n in ast.walk(t):
                if isinstance(n, ast.If) and n.lineno == _line:
                    n.test = ast.UnaryOp(op=ast.Not(), operand=n.test)
                    ast.fix_missing_locations(t)
                    break
            return t

        self.sites.append(_MutationSite(
            mutation_type=MutationType.NEGATE_COND,
            line=line,
            description=f"L{line}: negate if-condition",
            apply=_apply,
        ))
        self.generic_visit(node)


def generate_mutations(source: str, max_mutations: int = 8) -> list[Mutation]:
    """Generate up to *max_mutations* diverse AST mutations for *source*.

    Returns a list of Mutation objects, each containing the full mutated source.
    Selects a diverse subset across mutation types when more sites exist than
    the budget allows.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines()
    collector = _MutationCollector(lines)
    collector.visit(tree)

    sites = collector.sites
    if not sites:
        return []

    # Diverse selection: group by type, round-robin pick
    selected: list[_MutationSite]
    if len(sites) <= max_mutations:
        selected = sites
    else:
        by_type: dict[MutationType, list[_MutationSite]] = {}
        for s in sites:
            by_type.setdefault(s.mutation_type, []).append(s)
        # Shuffle within each type for variety
        for group in by_type.values():
            random.shuffle(group)

        selected = []
        type_iters = {t: iter(g) for t, g in by_type.items()}
        types = list(type_iters.keys())
        idx = 0
        while len(selected) < max_mutations and type_iters:
            t = types[idx % len(types)]
            try:
                selected.append(next(type_iters[t]))
            except StopIteration:
                type_iters.pop(t)
                types = list(type_iters.keys())
                if not types:
                    break
            idx += 1

    mutations: list[Mutation] = []
    for site in selected:
        try:
            mutated_tree = site.apply(tree)
            mutated_source = ast.unparse(mutated_tree)
            # Verify the mutated source is valid Python
            ast.parse(mutated_source)
            mutations.append(Mutation(
                mutation_type=site.mutation_type,
                line=site.line,
                description=site.description,
                mutated_source=mutated_source,
            ))
        except (SyntaxError, ValueError, TypeError):
            continue

    return mutations
