"""Architect agent — design specification generation and task decomposition."""

from __future__ import annotations

import ast
import json
import re
import textwrap

from pmca.agents.base import BaseAgent
from pmca.models.config import AgentRole
from pmca.models.manager import ModelManager
from pmca.prompts import architect as prompts
from pmca.tasks.state import ReviewResult, TaskType
from pmca.tasks.tree import TaskNode


class ArchitectAgent(BaseAgent):
    role = AgentRole.ARCHITECT

    def __init__(
        self,
        model_manager: ModelManager,
        max_children: int = 6,
        project_mode: bool = False,
    ) -> None:
        super().__init__(model_manager)
        self._max_children = max_children
        self._project_mode = project_mode

    def _get_system_prompt(self, role: AgentRole | None = None) -> str:
        """Construct system prompt, injecting Architect skills."""
        system = prompts.SYSTEM_PROMPT
        from pmca.prompts import ARCHITECT_SKILLS
        if ARCHITECT_SKILLS not in system:
            system += "\n" + ARCHITECT_SKILLS
        return system

    async def generate_spec(self, task: TaskNode, context: str, think: bool | None = None) -> str:
        """Generate a design specification for a task."""
        prompt = prompts.DESIGN_SPEC_PROMPT.format(
            task_title=task.title,
            context=context,
        )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system, think=think)
        return response.strip()

    async def decompose(self, task: TaskNode, think: bool | None = None) -> list[dict]:
        """Decide whether to decompose and return subtask definitions."""
        if self._project_mode and task.depth == 0:
            prompt = prompts.DECOMPOSE_PROJECT_PROMPT.format(
                spec=task.spec,
                max_children=self._max_children,
            )
        else:
            prompt = prompts.DECOMPOSE_PROMPT.format(
                spec=task.spec,
                max_children=self._max_children,
            )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system, think=think)
        response = response.strip()

        if response.upper().startswith("LEAF") or "LEAF" in response.upper().split("\n")[0]:
            self._log.info(f"Task '{task.title}' is a leaf — no decomposition")
            return []

        subtasks = self._parse_subtasks(response)
        if len(subtasks) > self._max_children:
            subtasks = subtasks[: self._max_children]
            self._log.warning(
                f"Truncated subtasks to {self._max_children} for '{task.title}'"
            )
        return subtasks

    async def refine_spec(self, task: TaskNode, feedback: ReviewResult, think: bool | None = None) -> str:
        """Refine a specification based on reviewer feedback."""
        prompt = prompts.REFINE_SPEC_PROMPT.format(
            spec=task.spec,
            issues="\n".join(f"- {i}" for i in feedback.issues),
            suggestions="\n".join(f"- {s}" for s in feedback.suggestions),
        )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system, think=think)
        return response.strip()

    def _parse_subtasks(self, response: str) -> list[dict]:
        """Parse subtask JSON from the model response."""
        # Try to find JSON array in the response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                raw = json.loads(json_match.group())
                subtasks = []
                for item in raw:
                    if isinstance(item, dict) and "title" in item:
                        task_type = item.get("type", "function")
                        try:
                            tt = TaskType(task_type)
                        except ValueError:
                            tt = TaskType.FUNCTION
                        subtasks.append({
                            "title": item["title"],
                            "type": tt,
                            "description": item.get("description", ""),
                        })
                return subtasks
            except json.JSONDecodeError:
                self._log.warning("Failed to parse subtask JSON")

        self._log.warning("Could not parse subtasks from response, treating as leaf")
        return []

    @staticmethod
    def extract_interface_from_code(code: str, filepath: str = "") -> str:
        """Extract public interface (signatures only) from Python code using AST.

        Returns a string with class/function signatures and ``...`` bodies.
        Deterministic, zero LLM tokens.  Falls back to empty string on
        ``SyntaxError``.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ""

        lines: list[str] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                sig = ArchitectAgent._format_func_signature(node)
                lines.append(f"{sig}: ...")
            elif isinstance(node, ast.ClassDef):
                lines.append(f"class {node.name}:")
                has_methods = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                        sig = ArchitectAgent._format_func_signature(item)
                        lines.append(f"    {sig}: ...")
                        has_methods = True
                if not has_methods:
                    lines.append("    ...")
        return "\n".join(lines)

    @staticmethod
    def _format_func_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Format a function/method AST node into its signature string."""
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args_parts: list[str] = []

        # Regular args
        all_args = node.args.args
        # Defaults are right-aligned: last N args have defaults
        num_defaults = len(node.args.defaults)
        for i, arg in enumerate(all_args):
            ann = ""
            if arg.annotation:
                ann = f": {ast.unparse(arg.annotation)}"
            default_idx = i - (len(all_args) - num_defaults)
            if default_idx >= 0:
                default_val = ast.unparse(node.args.defaults[default_idx])
                args_parts.append(f"{arg.arg}{ann}={default_val}")
            else:
                args_parts.append(f"{arg.arg}{ann}")

        ret = ""
        if node.returns:
            ret = f" -> {ast.unparse(node.returns)}"

        return f"{prefix} {node.name}({', '.join(args_parts)}){ret}"
