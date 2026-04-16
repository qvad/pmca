"""Tester agent — test design, failure analysis, and edge case generation."""

from __future__ import annotations

import json
import re

from pmca.agents.base import BaseAgent
from pmca.models.config import AgentRole
from pmca.prompts import tester as prompts
from pmca.tasks.state import CodeFile, FailureAnalysis
from pmca.tasks.tree import TaskNode


class TesterAgent(BaseAgent):
    role = AgentRole.TESTER

    def __init__(self, model_manager, project_mode: bool = False) -> None:
        super().__init__(model_manager)
        self._project_mode = project_mode

    def _get_system_prompt(self) -> str:
        """Construct system prompt with Tester SOP."""
        system = prompts.SYSTEM_PROMPT
        from pmca.prompts import TESTER_SOP
        if TESTER_SOP not in system:
            system += "\n" + TESTER_SOP
        return system

    async def generate_tests(self, task: TaskNode, context: str = "") -> list[CodeFile]:
        """Design tests from specification using the 14B model."""
        safe_name = self._derive_short_name(task.title)
        prompt = prompts.GENERATE_TESTS_PROMPT.format(
            spec=task.spec,
            context=context,
            suggested_module=safe_name,
            suggested_test_path=f"tests/test_{safe_name}.py",
        )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system)
        return self._parse_code_blocks(response)

    async def analyze_failure(
        self,
        task: TaskNode,
        test_output: str,
        code_content: str,
        tests_content: str = "",
    ) -> FailureAnalysis:
        """Analyze test failures and determine root cause."""
        prompt = prompts.ANALYZE_FAILURE_PROMPT.format(
            spec=task.spec,
            code=code_content,
            tests=tests_content,
            test_output=test_output,
        )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system)
        return self._parse_failure_analysis(response)

    async def generate_edge_cases(
        self,
        task: TaskNode,
        code_content: str,
        existing_tests: str,
    ) -> list[CodeFile]:
        """Generate boundary/adversarial tests for passing code."""
        safe_name = self._derive_short_name(task.title)
        prompt = prompts.GENERATE_EDGE_CASES_PROMPT.format(
            spec=task.spec,
            code=code_content,
            existing_tests=existing_tests,
            test_path=f"tests/test_{safe_name}_edge.py",
        )
        system = self._get_system_prompt()
        response = await self._generate(prompt, system=system)
        return self._parse_code_blocks(response)

    def _parse_failure_analysis(self, response: str) -> FailureAnalysis:
        """Parse a FailureAnalysis from model response JSON."""
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return FailureAnalysis(
                    root_cause=data.get("root_cause", "unknown"),
                    explanation=data.get("explanation", ""),
                    suggested_fix_target=data.get("suggested_fix_target", "code"),
                    specific_issues=data.get("specific_issues", []),
                )
            except json.JSONDecodeError:
                pass

        # Fallback: return unknown analysis
        return FailureAnalysis(
            root_cause="unknown",
            explanation="Failed to parse failure analysis from model response",
            suggested_fix_target="code",
            specific_issues=[response[:500] if response else "Empty response"],
        )
