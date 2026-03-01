"""Reviewer agent — specification and code verification."""

from __future__ import annotations

import json
import re
from datetime import datetime

from pmca.agents.base import BaseAgent
from pmca.models.config import AgentRole
from pmca.prompts import reviewer as prompts
from pmca.tasks.state import ReviewResult
from pmca.tasks.tree import TaskNode


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    async def verify_spec(self, child_spec: str, parent_spec: str) -> ReviewResult:
        """Verify that a child spec aligns with the parent spec."""
        prompt = prompts.VERIFY_SPEC_PROMPT.format(
            parent_spec=parent_spec,
            child_spec=child_spec,
        )
        response = await self._generate(prompt, system=prompts.SYSTEM_PROMPT)
        return self._parse_review(response)

    async def verify_code(self, code: str, spec: str) -> ReviewResult:
        """Verify that an implementation matches a spec."""
        prompt = prompts.VERIFY_CODE_PROMPT.format(
            spec=spec,
            code=code,
        )
        response = await self._generate(prompt, system=prompts.SYSTEM_PROMPT)
        return self._parse_review(response)

    async def verify_tests(self, tests: str, spec: str, context: str = "") -> ReviewResult:
        """Verify test quality using the 14B reviewer model.

        Checks for fake tests, missing imports, wrong assertions, and
        missing edge cases. Returns a ReviewResult with specific issues.
        """
        prompt = prompts.VERIFY_TESTS_PROMPT.format(
            spec=spec,
            tests=tests,
            context=context,
        )
        response = await self._generate(prompt, system=prompts.SYSTEM_PROMPT)
        return self._parse_review(response)

    async def verify_integration(self, task: TaskNode, children_summary: str) -> ReviewResult:
        """Verify that all children integrate correctly."""
        prompt = prompts.VERIFY_INTEGRATION_PROMPT.format(
            parent_spec=task.spec,
            children_summary=children_summary,
        )
        response = await self._generate(prompt, system=prompts.SYSTEM_PROMPT)
        return self._parse_review(response)

    def _parse_review(self, response: str) -> ReviewResult:
        """Parse a review result from model response JSON."""
        # Try to extract JSON from the response
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return ReviewResult(
                    passed=bool(data.get("passed", False)),
                    issues=data.get("issues", []),
                    suggestions=data.get("suggestions", []),
                    timestamp=datetime.now(),
                    model_used=self.role.value,
                )
            except json.JSONDecodeError:
                pass

        # Fallback: assume failure if we can't parse
        return ReviewResult(
            passed=False,
            issues=["Failed to parse review response from model"],
            suggestions=[],
            timestamp=datetime.now(),
            model_used=self.role.value,
        )
