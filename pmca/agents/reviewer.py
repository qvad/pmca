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

# JSON schema for grammar-constrained reviewer output (Ollama format parameter).
# Guarantees the model emits valid JSON instead of prose.
_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "suggestions": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": ["passed", "issues", "suggestions"],
}


class ReviewerAgent(BaseAgent):
    role = AgentRole.REVIEWER

    def _get_system_prompt(self) -> str:
        """Construct system prompt with Reviewer SOP."""
        system = prompts.SYSTEM_PROMPT
        from pmca.prompts import REVIEWER_SOP
        if REVIEWER_SOP not in system:
            system += "\n" + REVIEWER_SOP
        return system

    async def verify_spec(self, child_spec: str, parent_spec: str) -> ReviewResult:
        """Verify that a child spec aligns with the parent spec."""
        prompt = prompts.VERIFY_SPEC_PROMPT.format(
            parent_spec=parent_spec,
            child_spec=child_spec,
        )
        system = self._get_system_prompt()
        response = await self._generate_structured(prompt, _REVIEW_SCHEMA, system=system)
        return self._parse_review(response)

    async def verify_code(
        self, code: str, spec: str, *, test_status: str = ""
    ) -> ReviewResult:
        """Verify that an implementation matches a spec."""
        prompt = prompts.VERIFY_CODE_PROMPT.format(
            spec=spec,
            code=code,
            test_status=test_status,
        )
        system = self._get_system_prompt()
        response = await self._generate_structured(prompt, _REVIEW_SCHEMA, system=system)
        return self._parse_review(response)

    async def verify_tests(self, tests: str, spec: str, context: str = "") -> ReviewResult:
        """Verify test quality using the reviewer model."""
        prompt = prompts.VERIFY_TESTS_PROMPT.format(
            spec=spec,
            tests=tests,
            context=context,
        )
        system = self._get_system_prompt()
        response = await self._generate_structured(prompt, _REVIEW_SCHEMA, system=system)
        return self._parse_review(response)

    async def verify_integration(self, task: TaskNode, children_summary: str) -> ReviewResult:
        """Verify that all children integrate correctly."""
        prompt = prompts.VERIFY_INTEGRATION_PROMPT.format(
            parent_spec=task.spec,
            children_summary=children_summary,
        )
        system = self._get_system_prompt()
        response = await self._generate_structured(prompt, _REVIEW_SCHEMA, system=system)
        return self._parse_review(response)

    def _parse_review(self, response: str) -> ReviewResult:
        """Parse a review result from model response JSON."""
        # Primary: direct JSON parse (structured output guarantees valid JSON)
        try:
            data = json.loads(response)
            return ReviewResult(
                passed=bool(data.get("passed", False)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                timestamp=datetime.now(),
                model_used=self.role.value,
            )
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract JSON object from prose (plain _generate path)
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

        # Last resort: assume failure if we can't parse
        return ReviewResult(
            passed=False,
            issues=["Failed to parse review response from model"],
            suggestions=[],
            timestamp=datetime.now(),
            model_used=self.role.value,
        )
