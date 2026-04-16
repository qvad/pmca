"""Prompt templates and skill definitions for PMCA agents."""

from .architect_skills import ARCHITECT_SKILLS
from .go_skills import GO_SKILLS
from .qwen_python_skills import QWEN_PYTHON_SKILLS
from .sop_mandates import REVIEWER_SOP, TESTER_SOP, WATCHER_SOP
from .typescript_skills import TYPESCRIPT_SKILLS

__all__ = [
    "ARCHITECT_SKILLS",
    "GO_SKILLS",
    "QWEN_PYTHON_SKILLS",
    "REVIEWER_SOP",
    "TESTER_SOP",
    "TYPESCRIPT_SKILLS",
    "WATCHER_SOP",
]
