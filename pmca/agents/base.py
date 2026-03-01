"""Base agent abstract class."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from pmca.models.config import AgentRole
from pmca.models.manager import ModelManager
from pmca.tasks.state import CodeFile
from pmca.utils.logger import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    role: AgentRole

    def __init__(self, model_manager: ModelManager) -> None:
        self._model = model_manager
        self._log = get_logger(f"agent.{self.role.value}")

    async def _generate(self, prompt: str, system: str = "", temperature: float | None = None) -> str:
        """Generate a response using this agent's assigned model."""
        self._log.debug(f"Generating response (prompt length: {len(prompt)})")
        response = await self._model.generate(self.role, prompt, system=system, temperature=temperature)
        self._log.debug(f"Got response (length: {len(response)})")
        return response

    async def _generate_structured(
        self, prompt: str, schema: dict, system: str = "", temperature: float | None = None,
    ) -> str:
        """Generate a structured JSON response constrained by a JSON schema.

        Uses Ollama's ``format`` parameter for grammar-constrained decoding.
        Returns the raw JSON string.
        """
        self._log.debug(f"Generating structured response (prompt length: {len(prompt)})")
        response = await self._model.generate(
            self.role, prompt, system=system, temperature=temperature, format=schema,
        )
        self._log.debug(f"Got structured response (length: {len(response)})")
        return response

    @staticmethod
    def _derive_short_name(title: str) -> str:
        """Derive a short, meaningful filename from a task title."""
        # Try to find a quoted filename in the title (e.g. "called calculator.py")
        name_match = re.search(r"called\s+(\w[\w.-]*)", title, re.IGNORECASE)
        if name_match:
            name = name_match.group(1)
            # Strip extension if present (we add .py ourselves)
            name = re.sub(r"\.\w+$", "", name)
            return re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")

        # Strip common file extensions before word extraction
        cleaned_title = re.sub(r"\.(py|js|ts|html|css|java|go|rs)(?:\s|$)", " ", title)

        # Extract key nouns — take first 3 meaningful words
        words = re.sub(r"[^a-z0-9\s]", "", cleaned_title.lower()).split()
        stop_words = {"a", "an", "the", "write", "create", "build", "make", "implement",
                      "that", "which", "with", "and", "for", "from", "into", "using",
                      "single", "simple", "python", "file", "called", "module", "function",
                      "include", "includes", "including", "proper", "type", "hints"}
        meaningful = [w for w in words if w not in stop_words and len(w) > 1]
        if meaningful:
            name = "_".join(meaningful[:3])
            return re.sub(r"_+", "_", name).strip("_")[:40]

        # Last resort
        safe = re.sub(r"[^a-z0-9_]", "_", title.lower()[:30])
        return re.sub(r"_+", "_", safe).strip("_")

    def _parse_code_blocks(self, response: str) -> list[CodeFile]:
        """Parse fenced code blocks with file paths from model response."""
        files: list[CodeFile] = []

        # Strategy 1: Match blocks with explicit filepath comments
        pattern = r"```\w*\s*\n#\s*filepath:\s*(.+?)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)
        for filepath, content in matches:
            filepath = filepath.strip()
            content = content.strip()
            if content and filepath:
                files.append(CodeFile(path=filepath, content=content))

        if files:
            return files

        # Strategy 2: Look for filename in a heading or line before the code block
        pattern2 = r"(?:^|\n)(?:#+\s*|(?:\*\*)?`?)([a-zA-Z0-9_/.-]+\.[a-zA-Z]+)`?(?:\*\*)?\s*:?\s*\n```\w*\s*\n(.*?)```"
        matches2 = re.findall(pattern2, response, re.DOTALL)
        for filepath, content in matches2:
            filepath = filepath.strip()
            content = content.strip()
            if content and filepath:
                files.append(CodeFile(path=filepath, content=content))

        if files:
            return files

        # Strategy 3: Fallback — extract any fenced code block
        code_blocks = re.findall(r"```\w*\s*\n(.*?)```", response, re.DOTALL)
        for i, block in enumerate(code_blocks):
            block = block.strip()
            if block:
                ext = self._guess_extension(block)
                files.append(CodeFile(path=f"generated_{i}{ext}", content=block))

        return files

    @staticmethod
    def _guess_extension(content: str) -> str:
        """Guess file extension from code content."""
        if content.lstrip().startswith(("<!DOCTYPE", "<html", "<head", "<div")):
            return ".html"
        if content.lstrip().startswith(("import ", "from ", "def ", "class ")):
            return ".py"
        if "function " in content or "const " in content or "=>" in content:
            return ".js"
        return ".py"
