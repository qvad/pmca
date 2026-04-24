"""Base agent abstract class."""

from __future__ import annotations

import ast
import re
from abc import ABC
from pathlib import PurePosixPath

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
        self.filename_normalizations: int = 0

    async def _generate(
        self, prompt: str, system: str = "", temperature: float | None = None,
        role: AgentRole | None = None, think: bool | None = None,
    ) -> str:
        """Generate a response using this agent's assigned model.

        Args:
            role: Override the agent's default role for model selection.
                  Used by adaptive routing to switch coder models per task.
            think: Control thinking mode for reasoning models.
        """
        effective_role = role or self.role

        # Inject Qwen persona if applicable
        model_cfg = self._model._config.get_model(effective_role)
        if model_cfg and "qwen3.5" in model_cfg.name.lower():
            from pmca.prompts.qwen_python_skills import QWEN_PERSONA
            if QWEN_PERSONA not in system:
                system = QWEN_PERSONA + "\n\n" + system

        self._log.debug(f"Generating response (prompt length: {len(prompt)}, role={effective_role.value})")
        response = await self._model.generate(effective_role, prompt, system=system, temperature=temperature, think=think)
        response = self._sanitize_response(response)
        self._log.debug(f"Got response (length: {len(response)})")
        return response

    @staticmethod
    def _sanitize_response(response: str) -> str:
        """Strip think blocks, EOS tokens, and repeated content from model output.

        MoE models with aggressive quantization (IQ3/Q2) produce artifacts:
        <think>...</think> blocks, <|endoftext|> tokens, and sometimes
        duplicate the response after EOS. This strips all of those.
        """
        # Strip <think>...</think> blocks (may contain reasoning)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        # Strip standalone <think> or </think> tags
        response = re.sub(r"</?think>", "", response)
        # Strip EOS/BOS tokens from various model families
        response = re.sub(r"<\|(?:endoftext|im_start|im_end|end)\|>", "", response)
        # Strip content after first EOS-like boundary (model repeats itself)
        eos_pos = response.find("<|endoftext|>")
        if eos_pos > 0:
            response = response[:eos_pos]
        return response.strip()

    async def _generate_structured(
        self, prompt: str, schema: dict, system: str = "", temperature: float | None = None,
        role: AgentRole | None = None, think: bool | None = None,
    ) -> str:
        """Generate a structured JSON response constrained by a JSON schema.

        Uses Ollama's ``format`` parameter for grammar-constrained decoding.
        Returns the raw JSON string.
        """
        effective_role = role or self.role
        self._log.debug(f"Generating structured response (prompt length: {len(prompt)})")
        response = await self._model.generate(
            effective_role, prompt, system=system, temperature=temperature, format=schema, think=think,
        )
        self._log.debug(f"Got structured response (length: {len(response)})")
        return response

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """Convert CamelCase/PascalCase to snake_case.

        Examples: LRUCache → lru_cache, DataPipeline → data_pipeline,
        BankAccount → bank_account, Calculator → calculator.
        """
        # Insert underscore between lowercase/digit and uppercase
        s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        # Insert underscore between consecutive uppercase and uppercase+lowercase
        s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)
        return s.lower()

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

        # Split CamelCase tokens before lowering (LRUCache → LRU_Cache → lru cache)
        cleaned_title = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            " ", cleaned_title,
        )

        # Extract key nouns — take first 3 meaningful words
        words = re.sub(r"[^a-z0-9\s]", "", cleaned_title.lower()).split()
        stop_words = {"a", "an", "the", "write", "create", "build", "make", "implement",
                      "that", "which", "with", "and", "for", "from", "into", "using",
                      "single", "simple", "python", "file", "called", "module", "function",
                      "include", "includes", "including", "proper", "type", "hints",
                      "class", "method", "methods"}
        meaningful = [w for w in words if w not in stop_words and len(w) > 1]
        if meaningful:
            name = "_".join(meaningful[:3])
            return re.sub(r"_+", "_", name).strip("_")[:40]

        # Last resort
        safe = re.sub(r"[^a-z0-9_]", "_", title.lower()[:30])
        return re.sub(r"_+", "_", safe).strip("_")

    def _normalize_filenames(self, files: list[CodeFile]) -> list[CodeFile]:
        """Normalize Python module filenames to snake_case based on code content.

        Inspects AST of each source file, finds the primary class, and derives
        the correct snake_case module name. Only renames when the current
        filename is the same word mangled (e.g. ``lrucache.py`` + class
        ``LRUCache`` → ``lru_cache.py``). Does NOT rename unrelated filenames.
        """
        rename_map = self._build_rename_map(files)
        if not rename_map:
            return files

        for old, new in rename_map.items():
            self._log.info(f"Normalized filename: {old}.py → {new}.py")
        self.filename_normalizations += len(rename_map)

        return [self._apply_rename(f, rename_map) for f in files]

    @staticmethod
    def _build_rename_map(files: list[CodeFile]) -> dict[str, str]:
        """Return {old_stem: new_stem} for files where stem mismatches primary class."""
        rename_map: dict[str, str] = {}
        for f in files:
            p = PurePosixPath(f.path)
            if p.suffix != ".py" or p.stem.startswith("test_") or p.stem == "__init__":
                continue
            class_name = BaseAgent._primary_public_class(f.content)
            if not class_name:
                continue
            correct_stem = BaseAgent._camel_to_snake(class_name)
            if correct_stem == p.stem:
                continue
            # Only rename when same word mangled (e.g. lrucache ≡ lru_cache).
            if p.stem.replace("_", "") == correct_stem.replace("_", ""):
                rename_map[p.stem] = correct_stem
        return rename_map

    @staticmethod
    def _primary_public_class(source: str) -> str | None:
        """Return the first top-level non-underscore class name, or None."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                return node.name
        return None

    @staticmethod
    def _apply_rename(f: CodeFile, rename_map: dict[str, str]) -> CodeFile:
        """Rewrite path and import statements for one CodeFile using ``rename_map``."""
        p = PurePosixPath(f.path)
        new_path = f.path
        if p.suffix == ".py":
            if p.stem in rename_map:
                new_path = str(p.with_stem(rename_map[p.stem]))
            elif p.stem.startswith("test_") and p.stem[5:] in rename_map:
                new_path = str(p.with_stem(f"test_{rename_map[p.stem[5:]]}"))

        content = f.content
        for old_stem, new_stem in rename_map.items():
            content = content.replace(f"from {old_stem} import", f"from {new_stem} import")
            content = content.replace(f"import {old_stem}", f"import {new_stem}")
        return CodeFile(path=new_path, content=content)

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
            return self._normalize_filenames(files)

        # Strategy 1b: Match blocks with "# filename.py" comment (Qwen 3.6 style)
        pattern1b = r"```\w*\s*\n#\s*([a-zA-Z0-9_/.-]+\.py)\s*\n(.*?)```"
        matches1b = re.findall(pattern1b, response, re.DOTALL)
        for filepath, content in matches1b:
            filepath = filepath.strip()
            content = content.strip()
            if content and filepath:
                files.append(CodeFile(path=filepath, content=content))

        if files:
            return self._normalize_filenames(files)

        # Strategy 2: Look for filename in a heading or line before the code block
        pattern2 = r"(?:^|\n)(?:#+\s*|(?:\*\*)?`?)([a-zA-Z0-9_/.-]+\.[a-zA-Z]+)`?(?:\*\*)?\s*:?\s*\n```\w*\s*\n(.*?)```"
        matches2 = re.findall(pattern2, response, re.DOTALL)
        for filepath, content in matches2:
            filepath = filepath.strip()
            content = content.strip()
            if content and filepath:
                files.append(CodeFile(path=filepath, content=content))

        if files:
            return self._normalize_filenames(files)

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
