"""Model definitions, role mappings, and configuration loading."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class AgentRole(str, enum.Enum):
    ARCHITECT = "architect"
    CODER = "coder"
    REVIEWER = "reviewer"
    WATCHER = "watcher"
    TESTER = "tester"


@dataclass
class ModelConfig:
    name: str
    context_window: int = 16384
    temperature: float = 0.3
    provider: str = "ollama"    # "ollama", "groq", "openai"
    api_base: str = ""          # Override API base URL (empty = default for provider)


@dataclass
class CascadeConfig:
    max_depth: int = 4
    max_retries: int = 3
    max_children: int = 6
    leaf_threshold: str = "method"
    best_of_n: int = 1           # Candidates per generation (1=disabled)
    fresh_start_after: int = 3    # Fix attempts before fresh start
    test_first: bool = False      # Enable test designer phase


@dataclass
class WorkspaceConfig:
    path: str = "./workspace"
    git_checkpoint: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./pmca.log"


@dataclass
class RAGConfig:
    enabled: bool = False
    docs_path: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    n_results: int = 3
    persist_dir: str = "~/.pmca/rag"


@dataclass
class MCPConfig:
    enabled: bool = False
    server_name: str = "pmca"


@dataclass
class LintConfig:
    mypy: bool = False
    ruff: bool = False


@dataclass
class Config:
    models: dict[AgentRole, ModelConfig] = field(default_factory=dict)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    lint: LintConfig = field(default_factory=LintConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> Config:
        models: dict[AgentRole, ModelConfig] = {}
        for role_name, mcfg in raw.get("models", {}).items():
            role = AgentRole(role_name)
            models[role] = ModelConfig(
                name=mcfg["name"],
                context_window=mcfg.get("context_window", 16384),
                temperature=mcfg.get("temperature", 0.3),
                provider=mcfg.get("provider", "ollama"),
                api_base=mcfg.get("api_base", ""),
            )

        cascade_raw = raw.get("cascade", {})
        cascade = CascadeConfig(
            max_depth=cascade_raw.get("max_depth", 4),
            max_retries=cascade_raw.get("max_retries", 3),
            max_children=cascade_raw.get("max_children", 6),
            leaf_threshold=cascade_raw.get("leaf_threshold", "method"),
            best_of_n=cascade_raw.get("best_of_n", 1),
            fresh_start_after=cascade_raw.get("fresh_start_after", 3),
            test_first=cascade_raw.get("test_first", False),
        )

        ws_raw = raw.get("workspace", {})
        workspace = WorkspaceConfig(
            path=ws_raw.get("path", "./workspace"),
            git_checkpoint=ws_raw.get("git_checkpoint", True),
        )

        log_raw = raw.get("logging", {})
        logging_cfg = LoggingConfig(
            level=log_raw.get("level", "INFO"),
            file=log_raw.get("file", "./pmca.log"),
        )

        rag_raw = raw.get("rag", {})
        rag = RAGConfig(
            enabled=rag_raw.get("enabled", False),
            docs_path=rag_raw.get("docs_path", ""),
            embedding_model=rag_raw.get("embedding_model", "all-MiniLM-L6-v2"),
            n_results=rag_raw.get("n_results", 3),
            persist_dir=rag_raw.get("persist_dir", "~/.pmca/rag"),
        )

        mcp_raw = raw.get("mcp", {})
        mcp = MCPConfig(
            enabled=mcp_raw.get("enabled", False),
            server_name=mcp_raw.get("server_name", "pmca"),
        )

        lint_raw = raw.get("lint", {})
        lint = LintConfig(
            mypy=lint_raw.get("mypy", False),
            ruff=lint_raw.get("ruff", False),
        )

        return cls(
            models=models,
            cascade=cascade,
            workspace=workspace,
            logging=logging_cfg,
            rag=rag,
            mcp=mcp,
            lint=lint,
        )

    @classmethod
    def default(cls) -> Config:
        default_path = Path(__file__).parent.parent.parent / "config" / "default.yaml"
        if default_path.exists():
            return cls.from_yaml(default_path)
        return cls(
            models={
                AgentRole.ARCHITECT: ModelConfig(
                    name="qwen2.5-coder:14b-instruct-q4_K_M",
                    temperature=0.3,
                ),
                AgentRole.CODER: ModelConfig(
                    name="qwen2.5-coder:7b-instruct-q4_K_M",
                    temperature=0.2,
                ),
                AgentRole.REVIEWER: ModelConfig(
                    name="qwen2.5-coder:14b-instruct-q4_K_M",
                    temperature=0.1,
                ),
                AgentRole.WATCHER: ModelConfig(
                    name="qwen2.5-coder:7b-instruct-q4_K_M",
                    temperature=0.1,
                ),
                AgentRole.TESTER: ModelConfig(
                    name="qwen2.5-coder:14b-instruct-q4_K_M",
                    temperature=0.2,
                ),
            },
        )

    def get_model(self, role: AgentRole) -> ModelConfig:
        return self.models[role]
