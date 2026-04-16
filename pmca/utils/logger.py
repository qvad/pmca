"""Structured logging for PMCA."""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_console = Console(
    theme=Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "agent.architect": "bold magenta",
            "agent.coder": "bold green",
            "agent.reviewer": "bold blue",
            "agent.watcher": "bold yellow",
        }
    )
)


def setup_logging(level: str = "INFO", log_file: str | None = None) -> logging.Logger:
    """Configure and return the root PMCA logger."""
    logger = logging.getLogger("pmca")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    rich_handler = RichHandler(
        console=_console,
        show_path=False,
        show_time=True,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(logging.DEBUG)
    logger.addHandler(rich_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "pmca") -> logging.Logger:
    """Get a named logger under the pmca namespace."""
    return logging.getLogger(f"pmca.{name}" if name != "pmca" else "pmca")


def get_console() -> Console:
    """Get the shared Rich console for direct output."""
    return _console
