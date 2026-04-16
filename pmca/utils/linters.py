"""External linter integration — mypy, ruff, and semgrep subprocess runners."""

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

from pmca.utils.logger import get_logger

log = get_logger("utils.linters")

_RULES_DIR = Path(__file__).parent.parent / "rules"


def _find_tool(name: str) -> str | None:
    """Find a linter executable, checking the current venv first."""
    # Check alongside the running Python (venv bin dir)
    venv_bin = Path(sys.executable).parent / name
    if venv_bin.is_file():
        return str(venv_bin)
    # Fallback to system PATH
    return shutil.which(name)


def is_mypy_available() -> bool:
    """Check if mypy is installed."""
    return _find_tool("mypy") is not None


def is_ruff_available() -> bool:
    """Check if ruff is installed."""
    return _find_tool("ruff") is not None


async def run_mypy(file_path: Path, workspace: Path) -> list[str]:
    """Run mypy on a file, return list of error strings.

    Returns an empty list if mypy is not installed or the file has no errors.
    """
    mypy_bin = _find_tool("mypy")
    if mypy_bin is None:
        return []

    try:
        proc = await asyncio.create_subprocess_exec(
            mypy_bin,
            "--no-color-output",
            "--no-error-summary",
            "--ignore-missing-imports",
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        log.warning(f"mypy timed out on {file_path}")
        return []
    except FileNotFoundError:
        log.warning(f"mypy binary not found at {mypy_bin}")
        return []

    if proc.returncode == 0:
        return []

    errors: list[str] = []
    for line in stdout.decode().strip().splitlines():
        line = line.strip()
        if line and ": error:" in line:
            errors.append(line)
    return errors


async def run_ruff(file_path: Path, workspace: Path) -> list[str]:
    """Run ruff check on a file, return list of error strings.

    Returns an empty list if ruff is not installed or the file has no errors.
    """
    ruff_bin = _find_tool("ruff")
    if ruff_bin is None:
        return []

    try:
        proc = await asyncio.create_subprocess_exec(
            ruff_bin,
            "check",
            "--no-fix",
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        log.warning(f"ruff timed out on {file_path}")
        return []
    except FileNotFoundError:
        log.warning(f"ruff binary not found at {ruff_bin}")
        return []

    if proc.returncode == 0:
        return []

    errors: list[str] = []
    for line in stdout.decode().strip().splitlines():
        line = line.strip()
        if line:
            errors.append(line)
    return errors


async def ruff_autofix(file_path: Path, workspace: Path) -> int:
    """Run ``ruff check --fix`` on a file to auto-fix safe issues.

    Fixes unused imports (F401), duplicate imports (F811), and other
    auto-fixable rules.  Returns the number of fixes applied, or 0 if
    ruff is not available or nothing was fixed.
    """
    ruff_bin = _find_tool("ruff")
    if ruff_bin is None:
        return 0

    try:
        proc = await asyncio.create_subprocess_exec(
            ruff_bin,
            "check",
            "--fix",
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except (asyncio.TimeoutError, FileNotFoundError):
        return 0

    output = stdout.decode()
    # ruff reports "Found N errors (M fixed, K remaining)."
    import re
    m = re.search(r"\((\d+) fixed", output)
    fixed = int(m.group(1)) if m else 0
    if fixed > 0:
        log.info(f"ruff auto-fixed {fixed} issue(s) in {file_path.name}")
    return fixed


def is_semgrep_available() -> bool:
    """Check if semgrep is installed."""
    return _find_tool("semgrep") is not None


async def semgrep_autofix(file_path: Path, workspace: Path) -> int:
    """Run ``semgrep --autofix`` with PMCA rules on a file.

    Uses the rule YAML files in ``pmca/rules/`` to detect and auto-fix
    known 7B anti-patterns (broad exceptions, print in library code, etc.).
    Returns the number of fixes applied, or 0 if semgrep is not available,
    no rules exist, or nothing was fixed.
    """
    semgrep_bin = _find_tool("semgrep")
    if semgrep_bin is None:
        return 0

    if not _RULES_DIR.is_dir():
        return 0

    rule_files = list(_RULES_DIR.glob("*.yaml")) + list(_RULES_DIR.glob("*.yml"))
    if not rule_files:
        return 0

    # Read file before to compare after
    try:
        content_before = file_path.read_text()
    except OSError:
        return 0

    try:
        # Build --config args for each rule file
        config_args: list[str] = []
        for rf in rule_files:
            config_args.extend(["--config", str(rf)])

        proc = await asyncio.create_subprocess_exec(
            semgrep_bin,
            *config_args,
            "--autofix",
            "--no-git-ignore",
            str(file_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
    except (asyncio.TimeoutError, FileNotFoundError):
        return 0

    # Count fixes by comparing file content before and after
    try:
        content_after = file_path.read_text()
    except OSError:
        return 0

    if content_before == content_after:
        return 0

    # Count changed lines as a rough fix count
    before_lines = content_before.splitlines()
    after_lines = content_after.splitlines()
    changes = sum(1 for a, b in zip(before_lines, after_lines) if a != b)
    changes += abs(len(after_lines) - len(before_lines))
    fixes = max(changes, 1)

    log.info(f"semgrep auto-fixed {fixes} issue(s) in {file_path.name}")
    return fixes
