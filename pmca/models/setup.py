"""Auto-install Ollama, pull models, validate setup."""

from __future__ import annotations

import asyncio
import shutil
import subprocess

from pmca.models.config import Config
from pmca.models.manager import ModelManager
from pmca.utils.logger import get_logger

log = get_logger("models.setup")


class OllamaSetup:
    """Handles Ollama installation, model pulling, and validation."""

    def __init__(self, config: Config) -> None:
        self._config = config

    async def ensure_ollama_installed(self) -> bool:
        """Check if ollama binary exists in PATH. If not, guide user."""
        if shutil.which("ollama") is not None:
            log.info("Ollama binary found")
            return True

        log.warning("Ollama not found in PATH")
        log.info(
            "Install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
        )
        return False

    async def ensure_ollama_running(self) -> bool:
        """Check if Ollama service is running. Try starting if not."""
        manager = ModelManager(self._config)
        try:
            if await manager.is_ollama_running():
                log.info("Ollama service is running")
                return True

            log.info("Starting Ollama service...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Wait a moment for it to start
            for _ in range(10):
                await asyncio.sleep(1)
                if await manager.is_ollama_running():
                    log.info("Ollama service started successfully")
                    return True

            log.error("Failed to start Ollama service")
            return False
        finally:
            await manager.close()

    async def pull_required_models(self) -> None:
        """Pull all required models."""
        manager = ModelManager(self._config)
        try:
            await manager.pull_missing()
        finally:
            await manager.close()

    async def validate_models(self) -> dict[str, bool]:
        """Validate each model responds to a test prompt."""
        manager = ModelManager(self._config)
        results: dict[str, bool] = {}
        try:
            for role, model_cfg in self._config.models.items():
                try:
                    response = await manager.generate(
                        role, "Respond with 'OK'.", system="You are a test."
                    )
                    results[model_cfg.name] = len(response.strip()) > 0
                    log.info(f"Model {model_cfg.name}: validated OK")
                except Exception as e:
                    results[model_cfg.name] = False
                    log.error(f"Model {model_cfg.name}: validation failed — {e}")
        finally:
            await manager.close()
        return results

    async def full_setup(self) -> bool:
        """Run the complete setup sequence. Returns True if everything is ready."""
        if not await self.ensure_ollama_installed():
            return False

        if not await self.ensure_ollama_running():
            return False

        await self.pull_required_models()

        results = await self.validate_models()
        all_ok = all(results.values())
        if all_ok:
            log.info("All models validated successfully")
        else:
            failed = [name for name, ok in results.items() if not ok]
            log.error(f"Some models failed validation: {failed}")
        return all_ok
