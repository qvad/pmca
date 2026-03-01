"""Tests for the model manager with mocked Ollama responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pmca.models.config import AgentRole, Config, ModelConfig
from pmca.models.manager import ModelManager


@pytest.fixture
def config():
    return Config(
        models={
            AgentRole.ARCHITECT: ModelConfig(
                name="test-model:14b", temperature=0.3
            ),
            AgentRole.CODER: ModelConfig(
                name="test-model:7b", temperature=0.2
            ),
            AgentRole.REVIEWER: ModelConfig(
                name="test-model:14b", temperature=0.1
            ),
            AgentRole.WATCHER: ModelConfig(
                name="test-model:7b", temperature=0.1
            ),
        }
    )


@pytest.fixture
def manager(config):
    return ModelManager(config, ollama_host="http://localhost:11434")


class TestModelManager:
    @pytest.mark.asyncio
    async def test_is_ollama_running_success(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(manager._ollama_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            assert await manager.is_ollama_running()

    @pytest.mark.asyncio
    async def test_is_ollama_running_failure(self, manager):
        with patch.object(manager._ollama_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            assert not await manager.is_ollama_running()

    @pytest.mark.asyncio
    async def test_check_available(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "test-model:14b"},
            ]
        }

        with patch.object(manager._ollama_client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            available = await manager.check_available()
            assert available["test-model:14b"] is True
            assert available["test-model:7b"] is False

    @pytest.mark.asyncio
    async def test_ensure_loaded(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "OK"}

        with patch.object(manager._ollama_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            name = await manager.ensure_loaded(AgentRole.ARCHITECT)
            assert name == "test-model:14b"
            assert manager._current_model == "test-model:14b"

    @pytest.mark.asyncio
    async def test_ensure_loaded_switches_model(self, manager):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"response": "OK"}

        with patch.object(manager._ollama_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await manager.ensure_loaded(AgentRole.ARCHITECT)
            assert manager._current_model == "test-model:14b"

            await manager.ensure_loaded(AgentRole.CODER)
            assert manager._current_model == "test-model:7b"

    @pytest.mark.asyncio
    async def test_generate(self, manager):
        mock_warmup = MagicMock()
        mock_warmup.status_code = 200
        mock_warmup.raise_for_status = MagicMock()
        mock_warmup.json.return_value = {"response": "OK"}

        mock_gen = MagicMock()
        mock_gen.status_code = 200
        mock_gen.raise_for_status = MagicMock()
        mock_gen.json.return_value = {
            "response": "Generated text here",
            "eval_count": 42,
        }

        with patch.object(manager._ollama_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [mock_warmup, mock_gen]
            result = await manager.generate(
                AgentRole.ARCHITECT, "Write hello world"
            )
            assert result == "Generated text here"

    @pytest.mark.asyncio
    async def test_unload_current(self, manager):
        manager._current_model = "test-model:14b"
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(manager._ollama_client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await manager.unload_current()
            assert manager._current_model is None

    @pytest.mark.asyncio
    async def test_unload_when_none(self, manager):
        await manager.unload_current()  # Should not raise

    @pytest.mark.asyncio
    async def test_close(self, manager):
        with patch.object(manager._ollama_client, "aclose", new_callable=AsyncMock) as mock_close:
            await manager.close()
            mock_close.assert_called_once()
