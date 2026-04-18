"""Model manager — supports Ollama (local) and OpenAI-compatible APIs (Groq, Liquid, etc.)."""

from __future__ import annotations

import json as _json
import os
import time

import httpx

from pmca.models.config import AgentRole, Config, ModelConfig
from pmca.utils.logger import get_logger

log = get_logger("models.manager")
_telemetry_log = get_logger("telemetry")

# Default API base URLs per provider
_PROVIDER_DEFAULTS: dict[str, str] = {
    "ollama": "http://localhost:11434",
    "groq": "https://api.groq.com/openai",
    "openai": "https://api.openai.com",
    "liquid": "https://api.liquid.ai",
}

# Environment variable names for API keys per provider
_API_KEY_ENV: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "liquid": "LIQUID_API_KEY",
}


def _is_openai_provider(provider: str) -> bool:
    """Check if provider uses the OpenAI-compatible chat completions API."""
    return provider in ("groq", "openai", "liquid")


class ModelManager:
    """Manages models across Ollama and OpenAI-compatible providers."""

    def __init__(self, config: Config, ollama_host: str = "") -> None:
        self._config = config
        # Resolve Ollama host: explicit arg > OLLAMA_HOST env > default
        self._ollama_host = (
            ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        ).rstrip("/")
        self._current_model: str | None = None
        # Ollama client (used for local models)
        self._ollama_client = httpx.AsyncClient(base_url=self._ollama_host, timeout=1800.0)
        # Separate clients for OpenAI-compatible providers (keyed by base_url)
        self._openai_clients: dict[str, httpx.AsyncClient] = {}
        # Telemetry aggregation
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_llm_calls: int = 0
        self.total_llm_duration_ms: float = 0.0

    def _get_api_base(self, model_cfg: ModelConfig) -> str:
        """Resolve the API base URL for a model config."""
        if model_cfg.api_base:
            return model_cfg.api_base.rstrip("/")
        return _PROVIDER_DEFAULTS.get(model_cfg.provider, self._ollama_host)

    def _get_api_key(self, model_cfg: ModelConfig) -> str | None:
        """Get API key from environment for the provider."""
        env_var = _API_KEY_ENV.get(model_cfg.provider)
        if env_var:
            key = os.environ.get(env_var)
            if not key:
                raise RuntimeError(
                    f"API key not found. Set {env_var} environment variable "
                    f"for provider '{model_cfg.provider}'"
                )
            return key
        return None

    def _get_openai_client(self, model_cfg: ModelConfig) -> httpx.AsyncClient:
        """Get or create an httpx client for an OpenAI-compatible provider."""
        base_url = self._get_api_base(model_cfg)
        if base_url not in self._openai_clients:
            api_key = self._get_api_key(model_cfg)
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            self._openai_clients[base_url] = httpx.AsyncClient(
                base_url=base_url,
                headers=headers,
                timeout=600.0,
            )
        return self._openai_clients[base_url]

    async def close(self) -> None:
        await self._ollama_client.aclose()
        for client in self._openai_clients.values():
            await client.aclose()
        self._openai_clients.clear()

    async def ensure_loaded(self, role: AgentRole) -> str:
        """Load the model for a role. For Ollama, unloads current if different."""
        model_cfg = self._config.get_model(role)
        model_name = model_cfg.name

        # OpenAI-compatible providers don't need local loading
        if _is_openai_provider(model_cfg.provider):
            # Validate API key is available early
            self._get_api_key(model_cfg)
            log.info(f"Using [bold]{model_name}[/bold] via {model_cfg.provider} for {role.value}")
            return model_name

        # Ollama: load/unload as before
        if self._current_model == model_name:
            log.debug(f"Model {model_name} already loaded")
            return model_name

        if self._current_model is not None:
            await self.unload_current()

        log.info(f"Loading model [bold]{model_name}[/bold] for role {role.value}")

        try:
            resp = await self._ollama_client.post(
                "/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Hello",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            resp.raise_for_status()
            self._current_model = model_name
            log.info(f"Model {model_name} loaded successfully")
        except httpx.HTTPError as e:
            log.error(f"Failed to load model {model_name}: {e}")
            raise RuntimeError(f"Failed to load model {model_name}: {e}") from e

        return model_name

    async def unload_current(self) -> None:
        """Unload current Ollama model to free VRAM."""
        if self._current_model is None:
            return

        log.info(f"Unloading model {self._current_model}")
        try:
            await self._ollama_client.post(
                "/api/generate",
                json={
                    "model": self._current_model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
            )
        except httpx.HTTPError:
            log.warning(f"Failed to cleanly unload {self._current_model}")
        self._current_model = None

    async def generate(
        self,
        role: AgentRole,
        prompt: str,
        system: str = "",
        temperature: float | None = None,
        format: dict | None = None,
        think: bool | None = None,
    ) -> str:
        """Generate completion from the model assigned to this role.

        Args:
            format: Optional JSON schema dict for structured output (Ollama only).
                    When provided, the model output is constrained to match the schema.
            think: Control thinking mode for reasoning models (Ollama only).
                   True enables chain-of-thought, False disables it, None uses model default.
        """
        model_cfg = self._config.get_model(role)
        await self.ensure_loaded(role)

        if _is_openai_provider(model_cfg.provider):
            return await self._generate_openai(model_cfg, prompt, system, temperature)
        return await self._generate_ollama(model_cfg, role, prompt, system, temperature, format=format, think=think)

    async def _generate_ollama(
        self,
        model_cfg: ModelConfig,
        role: AgentRole,
        prompt: str,
        system: str,
        temperature: float | None,
        format: dict | None = None,
        think: bool | None = None,
    ) -> str:
        """Generate via Ollama's /api/generate endpoint."""
        temp = temperature if temperature is not None else model_cfg.temperature
        options: dict = {
            "temperature": temp,
            "num_ctx": model_cfg.context_window,
        }
        if model_cfg.max_tokens is not None:
            options["num_predict"] = model_cfg.max_tokens
        payload: dict = {
            "model": model_cfg.name,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system
        if format is not None:
            payload["format"] = format
        # Model-level think (from config YAML) overrides call-site default
        effective_think = model_cfg.think if model_cfg.think is not None else think
        if effective_think is not None:
            payload["think"] = effective_think

        log.debug(f"Generating with {model_cfg.name} (role={role.value})")

        try:
            t0 = time.monotonic()
            resp = await self._ollama_client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            duration_ms = (time.monotonic() - t0) * 1000
            response_text = data.get("response", "")

            # Capture Ollama response metadata for telemetry
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_llm_calls += 1
            self.total_llm_duration_ms += duration_ms

            _telemetry_log.info(_json.dumps({
                "event": "llm_generate",
                "agent": role.value,
                "model": model_cfg.name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_ms": round(duration_ms, 1),
                "eval_duration_ms": round(data.get("eval_duration", 0) / 1e6, 1),
                "temperature": temp,
            }))

            log.debug(
                f"Generated {len(response_text)} chars, "
                f"tokens={prompt_tokens}+{completion_tokens}, "
                f"duration={duration_ms:.0f}ms"
            )
            return response_text
        except httpx.HTTPError as e:
            log.error(f"Generation failed for {model_cfg.name}: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    async def _generate_openai(
        self,
        model_cfg: ModelConfig,
        prompt: str,
        system: str,
        temperature: float | None,
    ) -> str:
        """Generate via OpenAI-compatible /v1/chat/completions endpoint."""
        client = self._get_openai_client(model_cfg)
        temp = temperature if temperature is not None else model_cfg.temperature

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_cfg.name,
            "messages": messages,
            "temperature": temp,
            "max_tokens": model_cfg.context_window,
        }

        log.debug(f"Generating with {model_cfg.name} via {model_cfg.provider}")

        try:
            t0 = time.monotonic()
            resp = await client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
            duration_ms = (time.monotonic() - t0) * 1000
            choices = data.get("choices", [])
            if not choices:
                log.warning(f"No choices returned from {model_cfg.provider}")
                return ""
            response_text = choices[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_llm_calls += 1
            self.total_llm_duration_ms += duration_ms

            _telemetry_log.info(_json.dumps({
                "event": "llm_generate",
                "provider": model_cfg.provider,
                "model": model_cfg.name,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "duration_ms": round(duration_ms, 1),
                "temperature": temp,
            }))

            log.debug(
                f"Generated {len(response_text)} chars via {model_cfg.provider}, "
                f"tokens={prompt_tokens}+{completion_tokens}, "
                f"duration={duration_ms:.0f}ms"
            )
            return response_text
        except httpx.HTTPError as e:
            log.error(f"Generation failed for {model_cfg.name} via {model_cfg.provider}: {e}")
            raise RuntimeError(f"Generation failed ({model_cfg.provider}): {e}") from e

    async def check_available(self) -> dict[str, bool]:
        """Check which required models are available."""
        # Check Ollama models
        ollama_models: set[str] = set()
        try:
            resp = await self._ollama_client.get("/api/tags")
            resp.raise_for_status()
            ollama_models = {m["name"] for m in resp.json().get("models", [])}
        except httpx.HTTPError as e:
            log.warning(f"Cannot reach Ollama: {e}")

        result: dict[str, bool] = {}
        for model_cfg in self._config.models.values():
            if _is_openai_provider(model_cfg.provider):
                # Cloud models are always "available" if API key is set
                env_var = _API_KEY_ENV.get(model_cfg.provider, "")
                result[model_cfg.name] = bool(os.environ.get(env_var))
            else:
                result[model_cfg.name] = model_cfg.name in ollama_models
        return result

    async def pull_model(self, model_name: str) -> None:
        """Pull a model from Ollama registry."""
        log.info(f"Pulling model {model_name}...")
        try:
            async with self._ollama_client.stream(
                "POST",
                "/api/pull",
                json={"name": model_name, "stream": True},
                timeout=None,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.strip():
                        data = _json.loads(line)
                        status = data.get("status", "")
                        if "completed" in data and "total" in data:
                            pct = int(data["completed"] / data["total"] * 100)
                            log.info(f"  {status}: {pct}%")
                        elif status:
                            log.info(f"  {status}")
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to pull {model_name}: {e}") from e

    async def pull_missing(self) -> None:
        """Pull any missing Ollama models."""
        available = await self.check_available()
        for model_name, is_available in available.items():
            if not is_available:
                # Only pull Ollama models, not cloud ones
                for cfg in self._config.models.values():
                    if cfg.name == model_name and cfg.provider == "ollama":
                        await self.pull_model(model_name)
                        break

    async def is_ollama_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            resp = await self._ollama_client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
