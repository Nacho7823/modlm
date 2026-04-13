from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llmlib import OpenAI as LocalOpenAI

from ..core.config import settings
from ..core.exceptions import LLMException
from ..domain.schemas import LLMProviderConfigSchema
from .protocols import LLMProviderProtocol

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self._providers: dict[str, LLMProviderProtocol] = {}
        self._active_provider: str | None = None

    def register_provider(
        self,
        name: str,
        config: LLMProviderConfigSchema,
    ) -> None:
        provider = OpenAIProvider(config)
        self._providers[name] = provider
        if self._active_provider is None:
            self._active_provider = name
        logger.info(f"Registered LLM provider: {name}")

    def set_active_provider(self, name: str) -> None:
        if name not in self._providers:
            raise LLMException(
                provider=name,
                reason="Provider not registered",
            )
        self._active_provider = name
        logger.info(f"Active provider set to: {name}")

    def get_active_provider(self) -> str | None:
        return self._active_provider

    def get_provider(self, name: str) -> LLMProviderProtocol | None:
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def generate(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        if not self._active_provider:
            raise LLMException(
                provider="none",
                reason="No active provider configured",
            )
        provider = self._providers.get(self._active_provider)
        if not provider:
            raise LLMException(
                provider=self._active_provider,
                reason="Provider not found",
            )
        return provider.generate(messages, **kwargs)


class OpenAIProvider(LLMProviderProtocol):
    def __init__(self, config: LLMProviderConfigSchema):
        self._config = config
        self._client = LocalOpenAI(
            api_key=config.api_key or None,
            base_url=config.base_url or None,
        )

    def generate(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=list(messages),
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LLMException(
                provider=self._config.provider,
                reason=str(e),
            )

    def get_model_name(self) -> str:
        return self._config.model

    def is_available(self) -> bool:
        try:
            self._client.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False


llm_service = LLMService()


def _register_default_provider() -> None:
    api_key = settings.openai_api_key.strip()

    default_config = LLMProviderConfigSchema(
        provider="openai",
        model=settings.default_model,
        api_key=api_key,
        base_url=settings.openai_api_base,
        temperature=settings.default_temperature,
        max_tokens=settings.default_max_tokens,
    )
    llm_service.register_provider("openai", default_config)


_register_default_provider()
