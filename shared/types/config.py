from dataclasses import dataclass, field
from typing import TypedDict


class LLMProviderConfigDict(TypedDict):
    provider: str
    model: str
    api_key: str
    base_url: str | None
    temperature: float
    max_tokens: int | None


class LLMConfigDict(TypedDict):
    providers: dict[str, LLMProviderConfigDict]
    active_provider: str


@dataclass
class LLMProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None

    def to_dict(self) -> LLMProviderConfigDict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: LLMProviderConfigDict) -> "LLMProviderConfig":
        return cls(
            provider=data["provider"],
            model=data["model"],
            api_key=data["api_key"],
            base_url=data.get("base_url"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens"),
        )


@dataclass
class LLMConfig:
    providers: dict[str, LLMProviderConfig] = field(default_factory=dict)
    active_provider: str = "openai"

    def to_dict(self) -> LLMConfigDict:
        return {
            "providers": {k: v.to_dict() for k, v in self.providers.items()},
            "active_provider": self.active_provider,
        }

    @classmethod
    def from_dict(cls, data: LLMConfigDict) -> "LLMConfig":
        providers = {
            k: LLMProviderConfig.from_dict(v)
            for k, v in data.get("providers", {}).items()
        }
        return cls(
            providers=providers,
            active_provider=data.get("active_provider", "openai"),
        )
