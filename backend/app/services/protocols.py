from abc import ABC, abstractmethod
from typing import Any


class LLMProviderProtocol(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass
