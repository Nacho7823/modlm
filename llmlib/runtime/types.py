"""Runtime type definitions for llmlib orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for LLM clients."""

    api_url: str
    api_key: str
    model: str
