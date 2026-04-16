"""High-level runtime facade for llmapp integration."""

from __future__ import annotations

from typing import Any, AsyncIterator

from .chat_orchestrator import ChatOrchestrator
from .llm_runtime import LLMRuntime
from .mcp_registry import MCPRegistry
from .types import ChatMessage, LLMSettings, StreamEvent


class ChatRuntime:
    """Facade that owns LLM runtime + MCP registry."""

    def __init__(self) -> None:
        self._llm = LLMRuntime()
        self._mcp = MCPRegistry()

    def configure(
        self, settings: LLMSettings, mcp_servers: dict[str, str]
    ) -> list[str]:
        """Configure LLM clients and MCP servers.

        Returns warning messages for MCP servers that fail to initialize.
        """
        self._llm.configure(settings)
        return self._mcp.load_servers(mcp_servers)

    def list_mcp_servers(self) -> dict[str, str]:
        return self._mcp.list_servers()

    def add_mcp_server(self, name: str, url: str) -> str:
        return self._mcp.add_server(name, url)

    def remove_mcp_server(self, name: str) -> bool:
        return self._mcp.remove_server(name)

    async def stream(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
        streaming_enabled: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream response using configured LLM + MCP clients."""
        client_async = self._llm.client_async
        if client_async is None:
            raise ValueError("LLM runtime is not configured")

        runtime_messages = self._normalize_messages(messages)

        orchestrator = ChatOrchestrator(
            client_async=client_async,
            mcp_clients=self._mcp.build_clients(),
        )
        try:
            async for event in orchestrator.stream(runtime_messages, streaming_enabled):
                yield event
        except Exception as error:
            yield StreamEvent.error(f"\nError: {error}")
            yield StreamEvent.done()

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]] | list[ChatMessage],
    ) -> list[ChatMessage]:
        if not messages:
            return []
        first = messages[0]
        if isinstance(first, ChatMessage):
            return list(messages)
        return [ChatMessage.from_dict(message) for message in messages]
