"""High-level runtime facade for llmapp integration."""

from __future__ import annotations

from typing import Any, AsyncIterator

from .llm.orchestrator import ChatOrchestrator
from .llm.runtime import LLMRuntime
from .mcp.registry import MCPRegistry
from llmlib.models import Message, LLMSettings, MCPServerConfig, StreamEvent


class ChatRuntime:
    """Facade that owns LLM runtime + MCP registry."""

    def __init__(self) -> None:
        self._llm = LLMRuntime()
        self._mcp = MCPRegistry()

    async def configure(
        self, settings: LLMSettings, mcp_servers: dict[str, MCPServerConfig]
    ) -> list[str]:
        """Configure LLM clients and MCP servers.

        Returns warning messages for MCP servers that fail to initialize.
        """
        self._llm.configure(settings)
        return await self._mcp.load_servers(mcp_servers)

    def list_mcp_servers(self) -> dict[str, MCPServerConfig]:
        return self._mcp.list_servers()

    async def add_mcp_server(self, config: MCPServerConfig) -> str:
        return await self._mcp.add_server(config)

    async def remove_mcp_server(self, name: str) -> bool:
        return await self._mcp.remove_server(name)

    async def shutdown(self) -> None:
        """Gracefully shutdown all persistent clients."""
        await self._mcp.shutdown_all()

    async def stream(
        self,
        messages: list[dict[str, Any]] | list[Message],
        streaming_enabled: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream response using configured LLM + MCP clients."""
        client_async = self._llm.client_async
        if client_async is None:
            raise ValueError("LLM runtime is not configured")

        runtime_messages = self._normalize_messages(messages)
        orchestrator = ChatOrchestrator(
            client_async=client_async,
            mcp_clients=self._mcp.get_clients(),
        )
        try:
            async for event in orchestrator.stream(runtime_messages, streaming_enabled):
                yield event
        except Exception as error:
            yield StreamEvent.error(f"\nError: {error}")
            yield StreamEvent.done()

    def _normalize_messages(
        self,
        messages: list[dict[str, Any]] | list[Message],
    ) -> list[Message]:
        if not messages:
            return []
        first = messages[0]
        if isinstance(first, Message):
            return list(messages)
        return [Message.from_dict(message) for message in messages]
