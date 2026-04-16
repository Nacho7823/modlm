"""Lightweight OpenAI-compatible client and MCP utilities."""

from .llm import (
    Chat,
    Completions,
    OpenAI,
    ChatOrchestrator,
    LLMRuntime,
)
from .mcp import (
    HTTPMCPClient,
    MCPClientBase,
    LocalMCPClient,
    RemoteMCPClient,
    create_mcp_client,
    create_mcp_client_from_config,
    ToolExecutor,
    MCPRegistry,
)
from .chat_runtime import ChatRuntime
from llmlib.models import (
    ChatCompletion,
    Choice,
    Message,
    ToolCall,
    ToolResult,
    LLMSettings,
    MCPServerConfig,
    StreamEvent,
    MCPConnectionError,
    MCPToolError,
)

__all__ = [
    "OpenAI",
    "Chat",
    "ChatCompletion",
    "Choice",
    "Completions",
    "Message",
    "Tool",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "MCPClientBase",
    "LocalMCPClient",
    "HTTPMCPClient",
    "RemoteMCPClient",
    "MCPConnectionError",
    "MCPToolError",
    "create_mcp_client",
    "create_mcp_client_from_config",
    "ChatOrchestrator",
    "ChatRuntime",
    "LLMRuntime",
    "MCPRegistry",
    "LLMSettings",
    "MCPServerConfig",
    "StreamEvent",
]
