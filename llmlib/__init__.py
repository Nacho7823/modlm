"""Lightweight OpenAI-compatible client and MCP utilities."""

from .llm import (
    Chat,
    ChatCompletion,
    Choice,
    Completions,
    Message,
    OpenAI,
    Tool,
    ToolCall,
    ToolExecutor,
    ToolResult,
)
from .mcp import (
    HTTPMCPClient,
    MCPClientBase,
    MCPConnectionError,
    MCPToolError,
    LocalMCPClient,
    MCPToolResult,
    RemoteMCPClient,
    ToolSchema,
    create_mcp_client,
    create_mcp_client_from_config,
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
    "ToolSchema",
    "MCPToolResult",
    "MCPConnectionError",
    "MCPToolError",
    "create_mcp_client",
    "create_mcp_client_from_config",
]
