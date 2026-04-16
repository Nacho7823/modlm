"""MCP client implementations and utilities."""

from .client import (
    HTTPMCPClient,
    MCPClientBase,
    LocalMCPClient,
    RemoteMCPClient,
)
from .factory import create_mcp_client, create_mcp_client_from_config
from .executor import ToolExecutor
from .registry import MCPRegistry
from llmlib.models import (
    ChatCompletion,
    Choice,
    Message,
    Tool,
    ToolCall,
    ToolResult,
    MCPError,
    MCPConnectionError,
    MCPToolError,
    MCPServerConfig,
)

# Legacy aliases for backward compatibility in tests
ToolSchema = Tool
MCPToolResult = ToolResult

__all__ = [
    "MCPClientBase",
    "LocalMCPClient",
    "HTTPMCPClient",
    "RemoteMCPClient",
    "ToolSchema",
    "MCPToolResult",
    "MCPConnectionError",
    "MCPToolError",
    "MCPError",
    "create_mcp_client",
    "create_mcp_client_from_config",
    "ChatCompletion",
    "Choice",
    "Message",
    "Tool",
    "ToolCall",
    "ToolResult",
    "ToolExecutor",
    "MCPRegistry",
    "MCPServerConfig",
]
