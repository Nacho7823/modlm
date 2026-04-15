"""MCP models: tool schemas and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MCPError(Exception):
    """Base exception for MCP operations."""

    def __init__(self, message: str, details: Any = None):
        super().__init__(message)
        self.details = details


class MCPConnectionError(MCPError):
    """Raised when MCP connection fails."""

    pass


class MCPToolError(MCPError):
    """Raised when MCP tool call fails."""

    pass


@dataclass
class ToolSchema:
    """Represents an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""

    content: list[Any] = field(default_factory=list)
    is_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        text_content = []
        for item in self.content:
            if hasattr(item, "text"):
                text_content.append({"type": "text", "text": item.text})
            elif hasattr(item, "data"):
                text_content.append({"type": "resource", "data": item.data})
            elif isinstance(item, str):
                text_content.append({"type": "text", "text": item})
            else:
                text_content.append({"type": "text", "text": str(item)})

        return {"content": text_content, "isError": self.is_error}
