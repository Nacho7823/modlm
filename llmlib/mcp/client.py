"""MCP client implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .models import MCPToolError, MCPConnectionError, ToolSchema, MCPToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_CONNECT_TIMEOUT = 10


class MCPClientBase(ABC):
    """Base class for MCP clients."""

    def __init__(self, name: str, timeout: int = DEFAULT_TIMEOUT):
        self.name = name
        self.timeout = timeout
        self._client_session: Optional[ClientSession] = None
        self._connected = False

    @abstractmethod
    async def _create_transport(self) -> tuple[Any, Any]:
        """Create transport (read, write) streams."""
        pass

    @abstractmethod
    async def _close_transport(self, transport: Any) -> None:
        """Close a specific transport."""
        pass

    async def connect(self, connect_timeout: int = DEFAULT_CONNECT_TIMEOUT) -> bool:
        """Establish connection to the MCP server."""
        try:
            read_stream, write_stream = await self._create_transport()

            self._client_session = ClientSession(read_stream, write_stream)
            await self._client_session.__aenter__()

            await asyncio.wait_for(
                self._client_session.initialize(), timeout=connect_timeout
            )
            self._connected = True
            logger.info(f"Connected to MCP server: {self.name}")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to {self.name} after {connect_timeout}s")
            await self._cleanup()
            raise MCPConnectionError(
                f"Timeout connecting to {self.name} after {connect_timeout}s"
            )
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            await self._cleanup()
            raise MCPConnectionError(f"Failed to connect to {self.name}: {e}") from e

    async def _cleanup(self) -> None:
        """Clean up resources on error."""
        try:
            if self._client_session:
                try:
                    await self._client_session.__aexit__(None, None, None)
                except Exception:
                    pass
                self._client_session = None
        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")
        finally:
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            if self._client_session:
                await self._client_session.__aexit__(None, None, None)
                self._client_session = None
        except Exception as e:
            logger.warning(f"Error disconnecting from {self.name}: {e}")
        finally:
            self._connected = False

    async def test_connection(self) -> dict[str, Any]:
        """Test the connection and return status."""
        try:
            if not self._connected:
                await self.connect()

            tools = await self.list_tools()
            return {
                "status": "connected",
                "tools": [t.name for t in tools],
                "tool_count": len(tools),
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "tools": [], "tool_count": 0}

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def list_tools(self) -> list[ToolSchema]:
        """List available tools from the server."""
        if not self._connected:
            raise MCPConnectionError(f"Not connected to MCP server: {self.name}")
        if not self._client_session:
            raise MCPConnectionError(f"No session for MCP server: {self.name}")

        logger.info(f"Listing tools from {self.name}...")

        try:
            result = await self._client_session.list_tools()
        except Exception as e:
            logger.error(f"list_tools failed for {self.name}: {e}")
            raise MCPToolError(f"Failed to list tools: {e}") from e

        tools = []
        for tool in result.tools:
            input_schema = {}
            if hasattr(tool, "inputSchema"):
                input_schema = tool.inputSchema
            elif hasattr(tool, "input_schema"):
                input_schema = tool.input_schema
            elif hasattr(tool, "parameters"):
                input_schema = tool.parameters

            tools.append(
                ToolSchema(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=input_schema,
                )
            )

        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """Call a tool on the server."""
        if not self._connected or not self._client_session:
            raise MCPConnectionError("Not connected to MCP server")

        try:
            result = await self._client_session.call_tool(tool_name, arguments)

            is_error = False
            if hasattr(result, "isError"):
                is_error = result.isError

            return MCPToolResult(content=result.content or [], is_error=is_error)
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return MCPToolResult(content=[str(e)], is_error=True)

    async def __aenter__(self) -> "MCPClientBase":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.disconnect()


class LocalMCPClient(MCPClientBase):
    """MCP client for local (stdio) servers."""

    def __init__(
        self,
        name: str,
        command: list[str],
        timeout: int = DEFAULT_TIMEOUT,
        env: Optional[dict[str, str]] = None,
    ):
        super().__init__(name, timeout)
        self.command = command
        self.env = env or {}
        self._stdio_context = None

    async def _create_transport(self) -> tuple[Any, Any]:
        """Create stdio transport."""
        server_params = StdioServerParameters(
            command=self.command[0],
            args=self.command[1:],
            env=self.env.copy() if self.env else None,
        )

        logger.info(f"Starting stdio client for {self.name}...")
        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()
        logger.info(f"Stdio transport ready for {self.name}")
        return read, write

    async def _close_transport(self, transport: Any) -> None:
        """Close stdio transport."""
        if self._stdio_context:
            await self._stdio_context.__aexit__(None, None, None)
            self._stdio_context = None


class HTTPMCPClient(MCPClientBase):
    """MCP client for remote (HTTP/SSE) servers."""

    def __init__(
        self,
        name: str,
        url: str,
        timeout: int = DEFAULT_TIMEOUT,
        headers: Optional[dict[str, str]] = None,
    ):
        super().__init__(name, timeout)
        self.url = url
        self.headers = headers or {}
        self._http_context = None

    async def _create_transport(self) -> tuple[Any, Any]:
        """Create HTTP transport."""
        from mcp.client.streamable_http import streamablehttp_client

        logger.info(f"Starting HTTP client for {self.name} at {self.url}...")

        self._http_context = streamablehttp_client(
            self.url,
            headers=self.headers,
        )
        read, write, _ = await self._http_context.__aenter__()
        logger.info(f"HTTP transport ready for {self.name}")
        return read, write

    async def _close_transport(self, transport: Any) -> None:
        """Close HTTP transport."""
        if self._http_context:
            await self._http_context.__aexit__(None, None, None)
            self._http_context = None


RemoteMCPClient = HTTPMCPClient
