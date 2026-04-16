"""MCP client implementations."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llmlib.models import MCPToolError, MCPConnectionError, Tool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_CONNECT_TIMEOUT = 30


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
    async def _close_transport(self) -> None:
        """Close the underlying transport."""
        pass

    async def connect(self, connect_timeout: int = DEFAULT_CONNECT_TIMEOUT) -> bool:
        """Establish connection to the MCP server."""
        try:
            read_stream, write_stream = await self._create_transport()
            await self._open_session(read_stream, write_stream, connect_timeout)
            self._connected = True
            logger.info(f"Connected to MCP server: {self.name}")
            return True
        except asyncio.TimeoutError:
            message = f"Timeout connecting to {self.name} after {connect_timeout}s"
            logger.error(message)
            await self._cleanup()
            raise MCPConnectionError(message)
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            await self._cleanup()
            raise MCPConnectionError(f"Failed to connect to {self.name}: {e}") from e

    async def _open_session(
        self,
        read_stream: Any,
        write_stream: Any,
        connect_timeout: int,
    ) -> None:
        self._client_session = ClientSession(read_stream, write_stream)
        await self._client_session.__aenter__()
        await asyncio.wait_for(
            self._client_session.initialize(), timeout=connect_timeout
        )

    async def _cleanup(self) -> None:
        """Clean up resources on error."""
        try:
            await self._close_session()
            await self._close_transport()
        except Exception as e:
            logger.warning(f"Error during session cleanup: {e}")
        finally:
            self._connected = False

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        try:
            await self._close_session()
            await self._close_transport()
        except Exception as e:
            logger.warning(f"Error disconnecting from {self.name}: {e}")
        finally:
            self._connected = False

    async def _close_session(self) -> None:
        if not self._client_session:
            return
        try:
            await self._client_session.__aexit__(None, None, None)
        except RuntimeError as e:
            if "cancel scope" in str(e):
                logger.debug(f"Task mismatch during _close_session for {self.name} (ignoring anyio error)")
            else:
                logger.warning(f"Error closing session for {self.name}: {e}")
        except Exception as e:
            logger.warning(f"Error closing session for {self.name}: {e}")
        finally:
            self._client_session = None

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

    async def list_tools(self) -> list[Tool]:
        """List available tools from the server."""
        try:
            if not self._connected:
                await self.connect()
            if not self._client_session:
                raise MCPConnectionError(f"No active session for {self.name}")
                
            result = await self._client_session.list_tools()
        except (MCPConnectionError, MCPToolError):
            raise
        except Exception as e:
            logger.error(f"list_tools failed for {self.name}: {e}")
            raise MCPToolError(f"Failed to list tools: {e}") from e

        tools = []
        for tool in result.tools:
            tools.append(Tool.from_mcp_raw(self.name, tool))
        return tools

    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> ToolResult:
        """Call a tool on the server."""
        try:
            if not self._connected:
                await self.connect()
            if not self._client_session:
                raise MCPConnectionError(f"No active session for {self.name}")
                
            result = await self._client_session.call_tool(tool_name, arguments)
            is_error = bool(getattr(result, "isError", False))
            return ToolResult(tool_call_id="", content=result.content or [], is_error=is_error)
        except (MCPConnectionError, MCPToolError):
            raise
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return ToolResult(tool_call_id="", content=[str(e)], is_error=True)


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

    async def _close_transport(self) -> None:
        """Close stdio transport."""
        if self._stdio_context:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope" in str(e):
                    logger.debug(f"Task mismatch during _close_transport for {self.name} (ignoring anyio error)")
                else:
                    logger.warning(f"Error closing transport for {self.name}: {e}")
            except Exception as e:
                logger.warning(f"Error closing transport for {self.name}: {e}")
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
        self._http_client: httpx.AsyncClient | None = None

    async def _create_transport(self) -> tuple[Any, Any]:
        """Create HTTP transport."""
        from mcp.client.streamable_http import streamable_http_client

        logger.info(f"Starting HTTP client for {self.name} at {self.url}...")

        self._http_client = httpx.AsyncClient(headers=self.headers)

        self._http_context = streamable_http_client(
            self.url,
            http_client=self._http_client,
        )
        read, write, _ = await self._http_context.__aenter__()
        logger.info(f"HTTP transport ready for {self.name}")
        return read, write

    async def _close_transport(self) -> None:
        """Close HTTP transport."""
        if self._http_context:
            try:
                await self._http_context.__aexit__(None, None, None)
            except RuntimeError as e:
                if "cancel scope" in str(e):
                    logger.debug(f"Task mismatch during _close_transport for {self.name} (ignoring anyio error)")
                else:
                    logger.warning(f"Error closing transport for {self.name}: {e}")
            except Exception as e:
                logger.warning(f"Error closing transport for {self.name}: {e}")
            self._http_context = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


RemoteMCPClient = HTTPMCPClient
