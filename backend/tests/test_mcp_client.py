"""Tests for MCP client implementations."""

import asyncio
import logging
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from llmlib import (
    HTTPMCPClient,
    LocalMCPClient,
    MCPClientBase,
    MCPToolResult,
    RemoteMCPClient,
    ToolSchema,
    create_mcp_client,
    create_mcp_client_from_config,
)
from llmlib.mcp import MCPConnectionError


class TestLocalMCPClientCreation:
    """Tests for creating LocalMCPClient instances."""

    def test_create_with_command(self):
        """Test basic creation with command list."""
        client = LocalMCPClient(
            name="test-server",
            command=["echo", "hello"],
            timeout=30,
        )
        assert client.name == "test-server"
        assert client.command == ["echo", "hello"]
        assert client.timeout == 30
        assert client.env == {}

    def test_create_with_env(self):
        """Test creation with environment variables."""
        client = LocalMCPClient(
            name="test-server",
            command=["python", "server.py"],
            env={"DEBUG": "true"},
        )
        assert client.env == {"DEBUG": "true"}

    def test_not_connected_by_default(self):
        """Test that client is not connected by default."""
        client = LocalMCPClient(
            name="test-server",
            command=["echo", "test"],
        )
        assert client.is_connected is False


class TestHTTPMCPClientCreation:
    """Tests for creating HTTPMCPClient instances."""

    def test_create_with_url(self):
        """Test basic creation with URL."""
        client = HTTPMCPClient(
            name="test-server",
            url="http://localhost:3000/mcp",
            timeout=30,
        )
        assert client.name == "test-server"
        assert client.url == "http://localhost:3000/mcp"
        assert client.timeout == 30
        assert client.headers == {}

    def test_create_with_headers(self):
        """Test creation with HTTP headers."""
        client = HTTPMCPClient(
            name="test-server",
            url="http://localhost:3000/mcp",
            headers={"Authorization": "Bearer token"},
        )
        assert client.headers == {"Authorization": "Bearer token"}

    def test_not_connected_by_default(self):
        """Test that client is not connected by default."""
        client = HTTPMCPClient(
            name="test-server",
            url="http://localhost:3000/mcp",
        )
        assert client.is_connected is False

    def test_remote_mcp_client_is_httpmcp_client(self):
        """Test that RemoteMCPClient is alias for HTTPMCPClient."""
        assert RemoteMCPClient is HTTPMCPClient


class TestCreateMCPClientFactory:
    """Tests for the create_mcp_client factory function."""

    def test_create_local_client(self):
        """Test creating local client via factory."""
        client = create_mcp_client(
            name="test",
            server_type="local",
            command=["echo", "test"],
        )
        assert isinstance(client, LocalMCPClient)
        assert client.name == "test"

    def test_create_stdio_client(self):
        """Test creating stdio client via factory."""
        client = create_mcp_client(
            name="test",
            server_type="stdio",
            command=["echo", "test"],
        )
        assert isinstance(client, LocalMCPClient)

    def test_create_http_client(self):
        """Test creating HTTP client via factory."""
        client = create_mcp_client(
            name="test",
            server_type="http",
            url="http://localhost:3000/mcp",
        )
        assert isinstance(client, HTTPMCPClient)
        assert client.name == "test"

    def test_create_remote_client(self):
        """Test creating remote client via factory."""
        client = create_mcp_client(
            name="test",
            server_type="remote",
            url="http://localhost:3000/mcp",
        )
        assert isinstance(client, HTTPMCPClient)

    def test_local_requires_command(self):
        """Test that local client requires command."""
        with pytest.raises(MCPConnectionError, match="command is required"):
            create_mcp_client(
                name="test",
                server_type="local",
            )

    def test_http_requires_url(self):
        """Test that HTTP client requires URL."""
        with pytest.raises(MCPConnectionError, match="url is required"):
            create_mcp_client(
                name="test",
                server_type="http",
            )

    def test_invalid_server_type(self):
        """Test that invalid server type raises error."""
        with pytest.raises(MCPConnectionError, match="Unknown MCP server type"):
            create_mcp_client(
                name="test",
                server_type="invalid",
            )


class TestCreateMCPClientFromConfig:
    """Tests for creating MCP client from config dictionary."""

    def test_from_local_config(self):
        """Test creating client from local config."""
        config = {
            "name": "my-server",
            "type": "local",
            "command": ["echo", "test"],
            "timeout": 60,
        }
        client = create_mcp_client_from_config(config)
        assert client.name == "my-server"
        assert client.timeout == 60

    def test_from_http_config(self):
        """Test creating client from HTTP config."""
        config = {
            "name": "my-server",
            "type": "http",
            "url": "http://localhost:3000/mcp",
            "headers": {"Authorization": "Bearer token"},
        }
        client = create_mcp_client_from_config(config)
        assert isinstance(client, HTTPMCPClient)
        assert client.headers == {"Authorization": "Bearer token"}

    def test_defaults(self):
        """Test that defaults are applied."""
        config = {"name": "test", "type": "local", "command": ["echo"]}
        client = create_mcp_client_from_config(config)
        assert client.timeout == 30  # default


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_creation(self):
        """Test ToolSchema creation."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {}},
        )
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"

    def test_to_dict(self):
        """Test to_dict method."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
        )
        d = schema.to_dict()
        assert d["name"] == "test_tool"
        assert d["description"] == "A test tool"
        assert d["inputSchema"] == {"type": "object"}


class TestMCPToolResult:
    """Tests for MCPToolResult."""

    def test_creation(self):
        """Test MCPToolResult creation."""
        result = MCPToolResult(content=["result"], is_error=False)
        assert result.content == ["result"]
        assert result.is_error is False

    def test_to_dict_text_content(self):
        """Test to_dict with text content."""
        result = MCPToolResult(content=["hello"])
        d = result.to_dict()
        assert d["content"] == [{"type": "text", "text": "hello"}]
        assert d["isError"] is False

    def test_to_dict_with_error(self):
        """Test to_dict with error."""
        result = MCPToolResult(content=["error"], is_error=True)
        d = result.to_dict()
        assert d["isError"] is True

    @pytest.mark.asyncio
    async def test_connect_with_mock(self):
        """Test connection with mocked session."""
        client = LocalMCPClient(
            name="mock-test",
            command=["echo", "test"],
        )

        # Mocking the stdio_client context manager and its returns
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        mock_read = AsyncMock()
        mock_write = AsyncMock()

        mock_stdio = MagicMock()
        mock_stdio.__aenter__ = AsyncMock(return_value=(mock_read, mock_write))
        mock_stdio.__aexit__ = AsyncMock()

        with (
            patch("llmlib.mcp.client.stdio_client", return_value=mock_stdio),
            patch("llmlib.mcp.client.ClientSession", return_value=mock_session),
        ):
            connected = await client.connect(connect_timeout=1)
            assert connected is True
            assert client.is_connected is True
            mock_session.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_tools_with_mock(self):
        """Test listing tools with mocked session."""
        client = LocalMCPClient(
            name="mock-test",
            command=["echo", "test"],
        )

        mock_session = AsyncMock()
        mock_list_result = MagicMock()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.input_schema = {"type": "object"}
        mock_list_result.tools = [mock_tool]
        mock_session.list_tools = AsyncMock(return_value=mock_list_result)

        client._client_session = mock_session
        client._connected = True

        tools = await client.list_tools()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        mock_session.list_tools.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_call_tool_with_mock(self):
        """Test calling a tool with mocked session."""
        client = LocalMCPClient(
            name="mock-test",
            command=["echo", "test"],
        )

        mock_session = AsyncMock()
        mock_call_result = MagicMock()
        mock_call_result.content = ["result"]
        mock_call_result.isError = False
        mock_session.call_tool = AsyncMock(return_value=mock_call_result)

        client._client_session = mock_session
        client._connected = True

        result = await client.call_tool("test_tool", {"arg": 1})
        assert result.content == ["result"]
        assert result.is_error is False
        mock_session.call_tool.assert_awaited_once_with("test_tool", {"arg": 1})


@pytest.mark.integration
class TestLocalMCPClientIntegration:
    """Integration tests for LocalMCPClient with echo server."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect_echo(self):
        """Test connection to echo command."""
        client = LocalMCPClient(
            name="echo-test",
            command=["echo", "test"],
        )
        try:
            connected = await client.connect(connect_timeout=5)
            assert connected is True
            assert client.is_connected is True
        finally:
            await client.disconnect()
            assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_list_tools_echo(self):
        """Test listing tools from echo server."""
        client = LocalMCPClient(
            name="echo-test",
            command=["echo", "test"],
        )
        try:
            await client.connect(connect_timeout=5)
            tools = await client.list_tools()
            assert isinstance(tools, list)
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """Test connection timeout."""
        client = LocalMCPClient(
            name="timeout-test",
            command=["sleep", "100"],
        )
        connected = await client.connect(connect_timeout=1)
        assert connected is False
        assert client.is_connected is False


@pytest.mark.integration
class TestHTTPMCPClientIntegration:
    """Integration tests for HTTPMCPClient."""

    @pytest.mark.asyncio
    async def test_connect_to_localhost(self):
        """Test connection to localhost HTTP server."""
        client = HTTPMCPClient(
            name="http-test",
            url="http://localhost:9999/mcp",
        )
        connected = await client.connect(connect_timeout=2)
        # Server not running, should fail
        assert connected is False
        assert client.is_connected is False
