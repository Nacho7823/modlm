"""Integration tests for MCP clients with notepad server."""

from __future__ import annotations

import asyncio
import platform
import socket

import pytest

from llmlib.mcp import LocalMCPClient, HTTPMCPClient, MCPConnectionError

NOTEPAD_HTTP_PORT = 19876


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is open on a host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


async def wait_for_port(host: str, port: int, timeout: float = 30.0) -> bool:
    """Wait for a port to become available."""
    import time

    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(host, port, timeout=1.0):
            return True
        await asyncio.sleep(0.5)
    return False


@pytest.mark.integration
class TestNotepadClientIntegration:
    """Integration tests for LocalMCPClient with notepad server."""

    @pytest.mark.asyncio
    async def test_connect_to_notepad_server(self, notepad_stdio_command):
        """Test connection to notepad MCP server via stdio."""
        client = LocalMCPClient(
            name="notepad-server",
            command=notepad_stdio_command,
        )
        try:
            connected = await client.connect(connect_timeout=30)
            assert connected is True
            assert client.is_connected is True
        finally:
            await client.disconnect()

    @pytest.mark.asyncio
    async def test_list_tools_notepad(self, notepad_client):
        """Test listing tools from notepad server."""
        tools = await notepad_client.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "create_note" in tool_names
        assert "read_note" in tool_names
        assert "list_notes" in tool_names

    @pytest.mark.asyncio
    async def test_create_note(self, notepad_client):
        """Test creating a note."""
        result = await notepad_client.call_tool(
            "create_note", {"name": "test_note", "content": "Hello World"}
        )
        assert result is not None
        assert result.is_error is False
        assert "test_note" in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_note(self, notepad_client):
        """Test reading an existing note."""
        await notepad_client.call_tool(
            "create_note", {"name": "read_test", "content": "Test content"}
        )
        result = await notepad_client.call_tool("read_note", {"name": "read_test"})
        assert result is not None
        assert result.is_error is False
        assert "Test content" in result.content[0].text

    @pytest.mark.asyncio
    async def test_read_nonexistent_note(self, notepad_client):
        """Test reading a note that doesn't exist."""
        result = await notepad_client.call_tool("read_note", {"name": "nonexistent"})
        assert result is not None
        assert "Error" in result.content[0].text

    @pytest.mark.asyncio
    async def test_create_and_read_note(self, notepad_client):
        """Test creating a note and then reading it."""
        await notepad_client.call_tool(
            "create_note", {"name": "workflow_test", "content": "Workflow content"}
        )
        result = await notepad_client.call_tool("read_note", {"name": "workflow_test"})
        assert "Workflow content" in result.content[0].text

    @pytest.mark.asyncio
    async def test_list_notes(self, notepad_client):
        """Test listing all notes."""
        await notepad_client.call_tool(
            "create_note", {"name": "note_a", "content": "A"}
        )
        result = await notepad_client.call_tool("list_notes", {})
        assert result is not None
        content_text = result.content[0].text
        assert "note_a" in content_text

    @pytest.mark.asyncio
    async def test_context_manager(self, notepad_stdio_command):
        """Test async context manager."""
        async with LocalMCPClient(
            name="notepad-server",
            command=notepad_stdio_command,
        ) as client:
            assert client.is_connected is True
            tools = await client.list_tools()
            assert len(tools) > 0
            assert "create_note" in [t.name for t in tools]
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, notepad_stdio_command):
        """Test disconnect closes connection."""
        client = LocalMCPClient(
            name="notepad-server",
            command=notepad_stdio_command,
        )
        await client.connect(connect_timeout=30)
        assert client.is_connected is True
        await client.disconnect()
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connection_timeout(self):
        """Test connection timeout with slow server."""
        client = LocalMCPClient(
            name="slow-server",
            command=["python", "-c", "import time; time.sleep(100)"],
        )
        with pytest.raises(MCPConnectionError):
            await client.connect(connect_timeout=1)
        assert client.is_connected is False


@pytest.mark.integration
class TestNotepadHTTPClientIntegration:
    """Integration tests for HTTPMCPClient with notepad server."""

    @pytest.mark.asyncio
    async def test_list_tools_notepad_http(self, notepad_http_client):
        """Test listing tools from notepad server via HTTP."""
        tools = await notepad_http_client.list_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "create_note" in tool_names
        assert "read_note" in tool_names

    @pytest.mark.asyncio
    async def test_create_note_http(self, notepad_http_client):
        """Test creating a note via HTTP."""
        result = await notepad_http_client.call_tool(
            "create_note", {"name": "http_test", "content": "HTTP content"}
        )
        assert result is not None
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_read_note_http(self, notepad_http_client):
        """Test reading a note via HTTP."""
        await notepad_http_client.call_tool(
            "create_note", {"name": "http_read", "content": "Read me"}
        )
        result = await notepad_http_client.call_tool("read_note", {"name": "http_read"})
        assert result is not None
        assert result.is_error is False
        assert "Read me" in result.content[0].text

    @pytest.mark.asyncio
    async def test_list_notes_http(self, notepad_http_client):
        """Test listing notes via HTTP."""
        await notepad_http_client.call_tool(
            "create_note", {"name": "http_list_1", "content": "1"}
        )
        result = await notepad_http_client.call_tool("list_notes", {})
        assert result is not None
        content_text = result.content[0].text
        assert "http_list_1" in content_text


@pytest.mark.integration
class TestMCPClientErrors:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_list_tools_not_connected(self, notepad_client_no_connect):
        """Test listing tools when not connected raises error."""
        client = notepad_client_no_connect
        with pytest.raises(MCPConnectionError):
            await client.list_tools()

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self, notepad_client_no_connect):
        """Test calling tool when not connected raises error."""
        client = notepad_client_no_connect
        with pytest.raises(MCPConnectionError):
            await client.call_tool("create_note", {})

    @pytest.mark.asyncio
    async def test_call_nonexistent_tool(self, notepad_client):
        """Test calling non-existent tool returns error."""
        result = await notepad_client.call_tool("nonexistent_tool", {})
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_http_connection_failure(self):
        """Test HTTP connection to unavailable server raises error."""
        client = HTTPMCPClient(
            name="unavailable",
            url="http://localhost:19999/mcp",
        )
        with pytest.raises(MCPConnectionError):
            await client.connect(connect_timeout=2)
        assert client.is_connected is False
