"""Tests for MCP client implementations."""

import pytest

from llmlib.mcp import (
    HTTPMCPClient,
    LocalMCPClient,
    MCPClientBase,
    MCPConnectionError,
    RemoteMCPClient,
    ToolSchema,
    create_mcp_client,
    create_mcp_client_from_config,
)


class TestLocalMCPClientCreation:
    """Tests for LocalMCPClient instantiation."""

    def test_create_with_command(self):
        """Test LocalMCPClient creation with command."""
        client = LocalMCPClient(
            name="test-server",
            command=["echo", "hello"],
            timeout=30,
        )
        assert client.name == "test-server"
        assert client.command == ["echo", "hello"]
        assert client.timeout == 30
        assert client.env == {}
        assert client.is_connected is False

    def test_create_with_env(self):
        """Test LocalMCPClient creation with environment variables."""
        client = LocalMCPClient(
            name="test-server",
            command=["python", "server.py"],
            env={"DEBUG": "true", "LOG_LEVEL": "info"},
        )
        assert client.env == {"DEBUG": "true", "LOG_LEVEL": "info"}

    def test_create_with_defaults(self):
        """Test LocalMCPClient with default timeout."""
        client = LocalMCPClient(
            name="default-timeout",
            command=["echo"],
        )
        assert client.timeout == 30
        assert client.is_connected is False


class TestHTTPMCPClientCreation:
    """Tests for HTTPMCPClient instantiation."""

    def test_create_with_url(self):
        """Test HTTPMCPClient creation with URL."""
        client = HTTPMCPClient(
            name="test-server",
            url="http://localhost:3000/mcp",
            timeout=30,
        )
        assert client.name == "test-server"
        assert client.url == "http://localhost:3000/mcp"
        assert client.timeout == 30
        assert client.headers == {}
        assert client.is_connected is False

    def test_create_with_headers(self):
        """Test HTTPMCPClient creation with HTTP headers."""
        client = HTTPMCPClient(
            name="test-server",
            url="http://localhost:3000/mcp",
            headers={"Authorization": "Bearer token123"},
        )
        assert client.headers == {"Authorization": "Bearer token123"}

    def test_create_with_defaults(self):
        """Test HTTPMCPClient with default timeout."""
        client = HTTPMCPClient(
            name="default-timeout",
            url="http://localhost:3000/mcp",
        )
        assert client.timeout == 30
        assert client.is_connected is False


class TestRemoteMCPClientAlias:
    """Tests for RemoteMCPClient alias."""

    def test_remote_is_http_alias(self):
        """Test that RemoteMCPClient is an alias for HTTPMCPClient."""
        assert RemoteMCPClient is HTTPMCPClient


class TestMCPClientBase:
    """Tests for MCPClientBase abstract class."""

    def test_is_abstract(self):
        """Test that MCPClientBase cannot be instantiated."""
        with pytest.raises(TypeError):
            MCPClientBase(name="test")

    def test_connected_property_default(self):
        """Test default is_connected property."""
        client = LocalMCPClient(name="test", command=["echo"])
        assert client.is_connected is False


class TestCreateMCPClient:
    """Tests for create_mcp_client factory function."""

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

    def test_stdio_requires_command(self):
        """Test that stdio client requires command."""
        with pytest.raises(MCPConnectionError, match="command is required"):
            create_mcp_client(
                name="test",
                server_type="stdio",
            )

    def test_http_requires_url(self):
        """Test that HTTP client requires URL."""
        with pytest.raises(MCPConnectionError, match="url is required"):
            create_mcp_client(
                name="test",
                server_type="http",
            )

    def test_remote_requires_url(self):
        """Test that remote client requires URL."""
        with pytest.raises(MCPConnectionError, match="url is required"):
            create_mcp_client(
                name="test",
                server_type="remote",
            )

    def test_invalid_server_type(self):
        """Test that invalid server type raises error."""
        with pytest.raises(MCPConnectionError, match="Unknown MCP server type"):
            create_mcp_client(
                name="test",
                server_type="invalid",
            )

    def test_custom_timeout(self):
        """Test custom timeout is passed to client."""
        client = create_mcp_client(
            name="test",
            server_type="local",
            command=["echo"],
            timeout=60,
        )
        assert client.timeout == 60


class TestCreateMCPClientFromConfig:
    """Tests for create_mcp_client_from_config function."""

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

    def test_config_defaults(self):
        """Test that defaults are applied from config."""
        config = {"name": "test", "type": "local", "command": ["echo"]}
        client = create_mcp_client_from_config(config)
        assert client.timeout == 30
        assert client.name == "test"

    def test_config_missing_name(self):
        """Test config with missing name uses default."""
        config = {"type": "local", "command": ["echo"]}
        client = create_mcp_client_from_config(config)
        assert client.name == "unnamed"

    def test_config_missing_type(self):
        """Test config with missing type uses local default."""
        config = {"name": "test", "command": ["echo"]}
        client = create_mcp_client_from_config(config)
        assert isinstance(client, LocalMCPClient)
