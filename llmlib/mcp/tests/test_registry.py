import pytest
from unittest.mock import AsyncMock, patch
from llmlib.models import MCPServerConfig
from llmlib.mcp.registry import MCPRegistry


@pytest.mark.asyncio
async def test_load_remote_server_from_config():
    reg = MCPRegistry()
    cfg = MCPServerConfig(name="myserver", server_type="remote", url="http://localhost:8080")
    
    with patch("llmlib.mcp.registry.HTTPMCPClient") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.test_connection = AsyncMock(return_value={"status": "connected", "tools": []})
        
        warnings = await reg.load_servers({"myserver": cfg})
        assert warnings == []
        assert "myserver" in reg.list_servers()


@pytest.mark.asyncio
async def test_load_local_server_from_config():
    reg = MCPRegistry()
    cfg = MCPServerConfig(name="local", server_type="local", command=["npx", "mcp-tool"])
    
    with patch("llmlib.mcp.registry.LocalMCPClient") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.test_connection = AsyncMock(return_value={"status": "connected", "tools": []})
        
        warnings = await reg.load_servers({"local": cfg})
        assert warnings == []
        assert "local" in reg.list_servers()


@pytest.mark.asyncio
async def test_add_remote_server():
    reg = MCPRegistry()
    cfg = MCPServerConfig(name="srv", server_type="remote", url="http://x")
    
    with patch("llmlib.mcp.registry.HTTPMCPClient") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.test_connection = AsyncMock(return_value={"status": "connected", "tools": []})
        
        msg = await reg.add_server(cfg)
        assert "srv" in reg.list_servers()
        assert "added" in msg


@pytest.mark.asyncio
async def test_remove_server():
    reg = MCPRegistry()
    cfg = MCPServerConfig(name="srv", server_type="remote", url="http://x")
    
    with patch("llmlib.mcp.registry.HTTPMCPClient") as mock_cls:
        mock_client = mock_cls.return_value
        mock_client.test_connection = AsyncMock(return_value={"status": "connected", "tools": []})
        
        await reg.add_server(cfg)
        assert await reg.remove_server("srv") is True
        assert await reg.remove_server("nonexistent") is False
        assert "srv" not in reg.list_servers()
