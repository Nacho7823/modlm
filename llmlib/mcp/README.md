# MCP Client Library

MCP (Model Context Protocol) client implementations with support for local (stdio) and remote (HTTP) servers.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Local (Stdio) Client

```python
from llmlib.mcp import LocalMCPClient

client = LocalMCPClient(
    name="my-server",
    command=["npx", "-y", "@modelcontextprotocol/server-memory"]
)

await client.connect()
tools = await client.list_tools()
result = await client.call_tool("memory_read", {"key": "test"})
await client.disconnect()
```

### HTTP Client

```python
from llmlib.mcp import HTTPMCPClient

client = HTTPMCPClient(
    name="remote-server",
    url="http://localhost:9901/mcp"
)

await client.connect()
tools = await client.list_tools()
result = await client.call_tool("tool_name", {"arg": "value"})
await client.disconnect()
```

### Async Context Manager

```python
from llmlib.mcp import LocalMCPClient

async with LocalMCPClient(name="server", command=["echo"]) as client:
    tools = await client.list_tools()
    # client auto-disconnects on exit
```

### Factory Functions

```python
from llmlib.mcp import create_mcp_client, create_mcp_client_from_config

# Direct creation
client = create_mcp_client(
    name="server",
    server_type="local",
    command=["npx", "-y", "@modelcontextprotocol/server-memory"]
)

# From config dict
config = {
    "name": "my-server",
    "type": "local",
    "command": ["echo", "hello"],
    "timeout": 30,
}
client = create_mcp_client_from_config(config)
```

## API Reference

### Models

| Class | Description |
|-------|-------------|
| `ToolSchema` | Represents an MCP tool with name, description, and input schema |
| `MCPToolResult` | Result from an MCP tool call |
| `MCPConnectionError` | Raised when connection fails |
| `MCPToolError` | Raised when tool call fails |

### Clients

| Class | Description |
|-------|-------------|
| `MCPClientBase` | Abstract base class for MCP clients |
| `LocalMCPClient` | Client for local stdio servers |
| `HTTPMCPClient` | Client for remote HTTP/SSE servers |
| `RemoteMCPClient` | Alias for `HTTPMCPClient` |

### Factory

| Function | Description |
|----------|-------------|
| `create_mcp_client()` | Create client by type |
| `create_mcp_client_from_config()` | Create client from config dict |

## Testing

```bash
# Unit tests only
pytest -m "not integration"

# Integration tests (requires servers)
pytest -m integration

# All tests
pytest
```

### Test Servers

**Stdio:**
```bash
npx -y @modelcontextprotocol/server-memory
```

**HTTP:**
```bash
cd llmlib/mcp/tests/memoryRemote
npm install
npm run start
```

## Project Structure

```
llmlib/mcp/
├── __init__.py      # Public exports
├── models.py        # ToolSchema, MCPToolResult, exceptions
├── client.py       # MCPClientBase, LocalMCPClient, HTTPMCPClient
├── factory.py      # create_mcp_client functions
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   ├── test_models.py      # Unit tests for models
│   ├── test_client.py      # Unit tests for clients
│   └── test_integration.py # Integration tests
└── README.md
```
