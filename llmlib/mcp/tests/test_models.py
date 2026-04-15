"""Tests for MCP models."""

import pytest

from llmlib.mcp import MCPConnectionError, MCPToolError, MCPToolResult, ToolSchema


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_creation(self):
        """Test ToolSchema creation with all fields."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool description",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        )
        assert schema.name == "test_tool"
        assert schema.description == "A test tool description"
        assert schema.input_schema == {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }

    def test_creation_default_input_schema(self):
        """Test ToolSchema with default input_schema."""
        schema = ToolSchema(name="simple_tool", description="Simple tool")
        assert schema.name == "simple_tool"
        assert schema.description == "Simple tool"
        assert schema.input_schema == {}

    def test_to_dict(self):
        """Test ToolSchema serialization."""
        schema = ToolSchema(
            name="test_tool",
            description="Description",
            input_schema={"type": "object"},
        )
        d = schema.to_dict()
        assert d["name"] == "test_tool"
        assert d["description"] == "Description"
        assert d["inputSchema"] == {"type": "object"}

    def test_to_dict_empty_schema(self):
        """Test ToolSchema serialization with empty schema."""
        schema = ToolSchema(name="empty", description="")
        d = schema.to_dict()
        assert d["name"] == "empty"
        assert d["description"] == ""
        assert d["inputSchema"] == {}


class TestMCPToolResult:
    """Tests for MCPToolResult dataclass."""

    def test_creation_default(self):
        """Test MCPToolResult with defaults."""
        result = MCPToolResult()
        assert result.content == []
        assert result.is_error is False

    def test_creation_with_content(self):
        """Test MCPToolResult with content."""
        result = MCPToolResult(content=["result1", "result2"], is_error=False)
        assert result.content == ["result1", "result2"]
        assert result.is_error is False

    def test_creation_with_error(self):
        """Test MCPToolResult with error flag."""
        result = MCPToolResult(content=["error message"], is_error=True)
        assert result.content == ["error message"]
        assert result.is_error is True

    def test_to_dict_text_content(self):
        """Test MCPToolResult serialization with string content."""
        result = MCPToolResult(content=["hello world"])
        d = result.to_dict()
        assert d["content"] == [{"type": "text", "text": "hello world"}]
        assert d["isError"] is False

    def test_to_dict_with_object(self):
        """Test MCPToolResult serialization with object content."""

        class MockContent:
            def __init__(self):
                self.text = "object text"

        result = MCPToolResult(content=[MockContent()])
        d = result.to_dict()
        assert d["content"] == [{"type": "text", "text": "object text"}]

    def test_to_dict_with_data(self):
        """Test MCPToolResult serialization with data attribute."""

        class MockResource:
            def __init__(self):
                self.data = {"key": "value"}

        result = MCPToolResult(content=[MockResource()])
        d = result.to_dict()
        assert d["content"] == [{"type": "resource", "data": {"key": "value"}}]

    def test_to_dict_with_error(self):
        """Test MCPToolResult serialization with error flag."""
        result = MCPToolResult(content=["error"], is_error=True)
        d = result.to_dict()
        assert d["isError"] is True


class TestMCPExceptions:
    """Tests for MCP custom exceptions."""

    def test_connection_error_creation(self):
        """Test MCPConnectionError creation."""
        error = MCPConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert error.details is None

    def test_connection_error_with_details(self):
        """Test MCPConnectionError with details."""
        error = MCPConnectionError("Connection failed", details={"code": 500})
        assert str(error) == "Connection failed"
        assert error.details == {"code": 500}

    def test_tool_error_creation(self):
        """Test MCPToolError creation."""
        error = MCPToolError("Tool execution failed")
        assert str(error) == "Tool execution failed"

    def test_tool_error_with_details(self):
        """Test MCPToolError with details."""
        error = MCPToolError("Tool failed", details={"tool": "test_tool"})
        assert str(error) == "Tool failed"
        assert error.details == {"tool": "test_tool"}

    def test_connection_error_is_exception(self):
        """Test MCPConnectionError inheritance."""
        error = MCPConnectionError("test")
        assert isinstance(error, Exception)

    def test_tool_error_is_exception(self):
        """Test MCPToolError inheritance."""
        error = MCPToolError("test")
        assert isinstance(error, Exception)
