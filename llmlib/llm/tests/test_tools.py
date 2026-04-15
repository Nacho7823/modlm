"""Tests for LLM tools with real API."""

from __future__ import annotations

import pytest

from llmlib.llm import Tool, ToolCall, ToolResult, ToolExecutor


class TestTool:
    def test_init(self):
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a city"
        assert tool.parameters == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }

    def test_init_default_parameters(self):
        tool = Tool(name="test", description="A test tool")
        assert tool.parameters == {}

    def test_to_openai_schema(self):
        tool = Tool(
            name="get_weather",
            description="Get weather for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert schema["function"]["description"] == "Get weather for a city"
        assert schema["function"]["parameters"] == {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        }

    def test_repr(self):
        tool = Tool(name="test_tool", description="A test")
        assert "test_tool" in repr(tool)


class TestToolCall:
    def test_init(self):
        tc = ToolCall(id="call_123", name="get_weather", arguments={"city": "Tokyo"})
        assert tc.id == "call_123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Tokyo"}

    def test_init_default_arguments(self):
        tc = ToolCall(id="call_123", name="test")
        assert tc.arguments == {}

    def test_from_dict(self):
        data = {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris", "unit": "celsius"}',
            },
        }
        tc = ToolCall.from_dict(data)
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Paris", "unit": "celsius"}

    def test_from_dict_with_dict_arguments(self):
        data = {
            "id": "call_abc",
            "function": {
                "name": "test",
                "arguments": {"key": "value"},
            },
        }
        tc = ToolCall.from_dict(data)
        assert tc.arguments == {"key": "value"}

    def test_repr(self):
        tc = ToolCall(id="call_1", name="test")
        assert "call_1" in repr(tc)


class TestToolResult:
    def test_init(self):
        tr = ToolResult(tool_call_id="call_123", content="Sunny, 25°C")
        assert tr.tool_call_id == "call_123"
        assert tr.content == "Sunny, 25°C"
        assert tr.is_error is False

    def test_init_with_error(self):
        tr = ToolResult(tool_call_id="call_123", content="Error message", is_error=True)
        assert tr.is_error is True

    def test_to_message(self):
        tr = ToolResult(tool_call_id="call_123", content="Result")
        msg = tr.to_message()
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == "Result"
        assert msg["role"] == "tool"

    def test_repr(self):
        tr = ToolResult(tool_call_id="call_1", content="test")
        assert "call_1" in repr(tr)


class TestToolExecutor:
    def test_init(self):
        executor = ToolExecutor()
        assert executor._functions == {}

    def test_register(self):
        executor = ToolExecutor()

        def get_weather(city: str) -> str:
            return f"Weather in {city}"

        executor.register("get_weather", get_weather)
        assert "get_weather" in executor._functions

    def test_execute_registered_function(self):
        executor = ToolExecutor()

        def get_weather(city: str) -> str:
            return f"Weather in {city}"

        executor.register("get_weather", get_weather)

        tool_call = ToolCall(
            id="call_1", name="get_weather", arguments={"city": "Tokyo"}
        )
        result = executor.execute(tool_call)

        assert result.tool_call_id == "call_1"
        assert result.content == "Weather in Tokyo"
        assert result.is_error is False

    def test_execute_unknown_tool(self):
        executor = ToolExecutor()
        tool_call = ToolCall(id="call_1", name="unknown_tool", arguments={})
        result = executor.execute(tool_call)

        assert result.is_error is True
        assert "Unknown tool" in result.content

    def test_execute_with_exception(self):
        executor = ToolExecutor()

        def failing_function() -> str:
            raise ValueError("Something went wrong")

        executor.register("failing", failing_function)

        tool_call = ToolCall(id="call_1", name="failing", arguments={})
        result = executor.execute(tool_call)

        assert result.is_error is True
        assert "Something went wrong" in result.content

    def test_execute_all(self):
        executor = ToolExecutor()

        def add(a: int, b: int) -> int:
            return a + b

        def multiply(a: int, b: int) -> int:
            return a * b

        executor.register("add", add)
        executor.register("multiply", multiply)

        tool_calls = [
            ToolCall(id="call_1", name="add", arguments={"a": 2, "b": 3}),
            ToolCall(id="call_2", name="multiply", arguments={"a": 4, "b": 5}),
        ]

        results = executor.execute_all(tool_calls)

        assert len(results) == 2
        assert results[0].content == "5"
        assert results[1].content == "20"


class TestToolIntegrationReal:
    """Real API integration tests for tools."""

    @pytest.fixture(autouse=True)
    def check_api(self, request):
        from llmlib.llm.tests.conftest import is_api_available

        if not is_api_available():
            pytest.skip("API not available")
        yield

    def test_tool_execution_full_flow(self, real_client, model):
        executor = ToolExecutor()

        def add(a: int, b: int) -> str:
            return str(a + b)

        executor.register("add", add)

        tools = [
            Tool(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            )
        ]

        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            tools=tools,
            max_tokens=100,
        )

        tool_calls = result.tool_calls
        if tool_calls:
            for tc in tool_calls:
                tool_call = ToolCall.from_dict(tc)
                tool_result = executor.execute(tool_call)
                assert tool_result.content == "8"

    def test_tool_with_string_arguments(self, real_client, model):
        executor = ToolExecutor()

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        executor.register("greet", greet)

        tools = [
            Tool(
                name="greet",
                description="Greet a person",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                    },
                    "required": ["name"],
                },
            )
        ]

        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello to Alice."}],
            tools=tools,
            max_tokens=100,
        )

        tool_calls = result.tool_calls
        if tool_calls:
            for tc in tool_calls:
                tool_call = ToolCall.from_dict(tc)
                tool_result = executor.execute(tool_call)
                assert "Alice" in tool_result.content
