"""Tests for LLM models."""

from __future__ import annotations

import pytest

from llmlib.llm import Message, Choice, ChatCompletion


class TestMessage:
    def test_init_default_role(self):
        msg = Message(content="Hello")
        assert msg.content == "Hello"
        assert msg.role == "assistant"

    def test_init_custom_role(self):
        msg = Message(content="Hi", role="user")
        assert msg.content == "Hi"
        assert msg.role == "user"

    def test_to_dict(self):
        msg = Message(content="Hello", role="user")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello"}

    def test_to_dict_with_reasoning(self):
        msg = Message(content="Answer", role="assistant", reasoning_content="Reasoning")
        result = msg.to_dict()
        assert result == {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Reasoning",
        }

    def test_repr(self):
        msg = Message(content="Hello", role="user")
        assert "Message" in repr(msg)
        assert "user" in repr(msg)


class TestChoice:
    def test_init_default_values(self):
        choice = Choice()
        assert choice.message.content == ""
        assert choice.message.role == "assistant"
        assert choice.index == 0
        assert choice.finish_reason is None
        assert choice.tool_calls == []

    def test_init_with_values(self):
        msg = Message(content="Response", role="assistant")
        choice = Choice(message=msg, index=1, finish_reason="stop")
        assert choice.message.content == "Response"
        assert choice.index == 1
        assert choice.finish_reason == "stop"

    def test_init_with_tool_calls(self):
        tool_calls = [{"id": "call_1", "function": {"name": "test"}}]
        choice = Choice(tool_calls=tool_calls)
        assert choice.tool_calls == tool_calls

    def test_repr(self):
        choice = Choice(index=0)
        assert "Choice" in repr(choice)


class TestChatCompletion:
    def test_init_with_choices(self):
        choice = Choice()
        completion = ChatCompletion(choices=[choice], model="gpt-4")
        assert completion.choices == [choice]
        assert completion.model == "gpt-4"

    def test_id_property(self):
        completion = ChatCompletion(choices=[], model="gpt-4")
        assert completion.id == "chatcmpl-local"

    def test_usage_property(self):
        completion = ChatCompletion(choices=[])
        assert completion.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_content_property_with_choices(self):
        choice = Choice(message=Message(content="Hello world"))
        completion = ChatCompletion(choices=[choice])
        assert completion.content == "Hello world"

    def test_content_property_empty_choices(self):
        completion = ChatCompletion(choices=[])
        assert completion.content is None

    def test_tool_calls_property_with_choices(self):
        tool_calls = [{"id": "call_1"}]
        choice = Choice(tool_calls=tool_calls)
        completion = ChatCompletion(choices=[choice])
        assert completion.tool_calls == tool_calls

    def test_tool_calls_property_empty_choices(self):
        completion = ChatCompletion(choices=[])
        assert completion.tool_calls == []

    def test_finish_reason_property(self):
        completion = ChatCompletion(choices=[], finish_reason="stop")
        assert completion.finish_reason == "stop"

    def test_repr(self):
        completion = ChatCompletion(choices=[], model="gpt-4")
        assert "ChatCompletion" in repr(completion)
