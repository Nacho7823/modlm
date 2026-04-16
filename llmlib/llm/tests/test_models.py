"""Tests for LLM models."""

from __future__ import annotations

import pytest

from llmlib.models import Message, Choice, ChatCompletion


class TestMessage:
    def test_init_with_role(self):
        msg = Message(role="assistant", content="Hello")
        assert msg.content == "Hello"
        assert msg.role == "assistant"

    def test_to_dict(self):
        msg = Message(role="user", content="Hello")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Hello"}

    def test_to_dict_with_reasoning(self):
        msg = Message(role="assistant", content="Answer", thinking="Reasoning")
        result = msg.to_dict()
        assert result == {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Reasoning",
        }
    
    def test_reasoning_content_property(self):
        msg = Message(role="assistant", content="Answer", thinking="Reasoning")
        assert msg.reasoning_content == "Reasoning"
        msg.reasoning_content = "New"
        assert msg.thinking == "New"

    def test_repr(self):
        msg = Message(role="user", content="Hello")
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
        msg = Message(role="assistant", content="Response")
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

    def test_repr(self):
        completion = ChatCompletion(choices=[], model="gpt-4")
        assert "ChatCompletion" in repr(completion)
