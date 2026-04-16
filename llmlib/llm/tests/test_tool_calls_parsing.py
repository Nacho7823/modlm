"""Regression tests for tool_calls parsing from completion responses."""

from llmlib.llm.client import _parse_completion


def test_parse_completion_reads_tool_calls_from_choice_level() -> None:
    data = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "foo", "arguments": "{}"},
                    }
                ],
                "finish_reason": "tool_calls",
            }
        ]
    }

    completion = _parse_completion(data, "test-model")

    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0]["function"]["name"] == "foo"


def test_parse_completion_reads_tool_calls_from_message_level() -> None:
    data = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\n",
                    "tool_calls": [
                        {
                            "id": "374885387",
                            "type": "function",
                            "function": {
                                "name": "websearch__web_search_exa",
                                "arguments": '{"query":"ia"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    completion = _parse_completion(data, "qwen3.5-4b")

    assert len(completion.tool_calls) == 1
    assert completion.tool_calls[0]["id"] == "374885387"
    assert completion.tool_calls[0]["function"]["name"] == "websearch__web_search_exa"
