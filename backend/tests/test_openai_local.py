"""Tests for the local OpenAI-compatible client."""

from __future__ import annotations

import httpx
import pytest
from llmlib.openai_local import OpenAI, ChatCompletion, Message, Choice


def test_openai_initialization():
    """Test OpenAI client initialization."""
    api_key = "test-key"
    base_url = "http://localhost:11434/v1"
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    assert client.api_key == api_key
    assert client.base_url == base_url
    assert isinstance(client._client, httpx.Client)
    assert str(client._client.base_url).rstrip("/") == base_url.rstrip("/")
    
    client.close()


def test_openai_default_base_url():
    """Test OpenAI client with default base URL."""
    client = OpenAI()
    assert client.base_url == "https://api.openai.com/v1"
    client.close()


def test_chat_completion_properties():
    """Test ChatCompletion response object properties."""
    choices = [
        Choice(Message(content="Hello", role="assistant"), index=0)
    ]
    completion = ChatCompletion(choices=choices, model="gpt-3.5-turbo")
    
    assert completion.id == "chatcmpl-local"
    assert completion.usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert completion.model == "gpt-3.5-turbo"
    assert len(completion.choices) == 1
    assert completion.choices[0].message.content == "Hello"


from unittest.mock import MagicMock, patch

def test_completions_create_mocked():
    """Test chat completion creation with mocked httpx response."""
    base_url = "http://localhost:11434/v1"
    client = OpenAI(base_url=base_url)
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": "Test response", "role": "assistant"},
                "index": 0
            }
        ]
    }
    
    with patch.object(client._client, 'post', return_value=mock_response) as mock_post:
        messages = [{"role": "user", "content": "Hi"}]
        completion = client.chat.completions.create(
            model="test-model",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        # Verify call
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "/chat/completions"
        assert kwargs["json"]["model"] == "test-model"
        assert kwargs["json"]["messages"] == messages
        assert kwargs["json"]["temperature"] == 0.7
        assert kwargs["json"]["max_tokens"] == 100
        
        # Verify result
        assert len(completion.choices) == 1
        assert completion.choices[0].message.content == "Test response"
        assert completion.choices[0].message.role == "assistant"
    
    client.close()

def test_completions_create_error():
    """Test chat completion handles HTTP errors."""
    client = OpenAI()
    
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=MagicMock(), response=mock_response
    )
    
    with patch.object(client._client, 'post', return_value=mock_response):
        with pytest.raises(httpx.HTTPStatusError):
            client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hi"}]
            )
    
    client.close()
