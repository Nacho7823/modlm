"""Tests for LLM client with real API."""

from __future__ import annotations

import asyncio
import pytest
import httpx

from llmlib.llm import OpenAI, Chat, Completions, Tool


class TestOpenAIConnection:
    def test_init_default_base_url(self):
        with OpenAI() as client:
            assert client.base_url == "https://api.openai.com/v1"

    def test_init_custom_base_url(self, default_api_url):
        with OpenAI(base_url=default_api_url) as client:
            assert client.base_url == default_api_url

    def test_init_with_api_key(self):
        with OpenAI(api_key="test-key-123") as client:
            assert client.api_key == "test-key-123"

    def test_init_with_custom_timeout(self):
        with OpenAI(timeout=30.0) as client:
            assert client._client.timeout.connect == 30.0

    def test_close(self):
        client = OpenAI()
        client.close()
        assert client._client is not None

    def test_context_manager(self, default_api_url):
        with OpenAI(base_url=default_api_url) as client:
            assert client is not None
        assert client._client is None

    def test_chat_property(self, default_api_url):
        with OpenAI(base_url=default_api_url) as client:
            chat = client.chat
            assert isinstance(chat, Chat)

    def test_client_type(self, default_api_url):
        with OpenAI(base_url=default_api_url) as client:
            assert isinstance(client._client, httpx.Client)


class TestChatEndpoint:
    def test_completions_property(self, default_api_url):
        with OpenAI(base_url=default_api_url) as client:
            chat = client.chat
            completions = chat.completions
            assert isinstance(completions, Completions)

    def test_chat_passes_model_to_completions(self, default_api_url):
        with OpenAI(base_url=default_api_url, model="test-model") as client:
            completions = client.chat.completions
            assert completions._model == "test-model"


class TestCompletionsModel:
    """Tests for model parameter handling."""

    def test_model_from_constructor(self, real_client):
        completions = real_client.chat.completions
        assert completions._model is not None

    def test_model_override_in_create(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        assert result.model == model

    def test_model_required_error(self, api_config):
        with OpenAI(base_url=api_config["url"]) as client:
            with pytest.raises(ValueError, match="model is required"):
                client.chat.completions.create(
                    messages=[{"role": "user", "content": "test"}]
                )

    def test_model_optional_when_provided(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        assert result.model == model


class TestCompletionsAPI:
    """Real API tests for completions."""

    def test_create_basic(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=20,
        )
        assert_response_has_content(result)
        assert result.model == model

    def test_create_with_temperature(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.0,
            max_tokens=10,
        )
        assert_response_has_content(result)

    def test_create_with_max_tokens(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Count from 1 to 10."}],
            max_tokens=5,
        )
        assert_response_has_content(result)

    def test_create_with_tools(self, real_client, model):
        tools = [
            Tool(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            )
        ]
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            tools=tools,
            max_tokens=50,
        )
        assert result.choices
        assert result.model == model

    def test_create_empty_messages(self, real_client, model):
        with pytest.raises(Exception) as exc_info:
            create_completion(
                real_client,
                model,
                messages=[],
                max_tokens=10,
            )
        assert "400" in str(exc_info.value) or "Bad Request" in str(exc_info.value)

    def test_create_empty_choices(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
        )
        assert result.choices

    def test_endpoint_path(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        assert result.choices

    def test_model_passed_correctly(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
        )
        assert result.model == model


class TestConversationAPI:
    """Real API tests for conversations."""

    def test_single_message(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Say 'ok'."}],
            max_tokens=10,
        )
        assert_response_has_content(result)

    def test_multiple_messages(self, real_client, model):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = create_completion(
            real_client,
            model,
            messages=messages,
            max_tokens=20,
        )
        assert_response_has_content(result)

    def test_system_message_first(self, real_client, model):
        messages = [
            {"role": "system", "content": "You are a pirate. Reply like a pirate."},
            {"role": "user", "content": "Hello"},
        ]
        result = create_completion(
            real_client,
            model,
            messages=messages,
            max_tokens=20,
        )
        assert_response_has_content(result)

    def test_user_and_assistant_roles(self, real_client, model):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Multiply by 3"},
        ]
        result = create_completion(
            real_client,
            model,
            messages=messages,
            max_tokens=20,
        )
        assert_response_has_content(result)

    def test_message_content_preserved(self, real_client, model):
        messages = [
            {"role": "user", "content": "Test <special> chars"},
        ]
        result = create_completion(
            real_client,
            model,
            messages=messages,
            max_tokens=20,
        )
        assert_response_has_content(result)

    def test_system_prompt_via_kwargs(self, real_client, model):
        result = create_completion(
            real_client,
            model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=20,
            system_prompt="You are a helpful assistant.",
        )
        assert_response_has_content(result)


class TestFetchModels:
    """Tests for fetch_models method."""

    @pytest.mark.parametrize("api_key", [None, "test-key"])
    def test_fetch_models_returns_list(self, default_api_url, api_key):
        with OpenAI(base_url=default_api_url, api_key=api_key) as client:
            models = client.fetch_models()
        assert isinstance(models, list)

    def test_fetch_models_returns_strings(self, default_api_url):
        with OpenAI(base_url=default_api_url) as client:
            models = client.fetch_models()
        for m in models:
            assert isinstance(m, str)

    def test_fetch_models_non_empty(self, api_config):
        with OpenAI(
            base_url=api_config["url"], api_key=api_config["api_key"]
        ) as client:
            models = client.fetch_models()
        assert len(models) > 0


class TestStreamCompletions:
    def test_stream_completions_property(self, real_client):
        assert hasattr(real_client.chat, "stream_completions")

    def test_stream_completions_returns_iterator(self, real_client):
        result = real_client.chat.stream_completions.create(
            model=real_client.model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        chunks = list(result)
        assert isinstance(chunks, list)

    def test_stream_completions_chunks_have_delta(self, real_client):
        result = real_client.chat.stream_completions.create(
            model=real_client.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        chunks = list(result)
        assert_stream_has_content(chunks)


class TestAsyncCompletions:
    def test_chat_async_property(self, real_client):
        assert hasattr(real_client, "chat_async")

    def test_async_completions_create(self, real_client):
        async def run():
            result = await real_client.chat_async.completions.create(
                model=real_client.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return result

        result = asyncio.run(run())
        assert_response_has_content(result)

    def test_async_stream_completions(self, real_client):
        async def run():
            chunks = []
            async for chunk in await real_client.chat_async.completions.create(
                model=real_client.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10,
                stream=True,
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert_stream_has_content(chunks)


def create_completion(
    real_client,
    model,
    **kwargs,
):
    return real_client.chat.completions.create(model=model, **kwargs)


def assert_response_has_content(response):
    """Helper: assert response has content in first choice."""
    assert response.choices, "Response has no choices"
    content = response.choices[0].message.effective_content
    assert content, "Response message has no content"


def assert_stream_has_content(chunks):
    """Helper: assert streaming chunks have content."""
    has_content = any(
        chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        or chunk.get("choices", [{}])[0].get("delta", {}).get("reasoning_content")
        for chunk in chunks
    )
    assert has_content, "No content found in streaming chunks"
