"""Tests for LLM client with real API."""

from __future__ import annotations

import pytest
import httpx

from llmlib.llm import OpenAI, Chat, Completions, Tool


class TestOpenAIConnection:
    def test_init_default_base_url(self):
        client = OpenAI()
        assert client.base_url == "https://api.openai.com/v1"
        client.close()

    def test_init_custom_base_url(self):
        client = OpenAI(base_url="http://localhost:11434/v1")
        assert client.base_url == "http://localhost:11434/v1"
        client.close()

    def test_init_with_api_key(self):
        client = OpenAI(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        client.close()

    def test_init_with_custom_timeout(self):
        client = OpenAI(timeout=30.0)
        assert client._client.timeout.connect == 30.0
        client.close()

    def test_close(self):
        client = OpenAI()
        client.close()
        assert client._client is not None

    def test_context_manager(self):
        with OpenAI(base_url="http://localhost:11434/v1") as client:
            assert client is not None
        assert client._client is None

    def test_chat_property(self):
        client = OpenAI(base_url="http://localhost:11434/v1")
        chat = client.chat
        assert isinstance(chat, Chat)
        client.close()

    def test_client_type(self):
        client = OpenAI(base_url="http://localhost:11434/v1")
        assert isinstance(client._client, httpx.Client)
        client.close()


class TestChatEndpoint:
    def test_completions_property(self):
        client = OpenAI(base_url="http://localhost:11434/v1")
        chat = client.chat
        completions = chat.completions
        assert isinstance(completions, Completions)
        client.close()

    def test_chat_passes_model_to_completions(self):
        client = OpenAI(base_url="http://localhost:11434/v1", model="test-model")
        completions = client.chat.completions
        assert completions._model == "test-model"
        client.close()


class TestCompletionsModel:
    """Tests for model parameter handling."""

    @pytest.fixture(autouse=True)
    def check_api(self, request):
        from llmlib.llm.tests.conftest import is_api_available

        if not is_api_available():
            pytest.skip("API not available")
        yield

    def test_model_from_constructor(self, real_client):
        completions = real_client.chat.completions
        assert completions._model is not None
        real_client.close()

    def test_model_override_in_create(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        assert result.model == model
        real_client.close()

    def test_model_required_error(self, api_config):
        client = OpenAI(base_url=api_config["url"])
        with pytest.raises(ValueError, match="model is required"):
            client.chat.completions.create(
                messages=[{"role": "user", "content": "test"}]
            )
        client.close()

    def test_model_optional_when_provided(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
        )
        assert result.model == model
        real_client.close()


class TestCompletionsReal:
    """Real API tests for completions."""

    @pytest.fixture(autouse=True)
    def check_api(self, request):
        from llmlib.llm.tests.conftest import is_api_available

        if not is_api_available():
            pytest.skip("API not available")
        yield

    def test_create_basic(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=20,
        )
        assert result.choices[0].message.content
        assert result.model == model

    def test_create_with_temperature(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            temperature=0.0,
            max_tokens=10,
        )
        assert result.choices[0].message.content

    def test_create_with_max_tokens(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count from 1 to 10."}],
            max_tokens=5,
        )
        assert result.choices[0].message.content

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
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            tools=tools,
            max_tokens=50,
        )
        assert result.choices
        assert result.model == model

    def test_create_empty_messages(self, real_client, model):
        with pytest.raises(Exception) as exc_info:
            real_client.chat.completions.create(
                model=model,
                messages=[],
                max_tokens=10,
            )
        assert "400" in str(exc_info.value) or "Bad Request" in str(exc_info.value)

    def test_create_empty_choices(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
        )
        assert result.choices

    def test_endpoint_path(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )
        assert result.choices

    def test_model_passed_correctly(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5,
        )
        assert result.model == model


class TestConversationReal:
    """Real API tests for conversations."""

    @pytest.fixture(autouse=True)
    def check_api(self, request):
        from llmlib.llm.tests.conftest import is_api_available

        if not is_api_available():
            pytest.skip("API not available")
        yield

    def test_single_message(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok'."}],
            max_tokens=10,
        )
        assert result.choices[0].message.content

    def test_multiple_messages(self, real_client, model):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = real_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=20,
        )
        assert result.choices[0].message.content

    def test_system_message_first(self, real_client, model):
        messages = [
            {"role": "system", "content": "You are a pirate. Reply like a pirate."},
            {"role": "user", "content": "Hello"},
        ]
        result = real_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=20,
        )
        assert result.choices[0].message.content

    def test_user_and_assistant_roles(self, real_client, model):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Multiply by 3"},
        ]
        result = real_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=20,
        )
        assert result.choices[0].message.content

    def test_message_content_preserved(self, real_client, model):
        messages = [
            {"role": "user", "content": "Test <special> chars"},
        ]
        result = real_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=20,
        )
        assert result.choices[0].message.content

    def test_system_prompt_via_kwargs(self, real_client, model):
        result = real_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=20,
            system_prompt="You are a helpful assistant.",
        )
        assert result.choices[0].message.content


class TestFetchModels:
    """Tests for fetch_models method."""

    @pytest.fixture(autouse=True)
    def check_api(self, request):
        from llmlib.llm.tests.conftest import is_api_available

        if not is_api_available():
            pytest.skip("API not available")
        yield

    def test_fetch_models_ollama(self):
        client = OpenAI(base_url="http://localhost:11434/v1")
        models = client.fetch_models()
        client.close()
        assert isinstance(models, list)

    def test_fetch_models_returns_strings(self):
        client = OpenAI(base_url="http://localhost:11434/v1")
        models = client.fetch_models()
        client.close()
        for m in models:
            assert isinstance(m, str)

    def test_fetch_models_non_empty(self, api_config):
        client = OpenAI(base_url=api_config["url"], api_key=api_config["api_key"])
        models = client.fetch_models()
        client.close()
        assert len(models) > 0

    def test_fetch_models_with_api_key(self):
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="test-key")
        models = client.fetch_models()
        client.close()


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
        real_client.close()

    def test_stream_completions_chunks_have_delta(self, real_client):
        result = real_client.chat.stream_completions.create(
            model=real_client.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        chunks = list(result)
        has_content = any(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            for chunk in chunks
        )
        assert has_content
        real_client.close()


class TestAsyncCompletions:
    def test_chat_async_property(self, real_client):
        assert hasattr(real_client, "chat_async")

    def test_async_completions_create(self, real_client):
        import asyncio

        async def run():
            result = await real_client.chat_async.completions.create(
                model=real_client.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return result

        result = asyncio.run(run())
        assert result.choices[0].message.content
        real_client.close()

    def test_async_stream_completions(self, real_client):
        import asyncio

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
        has_content = any(
            chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            for chunk in chunks
        )
        assert has_content
        real_client.close()
