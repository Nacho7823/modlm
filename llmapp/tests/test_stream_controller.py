"""Unit tests for stream controller behavior."""

from __future__ import annotations

import asyncio

from llmlib.runtime import StreamEvent

from llmapp.controllers.stream_controller import StreamController


class _FakeView:
    def __init__(self) -> None:
        self.content = ""
        self.thinking = ""

    def append_content(self, text: str) -> None:
        self.content += text

    def append_thinking(self, text: str) -> None:
        self.thinking += text


class _FakeContainer:
    def __init__(self, view: _FakeView) -> None:
        self._view = view

    def add_message(self, role: str, content: str) -> None:
        return None

    def last_assistant_view(self) -> _FakeView:
        return self._view


class _FakeConfig:
    def get(self, key: str, default: bool) -> bool:
        return default


class _FakeRuntime:
    async def stream(self, messages, streaming_enabled=True):
        yield StreamEvent.thinking("pensando")
        yield StreamEvent.content("hola")
        yield StreamEvent.done()


class _FakeApp:
    def __init__(self) -> None:
        self.config_manager = _FakeConfig()
        self.runtime = _FakeRuntime()
        self.messages = [{"role": "assistant", "content": ""}]
        self._streaming_active = True
        self._streaming_task = None
        self._view = _FakeView()
        self._container = _FakeContainer(self._view)

    def _get_chat_container(self) -> _FakeContainer:
        return self._container

    def _add_message(self, role: str, content: str) -> None:
        return None


def test_run_stream_consumes_typed_stream_events() -> None:
    app = _FakeApp()
    controller = StreamController(app)

    asyncio.run(controller._run_stream(0))

    assert app.messages[0]["content"] == "hola"
    assert app.messages[0]["thinking"] == "pensando"
    assert app._streaming_active is False
