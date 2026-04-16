"""Streaming lifecycle controller for ChatApp."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from llmapp.constants import THINKING_PLACEHOLDER
from llmlib import StreamEvent

if TYPE_CHECKING:
    from llmapp.app import ChatApp


class StreamController:
    """Handles send/stream/update lifecycle for assistant responses."""

    def __init__(self, app: "ChatApp") -> None:
        self._app = app

    async def send_message(self, text: str) -> None:
        self._app._add_message("user", text)
        msg_idx = self._create_assistant_placeholder()
        self._app._streaming_active = True
        self._app._streaming_task = asyncio.create_task(self._run_stream(msg_idx))

    def cancel_active_stream(self, notify: bool = True) -> None:
        task = self._app._streaming_task
        if task and not task.done():
            task.cancel()
            self._app._streaming_active = False
            if notify:
                self._app._add_message("system", "Generation cancelled.")
        elif self._app._streaming_active:
            self._app._streaming_active = False

    def _create_assistant_placeholder(self) -> int:
        msg_idx = len(self._app.messages)
        self._app.messages.append({"role": "assistant", "content": "", "thinking": ""})
        container = self._app._get_chat_container()
        container.add_message("assistant", "")
        self._append_thinking(THINKING_PLACEHOLDER)
        return msg_idx

    async def _run_stream(self, msg_idx: int) -> None:
        got_output = False
        try:
            streaming_enabled = bool(self._app.config_manager.get("streaming", True))
            async for event in self._app.runtime.stream(
                self._app.messages,
                streaming_enabled=streaming_enabled,
            ):
                got_output = self._apply_event(
                    msg_idx,
                    event,
                    got_output,
                )
        except asyncio.CancelledError:
            self._apply_cancelled_fallback(msg_idx)
            # Re-raise to ensure the task is correctly stopped
            raise
        except Exception as error:
            error_msg = str(error)
            if "Attempted to exit cancel scope" in error_msg:
                # Masking the anyio task mismatch error for the user but logging it
                error_msg = "MCP Client Connection mismatch (internal error)"
            
            error_text = f"\n[bold red]⚠ Error:[/bold red] {error_msg}"
            self._app.messages[msg_idx]["content"] += error_text
            self._append_content(error_text)
            got_output = True
        finally:
            self._finalize_fallback(msg_idx, got_output)
            self._app._streaming_active = False

    def _apply_event(
        self,
        msg_idx: int,
        event: StreamEvent,
        got_output: bool,
    ) -> bool:
        if event.kind == "content" and event.text:
            got_output = True
            self._app.messages[msg_idx]["content"] += event.text
            self._append_content(event.text)

        if event.kind == "thinking" and event.text:
            got_output = True
            existing = self._app.messages[msg_idx].get("thinking", "")
            self._app.messages[msg_idx]["thinking"] = existing + event.text
            self._append_thinking(event.text)

        if event.kind == "error" and event.text:
            got_output = True
            self._app.messages[msg_idx]["content"] += event.text
            self._append_content(event.text)

        return got_output

    def _apply_cancelled_fallback(self, msg_idx: int) -> None:
        if self._app.messages[msg_idx].get("content", "") != "":
            return
        fallback = "Generation cancelled."
        self._app.messages[msg_idx]["content"] = fallback
        self._replace_last_view(fallback)

    def _finalize_fallback(self, msg_idx: int, got_output: bool) -> None:
        if got_output or self._app.messages[msg_idx].get("content"):
            return
        fallback = "No response returned by model (empty completion)."
        self._app.messages[msg_idx]["content"] = fallback
        self._replace_last_view(fallback)

    def _append_content(self, new_content: str) -> None:
        container = self._app._get_chat_container()
        view = container.last_assistant_view()
        if view:
            if view.thinking == THINKING_PLACEHOLDER:
                view.thinking = ""
            view.append_content(new_content)

    def _append_thinking(self, new_thinking: str) -> None:
        container = self._app._get_chat_container()
        view = container.last_assistant_view()
        if view:
            if view.thinking == THINKING_PLACEHOLDER:
                view.thinking = new_thinking
            else:
                view.append_thinking(new_thinking)

    def _replace_last_view(self, content: str) -> None:
        container = self._app._get_chat_container()
        view = container.last_assistant_view()
        if view:
            view.thinking = ""
            view.content = content
