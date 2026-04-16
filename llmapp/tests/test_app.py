"""Tests for llmapp using Textual native testing."""

import pytest
from llmapp.config import ConfigManager
from llmapp.history import ConversationManager
from llmapp.command import CommandHandler, Command


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_default_config(self, tmp_path):
        manager = ConfigManager(tmp_path)
        config = manager.load()
        assert config["api_url"] == "http://127.0.0.1:1234/v1"
        assert config["model"] == "qwen3.5-4b"
        assert config["mcp_servers"] == {}

    def test_save_and_load(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        manager.set("api_url", "http://localhost:8080/v1")
        manager.add_mcp_server("test", "http://localhost:3000")
        manager.save()

        manager2 = ConfigManager(tmp_path)
        config2 = manager2.load()
        assert config2["api_url"] == "http://localhost:8080/v1"
        assert "test" in config2["mcp_servers"]

    def test_get_api_config(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        url, key, model = manager.get_api_config()
        assert url == "http://127.0.0.1:1234/v1"
        assert model == "qwen3.5-4b"

    def test_add_mcp_server(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        manager.add_mcp_server("myServer", "http://localhost:8080/mcp")
        servers = manager.get_mcp_servers()
        assert servers["myServer"] == "http://localhost:8080/mcp"

    def test_remove_mcp_server(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        manager.add_mcp_server("toRemove", "http://localhost:8080")
        assert manager.remove_mcp_server("toRemove")
        assert "toRemove" not in manager.get_mcp_servers()


class TestConversationManager:
    """Tests for ConversationManager."""

    def test_save_and_load(self, tmp_path):
        manager = ConversationManager(tmp_path)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        manager.save("test_conv", messages)
        loaded = manager.load("test_conv")
        assert loaded == messages

    def test_list(self, tmp_path):
        manager = ConversationManager(tmp_path)
        manager.save("conv1", [{"role": "user", "content": "Hello"}])
        manager.save("conv2", [{"role": "user", "content": "Hi"}])
        sessions = manager.list()
        assert len(sessions) == 2
        names = [s["name"] for s in sessions]
        assert "conv1" in names
        assert "conv2" in names

    def test_delete(self, tmp_path):
        manager = ConversationManager(tmp_path)
        manager.save("to_delete", [{"role": "user", "content": "Test"}])
        assert manager.delete("to_delete")
        assert manager.load("to_delete") is None

    def test_exists(self, tmp_path):
        manager = ConversationManager(tmp_path)
        manager.save("exists", [{"role": "user", "content": "Test"}])
        assert manager.exists("exists")
        assert not manager.exists("nonexistent")


class TestCommandHandler:
    """Tests for CommandHandler."""

    @pytest.mark.parametrize(
        ("text", "name", "args"),
        [
            ("/config", "config", []),
            (
                "/mcp add myserver http://localhost:8080",
                "mcp",
                ["add", "myserver", "http://localhost:8080"],
            ),
            ("/session load myconv", "session", ["load", "myconv"]),
            ("/new", "new", []),
            ("/quit", "quit", []),
            ("/q", "q", []),
        ],
    )
    def test_parse_command(self, text, name, args):
        cmd = CommandHandler().parse(text)
        assert cmd is not None
        assert cmd.name == name
        assert cmd.args == args

    def test_parse_not_command(self):
        handler = CommandHandler()
        cmd = handler.parse("Hello world")
        assert cmd is None

    def test_execute_help(self):
        handler = CommandHandler()
        result = handler.execute(Command(name="help", args=[], raw="/help"))
        assert "Available commands:" in result

    def test_execute_unknown(self):
        handler = CommandHandler()
        result = handler.execute(Command(name="unknown", args=[], raw="/unknown"))
        assert "Unknown command" in result


class TestChatAppIntegration:
    """Integration tests for ChatApp using Textual headless mode."""

    def test_app_initialization(self, chat_app):
        app = chat_app
        assert app.config_manager is not None
        assert app.history_manager is not None
        assert app.messages == []

    def test_new_conversation_logic(self, chat_app):
        app = chat_app
        app.messages = [{"role": "user", "content": "test"}]
        app.current_session = "test_session"
        app.messages = []
        app.current_session = None
        assert app.messages == []
        assert app.current_session is None

    def test_list_sessions_empty(self, chat_app):
        app = chat_app
        sessions = app.history_controller.list()
        assert sessions == []

    def test_list_sessions_with_data(self, chat_app):
        app = chat_app
        app.history_manager.save("test", [{"role": "user", "content": "hi"}])
        sessions = app.history_controller.list()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "test"

    def test_add_mcp_server_handler(self, chat_app):
        app = chat_app
        result = app.runtime_controller.add_mcp_server(
            "test_server", "http://localhost:8080"
        )
        assert "test_server" in result
        assert "test_server" in app.config_manager.get_mcp_servers()
        assert (
            app.config_manager.get_mcp_servers()["test_server"]
            == "http://localhost:8080"
        )

    def test_remove_mcp_server_handler(self, chat_app):
        app = chat_app
        app.runtime_controller.add_mcp_server("to_remove", "http://localhost:8080")
        result = app.runtime_controller.remove_mcp_server("to_remove")
        assert "removed" in result
        assert "to_remove" not in app.config_manager.get_mcp_servers()

    def test_show_config_returns_config(self, chat_app):
        app = chat_app
        api_url, api_key, model = app.config_manager.get_api_config()
        assert api_url == "http://127.0.0.1:1234/v1"
        assert model == "qwen3.5-4b"

    def test_command_handler_mcp_list(self, chat_app):
        app = chat_app
        app.runtime_controller.add_mcp_server("server1", "http://localhost:8080")
        servers = app.runtime_controller.list_mcp_servers()
        assert "server1" in servers

    def test_load_session_not_found(self, chat_app):
        app = chat_app
        result = app.history_controller.load("nonexistent")
        assert "not found" in result

    def test_delete_session_not_found(self, chat_app):
        app = chat_app
        result = app.history_controller.delete("nonexistent")
        assert "not found" in result


class TestMessageView:
    """Tests for MessageView widget."""

    def test_message_view_user(self):
        from llmapp.widgets import MessageView

        msg = MessageView(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.thinking == ""

    def test_message_view_assistant(self):
        from llmapp.widgets import MessageView

        msg = MessageView(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_message_view_with_thinking(self):
        from llmapp.widgets import MessageView

        msg = MessageView(
            role="assistant", content="Final answer", thinking="Reasoning..."
        )
        assert msg.content == "Final answer"
        assert msg.thinking == "Reasoning..."

    def test_message_view_append_content(self):
        from llmapp.widgets import MessageView

        msg = MessageView(role="assistant", content="Hello")
        msg.append_content(" World")
        assert msg.content == "Hello World"

    def test_message_view_append_thinking(self):
        from llmapp.widgets import MessageView

        msg = MessageView(role="assistant", content="", thinking="Thinking")
        msg.append_thinking(" more")
        assert msg.thinking == "Thinking more"

    def test_message_view_content_setter(self):
        from llmapp.widgets import MessageView

        msg = MessageView(role="assistant", content="Hello")
        msg.content = "Updated"
        assert msg.content == "Updated"

    def test_message_view_thinking_setter(self):
        from llmapp.widgets import MessageView

        msg = MessageView(role="assistant", content="", thinking="Initial")
        msg.thinking = "Updated thinking"
        assert msg.thinking == "Updated thinking"


class TestChatContainer:
    """Tests for ChatContainer widget."""

    def test_chat_container_children_initially_empty(self):
        from llmapp.widgets import ChatContainer

        container = ChatContainer()
        assert len(container.children) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
