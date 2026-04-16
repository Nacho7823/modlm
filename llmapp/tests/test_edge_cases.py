"""Additional tests for llmapp - TUI and edge cases."""

import pytest
from llmapp.config import ConfigManager
from llmapp.history import ConversationManager
from llmapp.command import CommandHandler, Command


class TestConfigManagerEdgeCases:
    """Edge case tests for ConfigManager."""

    def test_load_with_missing_env_file(self, tmp_path):
        manager = ConfigManager(tmp_path)
        config = manager.load()
        assert config["api_url"] == "http://127.0.0.1:1234/v1"

    def test_set_and_get(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        manager.set("custom_key", "custom_value")
        assert manager.get("custom_key") == "custom_value"

    def test_get_with_default(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        assert manager.get("nonexistent", "default") == "default"

    def test_add_multiple_mcp_servers(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        manager._config["mcp_servers"] = {}
        manager.add_mcp_server("server1", "http://localhost:8080")
        manager.add_mcp_server("server2", "http://localhost:8081")
        servers = manager.get_mcp_servers()
        assert len(servers) == 2
        assert "server1" in servers
        assert "server2" in servers

    def test_remove_nonexistent_mcp_server(self, tmp_path):
        manager = ConfigManager(tmp_path)
        manager.load()
        result = manager.remove_mcp_server("nonexistent")
        assert result is False


class TestConversationManagerEdgeCases:
    """Edge case tests for ConversationManager."""

    def test_save_empty_messages(self, tmp_path):
        manager = ConversationManager(tmp_path)
        manager.save("empty_conv", [])
        loaded = manager.load("empty_conv")
        assert loaded == []

    def test_load_nonexistent(self, tmp_path):
        manager = ConversationManager(tmp_path)
        assert manager.load("nonexistent") is None

    def test_delete_nonexistent(self, tmp_path):
        manager = ConversationManager(tmp_path)
        assert manager.delete("nonexistent") is False

    def test_special_characters_in_name(self, tmp_path):
        manager = ConversationManager(tmp_path)
        manager.save("test-conversation_123", [{"role": "user", "content": "test"}])
        assert manager.exists("test-conversation_123")

    def test_message_with_special_characters(self, tmp_path):
        manager = ConversationManager(tmp_path)
        messages = [
            {"role": "user", "content": "Hello <world> & 'test'"},
            {"role": "assistant", "content": "Response with\nnewlines"},
        ]
        manager.save("special", messages)
        loaded = manager.load("special")
        assert loaded == messages


class TestCommandHandlerEdgeCases:
    """Edge case tests for CommandHandler."""

    def test_parse_empty_string(self):
        handler = CommandHandler()
        assert handler.parse("") is None

    def test_parse_whitespace_only(self):
        handler = CommandHandler()
        assert handler.parse("   ") is None

    def test_parse_command_with_extra_spaces(self):
        handler = CommandHandler()
        cmd = handler.parse("/mcp   add   server   url")
        assert cmd is not None
        assert cmd.name == "mcp"
        assert cmd.args == ["add", "server", "url"]

    def test_execute_mcp_without_args(self):
        handler = CommandHandler()
        result = handler.execute(Command(name="mcp", args=[], raw="/mcp"))
        assert "Usage" in result

    def test_execute_session_without_args(self):
        handler = CommandHandler()
        result = handler.execute(Command(name="session", args=[], raw="/session"))
        assert (
            "Usage" in result or "No saved" in result or "Saved conversations" in result
        )

    def test_execute_mcp_add_missing_args(self):
        handler = CommandHandler()
        result = handler.execute(Command(name="mcp", args=["add"], raw="/mcp add"))
        assert "Usage" in result

    def test_execute_mcp_add_only_name(self):
        handler = CommandHandler()
        result = handler.execute(
            Command(name="mcp", args=["add", "server"], raw="/mcp add server")
        )
        assert "Usage" in result


class TestMessageView:
    """Tests for MessageView widget."""

    def test_message_view_user_role(self):
        from llmapp.widgets import MessageView

        view = MessageView(role="user", content="Hello")
        assert view.role == "user"
        assert view.content == "Hello"

    def test_message_view_assistant_role(self):
        from llmapp.widgets import MessageView

        view = MessageView(role="assistant", content="Response")
        assert view.role == "assistant"
        assert view.content == "Response"

    def test_message_view_system_role(self):
        from llmapp.widgets import MessageView

        view = MessageView(role="system", content="System message")
        assert view.role == "system"
        assert view.content == "System message"


class TestChatContainer:
    """Tests for ChatContainer widget."""

    def test_chat_container_init(self):
        from llmapp.widgets import ChatContainer

        container = ChatContainer()
        assert container is not None


class TestChatInput:
    """Tests for ChatInput widget."""

    def test_chat_input_init(self):
        from llmapp.widgets import ChatInput

        inp = ChatInput()
        assert inp is not None

    def test_chat_input_init_with_custom_placeholder(self):
        from llmapp.widgets import ChatInput

        inp = ChatInput(placeholder="Custom placeholder")
        assert inp is not None


class TestCommandExecutionWithHandlers:
    """Tests for command execution with actual handlers."""

    def test_mcp_list_handler_returns_dict(self, chat_app):
        app = chat_app
        result = app._list_mcp_servers()
        assert isinstance(result, dict)

    def test_session_list_handler_returns_list(self, chat_app):
        app = chat_app
        result = app._list_sessions()
        assert isinstance(result, list)

    def test_load_and_save_session(self, chat_app):
        app = chat_app
        app.messages = [{"role": "user", "content": "test"}]
        app.history_manager.save("test_session", app.messages)
        loaded = app.history_manager.load("test_session")
        assert len(loaded) == 1
        assert loaded[0]["content"] == "test"

    def test_delete_and_verify(self, chat_app):
        app = chat_app
        app.history_manager.save("to_delete", [{"role": "user", "content": "test"}])
        app._delete_session("to_delete")
        assert app.history_manager.load("to_delete") is None


class TestConfigPersistence:
    """Tests for configuration persistence."""

    def test_config_survives_reload(self, tmp_path):
        manager1 = ConfigManager(tmp_path)
        manager1.load()
        manager1.set("test_key", "test_value")
        manager1.save()

        manager2 = ConfigManager(tmp_path)
        config2 = manager2.load()
        assert config2.get("test_key") == "test_value"

    def test_mcp_servers_persist(self, tmp_path):
        manager1 = ConfigManager(tmp_path)
        manager1.load()
        manager1.add_mcp_server("persist", "http://localhost:9999")
        manager1.save()

        manager2 = ConfigManager(tmp_path)
        config2 = manager2.load()
        assert "persist" in config2["mcp_servers"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
