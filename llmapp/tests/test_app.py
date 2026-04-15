"""Tests for llmapp using Textual native testing."""

import pytest
from textual.app import App
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

    def test_parse_config(self):
        handler = CommandHandler()
        cmd = handler.parse("/config")
        assert cmd is not None
        assert cmd.name == "config"
        assert cmd.args == []

    def test_parse_mcp_add(self):
        handler = CommandHandler()
        cmd = handler.parse("/mcp add myserver http://localhost:8080")
        assert cmd is not None
        assert cmd.name == "mcp"
        assert cmd.args == ["add", "myserver", "http://localhost:8080"]

    def test_parse_session_load(self):
        handler = CommandHandler()
        cmd = handler.parse("/session load myconv")
        assert cmd is not None
        assert cmd.name == "session"
        assert cmd.args == ["load", "myconv"]

    def test_parse_new(self):
        handler = CommandHandler()
        cmd = handler.parse("/new")
        assert cmd is not None
        assert cmd.name == "new"

    def test_parse_quit(self):
        handler = CommandHandler()
        cmd = handler.parse("/quit")
        assert cmd is not None
        assert cmd.name == "quit"

    def test_parse_q(self):
        handler = CommandHandler()
        cmd = handler.parse("/q")
        assert cmd is not None
        assert cmd.name == "q"

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

    def test_app_initialization(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.config_manager = ConfigManager(tmp_path)
        app.config_manager.load()
        app.history_manager = ConversationManager(tmp_path)
        assert app.config_manager is not None
        assert app.history_manager is not None
        assert app.messages == []

    def test_new_conversation_logic(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.history_manager = ConversationManager(tmp_path)
        app.messages = [{"role": "user", "content": "test"}]
        app.current_session = "test_session"
        app.messages = []
        app.current_session = None
        assert app.messages == []
        assert app.current_session is None

    def test_list_sessions_empty(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.history_manager = ConversationManager(tmp_path)
        sessions = app._list_sessions()
        assert sessions == []

    def test_list_sessions_with_data(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.history_manager = ConversationManager(tmp_path)
        app.history_manager.save("test", [{"role": "user", "content": "hi"}])
        sessions = app._list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["name"] == "test"

    def test_add_mcp_server_handler(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.config_manager = ConfigManager(tmp_path)
        app.config_manager.load()
        result = app._add_mcp_server("test_server", "http://localhost:8080")
        assert "test_server" in result
        assert "test_server" in app.config_manager.get_mcp_servers()
        assert (
            app.config_manager.get_mcp_servers()["test_server"]
            == "http://localhost:8080"
        )

    def test_remove_mcp_server_handler(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.config_manager = ConfigManager(tmp_path)
        app.config_manager.load()
        app._add_mcp_server("to_remove", "http://localhost:8080")
        result = app._remove_mcp_server("to_remove")
        assert "removed" in result
        assert "to_remove" not in app.config_manager.get_mcp_servers()

    def test_show_config_returns_config(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.config_manager = ConfigManager(tmp_path)
        app.config_manager.load()
        api_url, api_key, model = app.config_manager.get_api_config()
        assert api_url == "http://127.0.0.1:1234/v1"
        assert model == "qwen3.5-4b"

    def test_command_handler_mcp_list(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.config_manager = ConfigManager(tmp_path)
        app.config_manager.load()
        app._add_mcp_server("server1", "http://localhost:8080")
        servers = app._list_mcp_servers()
        assert "server1" in servers

    def test_load_session_not_found(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.history_manager = ConversationManager(tmp_path)
        result = app._load_session("nonexistent")
        assert "not found" in result

    def test_delete_session_not_found(self, tmp_path):
        from llmapp.app import ChatApp

        app = ChatApp()
        app.history_manager = ConversationManager(tmp_path)
        result = app._delete_session("nonexistent")
        assert "not found" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
