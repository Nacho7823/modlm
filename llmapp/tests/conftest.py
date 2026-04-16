"""Test configuration for llmapp."""

import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

from llmapp.app import ChatApp
from llmapp.config import ConfigManager
from llmapp.history import ConversationManager

project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

sys.path.insert(0, str(project_root))


@pytest.fixture
def chat_app(tmp_path):
    app = ChatApp()
    app.config_manager = ConfigManager(tmp_path)
    app.config_manager.load()
    app.history_manager = ConversationManager(tmp_path)
    return app
