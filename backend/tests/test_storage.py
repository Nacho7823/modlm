import pytest
import os
import tempfile
from pathlib import Path
import shutil


class TestChatStorage:
    @pytest.fixture
    def storage(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        os.environ["MODLLM_DATA_DIR"] = str(data_dir)

        from backend.app.services import chat_storage

        original_dir = chat_storage._data_dir
        chat_storage._data_dir = data_dir

        yield chat_storage

        chat_storage._data_dir = original_dir

    def test_create_chat(self, storage):
        chat = storage.create_chat()
        assert chat.id is not None
        assert chat.title.startswith("Chat")

    def test_get_chat_exists(self, storage):
        chat = storage.create_chat()
        retrieved = storage.get_chat(chat.id)
        assert retrieved.id == chat.id

    def test_not_found(self, storage):
        with pytest.raises(Exception):
            storage.get_chat("invalid-id")
