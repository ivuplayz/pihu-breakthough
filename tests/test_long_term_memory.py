"""Unit and integration test suite for Intelligent Long-Term Memory.

Tests:
- ConversationRepository.save_memory() and get_memories() with parameterized SQL.
- save_core_memory tool registration, schema, and execution in ToolRegistry.
- Automatic memory retrieval and dynamic system prompt injection in POST /api/chat (both sync and SSE).
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
import uuid

from app import create_app
from pihu_core.config import Config
from pihu_core.repository import ConversationRepository
from pihu_core.tools import save_core_memory, tool_registry


class TestConversationRepositoryMemory(unittest.TestCase):
    """Unit tests for ConversationRepository memory methods."""

    def test_save_memory_success(self) -> None:
        """save_memory executes parameterized SQL insert and returns record."""
        mock_db = MagicMock()
        mock_db.is_configured.return_value = True

        test_user_id = str(uuid.uuid4())
        test_mem_id = str(uuid.uuid4())
        mock_db.execute_query.return_value = [
            {
                "id": uuid.UUID(test_mem_id),
                "user_id": uuid.UUID(test_user_id),
                "category": "preference",
                "content": "Prefers Python over JavaScript",
                "timestamp": "2026-09-23T19:00:00Z",
            }
        ]

        repo = ConversationRepository(database_manager=mock_db)
        saved = repo.save_memory(
            user_id=test_user_id,
            category="Preference",
            content="Prefers Python over JavaScript",
        )

        self.assertEqual(saved["id"], test_mem_id)
        self.assertEqual(saved["user_id"], test_user_id)
        self.assertEqual(saved["category"], "preference")
        self.assertEqual(saved["content"], "Prefers Python over JavaScript")

        # Verify SQL parameter binding
        call_args = mock_db.execute_query.call_args[0]
        sql = call_args[0]
        params = call_args[1]
        self.assertIn("INSERT INTO memories", sql)
        self.assertEqual(params, (test_user_id, "preference", "Prefers Python over JavaScript"))

    def test_save_memory_empty_content_raises(self) -> None:
        """save_memory raises ValueError when content is blank."""
        mock_db = MagicMock()
        mock_db.is_configured.return_value = True
        repo = ConversationRepository(database_manager=mock_db)

        with self.assertRaises(ValueError):
            repo.save_memory(content="   ")

    def test_save_memory_unconfigured_raises(self) -> None:
        """save_memory raises RuntimeError if database is unconfigured."""
        mock_db = MagicMock()
        mock_db.is_configured.return_value = False
        repo = ConversationRepository(database_manager=mock_db)

        with self.assertRaises(RuntimeError):
            repo.save_memory(content="Test fact")

    def test_get_memories_success(self) -> None:
        """get_memories retrieves memories ordered by timestamp descending."""
        mock_db = MagicMock()
        mock_db.is_configured.return_value = True

        user_id = str(uuid.uuid4())
        mem_id_1 = str(uuid.uuid4())
        mem_id_2 = str(uuid.uuid4())

        mock_db.execute_query.return_value = [
            {
                "id": uuid.UUID(mem_id_1),
                "user_id": uuid.UUID(user_id),
                "category": "project",
                "content": "Building Pihu-BreakThough on Vercel",
                "timestamp": "2026-09-23T19:05:00Z",
            },
            {
                "id": uuid.UUID(mem_id_2),
                "user_id": uuid.UUID(user_id),
                "category": "preference",
                "content": "Prefers dark mode UI",
                "timestamp": "2026-09-23T19:00:00Z",
            },
        ]

        repo = ConversationRepository(database_manager=mock_db)
        memories = repo.get_memories(user_id=user_id, limit=10)

        self.assertEqual(len(memories), 2)
        self.assertEqual(memories[0]["id"], mem_id_1)
        self.assertEqual(memories[1]["id"], mem_id_2)

        call_args = mock_db.execute_query.call_args[0]
        sql = call_args[0]
        params = call_args[1]
        self.assertIn("SELECT id, user_id, category, content, timestamp", sql)
        self.assertIn("FROM memories", sql)
        self.assertEqual(params, (user_id, 10))


class TestSaveCoreMemoryTool(unittest.TestCase):
    """Unit tests for save_core_memory tool and ToolRegistry integration."""

    def test_tool_is_registered(self) -> None:
        """save_core_memory tool is registered in global tool_registry."""
        tool = tool_registry.get_tool("save_core_memory")
        self.assertIsNotNone(tool)
        schema = tool_registry.get_schema("save_core_memory")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["name"], "save_core_memory")
        self.assertIn("category", schema["parameters"]["properties"])
        self.assertIn("content", schema["parameters"]["properties"])

    @patch("pihu_core.repository.ConversationRepository")
    def test_save_core_memory_execution_success(self, mock_repo_cls: MagicMock) -> None:
        """Executing save_core_memory via tool_registry successfully saves to repo."""
        mock_repo = MagicMock()
        mock_repo.is_available.return_value = True
        mock_repo.save_memory.return_value = {"id": str(uuid.uuid4())}
        mock_repo_cls.return_value = mock_repo

        result = tool_registry.execute(
            "save_core_memory",
            category="preference",
            content="User prefers concise answers",
        )

        self.assertIn("Successfully saved to long-term memory", result)
        self.assertIn("preference", result)
        self.assertIn("User prefers concise answers", result)
        mock_repo.save_memory.assert_called_once_with(
            category="preference",
            content="User prefers concise answers",
        )

    def test_save_core_memory_empty_content_returns_error(self) -> None:
        """Empty content returns descriptive error message without raising exception."""
        result = save_core_memory(category="fact", content="   ")
        self.assertIn("Error: Memory content cannot be empty", result)

    @patch("pihu_core.repository.ConversationRepository")
    def test_save_core_memory_unconfigured_db_message(self, mock_repo_cls: MagicMock) -> None:
        """Unconfigured database returns friendly status string."""
        mock_repo = MagicMock()
        mock_repo.is_available.return_value = False
        mock_repo_cls.return_value = mock_repo

        result = save_core_memory(category="fact", content="Some fact")
        self.assertIn("Memory storage is currently unavailable", result)


class TestMemoryContextInjection(unittest.TestCase):
    """Integration tests for memory injection into system prompt in POST /api/chat."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()

    @patch("api.chat.router")
    def test_sync_chat_injects_memories_into_system_prompt(self, mock_router: MagicMock) -> None:
        """Existing user memories are retrieved and formatted into system_prompt for router.route_chat."""
        conv_uuid = str(uuid.uuid4())
        user_uuid = str(uuid.uuid4())

        mock_repo = MagicMock()
        mock_repo.is_available.return_value = True
        mock_repo.create_conversation.return_value = {"id": conv_uuid, "user_id": user_uuid, "title": "Chat"}
        mock_repo.get_messages.return_value = []
        mock_repo.save_message.return_value = {"id": str(uuid.uuid4())}
        mock_repo.get_memories.return_value = [
            {"category": "preference", "content": "Prefers Python"},
            {"category": "project", "content": "Developing Pihu AI"},
        ]

        mock_router.route_chat.return_value = {
            "ok": True,
            "response": "I know you're developing Pihu AI in Python!",
            "provider": "stage1-deterministic",
            "model": "deterministic-v1",
            "metadata": {},
        }

        with patch("api.chat.conversation_repo", mock_repo), \
             patch.object(Config, "GEMINI_API_KEY", ""), \
             patch.object(Config, "GROQ_API_KEY", ""):
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": "What am I working on?", "stream": False}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)

            # Verify memories were queried for the user
            mock_repo.get_memories.assert_called_once_with(user_id=user_uuid)

            # Verify system_prompt passed to route_chat contains the injected memories
            call_kwargs = mock_router.route_chat.call_args[1]
            sys_prompt = call_kwargs.get("system_prompt")
            self.assertIsNotNone(sys_prompt)
            self.assertIn("User Context/Memories:", sys_prompt)
            self.assertIn("Prefers Python", sys_prompt)
            self.assertIn("Developing Pihu AI", sys_prompt)
            self.assertIn("save_core_memory", sys_prompt)

    @patch("api.chat.router")
    def test_sse_stream_injects_memories_into_system_prompt(self, mock_router: MagicMock) -> None:
        """Existing user memories are injected into system_prompt for router.route_stream."""
        conv_uuid = str(uuid.uuid4())
        user_uuid = str(uuid.uuid4())

        mock_repo = MagicMock()
        mock_repo.is_available.return_value = True
        mock_repo.create_conversation.return_value = {"id": conv_uuid, "user_id": user_uuid, "title": "Stream"}
        mock_repo.get_messages.return_value = []
        mock_repo.save_message.return_value = {"id": str(uuid.uuid4())}
        mock_repo.get_memories.return_value = [
            {"category": "fact", "content": "User is named Ivan"},
        ]

        mock_router.route_stream.return_value = [
            {"ok": True, "text": "Hello Ivan!", "is_final": True, "provider": "stage1-deterministic"}
        ]

        with patch("api.chat.conversation_repo", mock_repo), \
             patch.object(Config, "GEMINI_API_KEY", ""), \
             patch.object(Config, "GROQ_API_KEY", ""):
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": "Greet me", "stream": True}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "text/event-stream")

            # Verify system_prompt passed to route_stream contains User Context/Memories
            call_kwargs = mock_router.route_stream.call_args[1]
            sys_prompt = call_kwargs.get("system_prompt")
            self.assertIsNotNone(sys_prompt)
            self.assertIn("User Context/Memories:", sys_prompt)
            self.assertIn("User is named Ivan", sys_prompt)


if __name__ == "__main__":
    unittest.main()
