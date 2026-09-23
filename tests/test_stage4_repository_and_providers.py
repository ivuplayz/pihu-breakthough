"""Integration and unit tests for Stage 4: Database Schemas, Repository, and Live AI Providers."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
import uuid

from app import create_app
from pihu_core.config import Config
from pihu_core.database import DatabaseManager, db_manager
from pihu_core.providers import GeminiProvider, GroqProvider, ProviderResponse, StreamChunk
from pihu_core.repository import ConversationRepository, conversation_repo


class TestStage4RepositoryAndProviders(unittest.TestCase):
    """Test suite covering database repository persistence and live AI providers."""

    def setUp(self) -> None:
        """Initialize test client and mock database manager."""
        class TestConfig(Config):
            TESTING = True
            DEBUG = False
            DATABASE_URL = None

        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

    # --------------------------------------------------------------------------
    # 1. Database Migrations & Schema Execution Tests
    # --------------------------------------------------------------------------
    @patch.object(DatabaseManager, "is_configured", return_value=True)
    def test_run_migrations(self, mock_is_configured: MagicMock) -> None:
        """run_migrations must read schema.sql and execute it on the connection cursor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

        mgr = DatabaseManager(database_url="postgresql://user:pass@host/db")
        with patch.object(mgr, "get_connection") as mock_get_conn:
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            success = mgr.run_migrations()
            self.assertTrue(success)
            mock_cursor.execute.assert_called_once()
            executed_sql = mock_cursor.execute.call_args[0][0]
            self.assertIn("CREATE TABLE IF NOT EXISTS users", executed_sql)
            self.assertIn("CREATE TABLE IF NOT EXISTS conversations", executed_sql)
            self.assertIn("CREATE TABLE IF NOT EXISTS messages", executed_sql)
            self.assertIn("CREATE TABLE IF NOT EXISTS memories", executed_sql)

    def test_run_migrations_unconfigured_raises(self) -> None:
        """run_migrations must raise RuntimeError if database is unconfigured."""
        mgr = DatabaseManager(database_url=None)
        with self.assertRaises(RuntimeError):
            mgr.run_migrations()

    # --------------------------------------------------------------------------
    # 2. ConversationRepository Unit Tests
    # --------------------------------------------------------------------------
    def test_repository_get_or_create_default_user_existing(self) -> None:
        """Repository retrieves existing default user if one exists."""
        mock_mgr = MagicMock()
        user_uuid = uuid.uuid4()
        mock_mgr.is_configured.return_value = True
        mock_mgr.execute_query.return_value = [{"id": user_uuid, "created_at": "2026-01-01", "preferences": {}}]

        repo = ConversationRepository(database_manager=mock_mgr)
        user = repo.get_or_create_default_user()
        self.assertEqual(user["id"], str(user_uuid))
        mock_mgr.execute_query.assert_called_once()

    def test_repository_get_or_create_default_user_creates_new(self) -> None:
        """Repository creates a default user if none exists."""
        mock_mgr = MagicMock()
        new_uuid = uuid.uuid4()
        mock_mgr.is_configured.return_value = True
        # First query returns empty, second query (insert) returns created user
        mock_mgr.execute_query.side_effect = [
            [],
            [{"id": new_uuid, "created_at": "2026-01-01", "preferences": {"name": "Default User"}}],
        ]

        repo = ConversationRepository(database_manager=mock_mgr)
        user = repo.get_or_create_default_user()
        self.assertEqual(user["id"], str(new_uuid))
        self.assertEqual(mock_mgr.execute_query.call_count, 2)

    def test_repository_create_conversation(self) -> None:
        """Repository creates a conversation and returns formatted UUIDs."""
        mock_mgr = MagicMock()
        conv_uuid = uuid.uuid4()
        user_uuid = uuid.uuid4()
        mock_mgr.is_configured.return_value = True
        mock_mgr.execute_query.return_value = [
            {"id": conv_uuid, "user_id": user_uuid, "title": "Test Chat", "created_at": "2026-01-01"}
        ]

        repo = ConversationRepository(database_manager=mock_mgr)
        conv = repo.create_conversation(user_id=str(user_uuid), title="Test Chat")
        self.assertEqual(conv["id"], str(conv_uuid))
        self.assertEqual(conv["user_id"], str(user_uuid))

    def test_repository_save_and_get_messages(self) -> None:
        """Repository saves messages and retrieves history in chronological order."""
        mock_mgr = MagicMock()
        msg_uuid = uuid.uuid4()
        conv_uuid = uuid.uuid4()
        mock_mgr.is_configured.return_value = True

        mock_mgr.execute_query.side_effect = [
            [{"id": msg_uuid, "conversation_id": conv_uuid, "role": "user", "content": "Hi", "provider_used": None, "timestamp": "2026-01-01"}],
            [{"id": msg_uuid, "conversation_id": conv_uuid, "role": "user", "content": "Hi", "provider_used": None, "timestamp": "2026-01-01"}],
        ]

        repo = ConversationRepository(database_manager=mock_mgr)
        saved = repo.save_message(conversation_id=str(conv_uuid), role="user", content="Hi")
        self.assertEqual(saved["id"], str(msg_uuid))

        history = repo.get_messages(conversation_id=str(conv_uuid))
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["content"], "Hi")

    # --------------------------------------------------------------------------
    # 3. Live AI SDK Provider Tests (Mocked External Calls)
    # --------------------------------------------------------------------------
    @patch("pihu_core.providers.genai.Client")
    def test_gemini_provider_generate(self, mock_client_cls: MagicMock) -> None:
        """GeminiProvider correctly formats contents and returns ProviderResponse."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini 2.5 Flash!"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "GEMINI_API_KEY", "mock-gemini-key"):
            provider = GeminiProvider(model_name="gemini-2.5-flash")
            self.assertTrue(provider.is_available())

            res = provider.generate(
                message="Tell me a joke",
                history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            )
            self.assertEqual(res.content, "Hello from Gemini 2.5 Flash!")
            self.assertEqual(res.provider, "gemini")
            self.assertEqual(res.model, "gemini-2.5-flash")

            # Verify contents mapping
            call_kwargs = mock_client.models.generate_content.call_args[1]
            contents = call_kwargs["contents"]
            self.assertEqual(len(contents), 3)
            self.assertEqual(contents[0]["role"], "user")
            self.assertEqual(contents[1]["role"], "model")  # Assistant mapped to model
            self.assertEqual(contents[2]["role"], "user")

    @patch("pihu_core.providers.genai.Client")
    def test_gemini_provider_stream_generate(self, mock_client_cls: MagicMock) -> None:
        """GeminiProvider yields stream chunks correctly."""
        mock_client = MagicMock()
        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "world!"
        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "GEMINI_API_KEY", "mock-gemini-key"):
            provider = GeminiProvider()
            chunks = list(provider.stream_generate(message="Hi"))
            self.assertEqual(len(chunks), 3)  # chunk1, chunk2, is_final chunk
            self.assertEqual(chunks[0].text, "Hello ")
            self.assertEqual(chunks[1].text, "world!")
            self.assertTrue(chunks[2].is_final)

    @patch("pihu_core.providers.Groq")
    def test_groq_provider_generate(self, mock_groq_cls: MagicMock) -> None:
        """GroqProvider correctly formats messages and returns ProviderResponse."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Ultra fast reply from Groq!"
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=8, total_tokens=18)
        mock_client.chat.completions.create.return_value = mock_completion
        mock_groq_cls.return_value = mock_client

        with patch.object(Config, "GROQ_API_KEY", "mock-groq-key"):
            provider = GroqProvider(model_name="llama-3.3-70b-versatile")
            self.assertTrue(provider.is_available())

            res = provider.generate(
                message="Status report",
                history=[{"role": "user", "content": "Hello"}],
                system_prompt="You are a helpful assistant.",
            )
            self.assertEqual(res.content, "Ultra fast reply from Groq!")
            self.assertEqual(res.provider, "groq")
            self.assertEqual(res.model, "llama-3.3-70b-versatile")

            messages_arg = mock_client.chat.completions.create.call_args[1]["messages"]
            self.assertEqual(len(messages_arg), 3)
            self.assertEqual(messages_arg[0]["role"], "system")
            self.assertEqual(messages_arg[1]["role"], "user")
            self.assertEqual(messages_arg[2]["role"], "user")

    @patch("pihu_core.providers.Groq")
    def test_groq_provider_stream_generate(self, mock_groq_cls: MagicMock) -> None:
        """GroqProvider yields stream chunks correctly."""
        mock_client = MagicMock()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Fast "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="stream!"))]
        mock_client.chat.completions.create.return_value = [chunk1, chunk2]
        mock_groq_cls.return_value = mock_client

        with patch.object(Config, "GROQ_API_KEY", "mock-groq-key"):
            provider = GroqProvider()
            chunks = list(provider.stream_generate(message="Hi"))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].text, "Fast ")
            self.assertEqual(chunks[1].text, "stream!")
            self.assertTrue(chunks[2].is_final)

    @patch("duckduckgo_search.DDGS")
    @patch("pihu_core.providers.genai.Client")
    def test_gemini_provider_tool_calling_loop(self, mock_client_cls: MagicMock, mock_ddgs_cls: MagicMock) -> None:
        """GeminiProvider executes tool calling loop when LLM requests search_web."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Latest AI News", "body": "Gemini 2.5 Flash released.", "href": "https://ai.google.dev"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        mock_client = MagicMock()
        # 1st response: function call to search_web
        mock_tool_call_resp = MagicMock()
        mock_fc = MagicMock()
        mock_fc.name = "search_web"
        mock_fc.args = {"query": "Gemini 2.5"}
        mock_tool_call_resp.function_calls = [mock_fc]
        mock_tool_call_resp.candidates = [MagicMock(content={"role": "model", "parts": []})]

        # 2nd response: final grounded answer
        mock_grounded_resp = MagicMock()
        mock_grounded_resp.text = "Based on web search, Gemini 2.5 Flash is now available."
        mock_grounded_resp.function_calls = None

        mock_client.models.generate_content.side_effect = [mock_tool_call_resp, mock_grounded_resp]
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "GEMINI_API_KEY", "mock-gemini-key"):
            provider = GeminiProvider()
            res = provider.generate(message="What's new in Gemini?")
            self.assertEqual(res.content, "Based on web search, Gemini 2.5 Flash is now available.")
            mock_ddgs_instance.text.assert_called_once_with("Gemini 2.5", max_results=5)
            self.assertEqual(mock_client.models.generate_content.call_count, 2)

    @patch("duckduckgo_search.DDGS")
    @patch("pihu_core.providers.Groq")
    def test_groq_provider_tool_calling_loop(self, mock_groq_cls: MagicMock, mock_ddgs_cls: MagicMock) -> None:
        """GroqProvider executes tool calling loop when LLM requests search_web."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Groq LPUs", "body": "Groq provides ultra-fast inference speed.", "href": "https://groq.com"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        mock_client = MagicMock()
        # 1st response: tool_calls requesting search_web
        mock_completion_1 = MagicMock()
        mock_choice_1 = MagicMock()
        mock_tc = MagicMock()
        mock_tc.id = "call_search_1"
        mock_tc.function = MagicMock(name="search_web", arguments='{"query": "Groq speed"}')
        mock_tc.function.name = "search_web"
        mock_choice_1.message = MagicMock(tool_calls=[mock_tc], content=None)
        mock_completion_1.choices = [mock_choice_1]

        # 2nd response: grounded final response
        mock_completion_2 = MagicMock()
        mock_choice_2 = MagicMock()
        mock_choice_2.message = MagicMock(tool_calls=None, content="Groq provides ultra-fast inference speeds via LPUs.")
        mock_completion_2.choices = [mock_choice_2]
        mock_completion_2.usage = MagicMock(prompt_tokens=20, completion_tokens=15, total_tokens=35)

        mock_client.chat.completions.create.side_effect = [mock_completion_1, mock_completion_2]
        mock_groq_cls.return_value = mock_client

        with patch.object(Config, "GROQ_API_KEY", "mock-groq-key"):
            provider = GroqProvider()
            res = provider.generate(message="Tell me about Groq speed")
            self.assertEqual(res.content, "Groq provides ultra-fast inference speeds via LPUs.")
            mock_ddgs_instance.text.assert_called_once_with("Groq speed", max_results=5)
            self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    # --------------------------------------------------------------------------
    # 4. Overhauled POST /api/chat Workflow with Persistent Memory
    # --------------------------------------------------------------------------
    def test_chat_creates_conversation_and_persists_messages(self) -> None:
        """POST /api/chat persists conversation history and returns conversation_id."""
        conv_uuid = str(uuid.uuid4())
        mock_repo = MagicMock()
        mock_repo.is_available.return_value = True
        mock_repo.create_conversation.return_value = {"id": conv_uuid, "title": "Test Message"}
        mock_repo.get_messages.return_value = []
        mock_repo.save_message.return_value = {"id": str(uuid.uuid4())}

        with patch("api.chat.conversation_repo", mock_repo), patch.object(Config, "GEMINI_API_KEY", ""), patch.object(Config, "GROQ_API_KEY", ""):
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": "Hello Pihu with memory"}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data["ok"])
            self.assertEqual(data["conversation_id"], conv_uuid)
            self.assertIn("response", data)
            self.assertIn("provider", data)

            # Ensure save_message was called for both user and assistant
            self.assertEqual(mock_repo.save_message.call_count, 2)
            first_save_call = mock_repo.save_message.call_args_list[0]
            self.assertEqual(first_save_call[1]["role"], "user")
            self.assertEqual(first_save_call[1]["content"], "Hello Pihu with memory")

            second_save_call = mock_repo.save_message.call_args_list[1]
            self.assertEqual(second_save_call[1]["role"], "assistant")

    def test_chat_uses_existing_conversation_and_db_history(self) -> None:
        """POST /api/chat fetches persistent history for an existing conversation_id."""
        conv_uuid = str(uuid.uuid4())
        mock_repo = MagicMock()
        mock_repo.is_available.return_value = True
        mock_repo.get_conversation.return_value = {"id": conv_uuid}
        mock_repo.get_messages.return_value = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]

        with patch("api.chat.conversation_repo", mock_repo), patch.object(Config, "GEMINI_API_KEY", ""), patch.object(Config, "GROQ_API_KEY", ""):
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": "Follow up question", "conversation_id": conv_uuid}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["conversation_id"], conv_uuid)
            mock_repo.get_messages.assert_called_with(conv_uuid, limit=50)


if __name__ == "__main__":
    unittest.main()
