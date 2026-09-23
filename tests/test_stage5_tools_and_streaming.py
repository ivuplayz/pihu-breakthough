"""Unit and integration test suite for Stage 5 Tools and SSE Streaming.

Covers:
- Unified ToolRegistry and schema conversion (OpenAI, Gemini formats).
- Real-time search_web tool using DuckDuckGo search (mocked).
- Function calling loops in GeminiProvider and GroqProvider.
- SSE Streaming endpoint (/api/chat) with full message reconstruction and DB persistence.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch
import uuid

from app import create_app
from pihu_core.config import Config
from pihu_core.providers import GeminiProvider, GroqProvider, StreamChunk
from pihu_core.tools import ToolRegistry, search_web, tool_registry


class TestToolRegistry(unittest.TestCase):
    """Unit tests for ToolRegistry and schema generation."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()

    def test_register_and_retrieve_tool(self) -> None:
        """Registering a function registers both the callable and its schema."""
        def sample_add(a: int, b: int = 5) -> int:
            """Add two numbers together."""
            return a + b

        self.registry.register(sample_add)
        tool = self.registry.get_tool("sample_add")
        self.assertIsNotNone(tool)
        self.assertEqual(tool(2, 3), 5)

        schema = self.registry.get_schema("sample_add")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["name"], "sample_add")
        self.assertEqual(schema["description"], "Add two numbers together.")
        self.assertIn("a", schema["parameters"]["properties"])
        self.assertEqual(schema["parameters"]["properties"]["a"]["type"], "integer")
        self.assertEqual(schema["parameters"]["properties"]["b"]["default"], 5)
        self.assertIn("a", schema["parameters"]["required"])
        self.assertNotIn("b", schema["parameters"]["required"])

    def test_execute_tool(self) -> None:
        """execute() runs registered tools and catches runtime exceptions cleanly."""
        def failing_tool() -> None:
            raise ValueError("Something went wrong inside tool")

        self.registry.register(failing_tool)
        res = self.registry.execute("failing_tool")
        self.assertIn("Error executing tool 'failing_tool'", res)

        unregistered = self.registry.execute("non_existent_tool")
        self.assertIn("Tool 'non_existent_tool' is not registered.", unregistered)

    def test_to_openai_and_gemini_formats(self) -> None:
        """ToolRegistry correctly converts schemas to OpenAI/Groq and Gemini formats."""
        def dummy_calc(x: float) -> float:
            """Calculate square."""
            return x * x

        self.registry.register(dummy_calc)
        openai_tools = self.registry.to_openai_format()
        self.assertEqual(len(openai_tools), 1)
        self.assertEqual(openai_tools[0]["type"], "function")
        self.assertEqual(openai_tools[0]["function"]["name"], "dummy_calc")

        gemini_tools = self.registry.to_gemini_format()
        self.assertEqual(len(gemini_tools), 1)
        self.assertEqual(gemini_tools[0], dummy_calc)


class TestSearchWebTool(unittest.TestCase):
    """Unit tests for search_web DuckDuckGo search integration."""

    @patch("duckduckgo_search.DDGS")
    def test_search_web_success(self, mock_ddgs_cls: MagicMock) -> None:
        """search_web formats DuckDuckGo results into structured multi-line text."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Neon Serverless Postgres",
                "body": "Neon is a serverless open-source alternative to AWS Aurora Postgres.",
                "href": "https://neon.tech",
            },
            {
                "title": "PostgreSQL Documentation",
                "body": "Official documentation for the PostgreSQL database.",
                "href": "https://postgresql.org",
            },
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        results = search_web("Neon Postgres", max_results=2)
        self.assertIn("[1] Neon Serverless Postgres", results)
        self.assertIn("Summary: Neon is a serverless open-source alternative", results)
        self.assertIn("Source: https://neon.tech", results)
        self.assertIn("[2] PostgreSQL Documentation", results)
        mock_ddgs_instance.text.assert_called_once_with("Neon Postgres", max_results=2)

    def test_search_web_empty_query(self) -> None:
        """search_web handles blank queries gracefully without making network calls."""
        res = search_web("   ")
        self.assertEqual(res, "Search query cannot be empty.")

    @patch("duckduckgo_search.DDGS")
    def test_search_web_no_results(self, mock_ddgs_cls: MagicMock) -> None:
        """search_web handles empty result lists gracefully."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs_cls.return_value = mock_ddgs_instance

        res = search_web("obscure query with no results")
        self.assertIn("No web search results found", res)

    @patch("duckduckgo_search.DDGS")
    def test_search_web_exception_handling(self, mock_ddgs_cls: MagicMock) -> None:
        """search_web catches DDGS exceptions and returns an error string."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.side_effect = Exception("Rate limit reached")
        mock_ddgs_cls.return_value = mock_ddgs_instance

        res = search_web("Rate limit test")
        self.assertIn("Web search error", res)
        self.assertIn("Rate limit reached", res)


class TestProvidersFunctionCalling(unittest.TestCase):
    """Unit tests for tool calling in GeminiProvider and GroqProvider."""

    @patch("duckduckgo_search.DDGS")
    @patch("pihu_core.providers.genai.Client")
    def test_gemini_provider_generate_with_tool_call(
        self, mock_client_cls: MagicMock, mock_ddgs_cls: MagicMock
    ) -> None:
        """GeminiProvider executes tool calling loop and returns grounded response."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Weather in Tokyo", "body": "Tokyo weather is 22C and sunny.", "href": "https://weather.com"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        mock_client = MagicMock()
        # 1. First model call: requests search_web function call
        mock_call_resp = MagicMock()
        mock_fc = MagicMock()
        mock_fc.name = "search_web"
        mock_fc.args = {"query": "Tokyo weather"}
        mock_call_resp.function_calls = [mock_fc]
        mock_call_resp.candidates = [MagicMock(content={"role": "model", "parts": []})]

        # 2. Second model call: returns final answer grounded in search
        mock_final_resp = MagicMock()
        mock_final_resp.text = "The weather in Tokyo is currently 22°C and sunny."
        mock_final_resp.function_calls = None

        mock_client.models.generate_content.side_effect = [mock_call_resp, mock_final_resp]
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "GEMINI_API_KEY", "mock-gemini-key"):
            provider = GeminiProvider()
            res = provider.generate(message="What's the weather in Tokyo?")
            self.assertEqual(res.content, "The weather in Tokyo is currently 22°C and sunny.")
            self.assertEqual(mock_client.models.generate_content.call_count, 2)
            mock_ddgs_instance.text.assert_called_once_with("Tokyo weather", max_results=5)

    @patch("duckduckgo_search.DDGS")
    @patch("pihu_core.providers.genai.Client")
    def test_gemini_provider_stream_generate_with_tool_call(
        self, mock_client_cls: MagicMock, mock_ddgs_cls: MagicMock
    ) -> None:
        """GeminiProvider stream_generate executes tool calling probe before streaming."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Mars Rover", "body": "Perseverance rover found signs of ancient water.", "href": "https://nasa.gov"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        mock_client = MagicMock()
        mock_probe_resp = MagicMock()
        mock_fc = MagicMock()
        mock_fc.name = "search_web"
        mock_fc.args = {"query": "Mars water"}
        mock_probe_resp.function_calls = [mock_fc]
        mock_probe_resp.candidates = [MagicMock(content={"role": "model", "parts": []})]

        mock_client.models.generate_content.return_value = mock_probe_resp

        chunk1 = MagicMock(text="Mars rover found ")
        chunk2 = MagicMock(text="ancient water signs.")
        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "GEMINI_API_KEY", "mock-gemini-key"):
            provider = GeminiProvider()
            chunks = list(provider.stream_generate(message="Mars water update"))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].text, "Mars rover found ")
            self.assertEqual(chunks[1].text, "ancient water signs.")
            self.assertTrue(chunks[2].is_final)
            mock_ddgs_instance.text.assert_called_once_with("Mars water", max_results=5)

    @patch("duckduckgo_search.DDGS")
    @patch("pihu_core.providers.Groq")
    def test_groq_provider_generate_with_tool_call(
        self, mock_groq_cls: MagicMock, mock_ddgs_cls: MagicMock
    ) -> None:
        """GroqProvider executes tool calling loop and returns grounded response."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Llama 3.3", "body": "Llama 3.3 70B versatile model released.", "href": "https://meta.com"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        mock_client = MagicMock()
        # 1. First call: tool_calls
        mock_tc = MagicMock()
        mock_tc.id = "call_llama_1"
        mock_tc.function = MagicMock(name="search_web", arguments='{"query": "Llama 3.3"}')
        mock_tc.function.name = "search_web"
        mock_comp_1 = MagicMock()
        mock_comp_1.choices = [MagicMock(message=MagicMock(tool_calls=[mock_tc], content=None))]

        # 2. Second call: grounded final content
        mock_comp_2 = MagicMock()
        mock_comp_2.choices = [MagicMock(message=MagicMock(tool_calls=None, content="Llama 3.3 70B is available now."))]
        mock_comp_2.usage = MagicMock(prompt_tokens=25, completion_tokens=10, total_tokens=35)

        mock_client.chat.completions.create.side_effect = [mock_comp_1, mock_comp_2]
        mock_groq_cls.return_value = mock_client

        with patch.object(Config, "GROQ_API_KEY", "mock-groq-key"):
            provider = GroqProvider()
            res = provider.generate(message="Tell me about Llama 3.3")
            self.assertEqual(res.content, "Llama 3.3 70B is available now.")
            self.assertEqual(mock_client.chat.completions.create.call_count, 2)
            mock_ddgs_instance.text.assert_called_once_with("Llama 3.3", max_results=5)

    @patch("duckduckgo_search.DDGS")
    @patch("pihu_core.providers.Groq")
    def test_groq_provider_stream_generate_with_tool_call(
        self, mock_groq_cls: MagicMock, mock_ddgs_cls: MagicMock
    ) -> None:
        """GroqProvider stream_generate executes tool calling probe before streaming."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {"title": "Groq Benchmarks", "body": "Groq achieves 300+ tokens per second.", "href": "https://groq.com"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        mock_client = MagicMock()
        # 1. Probe call: tool calls
        mock_tc = MagicMock()
        mock_tc.id = "call_bench_1"
        mock_tc.function = MagicMock(name="search_web", arguments='{"query": "Groq TPS"}')
        mock_tc.function.name = "search_web"
        mock_probe = MagicMock()
        mock_probe.choices = [MagicMock(message=MagicMock(tool_calls=[mock_tc], content=None))]

        # 2. Second call: streaming chunks
        chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Groq speeds reach "))])
        chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content="300 tokens/s."))])

        mock_client.chat.completions.create.side_effect = [mock_probe, [chunk1, chunk2]]
        mock_groq_cls.return_value = mock_client

        with patch.object(Config, "GROQ_API_KEY", "mock-groq-key"):
            provider = GroqProvider()
            chunks = list(provider.stream_generate(message="How fast is Groq?"))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].text, "Groq speeds reach ")
            self.assertEqual(chunks[1].text, "300 tokens/s.")
            self.assertTrue(chunks[2].is_final)
            mock_ddgs_instance.text.assert_called_once_with("Groq TPS", max_results=5)


class TestSSEStreamingAPI(unittest.TestCase):
    """Integration tests for Server-Sent Events (SSE) streaming in POST /api/chat."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()

    def test_post_chat_stream_false_returns_json(self) -> None:
        """stream=false returns standard JSON response."""
        with patch.object(Config, "GEMINI_API_KEY", ""), patch.object(Config, "GROQ_API_KEY", ""):
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": "Hello sync chat", "stream": False}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "application/json")
            data = response.get_json()
            self.assertTrue(data["ok"])
            self.assertIn("response", data)
            self.assertIn("provider", data)

    def test_post_chat_stream_invalid_type_returns_400(self) -> None:
        """Non-boolean stream parameter returns 400 with INVALID_STREAM_TYPE."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps({"message": "Hello", "stream": "true"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_STREAM_TYPE")

    def test_post_chat_stream_true_returns_event_stream_and_persists(self) -> None:
        """stream=true yields Server-Sent Events and reconstructs full message to database."""
        conv_uuid = str(uuid.uuid4())
        mock_repo = MagicMock()
        mock_repo.is_available.return_value = True
        mock_repo.create_conversation.return_value = {"id": conv_uuid, "title": "Stream Test"}
        mock_repo.get_messages.return_value = []
        mock_repo.save_message.return_value = {"id": str(uuid.uuid4())}

        with patch("api.chat.conversation_repo", mock_repo), patch.object(Config, "GEMINI_API_KEY", ""), patch.object(Config, "GROQ_API_KEY", ""):
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": "Tell me a story", "stream": True}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "text/event-stream")

            raw_stream = response.get_data(as_text=True)
            self.assertIn("data: ", raw_stream)

            # Parse lines
            events = []
            for line in raw_stream.split("\n"):
                clean_line = line.strip()
                if clean_line.startswith("data: "):
                    payload = json.loads(clean_line[len("data: "):])
                    events.append(payload)

            self.assertGreater(len(events), 0)
            for ev in events:
                self.assertIn("chunk", ev)
                self.assertIn("provider", ev)

            # Verify that save_message was called for user message and assistant message
            self.assertEqual(mock_repo.save_message.call_count, 2)
            user_save = mock_repo.save_message.call_args_list[0]
            self.assertEqual(user_save[1]["role"], "user")
            self.assertEqual(user_save[1]["content"], "Tell me a story")

            assistant_save = mock_repo.save_message.call_args_list[1]
            self.assertEqual(assistant_save[1]["role"], "assistant")
            self.assertTrue(len(assistant_save[1]["content"]) > 0)


if __name__ == "__main__":
    unittest.main()
