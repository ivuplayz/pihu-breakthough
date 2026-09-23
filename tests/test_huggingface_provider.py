"""Unit and integration test suite for HuggingFaceProvider and Summarization Routing.

Tests:
- HuggingFaceProvider availability, generate, and stream_generate with InferenceClient mocks.
- Secret sanitization and error wrapping.
- Summarization intent classification ("summarize", "tldr", "summary", etc.).
- Multi-tier routing cascade prioritizing Hugging Face for summarization.
- Failover mechanics and circuit breaker trips on Hugging Face API errors.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pihu_core.config import Config
from pihu_core.providers import (
    HuggingFaceProvider,
    IntentType,
    ProviderStatus,
    RoutingStrategy,
    StreamChunk,
)
from pihu_core.router import ProviderRouter, classify_intent


class TestHuggingFaceProvider(unittest.TestCase):
    """Unit tests for HuggingFaceProvider."""

    def test_availability(self) -> None:
        """is_available() is True only when HF_API_KEY is configured and SDK is present."""
        provider = HuggingFaceProvider()
        with patch.object(Config, "HF_API_KEY", "hf_test_token_123"):
            self.assertTrue(provider.is_available())

        with patch.object(Config, "HF_API_KEY", None):
            self.assertFalse(provider.is_available())

        with patch.object(Config, "HF_API_KEY", "   "):
            self.assertFalse(provider.is_available())

    def test_model_name_default_and_override(self) -> None:
        """Provider uses configured default or custom model name."""
        with patch.object(Config, "HF_MODEL", "meta-llama/Llama-3.2-3B-Instruct"):
            p1 = HuggingFaceProvider()
            self.assertEqual(p1.model_name, "meta-llama/Llama-3.2-3B-Instruct")

            p2 = HuggingFaceProvider(model_name="facebook/bart-large-cnn")
            self.assertEqual(p2.model_name, "facebook/bart-large-cnn")

    @patch("pihu_core.providers.InferenceClient")
    def test_generate_success(self, mock_client_cls: MagicMock) -> None:
        """generate() invokes chat completions with proper messages and returns ProviderResponse."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Summary: Pihu is an advanced modular AI architecture."
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=45, completion_tokens=15, total_tokens=60)
        mock_client.chat.completions.create.return_value = mock_completion
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "HF_API_KEY", "hf_valid_key_xyz"):
            provider = HuggingFaceProvider(model_name="meta-llama/Llama-3.2-3B-Instruct")
            response = provider.generate(
                message="Summarize the Pihu architecture",
                history=[{"role": "user", "content": "What is Pihu?"}, {"role": "assistant", "content": "An AI assistant."}],
                system_prompt="You are an expert summarizer.",
            )

            self.assertEqual(response.content, "Summary: Pihu is an advanced modular AI architecture.")
            self.assertEqual(response.provider, "huggingface")
            self.assertEqual(response.model, "meta-llama/Llama-3.2-3B-Instruct")
            self.assertEqual(response.metadata["usage"]["total_tokens"], 60)

            # Check messages argument structure
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            self.assertEqual(len(messages), 4)
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[1]["role"], "user")
            self.assertEqual(messages[2]["role"], "assistant")
            self.assertEqual(messages[3]["role"], "user")

    @patch("pihu_core.providers.InferenceClient")
    def test_stream_generate_success(self, mock_client_cls: MagicMock) -> None:
        """stream_generate() yields stream tokens from chat completions stream."""
        mock_client = MagicMock()
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="Concise "))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="summary."))]
        mock_client.chat.completions.create.return_value = [chunk1, chunk2]
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "HF_API_KEY", "hf_valid_key_xyz"):
            provider = HuggingFaceProvider()
            chunks = list(provider.stream_generate(message="TLDR"))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].text, "Concise ")
            self.assertEqual(chunks[1].text, "summary.")
            self.assertTrue(chunks[2].is_final)

    @patch("pihu_core.providers.InferenceClient")
    def test_error_handling_sanitizes_token(self, mock_client_cls: MagicMock) -> None:
        """Provider wraps exceptions cleanly and prevents token leakage."""
        secret_key = "hf_super_secret_token_12345"
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(f"HTTP 401 Unauthorized for {secret_key}")
        mock_client_cls.return_value = mock_client

        with patch.object(Config, "HF_API_KEY", secret_key):
            provider = HuggingFaceProvider()
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate(message="Hello")

            err_msg = str(ctx.exception)
            self.assertNotIn(secret_key, err_msg)
            self.assertIn("***", err_msg)
            self.assertIn("HuggingFace API error", err_msg)


class TestSummarizationRouting(unittest.TestCase):
    """Unit tests for summarization intent classification and router cascade."""

    def test_classify_intent_summarization(self) -> None:
        """classify_intent detects summarization keywords accurately."""
        prompts = [
            "Please summarize this long document for me.",
            "TLDR of the quarterly revenue results",
            "give me a brief summary of quantum mechanics",
            "tl;dr of the announcement",
            "Can you recap what happened in the meeting?",
            "What is the gist of this article?",
            "Brief overview of the project",
            "Give me the key takeaways from the research paper.",
        ]
        for p in prompts:
            with self.subTest(prompt=p):
                self.assertEqual(classify_intent(p), IntentType.SUMMARIZATION)

    def test_chain_resolution_summarization_prioritizes_huggingface(self) -> None:
        """resolve_provider_chain places huggingface at index 0 for summarization."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.SUMMARIZATION,
            strategy=RoutingStrategy.AUTO,
        )
        self.assertEqual(chain[0].name, "huggingface")
        self.assertEqual(chain[-1].name, "stage1-deterministic")

    def test_chain_resolution_summarization_strategy(self) -> None:
        """Explicit RoutingStrategy.SUMMARIZATION places huggingface at index 0."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.FAST_CHAT,
            strategy=RoutingStrategy.SUMMARIZATION,
        )
        self.assertEqual(chain[0].name, "huggingface")

    def test_global_fallback_cascade_order(self) -> None:
        """Default fallback cascade order is Gemini -> Groq -> HuggingFace -> Deterministic."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.COMPLEX_REASONING,
            strategy=RoutingStrategy.AUTO,
        )
        names = [p.name for p in chain]
        self.assertEqual(names, ["gemini", "groq", "huggingface", "stage1-deterministic"])

    @patch("pihu_core.providers.InferenceClient")
    def test_router_executes_huggingface_for_summarization(self, mock_client_cls: MagicMock) -> None:
        """route_chat seamlessly routes summarization query to HuggingFaceProvider."""
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Executive summary of the document."
        mock_completion.choices = [mock_choice]
        mock_completion.usage = MagicMock(prompt_tokens=50, completion_tokens=20, total_tokens=70)
        mock_client.chat.completions.create.return_value = mock_completion
        mock_client_cls.return_value = mock_client

        test_router = ProviderRouter()
        with patch.object(Config, "HF_API_KEY", "mock-hf-key"):
            result = test_router.route_chat(message="Summarize the quarterly results")
            self.assertTrue(result["ok"])
            self.assertEqual(result["provider"], "huggingface")
            self.assertEqual(result["response"], "Executive summary of the document.")
            self.assertFalse(result["metadata"]["fallback_occurred"])

    @patch("pihu_core.providers.InferenceClient")
    def test_router_failover_when_huggingface_fails(self, mock_client_cls: MagicMock) -> None:
        """When HuggingFace fails, router cascades cleanly to the next candidate."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Hugging Face API rate limited (503)")
        mock_client_cls.return_value = mock_client

        test_router = ProviderRouter()
        # Ensure Gemini and Groq are not configured so it cascades down to deterministic fallback
        with patch.object(Config, "HF_API_KEY", "mock-hf-key"), \
             patch.object(Config, "GEMINI_API_KEY", ""), \
             patch.object(Config, "GROQ_API_KEY", ""):
            result = test_router.route_chat(message="TLDR of this document")
            self.assertTrue(result["ok"])
            self.assertEqual(result["provider"], "stage1-deterministic")
            self.assertTrue(result["metadata"]["fallback_occurred"])
            self.assertIn("huggingface", result["metadata"]["fallback_chain"])


if __name__ == "__main__":
    unittest.main()
