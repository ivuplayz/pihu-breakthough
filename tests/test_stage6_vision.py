"""Comprehensive unit tests for Stage 6: Multimodal Vision Support in Pihu-BreakThough.

Covers:
- Image payload validation (supported formats, MIME types, size limits, error codes).
- GeminiProvider native vision integration with google-genai types.Part.from_bytes.
- ProviderRouter multimodal intent classification and Tier 1 vision pinning.
- Lean Neon database history persistence (image presence annotation without raw base64).
- Synchronous and SSE streaming chat endpoints with image attachments.
"""

from __future__ import annotations

import base64
import json
import unittest
from unittest.mock import MagicMock, patch

from app import create_app
from pihu_core.config import Config
from pihu_core.providers import (
    GeminiProvider,
    IntentType,
    ProviderCapabilities,
    ProviderResponse,
    RoutingStrategy,
    Stage1DeterministicProvider,
    StreamChunk,
)
from pihu_core.router import ProviderRouter, classify_intent

# Sample 1x1 transparent PNG base64 string
SAMPLE_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
SAMPLE_PNG_B64 = base64.b64encode(SAMPLE_PNG_BYTES).decode("utf-8")
SAMPLE_JPEG_B64 = base64.b64encode(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\xff\xd9").decode("utf-8")


class TestVisionPayloadValidation(unittest.TestCase):
    """Test schema and payload validation for multimodal vision input on /api/chat."""

    def setUp(self) -> None:
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_valid_single_image_payload(self) -> None:
        """Verify successful 200 response with a valid base64 image."""
        payload = {
            "message": "What is in this image?",
            "images": [
                {
                    "data": SAMPLE_PNG_B64,
                    "mime_type": "image/png",
                }
            ],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("response", data)
        self.assertTrue(data["metadata"].get("has_images"))
        self.assertEqual(data["metadata"].get("image_count"), 1)

    def test_valid_multiple_images_payload(self) -> None:
        """Verify successful response with up to 3 valid images."""
        payload = {
            "message": "Compare these three images",
            "images": [
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
                {"data": SAMPLE_JPEG_B64, "mime_type": "image/jpeg"},
                {"data": SAMPLE_PNG_B64, "mime_type": "image/webp"},
            ],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["metadata"].get("image_count"), 3)

    def test_valid_data_uri_payload(self) -> None:
        """Verify data URI format (data:image/png;base64,...) is correctly parsed."""
        payload = {
            "message": "Analyze diagram",
            "images": [
                {
                    "data": f"data:image/png;base64,{SAMPLE_PNG_B64}",
                    "mime_type": "image/png",
                }
            ],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])

    def test_valid_url_data_uri_payload(self) -> None:
        """Verify image provided via 'url' key with data URI is accepted."""
        payload = {
            "message": "Analyze chart",
            "images": [
                {
                    "url": f"data:image/jpeg;base64,{SAMPLE_JPEG_B64}",
                }
            ],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])

    def test_remote_url_fetch(self) -> None:
        """Verify image fetching from remote HTTP/HTTPS URL."""
        mock_resp = MagicMock()
        mock_resp.read.return_value = SAMPLE_PNG_BYTES
        mock_resp.headers.get_content_type.return_value = "image/png"
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_resp):
            payload = {
                "message": "Inspect from URL",
                "images": [
                    {
                        "url": "https://example.com/photo.png",
                    }
                ],
            }
            response = self.client.post("/api/chat", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertTrue(data["ok"])

    def test_rejects_non_list_images(self) -> None:
        """Verify 400 when 'images' is not an array."""
        payload = {
            "message": "Test message",
            "images": "invalid_string_not_list",
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_IMAGES_TYPE")

    def test_rejects_more_than_three_images(self) -> None:
        """Verify 400 when more than 3 images are provided."""
        payload = {
            "message": "Too many images",
            "images": [
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
            ],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "TOO_MANY_IMAGES")

    def test_rejects_non_dict_image_item(self) -> None:
        """Verify 400 when image item is not a dictionary."""
        payload = {
            "message": "Test message",
            "images": ["not_a_dict"],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_IMAGE_ITEM")

    def test_rejects_missing_image_payload(self) -> None:
        """Verify 400 when image object lacks both 'data' and 'url'."""
        payload = {
            "message": "Test message",
            "images": [{"mime_type": "image/png"}],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "MISSING_IMAGE_PAYLOAD")

    def test_rejects_empty_image_data(self) -> None:
        """Verify 400 when image data is empty string."""
        payload = {
            "message": "Test message",
            "images": [{"data": "   ", "mime_type": "image/png"}],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "EMPTY_IMAGE_DATA")

    def test_rejects_missing_mime_type(self) -> None:
        """Verify 400 when mime_type is omitted and cannot be inferred."""
        payload = {
            "message": "Test message",
            "images": [{"data": SAMPLE_PNG_B64}],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "MISSING_IMAGE_MIME_TYPE")

    def test_rejects_unsupported_mime_type(self) -> None:
        """Verify 400 when image MIME type is unsupported (e.g. application/pdf, text/html)."""
        payload = {
            "message": "Test message",
            "images": [{"data": SAMPLE_PNG_B64, "mime_type": "application/pdf"}],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "UNSUPPORTED_IMAGE_MIME_TYPE")

    def test_rejects_invalid_base64_data(self) -> None:
        """Verify 400 when base64 string is corrupted or invalid."""
        payload = {
            "message": "Test message",
            "images": [{"data": "!!!corrupted_base64_string$$$", "mime_type": "image/png"}],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_BASE64_IMAGE")

    def test_rejects_oversized_image(self) -> None:
        """Verify 400 when decoded image exceeds 5MB safe limit."""
        huge_bytes = b"0" * (5 * 1024 * 1024 + 10)
        huge_b64 = base64.b64encode(huge_bytes).decode("utf-8")
        payload = {
            "message": "Test oversized",
            "images": [{"data": huge_b64, "mime_type": "image/png"}],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "IMAGE_TOO_LARGE")


class TestGeminiProviderVisionIntegration(unittest.TestCase):
    """Test GeminiProvider vision handling and Part.from_bytes conversion."""

    def test_gemini_capabilities_declare_vision(self) -> None:
        """Verify GeminiProvider declares supports_vision=True."""
        provider = GeminiProvider()
        self.assertTrue(provider.capabilities.supports_vision)
        self.assertTrue(provider.capabilities.to_dict()["supports_vision"])

    def test_build_contents_with_images_and_types_part(self) -> None:
        """Verify _build_contents creates native Part.from_bytes using google-genai types."""
        provider = GeminiProvider()
        images = [
            {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
            {"data": SAMPLE_JPEG_B64, "mime_type": "image/jpeg"},
        ]
        contents = provider._build_contents(
            message="Describe these images",
            history=[{"role": "user", "content": "Prior message"}],
            images=images,
        )

        self.assertEqual(len(contents), 2)
        # History turn
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[0]["parts"][0]["text"], "Prior message")

        # Current multimodal turn
        user_turn = contents[1]
        self.assertEqual(user_turn["role"], "user")
        self.assertEqual(len(user_turn["parts"]), 3)  # 2 images + 1 text prompt

        # Verify last part is text
        self.assertEqual(user_turn["parts"][-1], {"text": "Describe these images"})

    @patch("pihu_core.providers.genai")
    @patch("pihu_core.providers.Config")
    def test_gemini_generate_with_images(self, mock_config, mock_genai) -> None:
        """Verify generate passes multimodal contents to client.models.generate_content."""
        mock_config.GEMINI_API_KEY = "test-api-key"
        mock_config.STAGE = "stage-6"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This image shows a blue icon."
        mock_response.function_calls = None
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider()
        images = [{"data": SAMPLE_PNG_B64, "mime_type": "image/png"}]

        resp = provider.generate(
            message="What is this?",
            images=images,
            tools=[],
        )

        self.assertEqual(resp.content, "This image shows a blue icon.")
        self.assertEqual(resp.provider, "gemini")
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        contents = call_kwargs["contents"]
        # Multimodal parts present
        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(len(contents[0]["parts"]), 2)

    @patch("pihu_core.providers.genai")
    @patch("pihu_core.providers.Config")
    def test_gemini_stream_generate_with_images(self, mock_config, mock_genai) -> None:
        """Verify stream_generate passes multimodal contents to client.models.generate_content_stream."""
        mock_config.GEMINI_API_KEY = "test-api-key"
        mock_config.STAGE = "stage-6"

        mock_client = MagicMock()
        chunk1 = MagicMock()
        chunk1.text = "Identified "
        chunk2 = MagicMock()
        chunk2.text = "diagram."
        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]
        mock_genai.Client.return_value = mock_client

        provider = GeminiProvider()
        images = [{"data": SAMPLE_PNG_B64, "mime_type": "image/png"}]

        chunks = list(
            provider.stream_generate(
                message="Explain graphic",
                images=images,
                tools=[],
            )
        )

        texts = [c.text for c in chunks]
        self.assertIn("Identified ", texts)
        self.assertIn("diagram.", texts)
        self.assertTrue(chunks[-1].is_final)
        mock_client.models.generate_content_stream.assert_called_once()


class TestRouterVisionPinning(unittest.TestCase):
    """Test intent classification and provider pinning for vision requests."""

    def test_classify_intent_multimodal(self) -> None:
        """Verify classify_intent returns IntentType.MULTIMODAL when has_images=True."""
        self.assertEqual(
            classify_intent("Look at this", has_images=True),
            IntentType.MULTIMODAL,
        )
        self.assertEqual(
            classify_intent("Summarize this doc", has_images=True),
            IntentType.MULTIMODAL,
        )
        self.assertEqual(
            classify_intent("Quick question", has_images=False),
            IntentType.FAST_CHAT,
        )

    def test_resolve_provider_chain_pins_gemini_for_vision(self) -> None:
        """Verify resolve_provider_chain prioritizes GeminiProvider as Tier 1 for vision."""
        router = ProviderRouter()
        chain = router.resolve_provider_chain(
            intent=IntentType.MULTIMODAL,
            strategy=RoutingStrategy.VISION,
        )
        self.assertGreater(len(chain), 0)
        self.assertEqual(chain[0].name, "gemini")

    def test_route_chat_auto_upgrades_strategy_to_vision(self) -> None:
        """Verify route_chat with images auto-selects RoutingStrategy.VISION."""
        router = ProviderRouter()
        images = [{"data": SAMPLE_PNG_B64, "mime_type": "image/png"}]

        result = router.route_chat(
            message="Inspect diagram",
            images=images,
        )

        self.assertTrue(result["ok"])
        metadata = result["metadata"]
        self.assertEqual(metadata["intent"], "multimodal")
        self.assertEqual(metadata["strategy"], "vision")
        self.assertTrue(metadata["has_images"])
        self.assertEqual(metadata["image_count"], 1)

    def test_route_stream_with_images(self) -> None:
        """Verify route_stream yields chunks successfully with image inputs."""
        router = ProviderRouter()
        images = [{"data": SAMPLE_PNG_B64, "mime_type": "image/png"}]

        events = list(
            router.route_stream(
                message="Stream inspect",
                images=images,
            )
        )
        self.assertGreater(len(events), 0)
        self.assertTrue(any(e.get("is_final") for e in events))


class TestLeanDatabaseImagePersistence(unittest.TestCase):
    """Test that message history persistence annotates image presence cleanly without raw base64 bloat."""

    def setUp(self) -> None:
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    @patch("api.chat.conversation_repo")
    def test_db_annotates_image_presence_without_bloat(self, mock_repo) -> None:
        """Verify database save_message receives clean tag like '[Attached 1 Image]' without raw base64."""
        mock_repo.is_available.return_value = True
        mock_repo.get_conversation.return_value = {"id": "conv-123", "user_id": "usr-456"}
        mock_repo.get_messages.return_value = []
        mock_repo.get_memories.return_value = []

        payload = {
            "message": "Please review this architectural blueprint",
            "conversation_id": "conv-123",
            "images": [
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
            ],
        }

        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)

        # Check call to save_message for the user's incoming message
        user_save_calls = [
            call for call in mock_repo.save_message.call_args_list
            if call[1].get("role") == "user"
        ]
        self.assertEqual(len(user_save_calls), 1)
        saved_content = user_save_calls[0][1]["content"]

        # Content must contain clean tag
        self.assertIn("[Attached 1 Image]", saved_content)
        # Content must NOT contain the massive raw base64 string
        self.assertNotIn(SAMPLE_PNG_B64, saved_content)

    @patch("api.chat.conversation_repo")
    def test_db_annotates_multiple_images_plural(self, mock_repo) -> None:
        """Verify database save_message receives '[Attached 2 Images]' for multiple attachments."""
        mock_repo.is_available.return_value = True
        mock_repo.get_conversation.return_value = {"id": "conv-789", "user_id": "usr-456"}
        mock_repo.get_messages.return_value = []
        mock_repo.get_memories.return_value = []

        payload = {
            "message": "Compare both mockups",
            "conversation_id": "conv-789",
            "images": [
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
                {"data": SAMPLE_JPEG_B64, "mime_type": "image/jpeg"},
            ],
        }

        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)

        user_save_calls = [
            call for call in mock_repo.save_message.call_args_list
            if call[1].get("role") == "user"
        ]
        self.assertEqual(len(user_save_calls), 1)
        saved_content = user_save_calls[0][1]["content"]
        self.assertIn("[Attached 2 Images]", saved_content)
        self.assertNotIn(SAMPLE_PNG_B64, saved_content)


class TestSSEStreamingMultimodal(unittest.TestCase):
    """Test SSE streaming endpoint with image input."""

    def setUp(self) -> None:
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_sse_stream_with_image_input(self) -> None:
        """Verify /api/chat?stream=true returns text/event-stream chunks when images are provided."""
        payload = {
            "message": "Describe image streamingly",
            "stream": True,
            "images": [
                {"data": SAMPLE_PNG_B64, "mime_type": "image/png"},
            ],
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.content_type)

        raw_data = response.get_data(as_text=True)
        self.assertIn("data: ", raw_data)


if __name__ == "__main__":
    unittest.main()
