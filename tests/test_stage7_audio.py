"""Comprehensive unit tests for Stage 7: Voice & Audio Processing via Groq Whisper.

Tests:
- AudioTranscriber base64 decoding and MIME validation.
- Mocking Groq Whisper API (whisper-large-v3-turbo) transcription.
- Handling invalid, unsupported, and oversized audio formats gracefully.
- /api/chat endpoint audio processing, voice-to-text injection, and metadata.
- Database message history annotation with [🎤 Voice Message].
- SSE streaming with audio inputs.
"""

from __future__ import annotations

import base64
import io
import json
import unittest
from unittest.mock import MagicMock, patch

from app import create_app
from pihu_core.audio import (
    AudioTranscriber,
    AudioTranscriptionError,
    DEFAULT_WHISPER_MODEL,
    SUPPORTED_AUDIO_MIMES,
)
from pihu_core.config import Config

# Dummy sample audio bytes (e.g. mock WebM header)
SAMPLE_AUDIO_BYTES = b"\x1aE\xdf\xa3dummy_audio_stream_data_bytes_12345"
SAMPLE_AUDIO_B64 = base64.b64encode(SAMPLE_AUDIO_BYTES).decode("utf-8")


class TestAudioTranscriber(unittest.TestCase):
    """Unit tests for AudioTranscriber logic and Groq Whisper integration."""

    def setUp(self) -> None:
        self.transcriber = AudioTranscriber()

    def test_supported_mimes_contains_standard_formats(self) -> None:
        """Verify standard formats (mp3, wav, webm, ogg, m4a, etc.) are supported."""
        for mime in ("audio/mp3", "audio/wav", "audio/webm", "audio/ogg", "audio/m4a"):
            self.assertIn(mime, SUPPORTED_AUDIO_MIMES)

    def test_rejects_unsupported_mime_type(self) -> None:
        """Verify AudioTranscriptionError when MIME type is not supported."""
        with self.assertRaises(AudioTranscriptionError) as ctx:
            self.transcriber.transcribe(
                audio_data=SAMPLE_AUDIO_B64,
                mime_type="audio/unsupported_format",
            )
        self.assertIn("Unsupported audio MIME type", str(ctx.exception))

    def test_rejects_empty_audio_data(self) -> None:
        """Verify AudioTranscriptionError when audio data is empty."""
        with self.assertRaises(AudioTranscriptionError) as ctx:
            self.transcriber.transcribe(audio_data="", mime_type="audio/webm")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_rejects_invalid_base64_data(self) -> None:
        """Verify AudioTranscriptionError when base64 string is corrupted."""
        with self.assertRaises(AudioTranscriptionError) as ctx:
            self.transcriber.transcribe(
                audio_data="!!!invalid_b64$$$",
                mime_type="audio/webm",
            )
        self.assertIn("Invalid base64", str(ctx.exception))

    def test_rejects_oversized_audio_data(self) -> None:
        """Verify AudioTranscriptionError when audio exceeds 25MB limit."""
        huge_bytes = b"0" * (25 * 1024 * 1024 + 10)
        with self.assertRaises(AudioTranscriptionError) as ctx:
            self.transcriber.transcribe(audio_data=huge_bytes, mime_type="audio/wav")
        self.assertIn("exceeds maximum allowed size", str(ctx.exception))

    @patch("pihu_core.audio.Groq")
    @patch("pihu_core.audio.Config")
    def test_transcribe_base64_success(self, mock_config, mock_groq_class) -> None:
        """Verify successful base64 transcription calls Groq Whisper with expected parameters."""
        mock_config.GROQ_API_KEY = "test-groq-key"

        mock_client = MagicMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "Hello, what is the weather today?"
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        mock_groq_class.return_value = mock_client

        result = self.transcriber.transcribe(
            audio_data=SAMPLE_AUDIO_B64,
            mime_type="audio/webm",
        )

        self.assertEqual(result, "Hello, what is the weather today?")
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], DEFAULT_WHISPER_MODEL)
        self.assertEqual(call_kwargs["file"][0], "audio.webm")
        self.assertEqual(call_kwargs["file"][2], "audio/webm")

    @patch("pihu_core.audio.Groq")
    @patch("pihu_core.audio.Config")
    def test_transcribe_data_uri_success(self, mock_config, mock_groq_class) -> None:
        """Verify data URI (data:audio/mp3;base64,...) is correctly parsed and transcribed."""
        mock_config.GROQ_API_KEY = "test-groq-key"

        mock_client = MagicMock()
        mock_transcription = MagicMock()
        mock_transcription.text = "Testing data URI transcription."
        mock_client.audio.transcriptions.create.return_value = mock_transcription
        mock_groq_class.return_value = mock_client

        data_uri = f"data:audio/mp3;base64,{SAMPLE_AUDIO_B64}"
        result = self.transcriber.transcribe(audio_data=data_uri)

        self.assertEqual(result, "Testing data URI transcription.")
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(call_kwargs["file"][0], "audio.mp3")

    @patch("pihu_core.audio.Groq")
    @patch("pihu_core.audio.Config")
    def test_transcribe_failure_raises_clean_error(self, mock_config, mock_groq_class) -> None:
        """Verify upstream Groq API exceptions are wrapped in AudioTranscriptionError without crashing."""
        mock_config.GROQ_API_KEY = "test-groq-key"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.side_effect = RuntimeError("Upstream Whisper API rate limit")
        mock_groq_class.return_value = mock_client

        with self.assertRaises(AudioTranscriptionError) as ctx:
            self.transcriber.transcribe(audio_data=SAMPLE_AUDIO_BYTES, mime_type="audio/wav")
        self.assertIn("Groq Whisper transcription failed", str(ctx.exception))


class TestApiChatAudioIntegration(unittest.TestCase):
    """Integration tests for POST /api/chat with audio payloads."""

    def setUp(self) -> None:
        self.app = create_app()
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    @patch("api.chat.audio_transcriber")
    def test_voice_only_request_success(self, mock_transcriber) -> None:
        """Verify request with only audio payload transcribes and generates response."""
        mock_transcriber.transcribe.return_value = "Explain quantum computing briefly"

        payload = {
            "audio": {
                "data": SAMPLE_AUDIO_B64,
                "mime_type": "audio/webm",
            }
        }

        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertIn("response", data)
        self.assertTrue(data["metadata"].get("audio_transcribed"))
        self.assertEqual(data["metadata"].get("transcription"), "Explain quantum computing briefly")
        mock_transcriber.transcribe.assert_called_once()

    @patch("api.chat.audio_transcriber")
    def test_hybrid_text_and_audio_request(self, mock_transcriber) -> None:
        """Verify text message and audio transcription are merged seamlessly."""
        mock_transcriber.transcribe.return_value = "how does photosynthesis work?"

        payload = {
            "message": "Please explain:",
            "audio": {
                "data": SAMPLE_AUDIO_B64,
                "mime_type": "audio/wav",
            },
        }

        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["ok"])
        self.assertTrue(data["metadata"].get("audio_transcribed"))

    def test_rejects_non_dict_audio(self) -> None:
        """Verify 400 when 'audio' is not an object."""
        payload = {
            "audio": "not_a_dict_payload",
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_AUDIO_TYPE")

    def test_rejects_missing_audio_data(self) -> None:
        """Verify 400 when 'audio' object is missing 'data'."""
        payload = {
            "audio": {
                "mime_type": "audio/webm",
            }
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "MISSING_AUDIO_DATA")

    def test_rejects_invalid_audio_data_type(self) -> None:
        """Verify 400 when 'audio.data' is not a string."""
        payload = {
            "audio": {
                "data": 12345,
                "mime_type": "audio/webm",
            }
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "INVALID_AUDIO_DATA")

    @patch("api.chat.audio_transcriber")
    def test_transcription_failure_returns_clean_400(self, mock_transcriber) -> None:
        """Verify graceful 400 error when AudioTranscriber raises AudioTranscriptionError."""
        mock_transcriber.transcribe.side_effect = AudioTranscriptionError("Simulated Whisper failure")

        payload = {
            "audio": {
                "data": SAMPLE_AUDIO_B64,
                "mime_type": "audio/webm",
            }
        }
        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["error"]["code"], "AUDIO_TRANSCRIPTION_FAILED")
        self.assertIn("Simulated Whisper failure", data["error"]["message"])

    @patch("api.chat.conversation_repo")
    @patch("api.chat.audio_transcriber")
    def test_database_annotates_voice_message(self, mock_transcriber, mock_repo) -> None:
        """Verify user message saved to database is prepended with [🎤 Voice Message]."""
        mock_transcriber.transcribe.return_value = "What time is it in Tokyo?"
        mock_repo.is_available.return_value = True
        mock_repo.get_conversation.return_value = {"id": "conv-voice", "user_id": "usr-voice"}
        mock_repo.get_messages.return_value = []
        mock_repo.get_memories.return_value = []

        payload = {
            "conversation_id": "conv-voice",
            "audio": {
                "data": SAMPLE_AUDIO_B64,
                "mime_type": "audio/webm",
            },
        }

        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)

        # Check call to save_message for the user message
        user_save_calls = [
            call for call in mock_repo.save_message.call_args_list
            if call[1].get("role") == "user"
        ]
        self.assertEqual(len(user_save_calls), 1)
        saved_content = user_save_calls[0][1]["content"]

        self.assertTrue(saved_content.startswith("[🎤 Voice Message]"))
        self.assertIn("What time is it in Tokyo?", saved_content)

    @patch("api.chat.audio_transcriber")
    def test_sse_streaming_with_audio(self, mock_transcriber) -> None:
        """Verify /api/chat?stream=true works with voice audio payloads."""
        mock_transcriber.transcribe.return_value = "Stream a poem about the sea"

        payload = {
            "stream": True,
            "audio": {
                "data": SAMPLE_AUDIO_B64,
                "mime_type": "audio/webm",
            },
        }

        response = self.client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.content_type)
        raw_data = response.get_data(as_text=True)
        self.assertIn("data: ", raw_data)


if __name__ == "__main__":
    unittest.main()
