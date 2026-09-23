"""Speech-to-text audio processing using Groq's official API for Pihu-BreakThough.

Stage 7 implements fast, serverless-friendly audio transcription using
whisper-large-v3-turbo via the official groq SDK without local ffmpeg/torch dependencies.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, Optional, Union

from pihu_core.config import Config

logger = logging.getLogger(__name__)

# Safe import for Groq SDK
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore
    GROQ_AVAILABLE = False

DEFAULT_WHISPER_MODEL = "whisper-large-v3-turbo"

SUPPORTED_AUDIO_MIMES: Dict[str, str] = {
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mp4": "mp4",
    "audio/flac": "flac",
    "audio/aac": "aac",
}

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB


class AudioTranscriptionError(RuntimeError):
    """Raised when audio transcription validation or API execution fails."""
    pass


class AudioTranscriber:
    """Manages audio validation, decoding, and speech-to-text via Groq Whisper."""

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL) -> None:
        self.model_name = model_name

    def is_available(self) -> bool:
        """Check presence of GROQ_API_KEY and groq SDK availability."""
        return bool(Config.GROQ_API_KEY and Config.GROQ_API_KEY.strip() and GROQ_AVAILABLE)

    def transcribe(
        self,
        audio_data: Union[bytes, io.BufferedIOBase, io.BytesIO, str],
        mime_type: str = "audio/webm",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Transcribe speech audio into text using Groq Whisper.

        Args:
            audio_data: Raw audio bytes, file-like BytesIO stream, or base64 string.
            mime_type: Audio MIME format (audio/webm, audio/mp3, audio/wav, audio/ogg, audio/m4a, etc.).
            language: Optional ISO 639-1 language code hint.
            prompt: Optional transcription prompt for domain guidance.
            **kwargs: Additional parameters passed to Groq audio transcription.

        Returns:
            Transcribed text string.

        Raises:
            AudioTranscriptionError: If format is invalid or upstream transcription fails.
        """
        clean_mime = (mime_type or "audio/webm").strip().lower()

        # Handle base64 string and data URIs
        raw_bytes: bytes
        if isinstance(audio_data, str):
            b64_str = audio_data.strip()
            if b64_str.startswith("data:"):
                try:
                    header, b64_part = b64_str.split(",", 1)
                    b64_str = b64_part
                    if ";base64" in header:
                        extracted_mime = header.split(":", 1)[1].split(";", 1)[0].strip().lower()
                        if extracted_mime in SUPPORTED_AUDIO_MIMES:
                            clean_mime = extracted_mime
                except Exception as uri_err:
                    raise AudioTranscriptionError(f"Malformed audio data URI: {uri_err}") from uri_err

            try:
                raw_bytes = base64.b64decode(b64_str, validate=True)
            except Exception as b64_err:
                raise AudioTranscriptionError(f"Invalid base64 audio encoding: {b64_err}") from b64_err

        elif isinstance(audio_data, (bytes, bytearray)):
            raw_bytes = bytes(audio_data)
        elif hasattr(audio_data, "read"):
            raw_bytes = audio_data.read()
        else:
            raise AudioTranscriptionError("Unsupported audio data type. Must be bytes, BytesIO, or base64 string.")

        if not raw_bytes or len(raw_bytes) == 0:
            raise AudioTranscriptionError("Audio data cannot be empty.")

        if len(raw_bytes) > MAX_AUDIO_BYTES:
            raise AudioTranscriptionError(
                f"Audio payload ({len(raw_bytes)} bytes) exceeds maximum allowed size of {MAX_AUDIO_BYTES} bytes (25MB)."
            )

        if clean_mime not in SUPPORTED_AUDIO_MIMES:
            raise AudioTranscriptionError(
                f"Unsupported audio MIME type '{clean_mime}'. Supported formats: {sorted(list(SUPPORTED_AUDIO_MIMES.keys()))}."
            )

        if not self.is_available():
            raise AudioTranscriptionError(
                "GROQ_API_KEY is not configured or groq SDK is unavailable."
            )

        ext = SUPPORTED_AUDIO_MIMES.get(clean_mime, "webm")
        file_tuple = (f"audio.{ext}", io.BytesIO(raw_bytes), clean_mime)

        try:
            client = Groq(api_key=Config.GROQ_API_KEY, timeout=kwargs.get("timeout", 60))
            call_params: Dict[str, Any] = {
                "model": self.model_name,
                "file": file_tuple,
                "response_format": "json",
            }
            if language:
                call_params["language"] = language
            if prompt:
                call_params["prompt"] = prompt

            transcription = client.audio.transcriptions.create(**call_params)
            text = getattr(transcription, "text", str(transcription)) or ""
            return text.strip()

        except Exception as exc:
            sanitized_err = str(exc).replace(Config.GROQ_API_KEY or "", "***") if Config.GROQ_API_KEY else str(exc)
            logger.error("Groq Whisper audio transcription failed: %s", sanitized_err)
            raise AudioTranscriptionError(f"Groq Whisper transcription failed: {sanitized_err}") from exc


# Global singleton instance
audio_transcriber = AudioTranscriber()
