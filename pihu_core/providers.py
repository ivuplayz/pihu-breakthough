"""Abstract and concrete LLM provider interfaces for Pihu-BreakThough.

Stage 3 introduces production-ready provider abstractions, capability matrices,
stream chunk specifications, and intent/strategy taxonomies.
Actual network execution to external vendor APIs is reserved for Stage 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import math
from typing import Any, Dict, Generator, List, Optional

from pihu_core.config import Config

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """Classified user intent determining optimal provider routing."""

    FAST_CHAT = "fast_chat"
    COMPLEX_REASONING = "complex_reasoning"
    CODING_SYSTEM = "coding_system"
    LOCAL_PRIVATE = "local_private"
    FALLBACK_SAFE = "fallback_safe"


class RoutingStrategy(str, Enum):
    """Strategic routing behavior configured per request or globally."""

    AUTO = "auto"
    LOW_LATENCY = "low_latency"
    HIGH_QUALITY = "high_quality"
    OFFLINE_ONLY = "offline_only"
    MANUAL = "manual"


class ProviderStatus(str, Enum):
    """Operational health status of an AI provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    TRIPPED = "tripped"
    UNAVAILABLE = "unavailable"


@dataclass
class ProviderCapabilities:
    """Declared capabilities and performance characteristics of a model provider."""

    supports_streaming: bool = True
    supports_tools: bool = False
    max_context_tokens: int = 8192
    typical_latency_ms: int = 500
    is_local: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert capability set to dictionary format."""
        return {
            "supports_streaming": self.supports_streaming,
            "supports_tools": self.supports_tools,
            "max_context_tokens": self.max_context_tokens,
            "typical_latency_ms": self.typical_latency_ms,
            "is_local": self.is_local,
        }


@dataclass
class ProviderResponse:
    """Standardized response from an LLM provider."""

    content: str
    provider: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary format."""
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
        }


@dataclass
class StreamChunk:
    """Standardized streaming chunk emitted during token generation."""

    text: str
    is_final: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "text": self.text,
            "is_final": self.is_final,
            "metadata": self.metadata,
        }


class BaseProvider(ABC):
    """Abstract base class for all LLM providers in Pihu-BreakThough."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Active model identifier."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Model capabilities and limits."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials/endpoints are available in environment."""
        pass

    def health_check(self) -> bool:
        """Perform a quick active health probe. Default returns availability status."""
        return self.is_available()

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string using standard ~4 chars/token heuristic."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))

    @abstractmethod
    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a complete synchronous response."""
        pass

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Generate response tokens as a stream.

        Default implementation falls back to emitting full generate() content as a single chunk.
        """
        resp = self.generate(message=message, history=history, system_prompt=system_prompt, **kwargs)
        yield StreamChunk(
            text=resp.content,
            is_final=True,
            metadata=resp.metadata,
        )


class Stage1DeterministicProvider(BaseProvider):
    """Deterministic local provider for core foundation and zero-failure safety net."""

    @property
    def name(self) -> str:
        return "stage1-deterministic"

    @property
    def model_name(self) -> str:
        return "core-deterministic-v1"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=16384,
            typical_latency_ms=5,
            is_local=True,
        )

    def is_available(self) -> bool:
        return True

    def health_check(self) -> bool:
        return True

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        history_len = len(history) if history else 0
        timestamp = datetime.now(timezone.utc).isoformat()

        reply = (
            f"Pihu-BreakThough Stage 1 Core is operational. "
            f"Received message ({len(message)} chars): \"{message.strip()}\". "
            f"History context contains {history_len} items."
        )

        return ProviderResponse(
            content=reply,
            provider=self.name,
            model=self.model_name,
            metadata={
                "received_chars": len(message),
                "history_length": history_len,
                "timestamp": timestamp,
                "stage": Config.STAGE,
            },
        )

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        resp = self.generate(message=message, history=history, system_prompt=system_prompt, **kwargs)
        words = resp.content.split(" ")
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            chunk_text = word if is_last else word + " "
            yield StreamChunk(
                text=chunk_text,
                is_final=is_last,
                metadata=resp.metadata if is_last else {},
            )


# Safe imports for AI vendor SDKs
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    GENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    Groq = None  # type: ignore
    GROQ_AVAILABLE = False


class GeminiProvider(BaseProvider):
    """Google Gemini provider (Tier 1 Primary Frontier Engine) using google.generativeai SDK."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            max_context_tokens=1048576,
            typical_latency_ms=450,
            is_local=False,
        )

    def is_available(self) -> bool:
        """Check presence of GEMINI_API_KEY and SDK availability."""
        return bool(Config.GEMINI_API_KEY and Config.GEMINI_API_KEY.strip() and GENAI_AVAILABLE)

    def _build_contents(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Map standard chat history to Gemini contents structure."""
        contents: List[Dict[str, Any]] = []
        if history:
            for item in history:
                # Gemini roles are 'user' and 'model'
                role = "model" if item.get("role") in ("assistant", "model") else "user"
                content_text = item.get("content", "")
                contents.append({"role": role, "parts": [content_text]})

        contents.append({"role": "user", "parts": [message]})
        return contents

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response via Google Gemini API."""
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not configured or google.generativeai SDK is unavailable.")

        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model_kwargs: Dict[str, Any] = {}
            if system_prompt:
                model_kwargs["system_instruction"] = system_prompt

            model = genai.GenerativeModel(self.model_name, **model_kwargs)
            contents = self._build_contents(message=message, history=history)
            timeout = kwargs.get("timeout", 30)

            response = model.generate_content(
                contents,
                request_options={"timeout": timeout},
            )

            text_output = response.text if response and hasattr(response, "text") else ""
            timestamp = datetime.now(timezone.utc).isoformat()

            return ProviderResponse(
                content=text_output,
                provider=self.name,
                model=self.model_name,
                metadata={
                    "stage": Config.STAGE,
                    "timestamp": timestamp,
                    "provider": self.name,
                    "model": self.model_name,
                },
            )
        except Exception as exc:
            logger.error("Gemini API call failed: %s", exc)
            raise RuntimeError(f"Gemini API error: {exc}") from exc

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Stream response tokens via Google Gemini API."""
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not configured or google.generativeai SDK is unavailable.")

        try:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            model_kwargs: Dict[str, Any] = {}
            if system_prompt:
                model_kwargs["system_instruction"] = system_prompt

            model = genai.GenerativeModel(self.model_name, **model_kwargs)
            contents = self._build_contents(message=message, history=history)
            timeout = kwargs.get("timeout", 30)

            response = model.generate_content(
                contents,
                stream=True,
                request_options={"timeout": timeout},
            )

            for chunk in response:
                chunk_text = chunk.text if hasattr(chunk, "text") else ""
                if chunk_text:
                    yield StreamChunk(text=chunk_text, is_final=False)

            yield StreamChunk(text="", is_final=True)
        except Exception as exc:
            logger.error("Gemini stream failed: %s", exc)
            raise RuntimeError(f"Gemini streaming error: {exc}") from exc


class GroqProvider(BaseProvider):
    """Groq provider (Tier 2 Ultra-Low-Latency Inference Engine) using official groq SDK."""

    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            max_context_tokens=32768,
            typical_latency_ms=180,
            is_local=False,
        )

    def is_available(self) -> bool:
        """Check presence of GROQ_API_KEY and SDK availability."""
        return bool(Config.GROQ_API_KEY and Config.GROQ_API_KEY.strip() and GROQ_AVAILABLE)

    def _build_messages(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Map chat history and system prompt to OpenAI/Groq messages format."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            for item in history:
                role = item.get("role", "user")
                if role not in ("system", "user", "assistant"):
                    role = "user"
                messages.append({"role": role, "content": item.get("content", "")})

        messages.append({"role": "user", "content": message})
        return messages

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate response via Groq API."""
        if not self.is_available():
            raise RuntimeError("GROQ_API_KEY is not configured or groq SDK is unavailable.")

        try:
            client = Groq(api_key=Config.GROQ_API_KEY, timeout=kwargs.get("timeout", 30))
            messages = self._build_messages(message=message, history=history, system_prompt=system_prompt)
            temperature = kwargs.get("temperature", 0.7)

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )

            reply_text = completion.choices[0].message.content or ""
            timestamp = datetime.now(timezone.utc).isoformat()

            usage_meta: Dict[str, Any] = {}
            if hasattr(completion, "usage") and completion.usage:
                usage_meta = {
                    "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                    "total_tokens": getattr(completion.usage, "total_tokens", None),
                }

            return ProviderResponse(
                content=reply_text,
                provider=self.name,
                model=self.model_name,
                metadata={
                    "stage": Config.STAGE,
                    "timestamp": timestamp,
                    "usage": usage_meta,
                    "provider": self.name,
                    "model": self.model_name,
                },
            )
        except Exception as exc:
            logger.error("Groq API call failed: %s", exc)
            raise RuntimeError(f"Groq API error: {exc}") from exc

    def stream_generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[StreamChunk, None, None]:
        """Stream response tokens via Groq API."""
        if not self.is_available():
            raise RuntimeError("GROQ_API_KEY is not configured or groq SDK is unavailable.")

        try:
            client = Groq(api_key=Config.GROQ_API_KEY, timeout=kwargs.get("timeout", 30))
            messages = self._build_messages(message=message, history=history, system_prompt=system_prompt)
            temperature = kwargs.get("temperature", 0.7)

            stream = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else ""
                if delta:
                    yield StreamChunk(text=delta, is_final=False)

            yield StreamChunk(text="", is_final=True)
        except Exception as exc:
            logger.error("Groq stream failed: %s", exc)
            raise RuntimeError(f"Groq streaming error: {exc}") from exc



class HuggingFaceProvider(BaseProvider):
    """Hugging Face Inference provider (Tier 3 Specialized Open-Weights Engine)."""

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return "Qwen/Qwen2.5-72B-Instruct"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=32768,
            typical_latency_ms=750,
            is_local=False,
        )

    def is_available(self) -> bool:
        return bool(Config.HF_API_KEY and Config.HF_API_KEY.strip())

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if not self.is_available():
            raise RuntimeError("HF_API_KEY is not configured.")

        raise NotImplementedError(
            "Hugging Face live API client execution is scheduled for Stage 4."
        )


class OllamaProvider(BaseProvider):
    """Local Ollama provider (Tier 4 Private & Offline Local Engine)."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "llama3"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
            max_context_tokens=8192,
            typical_latency_ms=600,
            is_local=True,
        )

    def is_available(self) -> bool:
        return bool(Config.OLLAMA_BASE_URL and Config.OLLAMA_BASE_URL.strip())

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if not self.is_available():
            raise RuntimeError("OLLAMA_BASE_URL is not configured.")

        raise NotImplementedError(
            "Ollama local API client execution is scheduled for Stage 4."
        )
