"""Abstract and concrete LLM provider interfaces for Pihu-BreakThough.

Stage 1 uses a deterministic/local provider foundation that requires
zero external API keys or remote connections. Stubs for future providers
(Gemini, Groq, Hugging Face, Ollama) are structured for subsequent stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pihu_core.config import Config


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

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider credentials/endpoints are available."""
        pass

    @abstractmethod
    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response given a prompt and optional conversation history."""
        pass


class Stage1DeterministicProvider(BaseProvider):
    """Deterministic local provider for Stage 1 foundation.

    Provides reliable, deterministic responses without external network dependencies.
    """

    @property
    def name(self) -> str:
        return "stage1-deterministic"

    @property
    def model_name(self) -> str:
        return "core-deterministic-v1"

    def is_available(self) -> bool:
        return True

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        history_len = len(history) if history else 0
        timestamp = datetime.now(timezone.utc).isoformat()

        # Deterministic stage 1 acknowledgement response
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
                "stage": "Stage 1",
            },
        )


class GeminiProvider(BaseProvider):
    """Google Gemini provider (Future Stage)."""

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return "gemini-2.5-flash"

    def is_available(self) -> bool:
        # In Stage 1, external providers are stubs; active execution is deferred
        return False

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "Gemini provider execution is deferred to subsequent stages."
        )


class GroqProvider(BaseProvider):
    """Groq provider (Future Stage)."""

    @property
    def name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return "llama-3.3-70b-versatile"

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "Groq provider execution is deferred to subsequent stages."
        )


class HuggingFaceProvider(BaseProvider):
    """Hugging Face provider (Future Stage)."""

    @property
    def name(self) -> str:
        return "huggingface"

    @property
    def model_name(self) -> str:
        return "hf-inference-v1"

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "Hugging Face provider execution is deferred to subsequent stages."
        )


class OllamaProvider(BaseProvider):
    """Local Ollama provider (Future Stage)."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return "llama3"

    def is_available(self) -> bool:
        return False

    def generate(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        raise NotImplementedError(
            "Ollama provider execution is deferred to subsequent stages."
        )

