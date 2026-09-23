"""Request and provider routing layer for Pihu-BreakThough.

Manages provider selection, dynamic registration, and safe fallback.
Stage 1 defaults to the deterministic local provider.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pihu_core.config import Config
from pihu_core.providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    OllamaProvider,
    ProviderResponse,
    Stage1DeterministicProvider,
)

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Routes chat requests to appropriate providers with graceful fallback."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}
        self._default_provider_name: str = "stage1-deterministic"
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Register built-in providers."""
        self.register_provider(Stage1DeterministicProvider())
        self.register_provider(GeminiProvider())
        self.register_provider(GroqProvider())
        self.register_provider(HuggingFaceProvider())
        self.register_provider(OllamaProvider())

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a provider instance."""
        self._providers[provider.name] = provider

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Retrieve a registered provider by name."""
        return self._providers.get(name)

    def get_active_provider(self) -> BaseProvider:
        """Return the default active provider for the current stage."""
        return self._providers[self._default_provider_name]

    def list_available_providers(self) -> List[str]:
        """List names of providers that report as currently available."""
        return [
            name for name, provider in self._providers.items() if provider.is_available()
        ]

    def route_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        provider_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Route chat prompt to the selected or default provider.

        Falls back gracefully if the requested provider is unavailable.
        """
        target_provider: BaseProvider

        if provider_name and provider_name in self._providers:
            candidate = self._providers[provider_name]
            if candidate.is_available():
                target_provider = candidate
            else:
                logger.info(
                    "Requested provider '%s' unavailable, falling back to '%s'",
                    provider_name,
                    self._default_provider_name,
                )
                target_provider = self.get_active_provider()
        else:
            target_provider = self.get_active_provider()

        try:
            response = target_provider.generate(
                message=message,
                history=history,
                **kwargs,
            )
        except NotImplementedError:
            logger.info(
                "Provider '%s' not implemented, falling back to '%s'",
                target_provider.name,
                self._default_provider_name,
            )
            fallback_provider = self.get_active_provider()
            response = fallback_provider.generate(
                message=message,
                history=history,
                **kwargs,
            )

        return {
            "ok": True,
            "app": Config.APP_NAME,
            "response": response.content,
            "provider": response.provider,
            "model": response.model,
            "metadata": response.metadata,
        }



# Global singleton router instance
router = ProviderRouter()
