"""Request and AI provider routing engine for Pihu-BreakThough (Stage 3).

Features:
- Fast rule-based intent classification.
- Dynamic multi-tier fallback chain resolution.
- Circuit breaking with automatic cooldown quarantine.
- Telemetry tracking for latencies and fallback transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Dict, Generator, List, Optional

from pihu_core.config import Config
from pihu_core.providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    IntentType,
    OllamaProvider,
    ProviderResponse,
    ProviderStatus,
    RoutingStrategy,
    Stage1DeterministicProvider,
    StreamChunk,
)

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_MAX_FAILURES = 3
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60.0


class ProviderCircuitBreaker:
    """Manages failure counts and temporary quarantine for a provider."""

    def __init__(self, provider_name: str, cooldown: float = CIRCUIT_BREAKER_COOLDOWN_SECONDS) -> None:
        self.provider_name = provider_name
        self.cooldown = cooldown
        self.consecutive_failures: int = 0
        self.last_failure_time: float = 0.0
        self.status: ProviderStatus = ProviderStatus.HEALTHY

    def record_success(self) -> None:
        """Reset failure counter and set state to HEALTHY."""
        self.consecutive_failures = 0
        self.status = ProviderStatus.HEALTHY

    def record_failure(self, error: Exception) -> None:
        """Record an error and trip circuit if consecutive threshold is crossed."""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
        if self.consecutive_failures >= CIRCUIT_BREAKER_MAX_FAILURES:
            self.status = ProviderStatus.TRIPPED
            logger.warning(
                "Circuit breaker tripped for provider '%s' after %d consecutive failures. In cooldown for %0.1fs.",
                self.provider_name,
                self.consecutive_failures,
                self.cooldown,
            )

    def is_available_for_traffic(self) -> bool:
        """Check if provider is eligible to receive traffic."""
        if self.status != ProviderStatus.TRIPPED:
            return True
        # Check if cooldown has elapsed
        if time.time() - self.last_failure_time >= self.cooldown:
            logger.info("Cooldown elapsed for provider '%s'. Testing half-open probe.", self.provider_name)
            self.status = ProviderStatus.DEGRADED
            return True
        return False


def classify_intent(message: str) -> IntentType:
    """High-throughput heuristic classifier mapping prompts to IntentType.

    Avoids latency overhead from auxiliary classification models.
    """
    if not message:
        return IntentType.FALLBACK_SAFE

    lower = message.lower().strip()

    # 1. Local / Private checks
    if any(k in lower for k in ("offline only", "keep private", "confidential", "local only")):
        return IntentType.LOCAL_PRIVATE

    # 2. Summarization checks
    summarize_indicators = (
        "summarize", "summarise", "summary", "tldr", "tl;dr",
        "synopsis", "recap", "brief overview", "key takeaways",
        "condense", "gist", "in a nutshell", "give me the short version"
    )
    if any(k in lower for k in summarize_indicators):
        return IntentType.SUMMARIZATION

    # 3. Fast Chat checks (short queries, pleasantries)
    greetings = ("hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you", "bye", "who are you")
    if any(lower == g or lower.startswith(g + " ") or lower.startswith(g + "!") or lower.startswith(g + ".") for g in greetings):
        return IntentType.FAST_CHAT

    if len(lower) < 25 and not any(c in lower for c in ("def ", "class ", "sql", "explain", "derive")):
        return IntentType.FAST_CHAT

    # 4. Coding & System prompts
    code_indicators = (
        "def ", "class ", "function ", "import ", "sql", "select ", "insert ",
        "update ", "delete ", "json", "traceback", "syntaxerror", "python",
        "javascript", "typescript", "html", "css", "dockerfile", "git "
    )
    if any(k in lower for k in code_indicators) or "```" in message:
        return IntentType.CODING_SYSTEM

    # 5. Complex Reasoning
    reasoning_indicators = (
        "explain step by step", "prove that", "derive", "compare and contrast",
        "architectural design", "solve", "why does", "what are the implications",
        "detailed analysis"
    )
    if any(k in lower for k in reasoning_indicators) or len(message) > 600:
        return IntentType.COMPLEX_REASONING

    return IntentType.FAST_CHAT


class ProviderRouter:
    """Routes chat requests across an intelligent multi-tier provider hierarchy."""

    def __init__(self) -> None:
        self._providers: Dict[str, BaseProvider] = {}
        self._circuit_breakers: Dict[str, ProviderCircuitBreaker] = {}
        self._default_provider_name: str = "stage1-deterministic"
        self._register_default_providers()

    def _register_default_providers(self) -> None:
        """Initialize built-in providers and their circuit breakers."""
        self.register_provider(Stage1DeterministicProvider())
        self.register_provider(GeminiProvider())
        self.register_provider(GroqProvider())
        self.register_provider(HuggingFaceProvider())
        self.register_provider(OllamaProvider())

    def register_provider(self, provider: BaseProvider) -> None:
        """Register a provider instance and associate a circuit breaker."""
        self._providers[provider.name] = provider
        self._circuit_breakers[provider.name] = ProviderCircuitBreaker(provider.name)

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Retrieve a registered provider by name."""
        return self._providers.get(name)

    def get_active_provider(self) -> BaseProvider:
        """Return the default fallback provider."""
        return self._providers[self._default_provider_name]

    def list_available_providers(self) -> List[str]:
        """List names of providers that report available and not currently tripped."""
        available = []
        for name, provider in self._providers.items():
            if provider.is_available() and self._circuit_breakers[name].is_available_for_traffic():
                available.append(name)
        return available

    def resolve_provider_chain(
        self,
        intent: IntentType,
        strategy: RoutingStrategy,
        hint: Optional[str] = None,
    ) -> List[BaseProvider]:
        """Resolve ordered list of candidate providers based on intent and strategy."""
        candidates: List[BaseProvider] = []

        # If user explicitly requested a provider hint
        if hint and hint in self._providers:
            candidates.append(self._providers[hint])

        # Offline / Local Strategy
        if strategy == RoutingStrategy.OFFLINE_ONLY or intent == IntentType.LOCAL_PRIVATE:
            for name in ("ollama", "stage1-deterministic"):
                if name in self._providers and self._providers[name] not in candidates:
                    candidates.append(self._providers[name])
            return candidates

        # Summarization Strategy or Intent
        if strategy == RoutingStrategy.SUMMARIZATION or intent == IntentType.SUMMARIZATION:
            order = ("huggingface", "gemini", "groq", "stage1-deterministic")
        # Low Latency Strategy or Fast Chat
        elif strategy == RoutingStrategy.LOW_LATENCY or intent == IntentType.FAST_CHAT:
            order = ("groq", "gemini", "huggingface", "stage1-deterministic")
        # High Quality Strategy or Complex Reasoning / Coding
        elif strategy == RoutingStrategy.HIGH_QUALITY or intent in (IntentType.COMPLEX_REASONING, IntentType.CODING_SYSTEM):
            order = ("gemini", "groq", "huggingface", "stage1-deterministic")
        else:
            # Default balanced chain: Gemini -> Groq -> HuggingFace -> Deterministic fallback
            order = ("gemini", "groq", "huggingface", "stage1-deterministic")

        for name in order:
            if name in self._providers:
                p = self._providers[name]
                if p not in candidates:
                    candidates.append(p)

        return candidates

    def route_chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        strategy: str | RoutingStrategy = RoutingStrategy.AUTO,
        provider_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Route chat prompt across multi-tier hierarchy with transparent fallback.

        Args:
            message: User query string.
            history: Optional list of previous chat messages.
            strategy: Routing strategy (auto, low_latency, high_quality, offline_only, manual).
            provider_name: Optional explicit provider hint.
            system_prompt: Optional system personality/instruction string.

        Returns:
            Normalized dictionary containing response content and execution metadata.
        """
        start_time = time.time()

        # Parse routing strategy
        if isinstance(strategy, str):
            try:
                parsed_strategy = RoutingStrategy(strategy.lower())
            except ValueError:
                parsed_strategy = RoutingStrategy.AUTO
        else:
            parsed_strategy = strategy

        intent = classify_intent(message)
        chain = self.resolve_provider_chain(
            intent=intent,
            strategy=parsed_strategy,
            hint=provider_name,
        )

        attempted_chain: List[str] = []
        selected_provider: Optional[BaseProvider] = None
        response: Optional[ProviderResponse] = None

        for candidate in chain:
            attempted_chain.append(candidate.name)
            cb = self._circuit_breakers[candidate.name]

            # Skip candidate if unconfigured or currently quarantined by circuit breaker
            if not candidate.is_available():
                logger.debug("Provider '%s' skipped: credentials/endpoint not configured.", candidate.name)
                continue

            if not cb.is_available_for_traffic():
                logger.debug("Provider '%s' skipped: circuit breaker in cooldown.", candidate.name)
                continue

            try:
                response = candidate.generate(
                    message=message,
                    history=history,
                    system_prompt=system_prompt,
                    **kwargs,
                )
                cb.record_success()
                selected_provider = candidate
                break
            except NotImplementedError as exc:
                # Expected in Stage 3 for skeleton classes
                logger.info("Provider '%s' execution not implemented in this stage (%s).", candidate.name, exc)
                cb.record_failure(exc)
                continue
            except Exception as exc:
                logger.warning("Provider '%s' failed execution: %s", candidate.name, exc)
                cb.record_failure(exc)
                continue

        # If all candidates failed or skipped, execute deterministic safety net
        if response is None:
            fallback = self.get_active_provider()
            attempted_chain.append(fallback.name)
            response = fallback.generate(
                message=message,
                history=history,
                system_prompt=system_prompt,
                **kwargs,
            )
            selected_provider = fallback

        elapsed_ms = int((time.time() - start_time) * 1000)
        fallback_occurred = len(attempted_chain) > 1 and attempted_chain[0] != selected_provider.name

        combined_metadata = {
            **response.metadata,
            "intent": intent.value,
            "strategy": parsed_strategy.value,
            "latency_ms": elapsed_ms,
            "fallback_occurred": fallback_occurred,
            "fallback_chain": attempted_chain,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "ok": True,
            "app": Config.APP_NAME,
            "response": response.content,
            "provider": response.provider,
            "model": response.model,
            "metadata": combined_metadata,
        }

    def route_stream(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        strategy: str | RoutingStrategy = RoutingStrategy.AUTO,
        provider_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        """Route chat prompt yielding streaming chunks with transparent fallback."""
        start_time = time.time()

        if isinstance(strategy, str):
            try:
                parsed_strategy = RoutingStrategy(strategy.lower())
            except ValueError:
                parsed_strategy = RoutingStrategy.AUTO
        else:
            parsed_strategy = strategy

        intent = classify_intent(message)
        chain = self.resolve_provider_chain(
            intent=intent,
            strategy=parsed_strategy,
            hint=provider_name,
        )

        for candidate in chain:
            cb = self._circuit_breakers[candidate.name]
            if not candidate.is_available() or not cb.is_available_for_traffic():
                continue

            try:
                for chunk in candidate.stream_generate(
                    message=message,
                    history=history,
                    system_prompt=system_prompt,
                    **kwargs,
                ):
                    yield {
                        "ok": True,
                        "text": chunk.text,
                        "is_final": chunk.is_final,
                        "provider": candidate.name,
                        "model": candidate.model_name,
                        "metadata": chunk.metadata,
                    }
                cb.record_success()
                return
            except Exception as exc:
                logger.warning("Stream provider '%s' failed: %s", candidate.name, exc)
                cb.record_failure(exc)
                continue

        # Deterministic fallback stream
        fallback = self.get_active_provider()
        for chunk in fallback.stream_generate(
            message=message,
            history=history,
            system_prompt=system_prompt,
            **kwargs,
        ):
            yield {
                "ok": True,
                "text": chunk.text,
                "is_final": chunk.is_final,
                "provider": fallback.name,
                "model": fallback.model_name,
                "metadata": chunk.metadata,
            }


# Global singleton router instance
router = ProviderRouter()
