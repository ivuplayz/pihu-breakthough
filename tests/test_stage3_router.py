"""Unit and integration test suite for Stage 3 Brain and AI Provider Router."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from pihu_core.config import Config
from pihu_core.providers import (
    BaseProvider,
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    IntentType,
    OllamaProvider,
    ProviderCapabilities,
    ProviderResponse,
    ProviderStatus,
    RoutingStrategy,
    Stage1DeterministicProvider,
    StreamChunk,
)
from pihu_core.router import (
    CIRCUIT_BREAKER_MAX_FAILURES,
    ProviderCircuitBreaker,
    ProviderRouter,
    classify_intent,
    router,
)


class MockHealthyProvider(BaseProvider):
    """Mock provider returning simulated successful responses."""

    def __init__(self, name: str = "mock-healthy", model: str = "mock-v1") -> None:
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(supports_streaming=True)

    def is_available(self) -> bool:
        return True

    def generate(self, message: str, **kwargs) -> ProviderResponse:
        return ProviderResponse(
            content=f"Mock response from {self._name}",
            provider=self._name,
            model=self._model,
            metadata={"mock": True},
        )


class MockFailingProvider(BaseProvider):
    """Mock provider that always raises an operational exception."""

    def __init__(self, name: str = "mock-failing") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return "failing-model"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    def is_available(self) -> bool:
        return True

    def generate(self, message: str, **kwargs) -> ProviderResponse:
        raise RuntimeError(f"Simulated upstream outage for {self._name}")


class TestStage3BrainRouter(unittest.TestCase):
    """Test suite covering Stage 3 intent classification, circuit breaking, and routing."""

    # --------------------------------------------------------------------------
    # 1. Intent Classification Tests
    # --------------------------------------------------------------------------
    def test_classify_fast_chat_intents(self) -> None:
        """Greetings and brief questions map to FAST_CHAT."""
        self.assertEqual(classify_intent("Hello!"), IntentType.FAST_CHAT)
        self.assertEqual(classify_intent("hey there, how are you?"), IntentType.FAST_CHAT)
        self.assertEqual(classify_intent("thanks a lot"), IntentType.FAST_CHAT)
        self.assertEqual(classify_intent("what time is it?"), IntentType.FAST_CHAT)

    def test_classify_coding_intents(self) -> None:
        """Prompts containing code syntax map to CODING_SYSTEM."""
        self.assertEqual(classify_intent("def calculate_tax(amount): return amount * 0.2"), IntentType.CODING_SYSTEM)
        self.assertEqual(classify_intent("Write a python script to fetch JSON from an API"), IntentType.CODING_SYSTEM)
        self.assertEqual(classify_intent("Fix this syntaxerror in my javascript code"), IntentType.CODING_SYSTEM)
        self.assertEqual(classify_intent("SELECT * FROM users WHERE active = true;"), IntentType.CODING_SYSTEM)

    def test_classify_complex_reasoning_intents(self) -> None:
        """Prompts requesting detailed proofs, analysis, or lengthy queries map to COMPLEX_REASONING."""
        self.assertEqual(classify_intent("Explain step by step how RSA encryption works."), IntentType.COMPLEX_REASONING)
        self.assertEqual(classify_intent("Compare and contrast microservices and monolithic architectures."), IntentType.COMPLEX_REASONING)
        long_prompt = "Analyze the economic implications of automated AI coding assistants on modern software engineering workflows. " * 10
        self.assertEqual(classify_intent(long_prompt), IntentType.COMPLEX_REASONING)

    def test_classify_local_private_intents(self) -> None:
        """Queries with privacy or offline keywords map to LOCAL_PRIVATE."""
        self.assertEqual(classify_intent("Keep this confidential: analyze my private journal"), IntentType.LOCAL_PRIVATE)
        self.assertEqual(classify_intent("Run this offline only please"), IntentType.LOCAL_PRIVATE)

    def test_classify_empty_message(self) -> None:
        """Empty messages map safely to FALLBACK_SAFE."""
        self.assertEqual(classify_intent(""), IntentType.FALLBACK_SAFE)

    # --------------------------------------------------------------------------
    # 2. Circuit Breaker Lifecycle Tests
    # --------------------------------------------------------------------------
    def test_circuit_breaker_lifecycle(self) -> None:
        """Circuit breaker trips after consecutive failures and recovers on success."""
        cb = ProviderCircuitBreaker("test-provider", cooldown=0.1)
        self.assertEqual(cb.status, ProviderStatus.HEALTHY)
        self.assertTrue(cb.is_available_for_traffic())

        # 1-2 failures do not trip
        cb.record_failure(RuntimeError("fail 1"))
        cb.record_failure(RuntimeError("fail 2"))
        self.assertEqual(cb.consecutive_failures, 2)
        self.assertEqual(cb.status, ProviderStatus.HEALTHY)
        self.assertTrue(cb.is_available_for_traffic())

        # 3rd consecutive failure trips the circuit
        cb.record_failure(RuntimeError("fail 3"))
        self.assertEqual(cb.consecutive_failures, 3)
        self.assertEqual(cb.status, ProviderStatus.TRIPPED)
        self.assertFalse(cb.is_available_for_traffic())

        # Wait for cooldown to expire
        time.sleep(0.12)
        # Should now be eligible for half-open test
        self.assertTrue(cb.is_available_for_traffic())
        self.assertEqual(cb.status, ProviderStatus.DEGRADED)

        # Successful call resets circuit
        cb.record_success()
        self.assertEqual(cb.consecutive_failures, 0)
        self.assertEqual(cb.status, ProviderStatus.HEALTHY)
        self.assertTrue(cb.is_available_for_traffic())

    # --------------------------------------------------------------------------
    # 3. Strategy & Chain Resolution Tests
    # --------------------------------------------------------------------------
    def test_chain_resolution_low_latency(self) -> None:
        """LOW_LATENCY strategy prioritizes Groq at the front of the chain."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.FAST_CHAT,
            strategy=RoutingStrategy.LOW_LATENCY,
        )
        self.assertEqual(chain[0].name, "groq")
        self.assertEqual(chain[-1].name, "stage1-deterministic")

    def test_chain_resolution_high_quality(self) -> None:
        """HIGH_QUALITY strategy prioritizes Gemini at the front of the chain."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.COMPLEX_REASONING,
            strategy=RoutingStrategy.HIGH_QUALITY,
        )
        self.assertEqual(chain[0].name, "gemini")
        self.assertEqual(chain[-1].name, "stage1-deterministic")

    def test_chain_resolution_offline_only(self) -> None:
        """OFFLINE_ONLY strategy only includes local Ollama and deterministic safety net."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.LOCAL_PRIVATE,
            strategy=RoutingStrategy.OFFLINE_ONLY,
        )
        names = [p.name for p in chain]
        self.assertEqual(names, ["ollama", "stage1-deterministic"])

    def test_chain_resolution_with_hint(self) -> None:
        """Explicit provider hint places requested provider at index 0."""
        test_router = ProviderRouter()
        chain = test_router.resolve_provider_chain(
            intent=IntentType.FAST_CHAT,
            strategy=RoutingStrategy.AUTO,
            hint="groq",
        )
        self.assertEqual(chain[0].name, "groq")

    # --------------------------------------------------------------------------
    # 4. Multi-Tier Fallback Routing Execution Tests
    # --------------------------------------------------------------------------
    def test_routing_successful_primary_provider(self) -> None:
        """When primary provider succeeds, returns its response without fallback."""
        test_router = ProviderRouter()
        mock_p = MockHealthyProvider(name="mock-primary", model="primary-v1")
        test_router.register_provider(mock_p)

        result = test_router.route_chat(
            message="Hello",
            provider_name="mock-primary",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "mock-primary")
        self.assertEqual(result["model"], "primary-v1")
        self.assertFalse(result["metadata"]["fallback_occurred"])

    def test_routing_cascades_on_failing_provider(self) -> None:
        """When a candidate fails, router automatically cascades to the next healthy provider."""
        test_router = ProviderRouter()
        failing_p = MockFailingProvider(name="failing-primary")
        healthy_p = MockHealthyProvider(name="healthy-backup", model="backup-v1")
        test_router.register_provider(failing_p)
        test_router.register_provider(healthy_p)

        # Mock resolve_provider_chain to test exact cascade
        with patch.object(test_router, "resolve_provider_chain", return_value=[failing_p, healthy_p]):
            result = test_router.route_chat(message="Test query")
            self.assertTrue(result["ok"])
            self.assertEqual(result["provider"], "healthy-backup")
            self.assertTrue(result["metadata"]["fallback_occurred"])
            self.assertIn("failing-primary", result["metadata"]["fallback_chain"])
            self.assertIn("healthy-backup", result["metadata"]["fallback_chain"])

    def test_routing_all_providers_fail_safely_to_deterministic(self) -> None:
        """When all candidates fail, execution falls back cleanly to deterministic safety net."""
        test_router = ProviderRouter()
        failing_1 = MockFailingProvider(name="fail-1")
        failing_2 = MockFailingProvider(name="fail-2")
        test_router.register_provider(failing_1)
        test_router.register_provider(failing_2)

        with patch.object(test_router, "resolve_provider_chain", return_value=[failing_1, failing_2]):
            result = test_router.route_chat(message="Catastrophic test")
            self.assertTrue(result["ok"])
            self.assertEqual(result["provider"], "stage1-deterministic")
            self.assertTrue(result["metadata"]["fallback_occurred"])
            self.assertIn("stage1-deterministic", result["metadata"]["fallback_chain"])

    # --------------------------------------------------------------------------
    # 5. Token Estimation & Capabilities Tests
    # --------------------------------------------------------------------------
    def test_token_estimation_and_capabilities(self) -> None:
        """Providers provide sensible token estimation and capability reporting."""
        p = Stage1DeterministicProvider()
        tokens = p.estimate_tokens("Hello world, this is a test.")
        self.assertGreater(tokens, 0)
        self.assertEqual(p.estimate_tokens(""), 0)

        caps = p.capabilities
        self.assertTrue(caps.supports_streaming)
        self.assertTrue(caps.is_local)
        self.assertIsInstance(caps.to_dict(), dict)

    # --------------------------------------------------------------------------
    # 6. Streaming Generation & Tool-Calling Routing Tests
    # --------------------------------------------------------------------------
    def test_route_stream_deterministic_fallback(self) -> None:
        """route_stream yields chunks and concludes with is_final=True."""
        test_router = ProviderRouter()
        with patch.object(Config, "GEMINI_API_KEY", ""), patch.object(Config, "GROQ_API_KEY", ""):
            chunks = list(test_router.route_stream(message="Stream test message"))
            self.assertGreater(len(chunks), 0)
            self.assertTrue(chunks[-1]["is_final"])
            self.assertEqual(chunks[-1]["provider"], "stage1-deterministic")

    @patch("duckduckgo_search.DDGS")
    def test_router_with_tool_calling_loop(self, mock_ddgs_cls: MagicMock) -> None:
        """ProviderRouter executes tool calling loop when provider requests tool execution."""
        mock_ddgs_instance = MagicMock()
        mock_ddgs_instance.text.return_value = [
            {
                "title": "Python 3.13 Release Notes",
                "body": "Python 3.13 includes free-threaded CPython and a new JIT.",
                "href": "https://python.org/doc/3.13",
            }
        ]
        mock_ddgs_cls.return_value = mock_ddgs_instance

        class MockToolCallingProvider(BaseProvider):
            @property
            def name(self) -> str:
                return "mock-tool-caller"

            @property
            def model_name(self) -> str:
                return "mock-tool-v1"

            @property
            def capabilities(self) -> ProviderCapabilities:
                return ProviderCapabilities(supports_tools=True, supports_streaming=True)

            def is_available(self) -> bool:
                return True

            def generate(self, message: str, **kwargs: Any) -> ProviderResponse:
                from pihu_core.tools import tool_registry
                search_res = tool_registry.execute("search_web", query="Python 3.13")
                return ProviderResponse(
                    content=f"Grounded answer with search: {search_res}",
                    provider=self.name,
                    model=self.model_name,
                )

        test_router = ProviderRouter()
        tool_p = MockToolCallingProvider()
        test_router.register_provider(tool_p)

        result = test_router.route_chat(
            message="What is in Python 3.13?",
            provider_name="mock-tool-caller",
        )
        self.assertTrue(result["ok"])
        self.assertIn("Python 3.13 includes free-threaded CPython", result["response"])
        mock_ddgs_instance.text.assert_called_once_with("Python 3.13", max_results=5)


if __name__ == "__main__":
    unittest.main()
