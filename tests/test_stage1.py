"""Unit and integration test suite for Pihu-BreakThough Stage 1."""

from __future__ import annotations

import json
import unittest

from app import create_app
from pihu_core.config import Config
from pihu_core.database import DatabaseManager, db_manager
from pihu_core.providers import (
    GeminiProvider,
    GroqProvider,
    HuggingFaceProvider,
    OllamaProvider,
    Stage1DeterministicProvider,
)
from pihu_core.router import ProviderRouter


class TestStage1Core(unittest.TestCase):
    """Test suite verifying Stage 1 core backend architecture."""

    def setUp(self) -> None:
        """Initialize test client with isolated test settings."""

        class TestConfig(Config):
            TESTING = True
            DEBUG = False
            DATABASE_URL = None
            GEMINI_API_KEY = None
            GROQ_API_KEY = None
            HF_API_KEY = None

        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

    # --------------------------------------------------------------------------
    # 1. Root & Diagnostic Endpoints
    # --------------------------------------------------------------------------
    def test_root_endpoint(self) -> None:
        """GET / must return structured diagnostic information."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("app"), "Pihu-BreakThough")
        self.assertEqual(data.get("service"), "brain")

    def test_health_endpoint_without_database(self) -> None:
        """GET /api/health must report 'NOT CONFIGURED' for database without throwing."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("app"), "Pihu-BreakThough")
        self.assertEqual(data.get("stage"), Config.STAGE)
        self.assertEqual(data.get("database"), "NOT CONFIGURED")
        self.assertEqual(data.get("router"), "active")
        self.assertEqual(data.get("active_provider"), "stage1-deterministic")
        self.assertIn("stage1-deterministic", data.get("available_providers", []))
        self.assertIn("timestamp", data)

    # --------------------------------------------------------------------------
    # 2. Chat Endpoint - Happy Path
    # --------------------------------------------------------------------------
    def test_chat_valid_single_message(self) -> None:
        """POST /api/chat with valid payload must return 200 and deterministic response."""
        payload = {"message": "Hello, Pihu-BreakThough!"}
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("app"), "Pihu-BreakThough")
        self.assertIn("Pihu-BreakThough Stage 1 Core is operational", data.get("response", ""))
        self.assertEqual(data.get("provider"), "stage1-deterministic")
        self.assertEqual(data.get("model"), "core-deterministic-v1")
        metadata = data.get("metadata", {})
        self.assertEqual(metadata.get("history_length"), 0)
        self.assertEqual(metadata.get("received_chars"), len(payload["message"]))
        self.assertEqual(metadata.get("stage"), Config.STAGE)

    def test_chat_valid_with_history(self) -> None:
        """POST /api/chat with conversation history must accurately reflect context."""
        payload = {
            "message": "Continue the discussion",
            "history": [
                {"role": "user", "content": "Initial question"},
                {"role": "assistant", "content": "Initial reply"},
            ],
        }
        response = self.client.post(
            "/api/chat",
            data=json.dumps(payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data.get("ok"))
        metadata = data.get("metadata", {})
        self.assertEqual(metadata.get("history_length"), 2)

    # --------------------------------------------------------------------------
    # 3. Chat Endpoint - Strict Validation & Error Handling
    # --------------------------------------------------------------------------
    def test_chat_missing_content_type(self) -> None:
        """Missing or wrong Content-Type must return 400 INVALID_CONTENT_TYPE."""
        response = self.client.post(
            "/api/chat",
            data="message=hello",
            content_type="application/x-www-form-urlencoded",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_CONTENT_TYPE")

    def test_chat_malformed_json(self) -> None:
        """Malformed JSON string must return 400 INVALID_JSON."""
        response = self.client.post(
            "/api/chat",
            data="{'invalid_json': true,",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_JSON")

    def test_chat_non_object_root_json(self) -> None:
        """JSON payload that is a list instead of object must return 400 INVALID_JSON."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps(["message", "test"]),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_JSON")

    def test_chat_missing_message_field(self) -> None:
        """Payload missing 'message' key must return 400 MISSING_MESSAGE."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps({"wrong_key": "some content"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "MISSING_MESSAGE")

    def test_chat_non_string_message(self) -> None:
        """Payload with non-string 'message' must return 400 INVALID_MESSAGE_TYPE."""
        for invalid_val in [12345, True, {"text": "hello"}, [1, 2, 3]]:
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": invalid_val}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertFalse(data["ok"])
            self.assertEqual(data["error"]["code"], "INVALID_MESSAGE_TYPE")

    def test_chat_empty_or_whitespace_message(self) -> None:
        """Empty or whitespace-only 'message' must return 400 EMPTY_MESSAGE."""
        for empty_val in ["", "   ", "\n\t   \n"]:
            response = self.client.post(
                "/api/chat",
                data=json.dumps({"message": empty_val}),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 400)
            data = response.get_json()
            self.assertFalse(data["ok"])
            self.assertEqual(data["error"]["code"], "EMPTY_MESSAGE")

    def test_chat_oversized_message(self) -> None:
        """Message exceeding 4000 characters must return 400 MESSAGE_TOO_LONG."""
        oversized = "A" * 4001
        response = self.client.post(
            "/api/chat",
            data=json.dumps({"message": oversized}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "MESSAGE_TOO_LONG")

    def test_chat_invalid_history_type(self) -> None:
        """'history' provided as non-list must return 400 INVALID_HISTORY."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps({"message": "test", "history": "not a list"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_HISTORY")

    def test_chat_history_too_long(self) -> None:
        """'history' exceeding 50 items must return 400 HISTORY_TOO_LONG."""
        long_history = [{"role": "user", "content": f"msg {i}"} for i in range(51)]
        response = self.client.post(
            "/api/chat",
            data=json.dumps({"message": "test", "history": long_history}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "HISTORY_TOO_LONG")

    def test_chat_invalid_history_item_structure(self) -> None:
        """History item that is not a dict must return 400 INVALID_HISTORY_ITEM."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps({"message": "test", "history": ["string_item"]}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_HISTORY_ITEM")

    def test_chat_invalid_history_role(self) -> None:
        """History item with invalid role must return 400 INVALID_HISTORY_ROLE."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps(
                {
                    "message": "test",
                    "history": [{"role": "moderator", "content": "not allowed"}],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_HISTORY_ROLE")

    def test_chat_invalid_history_content(self) -> None:
        """History item with non-string content must return 400 INVALID_HISTORY_CONTENT."""
        response = self.client.post(
            "/api/chat",
            data=json.dumps(
                {
                    "message": "test",
                    "history": [{"role": "user", "content": 12345}],
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INVALID_HISTORY_CONTENT")

    # --------------------------------------------------------------------------
    # 4. Global HTTP Error Handlers
    # --------------------------------------------------------------------------
    def test_404_not_found(self) -> None:
        """Request to unknown route returns structured 404 JSON."""
        response = self.client.get("/non_existent_route")
        self.assertEqual(response.status_code, 404)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "NOT_FOUND")

    def test_405_method_not_allowed(self) -> None:
        """Using wrong HTTP method returns structured 405 JSON."""
        response = self.client.get("/api/chat")
        self.assertEqual(response.status_code, 405)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "METHOD_NOT_ALLOWED")

    # --------------------------------------------------------------------------
    # 5. Modular Component Unit Tests
    # --------------------------------------------------------------------------
    def test_database_manager_unconfigured(self) -> None:
        """DatabaseManager safely reports NOT CONFIGURED and avoids connection calls."""
        mgr = DatabaseManager(database_url=None)
        self.assertFalse(mgr.is_configured())
        self.assertEqual(mgr.get_status(), "NOT CONFIGURED")
        info = mgr.get_diagnostic_info()
        self.assertEqual(info["configured"], False)
        self.assertEqual(info["status"], "NOT CONFIGURED")

        # Unconfigured schema initialization raises RuntimeError
        with self.assertRaises(RuntimeError):
            mgr.initialize_schema()

    def test_provider_router_and_fallbacks(self) -> None:
        """ProviderRouter routes to Stage 1 deterministic provider and handles fallbacks."""
        router = ProviderRouter()
        active = router.get_active_provider()
        self.assertIsInstance(active, Stage1DeterministicProvider)

        # Gemini is not configured in Stage 1, requesting it should fall back cleanly
        routed = router.route_chat(
            message="Test prompt",
            provider_name="gemini",
        )
        self.assertTrue(routed["ok"])
        self.assertEqual(routed["provider"], "stage1-deterministic")

    def test_413_payload_too_large(self) -> None:
        """Exceeding MAX_CONTENT_LENGTH triggers structured 413 error."""
        # Create an app instance with a small MAX_CONTENT_LENGTH
        class SmallLimitConfig(Config):
            TESTING = True
            MAX_CONTENT_LENGTH = 100

        small_app = create_app(SmallLimitConfig)
        small_client = small_app.test_client()

        large_payload = {"message": "X" * 200}
        response = small_client.post(
            "/api/chat",
            data=json.dumps(large_payload),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 413)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "PAYLOAD_TOO_LARGE")

    def test_500_internal_error_no_stack_trace(self) -> None:
        """Internal server error handler returns clean JSON without exposing stack trace."""
        class ProdConfig(Config):
            TESTING = False
            DEBUG = False

        prod_app = create_app(ProdConfig)
        prod_client = prod_app.test_client()

        @prod_app.route("/force_error")
        def force_error() -> None:
            raise RuntimeError("Secret internal failure detail")

        response = prod_client.get("/force_error")
        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], "INTERNAL_SERVER_ERROR")
        # Ensure raw exception message or python traceback is not in JSON
        self.assertNotIn("Secret internal failure detail", str(data))
    def test_empty_port_and_env_variables(self) -> None:
        """Config must not crash if PORT or MAX_CONTENT_LENGTH is empty string (e.g. on Vercel)."""
        import os
        from unittest.mock import patch
        from pihu_core.config import _safe_int

        self.assertEqual(_safe_int("", 5000), 5000)
        self.assertEqual(_safe_int("   ", 5000), 5000)
        self.assertEqual(_safe_int(None, 5000), 5000)
        self.assertEqual(_safe_int("8080", 5000), 8080)
        self.assertEqual(_safe_int("invalid", 5000), 5000)

        with patch.dict(os.environ, {"PORT": "", "MAX_CONTENT_LENGTH": ""}):
            # Reload / check Config class attributes when PORT is empty string
            port = _safe_int(os.environ.get("PORT"), 5000)
            self.assertEqual(port, 5000)
            max_len = _safe_int(os.environ.get("MAX_CONTENT_LENGTH"), 1048576)
            self.assertEqual(max_len, 1048576)


if __name__ == "__main__":

    unittest.main()
