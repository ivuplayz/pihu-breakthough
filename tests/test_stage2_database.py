"""Unit tests for Stage 2 Neon Database Foundation in Pihu-BreakThough."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app import create_app
from pihu_core.config import Config
from pihu_core.database import (
    DatabaseManager,
    db_manager,
    sanitize_database_error,
)


class TestStage2Database(unittest.TestCase):
    """Test suite verifying Stage 2 Neon PostgreSQL connection management and health checks."""

    def setUp(self) -> None:
        """Create test client with clean configuration."""
        class TestConfig(Config):
            TESTING = True
            DEBUG = False
            DATABASE_URL = None

        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

    # --------------------------------------------------------------------------
    # 1. Credential Sanitization Unit Tests
    # --------------------------------------------------------------------------
    def test_sanitize_error_masks_uri_passwords(self) -> None:
        """Sanitizer must mask passwords in PostgreSQL URI strings."""
        raw_error = "Failed to connect to postgresql://myuser:super_secret_pwd_999@ep-sample.neon.tech:5432/neondb"
        sanitized = sanitize_database_error(raw_error)
        self.assertNotIn("super_secret_pwd_999", sanitized)
        self.assertIn("myuser:***@", sanitized)

    def test_sanitize_error_masks_keyword_passwords(self) -> None:
        """Sanitizer must mask password=xyz keyword arguments."""
        raw_error = "Connection rejected with params host=localhost password=super_hidden_token dbname=test"
        sanitized = sanitize_database_error(raw_error)
        self.assertNotIn("super_hidden_token", sanitized)
        self.assertIn("password=***", sanitized)

    def test_sanitize_error_categorizes_common_errors(self) -> None:
        """Sanitizer maps common network/auth errors to safe, uniform messages."""
        self.assertEqual(
            sanitize_database_error("FATAL: password authentication failed for user 'neondb'"),
            "Authentication failed: invalid database credentials.",
        )
        self.assertEqual(
            sanitize_database_error("could not translate host name 'ep-unknown.neon.tech' to address"),
            "Host resolution failed: could not resolve database host.",
        )
        self.assertEqual(
            sanitize_database_error("connection refused on port 5432"),
            "Connection refused: database server is unreachable.",
        )
        self.assertEqual(
            sanitize_database_error("operation timed out after 5000ms"),
            "Connection timed out while reaching database server.",
        )
        self.assertEqual(
            sanitize_database_error("SSL SYSCALL error: EOF detected"),
            "SSL handshake error: secure connection could not be established.",
        )

    # --------------------------------------------------------------------------
    # 2. DatabaseManager Connection Tests
    # --------------------------------------------------------------------------
    def test_unconfigured_database_behavior(self) -> None:
        """When DATABASE_URL is not set, DatabaseManager reports NOT CONFIGURED."""
        mgr = DatabaseManager(database_url=None)
        self.assertFalse(mgr.is_configured())

        check = mgr.check_connection()
        self.assertEqual(check["status"], "NOT CONFIGURED")
        self.assertFalse(check["connected"])
        self.assertNotIn("error", check)

        # create_connection must raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            mgr.create_connection()
        self.assertIn("not configured", str(ctx.exception).lower())

    @patch("psycopg.connect")
    def test_successful_database_connection(self, mock_connect: MagicMock) -> None:
        """Successful connection executes SELECT 1 and reports connected."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (1,)
        mock_connect.return_value = mock_conn

        mgr = DatabaseManager(database_url="postgresql://user:pass@ep-test.neon.tech/neondb")
        self.assertTrue(mgr.is_configured())

        check = mgr.check_connection()
        self.assertEqual(check["status"], "connected")
        self.assertTrue(check["connected"])
        self.assertNotIn("error", check)

        mock_cursor.execute.assert_called_with("SELECT 1;")

    @patch("psycopg.connect")
    def test_failed_database_connection_sanitizes_error(self, mock_connect: MagicMock) -> None:
        """Failed connection reports error status without leaking sensitive details."""
        sensitive_pass = "ultra_sensitive_password_xyz"
        mock_connect.side_effect = Exception(
            f"FATAL: password authentication failed for postgresql://neondb_user:{sensitive_pass}@ep-test.neon.tech/neondb"
        )

        mgr = DatabaseManager(database_url=f"postgresql://neondb_user:{sensitive_pass}@ep-test.neon.tech/neondb")
        check = mgr.check_connection()

        self.assertEqual(check["status"], "error")
        self.assertFalse(check["connected"])
        self.assertIn("error", check)
        self.assertNotIn(sensitive_pass, check["error"])

    @patch("psycopg.connect")
    def test_get_connection_context_manager_lifecycle(self, mock_connect: MagicMock) -> None:
        """get_connection commits on success and closes on exit."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = DatabaseManager(database_url="postgresql://user:pass@host/db")
        with mgr.get_connection() as conn:
            self.assertEqual(conn, mock_conn)

        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch("psycopg.connect")
    def test_get_connection_context_manager_rollback_on_error(self, mock_connect: MagicMock) -> None:
        """get_connection rolls back on exception and closes."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn

        mgr = DatabaseManager(database_url="postgresql://user:pass@host/db")
        with self.assertRaises(ValueError):
            with mgr.get_connection():
                raise ValueError("Query error")

        mock_conn.rollback.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_initialize_schema_deferred_to_stage_4(self) -> None:
        """Schema migrations must be strictly deferred to Stage 4."""
        mgr = DatabaseManager()
        with self.assertRaises(NotImplementedError) as ctx:
            mgr.initialize_schema()
        self.assertIn("Stage 4", str(ctx.exception))

    # --------------------------------------------------------------------------
    # 3. GET /api/health Dynamic Database Reporting Tests
    # --------------------------------------------------------------------------
    def test_health_endpoint_reports_not_configured(self) -> None:
        """When DATABASE_URL is not set, GET /api/health reports 'NOT CONFIGURED'."""
        with patch.object(db_manager, "check_connection", return_value={"status": "NOT CONFIGURED", "connected": False}):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["database"], "NOT CONFIGURED")
            self.assertNotIn("database_error", data)

    def test_health_endpoint_reports_connected(self) -> None:
        """When database connection succeeds, GET /api/health reports 'connected'."""
        with patch.object(db_manager, "check_connection", return_value={"status": "connected", "connected": True}):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["database"], "connected")
            self.assertNotIn("database_error", data)

    def test_health_endpoint_reports_error_with_safe_message(self) -> None:
        """When database connection fails, GET /api/health reports 'error' with safe reason."""
        safe_reason = "Connection timed out while reaching database server."
        with patch.object(
            db_manager,
            "check_connection",
            return_value={"status": "error", "connected": False, "error": safe_reason},
        ):
            response = self.client.get("/api/health")
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data["database"], "error")
            self.assertEqual(data["database_error"], safe_reason)


if __name__ == "__main__":
    unittest.main()
