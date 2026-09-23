"""Database connection and abstraction layer for Pihu-BreakThough.

Provides a clean foundation for Neon PostgreSQL (via psycopg 3).
Stage 1 does NOT require a database connection, does NOT create tables,
and gracefully reports 'NOT CONFIGURED' if DATABASE_URL is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from pihu_core.config import Config

logger = logging.getLogger(__name__)

# Safely import psycopg if installed
try:
    import psycopg

    PSYCOPG_AVAILABLE = True
except ImportError:
    psycopg = None  # type: ignore
    PSYCOPG_AVAILABLE = False


class DatabaseManager:
    """Manages database connection checks and future queries for Pihu-BreakThough."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url if database_url is not None else Config.DATABASE_URL

    def is_configured(self) -> bool:
        """Check whether a database URL is provided."""
        return bool(self.database_url and self.database_url.strip())

    def get_status(self) -> str:
        """Return the database status string for health checks.

        Returns 'NOT CONFIGURED' when no URL is provided, or connection state.
        """
        if not self.is_configured():
            return "NOT CONFIGURED"

        if not PSYCOPG_AVAILABLE:
            return "DRIVER_UNAVAILABLE"

        # Safe diagnostic ping when URL is explicitly provided
        return self._ping()

    def _ping(self) -> str:
        """Perform a quick ping to test database connectivity."""
        if not self.database_url:
            return "NOT CONFIGURED"

        try:
            # Set short connect_timeout to avoid hanging during health checks
            with psycopg.connect(self.database_url, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return "CONNECTED"
        except Exception as e:
            logger.warning("Database ping failed: %s", str(e))
            return "CONNECTION_FAILED"

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Return structured database diagnostic information."""
        configured = self.is_configured()
        return {
            "driver": "psycopg3" if PSYCOPG_AVAILABLE else "none",
            "configured": configured,
            "status": self.get_status(),
            "target": "Neon PostgreSQL" if configured else "none",
        }

    def initialize_schema(self) -> None:
        """Placeholder for future schema migrations.

        Stage 1 explicitly avoids any schema creation.
        """
        raise NotImplementedError(
            "Schema initialization and migrations are deferred to Stage 2."
        )


# Global singleton database manager instance
db_manager = DatabaseManager()
