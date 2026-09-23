"""Database connection and abstraction layer for Pihu-BreakThough.

Stage 2 establishes secure connection management and health verification
for Neon PostgreSQL using psycopg 3.
Schema and table creation are strictly deferred to Stage 4.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import re
from typing import Any, Dict, Generator, Optional

from pihu_core.config import Config

logger = logging.getLogger(__name__)

# Safely import psycopg if installed
try:
    import psycopg

    PSYCOPG_AVAILABLE = True
except ImportError:
    psycopg = None  # type: ignore
    PSYCOPG_AVAILABLE = False


def sanitize_database_error(error_msg: str) -> str:
    """Sanitize error messages to ensure passwords and DSN strings are never leaked.

    Masks URI credentials (e.g. postgresql://user:password@host) and keyword
    credentials (e.g. password=xyz) from output strings.
    """
    if not error_msg:
        return "Unknown database error."

    # Categorize common connection failures into safe, uniform descriptions
    lower_msg = error_msg.lower()
    if "password authentication failed" in lower_msg or "authentication failed" in lower_msg:
        return "Authentication failed: invalid database credentials."
    if "could not translate host name" in lower_msg or "name or service not known" in lower_msg:
        return "Host resolution failed: could not resolve database host."
    if "connection refused" in lower_msg:
        return "Connection refused: database server is unreachable."
    if "timeout" in lower_msg or "timed out" in lower_msg:
        return "Connection timed out while reaching database server."
    if "ssl" in lower_msg and ("error" in lower_msg or "failed" in lower_msg):
        return "SSL handshake error: secure connection could not be established."

    # Mask any postgresql://user:password@host patterns
    sanitized = re.sub(
        r"://([^:@\s]+):([^@\s]+)@",
        r"://\1:***@",
        error_msg,
    )

    # Mask password=xyz or secret=xyz
    sanitized = re.sub(
        r"(password|pwd|secret)=([^\s;,]+)",
        r"\1=***",
        sanitized,
        flags=re.IGNORECASE,
    )

    # Strip any potential path or socket details
    sanitized = sanitized.replace("\n", " ").strip()
    if len(sanitized) > 150:
        sanitized = sanitized[:147] + "..."

    return sanitized


class DatabaseManager:
    """Manages Neon PostgreSQL connection lifecycle and health verification."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self._custom_database_url = database_url

    @property
    def database_url(self) -> Optional[str]:
        """Dynamically resolve the database URL."""
        if self._custom_database_url is not None:
            return self._custom_database_url
        return Config.DATABASE_URL

    def is_configured(self) -> bool:
        """Check whether a database URL is present in the configuration."""
        url = self.database_url
        return bool(url and url.strip())

    def create_connection(self, timeout: int = 5) -> Any:
        """Create and return a raw psycopg connection.

        Args:
            timeout: Maximum seconds to wait for connection.

        Raises:
            RuntimeError: If psycopg is unavailable or DATABASE_URL is unconfigured.
            Exception: If psycopg.connect fails.
        """
        if not PSYCOPG_AVAILABLE:
            raise RuntimeError("psycopg driver is not installed.")

        if not self.is_configured():
            raise RuntimeError("DATABASE_URL is not configured.")

        # psycopg.connect accepts connect_timeout
        return psycopg.connect(self.database_url, connect_timeout=timeout)

    @contextmanager
    def get_connection(self, timeout: int = 5) -> Generator[Any, None, None]:
        """Context manager providing a safe database connection with auto-commit and cleanup.

        Usage:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("...")

        Yields:
            psycopg.Connection instance.
        """
        conn = self.create_connection(timeout=timeout)
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def check_connection(self, timeout: int = 5) -> Dict[str, Any]:
        """Perform a SELECT 1 query to verify database connectivity.

        Returns:
            Dict containing:
                status: 'NOT CONFIGURED' | 'connected' | 'error'
                connected: bool
                error: safe error string if status is 'error'
        """
        if not self.is_configured():
            return {
                "status": "NOT CONFIGURED",
                "connected": False,
            }

        if not PSYCOPG_AVAILABLE:
            return {
                "status": "error",
                "connected": False,
                "error": "psycopg driver is not installed in runtime.",
            }

        try:
            with self.get_connection(timeout=timeout) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            return {
                "status": "connected",
                "connected": True,
            }
        except Exception as exc:
            raw_msg = str(exc)
            safe_msg = sanitize_database_error(raw_msg)
            logger.warning("Database connection check failed: %s", safe_msg)
            return {
                "status": "error",
                "connected": False,
                "error": safe_msg,
            }

    def get_status(self) -> str:
        """Return the database status string for health checks.

        Returns 'NOT CONFIGURED', 'connected', or 'error'.
        """
        result = self.check_connection()
        return result["status"]

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Return structured database diagnostic information."""
        check_result = self.check_connection()
        info: Dict[str, Any] = {
            "driver": "psycopg3" if PSYCOPG_AVAILABLE else "none",
            "configured": self.is_configured(),
            "status": check_result["status"],
            "target": "Neon PostgreSQL" if self.is_configured() else "none",
        }
        if "error" in check_result:
            info["error"] = check_result["error"]
        return info

    def execute_query(
        self,
        query: str,
        params: Optional[Any] = None,
        timeout: int = 5,
    ) -> List[Dict[str, Any]]:
        """Execute a read query using strict parameterization to structurally prevent SQL injection.

        Args:
            query: The SQL query containing placeholders (%s or %(name)s). Dynamic string
                   concatenation or formatting of untrusted input is strictly prohibited.
            params: Sequence of values or dictionary mapping parameter names to values.
            timeout: Connection timeout in seconds.

        Returns:
            List of row dictionaries where keys correspond to column names.

        Raises:
            RuntimeError: If database is unconfigured or driver unavailable.
            Exception: If query execution fails.
        """
        if not self.is_configured():
            raise RuntimeError("Database is not configured.")

        with self.get_connection(timeout=timeout) as conn:
            # Use dict_row factory if psycopg 3 is available
            row_factory = getattr(getattr(psycopg, "rows", None), "dict_row", None)
            cursor_kwargs = {"row_factory": row_factory} if row_factory else {}
            with conn.cursor(**cursor_kwargs) as cur:
                cur.execute(query, params)
                records = cur.fetchall()
                if records and isinstance(records[0], dict):
                    return list(records)
                # Fallback column mapping if dict_row is unavailable
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    return [dict(zip(columns, row)) for row in records]
                return []

    def execute_statement(
        self,
        statement: str,
        params: Optional[Any] = None,
        timeout: int = 5,
    ) -> int:
        """Execute a write statement (INSERT/UPDATE/DELETE) using strict parameterization.

        Args:
            statement: The SQL statement containing placeholders (%s or %(name)s).
            params: Sequence of values or dictionary mapping parameter names to values.
            timeout: Connection timeout in seconds.

        Returns:
            Number of rows affected (rowcount).

        Raises:
            RuntimeError: If database is unconfigured or driver unavailable.
            Exception: If statement execution fails.
        """
        if not self.is_configured():
            raise RuntimeError("Database is not configured.")

        with self.get_connection(timeout=timeout) as conn:
            with conn.cursor() as cur:
                cur.execute(statement, params)
                return cur.rowcount if cur.rowcount is not None else 0

    def initialize_schema(self) -> None:
        """Placeholder for future schema migrations.

        Stage 2 explicitly avoids any table or schema creation (deferred to Stage 4).

        Raises:
            NotImplementedError: Schema and migration setup is strictly deferred to Stage 4.
        """
        raise NotImplementedError(
            "Database schema and table creation are deferred to Stage 4."
        )



# Global singleton database manager instance
db_manager = DatabaseManager()
