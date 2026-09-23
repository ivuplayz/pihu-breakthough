"""Data repository layer for Pihu-BreakThough.

Provides clean, parameterized SQL access to users, conversations, messages,
and long-term memories using psycopg 3.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import uuid

from pihu_core.database import DatabaseManager, db_manager

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository managing persistence for users, conversations, and chat messages."""

    def __init__(self, database_manager: Optional[DatabaseManager] = None) -> None:
        self.db = database_manager if database_manager is not None else db_manager

    def is_available(self) -> bool:
        """Check whether the underlying database is configured and available."""
        return self.db.is_configured()

    def get_or_create_default_user(self) -> Dict[str, Any]:
        """Fetch the primary default user, creating one if no users exist.

        Returns:
            Dictionary with user id, created_at, and preferences.

        Raises:
            RuntimeError: If database is unconfigured.
        """
        if not self.is_available():
            raise RuntimeError("Database is unconfigured. Cannot query users.")

        rows = self.db.execute_query(
            "SELECT id, created_at, preferences FROM users ORDER BY created_at ASC LIMIT 1;"
        )
        if rows:
            user = rows[0]
            if isinstance(user.get("id"), uuid.UUID):
                user["id"] = str(user["id"])
            return user

        # Create default user
        created_rows = self.db.execute_query(
            "INSERT INTO users (preferences) VALUES (%s) RETURNING id, created_at, preferences;",
            (json.dumps({"name": "Default User", "theme": "system"}),),
        )
        user = created_rows[0]
        if isinstance(user.get("id"), uuid.UUID):
            user["id"] = str(user["id"])
        return user

    def create_user(self, preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new user record with custom preferences.

        Args:
            preferences: Optional dictionary of user preferences.

        Returns:
            Dictionary with new user record.
        """
        if not self.is_available():
            raise RuntimeError("Database is unconfigured. Cannot create user.")

        prefs_json = json.dumps(preferences or {})
        rows = self.db.execute_query(
            "INSERT INTO users (preferences) VALUES (%s) RETURNING id, created_at, preferences;",
            (prefs_json,),
        )
        user = rows[0]
        if isinstance(user.get("id"), uuid.UUID):
            user["id"] = str(user["id"])
        return user

    def create_conversation(
        self,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation session.

        Args:
            user_id: Optional UUID string of the owner user. Defaults to primary user.
            title: Optional conversation title.

        Returns:
            Dictionary with conversation record.
        """
        if not self.is_available():
            raise RuntimeError("Database is unconfigured. Cannot create conversation.")

        if not user_id:
            user = self.get_or_create_default_user()
            user_id = user["id"]

        conv_title = title or "New Conversation"
        rows = self.db.execute_query(
            "INSERT INTO conversations (user_id, title) VALUES (%s, %s) RETURNING id, user_id, title, created_at;",
            (user_id, conv_title),
        )
        conv = rows[0]
        for key in ("id", "user_id"):
            if isinstance(conv.get(key), uuid.UUID):
                conv[key] = str(conv[key])
        return conv

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a conversation by its ID.

        Args:
            conversation_id: UUID string of conversation.

        Returns:
            Conversation dictionary or None if not found.
        """
        if not self.is_available():
            raise RuntimeError("Database is unconfigured. Cannot query conversation.")

        rows = self.db.execute_query(
            "SELECT id, user_id, title, created_at FROM conversations WHERE id = %s;",
            (conversation_id,),
        )
        if not rows:
            return None
        conv = rows[0]
        for key in ("id", "user_id"):
            if isinstance(conv.get(key), uuid.UUID):
                conv[key] = str(conv[key])
        return conv

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        provider_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Save a chat message associated with a conversation.

        Args:
            conversation_id: UUID string of the conversation.
            role: 'user', 'assistant', or 'system'.
            content: Text content of the message.
            provider_used: Optional name of the model provider used.

        Returns:
            Saved message dictionary.
        """
        if not self.is_available():
            raise RuntimeError("Database is unconfigured. Cannot save message.")

        rows = self.db.execute_query(
            """
            INSERT INTO messages (conversation_id, role, content, provider_used)
            VALUES (%s, %s, %s, %s)
            RETURNING id, conversation_id, role, content, provider_used, timestamp;
            """,
            (conversation_id, role, content, provider_used),
        )
        msg = rows[0]
        for key in ("id", "conversation_id"):
            if isinstance(msg.get(key), uuid.UUID):
                msg[key] = str(msg[key])
        return msg

    def get_messages(
        self,
        conversation_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve message history for a conversation in chronological order.

        Args:
            conversation_id: UUID string of the conversation.
            limit: Maximum number of recent messages to return.

        Returns:
            List of message dictionaries ordered chronologically.
        """
        if not self.is_available():
            raise RuntimeError("Database is unconfigured. Cannot query messages.")

        rows = self.db.execute_query(
            """
            SELECT id, conversation_id, role, content, provider_used, timestamp
            FROM messages
            WHERE conversation_id = %s
            ORDER BY timestamp ASC
            LIMIT %s;
            """,
            (conversation_id, limit),
        )
        for msg in rows:
            for key in ("id", "conversation_id"):
                if isinstance(msg.get(key), uuid.UUID):
                    msg[key] = str(msg[key])
        return rows


# Global singleton repository instance
conversation_repo = ConversationRepository()
