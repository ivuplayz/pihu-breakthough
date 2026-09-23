"""Chat endpoint for Pihu-BreakThough API with strict input validation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from werkzeug.exceptions import HTTPException

from pihu_core.router import router

logger = logging.getLogger(__name__)

chat_bp = Blueprint("chat", __name__)

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_ITEMS = 50
VALID_ROLES = {"user", "assistant", "system"}


def _error_response(code: str, message: str, status_code: int = 400) -> tuple[Any, int]:
    """Helper to return consistent, structured error JSON without stack traces."""
    return (
        jsonify(
            {
                "ok": False,
                "error": {
                    "code": code,
                    "message": message,
                },
            }
        ),
        status_code,
    )


@chat_bp.route("/chat", methods=["POST"])
def post_chat() -> tuple[Any, int]:
    """Process chat interactions with comprehensive validation.

    Accepts:
        {
            "message": "User query string",
            "history": [{"role": "user", "content": "..."}, ...],   (optional)
            "provider": "stage1-deterministic"                     (optional)
        }

    Returns:
        JSON response with deterministic response and future-compatible metadata.
    """
    # 1. Content-Type and JSON validity verification
    if not request.is_json:
        return _error_response(
            code="INVALID_CONTENT_TYPE",
            message="Content-Type must be 'application/json'.",
            status_code=400,
        )

    try:
        data = request.get_json(silent=False)
    except HTTPException as http_err:
        # Re-raise payload too large or server errors for global handling
        if http_err.code == 413 or (http_err.code and http_err.code >= 500):
            raise
        return _error_response(
            code="INVALID_JSON",
            message="Request body could not be parsed as valid JSON.",
            status_code=400,
        )
    except Exception:
        return _error_response(
            code="INVALID_JSON",
            message="Request body could not be parsed as valid JSON.",
            status_code=400,
        )



    if data is None or not isinstance(data, dict):
        return _error_response(
            code="INVALID_JSON",
            message="Request body must be a JSON object.",
            status_code=400,
        )

    # 2. 'message' field presence and type validation
    if "message" not in data:
        return _error_response(
            code="MISSING_MESSAGE",
            message="The 'message' field is required.",
            status_code=400,
        )

    message = data["message"]
    if not isinstance(message, str):
        return _error_response(
            code="INVALID_MESSAGE_TYPE",
            message="The 'message' field must be a string.",
            status_code=400,
        )

    # 3. 'message' content length and non-emptiness validation
    clean_message = message.strip()
    if not clean_message:
        return _error_response(
            code="EMPTY_MESSAGE",
            message="The 'message' field cannot be empty or contain only whitespace.",
            status_code=400,
        )

    if len(message) > MAX_MESSAGE_LENGTH:
        return _error_response(
            code="MESSAGE_TOO_LONG",
            message=f"Message exceeds maximum allowed length of {MAX_MESSAGE_LENGTH} characters.",
            status_code=400,
        )

    # 4. 'history' field validation (optional)
    history: Optional[List[Dict[str, str]]] = None
    if "history" in data and data["history"] is not None:
        raw_history = data["history"]
        if not isinstance(raw_history, list):
            return _error_response(
                code="INVALID_HISTORY",
                message="The 'history' field must be an array of message objects.",
                status_code=400,
            )

        if len(raw_history) > MAX_HISTORY_ITEMS:
            return _error_response(
                code="HISTORY_TOO_LONG",
                message=f"The 'history' field exceeds maximum allowed length of {MAX_HISTORY_ITEMS} items.",
                status_code=400,
            )

        sanitized_history: List[Dict[str, str]] = []
        for index, item in enumerate(raw_history):
            if not isinstance(item, dict):
                return _error_response(
                    code="INVALID_HISTORY_ITEM",
                    message=f"History item at index {index} must be an object with 'role' and 'content'.",
                    status_code=400,
                )

            role = item.get("role")
            content = item.get("content")

            if not isinstance(role, str) or role.lower() not in VALID_ROLES:
                return _error_response(
                    code="INVALID_HISTORY_ROLE",
                    message=f"History item at index {index} has invalid 'role'. Allowed values: {sorted(list(VALID_ROLES))}.",
                    status_code=400,
                )

            if not isinstance(content, str):
                return _error_response(
                    code="INVALID_HISTORY_CONTENT",
                    message=f"History item at index {index} must have a string 'content' field.",
                    status_code=400,
                )

            sanitized_history.append({"role": role.lower(), "content": content})

        history = sanitized_history

    # 5. Optional provider hint validation
    provider_name: Optional[str] = None
    if "provider" in data and data["provider"] is not None:
        if not isinstance(data["provider"], str):
            return _error_response(
                code="INVALID_PROVIDER_TYPE",
                message="The 'provider' field must be a string.",
                status_code=400,
            )
        provider_name = data["provider"].strip()

    # 6. Execute routed chat logic safely
    try:
        result = router.route_chat(
            message=clean_message,
            history=history,
            provider_name=provider_name,
        )
        return jsonify(result), 200
    except Exception as exc:
        logger.exception("Error processing chat message: %s", exc)
        return _error_response(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred while processing the chat request.",
            status_code=500,
        )
