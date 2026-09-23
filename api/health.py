"""Health check endpoint for Pihu-BreakThough API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from flask import Blueprint, jsonify

from pihu_core.config import Config
from pihu_core.database import db_manager
from pihu_core.router import router

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def get_health() -> tuple[Any, int]:
    """Return structured system and component health status.

    Returns:
        JSON response with system health, database status, and active provider info.
    """
    db_status = db_manager.get_status()

    payload: Dict[str, Any] = {
        "status": "healthy",
        "app": Config.APP_NAME,
        "stage": Config.STAGE,
        "database": db_status,
        "router": "active",
        "active_provider": router.get_active_provider().name,
        "available_providers": router.list_available_providers(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return jsonify(payload), 200
