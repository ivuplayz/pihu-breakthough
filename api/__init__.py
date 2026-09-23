"""API blueprint package for Pihu-BreakThough.

Combines health and chat routes under the '/api' prefix.
"""

from __future__ import annotations

from flask import Blueprint

from api.chat import chat_bp
from api.health import health_bp

api_bp = Blueprint("api", __name__, url_prefix="/api")
api_bp.register_blueprint(health_bp)
api_bp.register_blueprint(chat_bp)

__all__ = ["api_bp"]
