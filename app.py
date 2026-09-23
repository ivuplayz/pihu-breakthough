"""Application entry point for Pihu-BreakThough backend service.

Follows the Flask application-factory pattern. Operates cleanly in Stage 1
without external databases, local PC dependencies, or remote API keys.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import Flask, jsonify

from api import api_bp
from pihu_core.config import Config

# Configure baseline logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] in %(module)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config_class: type[Config] = Config) -> Flask:
    """Factory creating and configuring the Flask application instance."""
    app = Flask(__name__)

    # Apply configuration
    app.config.from_object(config_class)
    app.config["MAX_CONTENT_LENGTH"] = config_class.MAX_CONTENT_LENGTH

    # Root service endpoint
    @app.route("/", methods=["GET"])
    def root_endpoint() -> tuple[Any, int]:
        """Root diagnostic endpoint."""
        return (
            jsonify(
                {
                    "ok": True,
                    "app": Config.APP_NAME,
                    "service": "brain",
                }
            ),
            200,
        )

    # Register API blueprint
    app.register_blueprint(api_bp)

    # Global custom error handlers (preventing any stack trace exposure)
    @app.errorhandler(400)
    def handle_bad_request(error: Exception) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "BAD_REQUEST",
                        "message": str(error) if app.debug else "Bad request.",
                    },
                }
            ),
            400,
        )

    @app.errorhandler(404)
    def handle_not_found(error: Exception) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "The requested endpoint does not exist.",
                    },
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def handle_method_not_allowed(error: Exception) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "METHOD_NOT_ALLOWED",
                        "message": "The HTTP method is not allowed for this endpoint.",
                    },
                }
            ),
            405,
        )

    @app.errorhandler(413)
    def handle_payload_too_large(error: Exception) -> tuple[Any, int]:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "PAYLOAD_TOO_LARGE",
                        "message": "Request payload exceeds maximum allowed size.",
                    },
                }
            ),
            413,
        )

    @app.errorhandler(500)
    def handle_internal_error(error: Exception) -> tuple[Any, int]:
        logger.error("Unhandled internal server error: %s", error, exc_info=True)
        return (
            jsonify(
                {
                    "ok": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An internal server error occurred.",
                    },
                }
            ),
            500,
        )

    return app


# WSGI application handle for deployment (e.g. Vercel, Gunicorn)
app = create_app()

if __name__ == "__main__":
    logger.info(
        "Starting %s (Stage 1) on port %s [debug=%s]",
        Config.APP_NAME,
        Config.PORT,
        Config.DEBUG,
    )
    app.run(
        host="0.0.0.0",
        port=Config.PORT,
        debug=Config.DEBUG,
    )
