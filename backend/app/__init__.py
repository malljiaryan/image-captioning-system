import os
import time
import logging
from flask import Flask, request, g
from flask_cors import CORS
from app.config import Config
from app.utils.response import api_response


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def create_app():
    configure_logging()
    
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
    app = Flask(__name__, static_folder=frontend_dir, static_url_path="")
    app.config["FRONTEND_DIR"] = frontend_dir
    app.config.from_object(Config)

    # Initialize SQLAlchemy Database & Schema (3.13.5, 3.13.6)
    from app.database import init_db
    init_db(app)

    # Enable Cross-Origin Resource Sharing (CORS) for all routes (3.12.12)
    CORS(app, resources={r"/*": {"origins": "*"}})

    # Request logging & timing middleware (3.11.15)
    @app.before_request
    def start_timer():
        g.start_time = time.time()

    @app.after_request
    def apply_security_headers_and_log(response):
        # Security Hardening Headers (3.14.15)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        if request.path.startswith("/api"):
            duration_ms = round((time.time() - getattr(g, "start_time", time.time())) * 1000, 1)
            logging.getLogger("api").info(
                f"{request.method} {request.path} {response.status_code} - {duration_ms}ms"
            )
        return response

    # Global Centralized Error Handlers (3.11.11)
    @app.errorhandler(400)
    def bad_request(error):
        msg = getattr(error, "description", "Bad Request")
        return api_response(success=False, message=str(msg), error="BadRequest", status_code=400)

    @app.errorhandler(404)
    def not_found(error):
        return api_response(success=False, message="Resource not found.", error="NotFound", status_code=404)

    @app.errorhandler(405)
    def method_not_allowed(error):
        return api_response(success=False, message="HTTP method not allowed for this endpoint.", error="MethodNotAllowed", status_code=405)

    @app.errorhandler(413)
    def payload_too_large(error):
        return api_response(success=False, message="Uploaded file exceeds maximum permitted size limit.", error="PayloadTooLarge", status_code=413)

    @app.errorhandler(500)
    def internal_error(error):
        logging.getLogger("api").error(f"Internal server error: {error}", exc_info=True)
        return api_response(success=False, message="Internal server error occurred.", error="InternalServerError", status_code=500)

    @app.errorhandler(504)
    def gateway_timeout(error):
        return api_response(success=False, message="Inference request timed out.", error="GatewayTimeout", status_code=504)

    from app.routes import main
    app.register_blueprint(main)

    # Trigger model warm-up in background thread on server startup (unless in test mode)
    if not app.config.get("TESTING") and os.getenv("TESTING", "").lower() != "true":
        import threading
        from app.ai.model import warmup_model
        threading.Thread(target=warmup_model, daemon=True).start()

    return app