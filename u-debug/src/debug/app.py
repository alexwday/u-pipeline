"""Debug interface for the u-retriever pipeline."""

import logging
import os
from pathlib import Path

from flask import Flask
from retriever.utils.config_setup import load_config

from .api.queries import queries_bp
from .api.retrieval import retrieval_bp
from .api.traces import traces_bp

_STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Flask application with all blueprints registered
    """
    app = Flask(
        __name__,
        static_folder=str(_STATIC_DIR),
        static_url_path="/static",
    )
    app.register_blueprint(queries_bp, url_prefix="/api")
    app.register_blueprint(retrieval_bp, url_prefix="/api")
    app.register_blueprint(traces_bp, url_prefix="/api")

    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    return app


def main() -> None:
    """Run the debug server."""
    load_config()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    app = create_app()
    port = int(os.environ.get("DEBUG_PORT", "5001"))
    logger.info("Debug interface starting at http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
