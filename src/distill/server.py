"""Minimal Flask server for the distill browser extension."""

from __future__ import annotations

from flask import Flask, jsonify, request

from distill import __version__
from distill.extractors import extract_from_html
from distill.pipeline import Pipeline


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "version": __version__})

    @app.route("/score", methods=["POST", "OPTIONS"])
    def score():
        if request.method == "OPTIONS":
            return "", 204

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Request body must be JSON"}), 400

        url = data.get("url", "")
        metadata = {"url": url} if url else None

        # Accept either raw text or HTML
        if "text" in data:
            text = data["text"]
        elif "html" in data:
            extracted = extract_from_html(data["html"], url=url)
            text = extracted["text"]
        else:
            return jsonify({"error": "Provide 'text' or 'html' in request body"}), 400

        if not text or not text.strip():
            return jsonify({"error": "Empty content"}), 400

        pipeline = Pipeline()
        report = pipeline.score(text, metadata=metadata)
        return jsonify(report.to_dict())

    return app
