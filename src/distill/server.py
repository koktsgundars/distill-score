"""Minimal Flask server for the distill browser extension."""

from __future__ import annotations

from flask import Flask, jsonify, request

from distill import __version__
from distill.extractors import extract_from_html
from distill.pipeline import Pipeline
from distill.profiles import list_profiles as _list_profiles
from distill.scorer import list_scorers as _list_scorers


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

        # Pipeline options
        profile = data.get("profile")
        auto_profile = bool(data.get("auto_profile", False))
        highlights = bool(data.get("highlights", False))
        paragraphs = bool(data.get("paragraphs", False))

        scorers_opt = data.get("scorers")
        scorer_names = None
        if scorers_opt:
            if isinstance(scorers_opt, str):
                scorer_names = [s.strip() for s in scorers_opt.split(",")]
            elif isinstance(scorers_opt, list):
                scorer_names = scorers_opt

        # Validate profile name
        if profile:
            from distill.profiles import get_profile
            try:
                get_profile(profile)
            except KeyError as e:
                return jsonify({"error": str(e)}), 400

        try:
            pipeline = Pipeline(
                scorers=scorer_names, profile=profile, auto_profile=auto_profile,
            )
        except KeyError as e:
            return jsonify({"error": str(e)}), 400

        report = pipeline.score(text, metadata=metadata, include_paragraphs=paragraphs)
        result = report.to_dict(include_highlights=highlights)

        # Include auto-detection info
        if auto_profile and pipeline.detected_content_type:
            ct = pipeline.detected_content_type
            result["detected_type"] = ct.name
            result["detected_confidence"] = round(ct.confidence, 3)

        return jsonify(result)

    @app.route("/scorers", methods=["GET"])
    def scorers():
        return jsonify(_list_scorers())

    @app.route("/profiles", methods=["GET"])
    def profiles():
        return jsonify(_list_profiles())

    return app
