"""Tests for the distill Flask server."""

from __future__ import annotations

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed (pip install distill-score[server])")

from distill.server import create_app  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_cors_headers(self, client):
        resp = client.get("/health")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


class TestScore:
    def test_score_with_text(self, client):
        resp = client.post("/score", json={
            "text": (
                "We migrated our PostgreSQL cluster from 14 to 16. The process took "
                "3 weeks across our 12-node setup. Latency improved by approximately 18%."
            ),
            "url": "https://example.com/article",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "overall_score" in data
        assert "grade" in data
        assert "dimensions" in data

    def test_score_with_html(self, client):
        resp = client.post("/score", json={
            "html": (
                "<html><body><p>We migrated our PostgreSQL cluster from 14 to 16. "
                "The process took 3 weeks across our 12-node setup. Latency improved "
                "by approximately 18% on our analytical queries.</p></body></html>"
            ),
            "url": "https://example.com/article",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "overall_score" in data

    def test_score_missing_fields(self, client):
        resp = client.post("/score", json={"url": "https://example.com"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "error" in data

    def test_score_empty_text(self, client):
        resp = client.post("/score", json={"text": "   "})
        assert resp.status_code == 400

    def test_score_no_json(self, client):
        resp = client.post("/score", data="not json",
                           content_type="text/plain")
        assert resp.status_code == 400

    def test_score_cors_headers(self, client):
        resp = client.post("/score", json={"text": "Some content to score here."})
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")

    def test_score_options_preflight(self, client):
        resp = client.options("/score")
        assert resp.status_code == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
