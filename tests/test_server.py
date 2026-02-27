"""Tests for the distill Flask server."""

from __future__ import annotations

import pytest

flask = pytest.importorskip("flask", reason="Flask not installed (pip install distill-score[server])")

from distill.server import create_app  # noqa: E402


SAMPLE_TEXT = (
    "We migrated our PostgreSQL cluster from 14 to 16. The process took "
    "3 weeks across our 12-node setup. Latency improved by approximately 18%."
)


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
            "text": SAMPLE_TEXT,
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

    def test_score_with_profile(self, client):
        resp = client.post("/score", json={
            "text": SAMPLE_TEXT,
            "profile": "technical",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "overall_score" in data

    def test_score_with_invalid_profile(self, client):
        resp = client.post("/score", json={
            "text": SAMPLE_TEXT,
            "profile": "nonexistent",
        })
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_score_with_auto_profile(self, client):
        resp = client.post("/score", json={
            "text": SAMPLE_TEXT,
            "auto_profile": True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert "overall_score" in data
        # Technical text should be auto-detected
        if "detected_type" in data:
            assert "detected_confidence" in data

    def test_score_with_scorers_string(self, client):
        resp = client.post("/score", json={
            "text": SAMPLE_TEXT,
            "scorers": "substance,epistemic",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        dims = data["dimensions"]
        assert "substance" in dims
        assert "epistemic" in dims
        assert "readability" not in dims

    def test_score_with_scorers_list(self, client):
        resp = client.post("/score", json={
            "text": SAMPLE_TEXT,
            "scorers": ["substance"],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["dimensions"]) == 1
        assert "substance" in data["dimensions"]

    def test_score_with_highlights(self, client):
        resp = client.post("/score", json={
            "text": SAMPLE_TEXT,
            "highlights": True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # At least some dimension should have highlights
        has_highlights = any(
            "highlights" in dim for dim in data["dimensions"].values()
        )
        assert has_highlights

    def test_score_with_paragraphs(self, client):
        long_text = (SAMPLE_TEXT + "\n\n") * 5
        resp = client.post("/score", json={
            "text": long_text,
            "paragraphs": True,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        # May or may not have paragraphs depending on word count threshold
        assert "overall_score" in data


class TestDiscovery:
    def test_scorers_endpoint(self, client):
        resp = client.get("/scorers")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "substance" in data
        assert "epistemic" in data

    def test_profiles_endpoint(self, client):
        resp = client.get("/profiles")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, dict)
        assert "default" in data
        assert "technical" in data
        assert "news" in data
        assert "opinion" in data
