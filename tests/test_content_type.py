"""Tests for content type auto-detection."""

import json

from click.testing import CliRunner

from distill.content_type import ContentType, detect_content_type
from distill.pipeline import Pipeline


# --- Detection tests ---

TECHNICAL_TEXT = """
We deployed v2.3 of the API gateway and latency dropped from p99 340ms to 95ms.
The migration required updating our Kubernetes manifests and rebuilding the Docker
images. We tested with `locust` and found that throughput(req/s) improved by 40%.
The connection pool was sized at 200 concurrent connections. Our CI/CD pipeline
runs postgres integration tests before each deploy.
"""

NEWS_TEXT = """
By Sarah Johnson

According to sources familiar with the matter, the company announced on Tuesday
that it would be restructuring its operations. The spokesperson said that
"we are committed to long-term growth." Reuters reported that the deal, valued
at $2.3 billion, is expected to close by December 15. He added that the board
had unanimously approved the transaction.
"""

OPINION_TEXT = """
I think the current approach to software development is fundamentally broken.
In my experience, most teams spend too much time on ceremony and too little on
actual problem-solving. Personally, I'd argue that we should focus more on
outcomes than processes. That said, there are legitimate reasons why some
structure is needed. However, the problem with most frameworks is that they
optimize for the wrong things. Here's the thing: what most people don't realize
is that simplicity requires more effort than complexity.
"""

GENERIC_TEXT = """
Welcome to our website. We offer a variety of products and services.
Please browse our catalog and contact us if you have any questions.
Our team is here to help you find what you need. Thank you for visiting.
"""


def test_detect_technical():
    result = detect_content_type(TECHNICAL_TEXT)
    assert result.name == "technical"
    assert result.confidence > 0


def test_detect_news():
    result = detect_content_type(NEWS_TEXT)
    assert result.name == "news"
    assert result.confidence > 0


def test_detect_opinion():
    result = detect_content_type(OPINION_TEXT)
    assert result.name == "opinion"
    assert result.confidence > 0


def test_detect_generic_falls_back_to_default():
    result = detect_content_type(GENERIC_TEXT)
    assert result.name == "default"


def test_empty_text():
    result = detect_content_type("")
    assert result.name == "default"
    assert result.confidence == 0.0


def test_whitespace_only():
    result = detect_content_type("   \n\n  ")
    assert result.name == "default"
    assert result.confidence == 0.0


def test_content_type_dataclass():
    ct = ContentType(name="technical", confidence=0.5, signals={"code": 3})
    assert ct.name == "technical"
    assert ct.confidence == 0.5
    assert ct.signals == {"code": 3}


# --- Pipeline integration ---

def test_auto_profile_changes_weights():
    """auto_profile=True should apply profile weights based on detected type."""
    pipeline_default = Pipeline()
    pipeline_auto = Pipeline(auto_profile=True)

    pipeline_default.score(TECHNICAL_TEXT)
    pipeline_auto.score(TECHNICAL_TEXT)

    # Auto-profile should detect "technical" and apply different weights
    assert pipeline_auto.detected_content_type is not None
    assert pipeline_auto.detected_content_type.name == "technical"
    # Scores may differ due to different weights
    # (they could coincidentally be equal, so just check the type was detected)


def test_auto_profile_disabled_by_default():
    """Pipeline without auto_profile should not detect content type."""
    pipeline = Pipeline()
    pipeline.score(TECHNICAL_TEXT)
    assert pipeline.detected_content_type is None


def test_explicit_profile_overrides_auto():
    """When profile is explicit, auto_profile should be ignored."""
    pipeline = Pipeline(profile="news", auto_profile=True)
    pipeline.score(TECHNICAL_TEXT)
    # auto_profile is disabled when explicit profile is set
    assert pipeline.detected_content_type is None


# --- CLI integration ---

def test_cli_auto_profile_flag(tmp_path):
    """--auto-profile flag should work on the score command."""
    from distill.cli import main

    f = tmp_path / "tech.txt"
    f.write_text(TECHNICAL_TEXT)

    runner = CliRunner()
    result = runner.invoke(main, ["score", "--auto-profile", str(f)])
    assert result.exit_code == 0
    assert "Auto-detected:" in result.output


def test_cli_auto_profile_json(tmp_path):
    """--auto-profile with --json should include detected_type in output."""
    from distill.cli import main

    f = tmp_path / "tech.txt"
    f.write_text(TECHNICAL_TEXT)

    runner = CliRunner()
    result = runner.invoke(main, ["score", "--auto-profile", "--json", str(f)])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "detected_type" in data


def test_cli_auto_profile_and_profile_mutually_exclusive(tmp_path):
    """--auto-profile and --profile should not be used together."""
    from distill.cli import main

    f = tmp_path / "test.txt"
    f.write_text("test content")

    runner = CliRunner()
    result = runner.invoke(main, ["score", "--auto-profile", "--profile", "technical", str(f)])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()
