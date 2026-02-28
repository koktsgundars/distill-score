"""Tests for the watch command."""

from __future__ import annotations

import os
import threading
import time

import pytest
from click.testing import CliRunner

from distill.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "test_article.md"
    f.write_text(
        "We migrated our payment service from a monolith to a separate deployment. "
        "Latency dropped from p99 of 340ms to 95ms. The oversized connection pool "
        "was masking a connection leak in our retry logic. This is a concrete finding "
        "with specific metrics that demonstrate real engineering tradeoffs."
    )
    return str(f)


class TestWatchValidation:
    """Test input validation for the watch command."""

    def test_rejects_stdin(self, runner):
        result = runner.invoke(main, ["watch", "-"])
        assert result.exit_code != 0
        assert "stdin" in result.output.lower()

    def test_rejects_http_url(self, runner):
        result = runner.invoke(main, ["watch", "https://example.com"])
        assert result.exit_code != 0
        assert "URL" in result.output

    def test_rejects_https_url(self, runner):
        result = runner.invoke(main, ["watch", "http://example.com"])
        assert result.exit_code != 0
        assert "URL" in result.output

    def test_rejects_nonexistent_file(self, runner):
        result = runner.invoke(main, ["watch", "/nonexistent/file.md"])
        assert result.exit_code != 0


class TestWatchDebounce:
    """Test that debounce logic coalesces rapid file changes."""

    def test_debounce_coalesces_rapid_changes(self, sample_file):
        """Simulate rapid file modifications and verify scoring happens once after debounce."""
        from watchdog.events import FileModifiedEvent, FileSystemEventHandler

        change_count = 0
        change_event = threading.Event()
        timer = None
        timer_lock = threading.Lock()
        debounce = 0.3

        filepath = os.path.abspath(sample_file)

        def _on_change():
            nonlocal change_count
            change_count += 1
            change_event.set()

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if os.path.abspath(event.src_path) != filepath:
                    return
                nonlocal timer
                with timer_lock:
                    if timer is not None:
                        timer.cancel()
                    timer = threading.Timer(debounce, _on_change)
                    timer.daemon = True
                    timer.start()

        handler = Handler()
        event = FileModifiedEvent(filepath)

        # Simulate 5 rapid modifications
        for _ in range(5):
            handler.on_modified(event)
            time.sleep(0.05)

        # Wait for debounce to fire
        change_event.wait(timeout=2.0)
        time.sleep(0.1)

        assert change_count == 1

        with timer_lock:
            if timer is not None:
                timer.cancel()

    def test_separate_changes_score_separately(self, sample_file):
        """Two changes separated by more than debounce should each trigger scoring."""
        from watchdog.events import FileModifiedEvent, FileSystemEventHandler

        change_count = 0
        timer = None
        timer_lock = threading.Lock()
        debounce = 0.2

        filepath = os.path.abspath(sample_file)

        def _on_change():
            nonlocal change_count
            change_count += 1

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if os.path.abspath(event.src_path) != filepath:
                    return
                nonlocal timer
                with timer_lock:
                    if timer is not None:
                        timer.cancel()
                    timer = threading.Timer(debounce, _on_change)
                    timer.daemon = True
                    timer.start()

        handler = Handler()
        event = FileModifiedEvent(filepath)

        # First change
        handler.on_modified(event)
        time.sleep(debounce + 0.15)

        # Second change
        handler.on_modified(event)
        time.sleep(debounce + 0.15)

        assert change_count == 2

        with timer_lock:
            if timer is not None:
                timer.cancel()
