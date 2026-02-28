"""SQLite-backed score caching and history for distill."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


_DEFAULT_DB_DIR = Path.home() / ".distill"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "history.db"

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text_hash TEXT NOT NULL,
    source TEXT,
    profile TEXT DEFAULT 'default',
    scorer_set TEXT,
    overall_score REAL NOT NULL,
    grade TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    scores_json TEXT NOT NULL,
    metadata_json TEXT,
    scored_at TEXT NOT NULL,
    UNIQUE(text_hash, profile, scorer_set)
);
"""


class ScoreCache:
    """SQLite-backed score cache and history store.

    Args:
        db_path: Path to the SQLite database file. Defaults to ~/.distill/history.db.
    """

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _cache_key(
        text: str, profile: str | None, scorer_names: list[str] | None
    ) -> tuple[str, str, str]:
        """Compute cache key components.

        Returns:
            (text_hash, profile_str, scorer_set_str)
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        profile_str = profile or "default"
        scorer_set_str = ",".join(sorted(scorer_names)) if scorer_names else ""
        return text_hash, profile_str, scorer_set_str

    def get(
        self,
        text: str,
        profile: str | None = None,
        scorer_names: list[str] | None = None,
    ) -> dict | None:
        """Look up a cached score result.

        Returns:
            The stored report dict (from QualityReport.to_dict()), or None on cache miss.
        """
        text_hash, profile_str, scorer_set_str = self._cache_key(text, profile, scorer_names)
        row = self._conn.execute(
            "SELECT scores_json FROM score_history "
            "WHERE text_hash = ? AND profile = ? AND scorer_set = ?",
            (text_hash, profile_str, scorer_set_str),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["scores_json"])

    def put(
        self,
        text: str,
        report_dict: dict,
        source: str | None = None,
        profile: str | None = None,
        scorer_names: list[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Save a score result to the cache and history.

        Args:
            text: The scored text content.
            report_dict: The report as a dict (from QualityReport.to_dict()).
            source: Source label (URL, filename, or "stdin").
            profile: Profile name used for scoring.
            scorer_names: List of scorer names used.
            metadata: Optional source metadata.
        """
        text_hash, profile_str, scorer_set_str = self._cache_key(text, profile, scorer_names)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT OR REPLACE INTO score_history "
            "(text_hash, source, profile, scorer_set, overall_score, grade, "
            "word_count, scores_json, metadata_json, scored_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                text_hash,
                source,
                profile_str,
                scorer_set_str,
                report_dict["overall_score"],
                report_dict["grade"],
                report_dict["word_count"],
                json.dumps(report_dict),
                json.dumps(metadata) if metadata else None,
                now,
            ),
        )
        self._conn.commit()

    def history(
        self,
        source: str | None = None,
        limit: int = 20,
        since: str | None = None,
    ) -> list[dict]:
        """Query past scores.

        Args:
            source: Filter by source substring (case-insensitive).
            limit: Maximum number of entries to return.
            since: ISO 8601 date string; only return entries scored after this date.

        Returns:
            List of history entry dicts, newest first.
        """
        query = "SELECT * FROM score_history WHERE 1=1"
        params: list = []

        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")

        if since:
            query += " AND scored_at >= ?"
            params.append(since)

        query += " ORDER BY scored_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "id": row["id"],
                "source": row["source"],
                "profile": row["profile"],
                "overall_score": row["overall_score"],
                "grade": row["grade"],
                "word_count": row["word_count"],
                "scored_at": row["scored_at"],
                "scores": json.loads(row["scores_json"]),
            }
            for row in rows
        ]

    def clear(
        self,
        before: str | None = None,
        source: str | None = None,
    ) -> int:
        """Delete history entries.

        Args:
            before: ISO 8601 date string; delete entries scored before this date.
            source: Filter by source substring (case-insensitive).

        Returns:
            Number of entries deleted.
        """
        query = "DELETE FROM score_history WHERE 1=1"
        params: list = []

        if before:
            query += " AND scored_at < ?"
            params.append(before)

        if source:
            query += " AND source LIKE ?"
            params.append(f"%{source}%")

        cursor = self._conn.execute(query, params)
        self._conn.commit()
        return cursor.rowcount

    def stats(self) -> dict:
        """Return cache statistics.

        Returns:
            Dict with keys: count, size_bytes, oldest, newest.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) as count, MIN(scored_at) as oldest, MAX(scored_at) as newest "
            "FROM score_history"
        ).fetchone()

        try:
            size_bytes = os.path.getsize(self._db_path)
        except OSError:
            size_bytes = 0

        return {
            "count": row["count"],
            "size_bytes": size_bytes,
            "oldest": row["oldest"],
            "newest": row["newest"],
        }
