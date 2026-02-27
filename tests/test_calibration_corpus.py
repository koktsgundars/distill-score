"""Tests using the curated calibration corpus to catch scoring regressions."""

from __future__ import annotations

import pathlib

import pytest
import yaml

from distill.pipeline import Pipeline

CORPUS_PATH = pathlib.Path(__file__).parent / "corpus" / "calibration_corpus.yaml"
GRADE_ORDER = ["F", "D", "C", "B", "A"]


def _load_corpus():
    with open(CORPUS_PATH) as f:
        data = yaml.safe_load(f)
    return data["entries"]


def _grade_index(grade: str) -> int:
    return GRADE_ORDER.index(grade)


_corpus_entries = _load_corpus()
_corpus_ids = [e["id"] for e in _corpus_entries]


def pytest_generate_tests(metafunc):
    if "corpus_entry" in metafunc.fixturenames:
        metafunc.parametrize(
            "corpus_entry",
            _corpus_entries,
            ids=_corpus_ids,
        )


@pytest.fixture(scope="module")
def pipeline():
    return Pipeline()


class TestCalibrationCorpus:
    def test_overall_score_in_range(self, corpus_entry, pipeline):
        report = pipeline.score(corpus_entry["text"])
        min_score = corpus_entry["expected_min_score"]
        max_score = corpus_entry["expected_max_score"]
        assert min_score <= report.overall_score <= max_score, (
            f"[{corpus_entry['id']}] score {report.overall_score:.3f} "
            f"not in [{min_score}, {max_score}]"
        )

    def test_grade_in_range(self, corpus_entry, pipeline):
        report = pipeline.score(corpus_entry["text"])
        min_grade = corpus_entry["expected_min_grade"]
        max_grade = corpus_entry["expected_max_grade"]
        grade_idx = _grade_index(report.grade)
        lo = min(_grade_index(min_grade), _grade_index(max_grade))
        hi = max(_grade_index(min_grade), _grade_index(max_grade))
        assert lo <= grade_idx <= hi, (
            f"[{corpus_entry['id']}] grade {report.grade} "
            f"not in [{min_grade}, {max_grade}]"
        )

    def test_dimension_expectations(self, corpus_entry, pipeline):
        expectations = corpus_entry.get("dimension_expectations")
        if not expectations:
            pytest.skip("No dimension expectations for this entry")

        report = pipeline.score(corpus_entry["text"])
        score_map = {r.name: r.score for r in report.scores}

        for dim, bounds in expectations.items():
            if dim not in score_map:
                continue
            actual = score_map[dim]
            assert bounds["min"] <= actual <= bounds["max"], (
                f"[{corpus_entry['id']}] {dim} score {actual:.3f} "
                f"not in [{bounds['min']}, {bounds['max']}]"
            )
