from pathlib import Path

import pandas as pd

from event_video_recognition.validation import validate_dataset


LABELS = ["stand", "walk", "run", "jump", "push_ups", "squat", "bend", "other"]


def test_validation_catches_unknown_label(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [{"video": "missing.mp4", "start_sec": 0, "end_sec": 1, "label": "dance"}]
    ).to_csv(csv_path, index=False)

    report = validate_dataset(csv_path, raw, LABELS)

    assert not report.ok
    assert any("unknown label" in error for error in report.errors)


def test_validation_catches_bad_interval(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        [{"video": "missing.mp4", "start_sec": 2, "end_sec": 1, "label": "walk"}]
    ).to_csv(csv_path, index=False)

    report = validate_dataset(csv_path, raw, LABELS)

    assert not report.ok
    assert any("end_sec must be greater" in error for error in report.errors)
