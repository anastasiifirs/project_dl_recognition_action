from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import pandas as pd


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
REQUIRED_COLUMNS = {"video", "start_sec", "end_sec", "label"}


@dataclass
class VideoInfo:
    video: str
    path: str
    fps: float
    width: int
    height: int
    frames: int
    duration_sec: float


@dataclass
class ValidationReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    videos: list[VideoInfo]
    num_events: int
    events_by_label: dict[str, int]
    duration_by_label: dict[str, float]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["videos"] = [asdict(item) for item in self.videos]
        return data


def probe_video(path: Path) -> VideoInfo | None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    fps = fps if fps > 1e-6 else 30.0
    duration = frames / fps if frames > 0 else 0.0
    return VideoInfo(
        video=path.name,
        path=str(path),
        fps=round(fps, 3),
        width=width,
        height=height,
        frames=frames,
        duration_sec=round(duration, 3),
    )


def validate_dataset(
    annotations_csv: str | Path,
    raw_dir: str | Path,
    labels: list[str],
    tolerance_sec: float = 0.25,
) -> ValidationReport:
    annotations_path = Path(annotations_csv)
    raw_path = Path(raw_dir)
    errors: list[str] = []
    warnings: list[str] = []

    if not annotations_path.exists():
        return ValidationReport(False, [f"Annotations not found: {annotations_path}"], [], [], 0, {}, {})
    if not raw_path.exists():
        errors.append(f"Raw video directory not found: {raw_path}")

    df = pd.read_csv(annotations_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
        return ValidationReport(False, errors, warnings, [], len(df), {}, {})

    label_set = set(labels)
    video_infos: dict[str, VideoInfo] = {}
    for video_name in sorted(df["video"].astype(str).unique()):
        video_path = raw_path / video_name
        if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            errors.append(f"Unsupported extension for {video_name}; supported: {sorted(VIDEO_EXTENSIONS)}")
        if not video_path.exists():
            errors.append(f"Video referenced in CSV not found: {video_path}")
            continue
        info = probe_video(video_path)
        if info is None:
            errors.append(f"OpenCV cannot open video: {video_path}")
            continue
        video_infos[video_name] = info

    for idx, row in df.iterrows():
        prefix = f"row {idx + 2}"
        video = str(row["video"])
        label = str(row["label"])
        try:
            start = float(row["start_sec"])
            end = float(row["end_sec"])
        except (TypeError, ValueError):
            errors.append(f"{prefix}: start_sec/end_sec must be numeric")
            continue

        if label not in label_set:
            errors.append(f"{prefix}: unknown label '{label}'")
        if start < 0:
            errors.append(f"{prefix}: start_sec must be >= 0")
        if end <= start:
            errors.append(f"{prefix}: end_sec must be greater than start_sec")
        if video in video_infos and end > video_infos[video].duration_sec + tolerance_sec:
            errors.append(
                f"{prefix}: end_sec={end:.3f} exceeds video duration "
                f"{video_infos[video].duration_sec:.3f}s for {video}"
            )

    durations = (df["end_sec"].astype(float) - df["start_sec"].astype(float)).clip(lower=0)
    events_by_label = df.groupby("label").size().astype(int).to_dict()
    duration_by_label = (
        df.assign(duration_sec=durations).groupby("label")["duration_sec"].sum().round(3).to_dict()
    )
    for label in labels:
        events_by_label.setdefault(label, 0)
        duration_by_label.setdefault(label, 0.0)

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        videos=list(video_infos.values()),
        num_events=len(df),
        events_by_label={str(k): int(v) for k, v in events_by_label.items()},
        duration_by_label={str(k): float(v) for k, v in duration_by_label.items()},
    )
