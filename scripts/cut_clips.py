from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from event_video_recognition.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut annotated events into class folders.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--use-opencv", action="store_true", help="Force OpenCV cutting instead of ffmpeg.")
    return parser.parse_args()


def cut_with_ffmpeg(src: Path, dst: Path, start: float, end: float) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start),
        "-to",
        str(end),
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dst),
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def cut_with_opencv(src: Path, dst: Path, start: float, end: float) -> bool:
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        return False
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    start_frame = max(0, int(start * fps))
    end_frame = max(start_frame + 1, int(end * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    capture.release()
    return dst.exists() and dst.stat().st_size > 0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = Path(data_cfg["root"])
    raw_dir = Path(data_cfg["raw_dir"])
    annotations = pd.read_csv(data_cfg["annotations"])
    clips_dir = root / "clips"
    metadata_path = root / "clips_metadata.csv"
    ffmpeg_available = shutil.which("ffmpeg") is not None and not args.use_opencv

    rows = []
    for idx, row in tqdm(list(annotations.iterrows()), desc="Cutting clips"):
        label = str(row["label"])
        src = raw_dir / str(row["video"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        stem = Path(str(row["video"])).stem
        dst = clips_dir / label / f"{stem}_{idx:04d}_{label}.mp4"
        ok = cut_with_ffmpeg(src, dst, start, end) if ffmpeg_available else cut_with_opencv(src, dst, start, end)
        if not ok:
            raise RuntimeError(f"Failed to cut clip from {src}: {start}-{end}")
        rows.append(
            {
                "clip_path": str(dst),
                "label": label,
                "source_video": str(row["video"]),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 3),
            }
        )

    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    print(f"Saved clips metadata: {metadata_path}")


if __name__ == "__main__":
    main()
