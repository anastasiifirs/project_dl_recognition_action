from __future__ import annotations

import argparse
from pathlib import Path

from event_video_recognition.config import load_config
from event_video_recognition.metrics import events_to_dicts, read_events
from event_video_recognition.pipeline import render_annotated_video
from event_video_recognition.repetitions import add_repetition_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render annotated video from existing events.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--add-counts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    events = read_events(args.events)
    rows = events_to_dicts(events)
    if args.add_counts:
        rows = add_repetition_counts(rows, events, args.video, cfg.get("repetition_counting"))
    output_path = Path(args.output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_annotated_video(args.video, rows, output_path)
    print(f"Annotated video: {output_path}")


if __name__ == "__main__":
    main()
