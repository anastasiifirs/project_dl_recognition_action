from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from event_video_recognition.config import load_config
from event_video_recognition.metrics import events_to_dicts, read_events
from event_video_recognition.repetitions import add_repetition_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add repetition counts to an existing events.json.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--events-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["label", "start_sec", "end_sec", "avg_confidence", "max_confidence"]
    for optional in ["repetition_count", "repetition_confidence", "repetition_method"]:
        if any(optional in row for row in rows):
            fieldnames.append(optional)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    events_path = Path(args.events_json)
    with open(events_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    events = read_events(events_path)
    rows = add_repetition_counts(events_to_dicts(events), events, args.video, cfg.get("repetition_counting"))
    payload["repetition_counting"] = cfg.get("repetition_counting", {"enabled": False})
    payload["events"] = rows

    output_json = Path(args.output_json) if args.output_json else events_path
    output_csv = Path(args.output_csv) if args.output_csv else events_path.with_suffix(".csv")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    write_csv(output_csv, rows)
    print(json.dumps({"events_json": str(output_json), "events_csv": str(output_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
