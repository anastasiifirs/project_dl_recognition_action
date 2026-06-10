from __future__ import annotations

import argparse
import json

from event_video_recognition.config import load_config
from event_video_recognition.validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dev video dataset and annotations.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    report = validate_dataset(data_cfg["annotations"], data_cfg["raw_dir"], cfg["labels"])
    payload = report.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
