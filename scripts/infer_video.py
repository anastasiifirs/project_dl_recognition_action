from __future__ import annotations

import argparse
import json

import torch

from event_video_recognition.config import load_config
from event_video_recognition.pipeline import run_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register action events in a video file.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="outputs/dev")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = run_video_inference(args.video, cfg, args.checkpoint, args.output_dir, args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
