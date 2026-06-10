from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from event_video_recognition.config import ensure_dir, load_config
from event_video_recognition.metrics import event_level_metrics, read_events, read_ground_truth
from event_video_recognition.pipeline import run_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and evaluation for every annotated video.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="outputs/dev/evaluate_all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def average(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return round(sum(values) / max(1, len(values)), 4)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    raw_dir = Path(cfg["data"]["raw_dir"])
    annotations = pd.read_csv(cfg["data"]["annotations"])
    video_names = sorted(annotations["video"].drop_duplicates().tolist())
    iou_threshold = float(cfg.get("evaluation", {}).get("iou_threshold", 0.3))

    rows = []
    totals = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "num_ground_truth": 0,
        "num_predicted": 0,
    }
    for video_name in video_names:
        video_path = raw_dir / video_name
        video_out = ensure_dir(output_dir / Path(video_name).stem)
        predicted_path = video_out / "events.json"
        if not args.reuse_existing or not predicted_path.exists():
            run_video_inference(video_path, cfg, args.checkpoint, video_out, args.device)

        predicted = read_events(predicted_path)
        ground_truth = read_ground_truth(cfg["data"]["annotations"], video_name)
        metrics = event_level_metrics(ground_truth, predicted, iou_threshold=iou_threshold)
        row = {"video": video_name, **metrics}
        rows.append(row)
        for key in totals:
            totals[key] += int(metrics.get(key, 0))
        with open(video_out / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

    precision = totals["true_positives"] / max(1, totals["true_positives"] + totals["false_positives"])
    recall = totals["true_positives"] / max(1, totals["true_positives"] + totals["false_negatives"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    summary = {
        **totals,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": average(rows, "mean_iou"),
        "mean_latency_sec": average(rows, "mean_latency_sec"),
        "videos": rows,
    }

    pd.DataFrame(rows).to_csv(output_dir / "per_video_metrics.csv", index=False)
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
