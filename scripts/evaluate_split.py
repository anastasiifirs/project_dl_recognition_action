from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from event_video_recognition.config import ensure_dir, load_config
from event_video_recognition.metrics import event_level_metrics, match_events, read_events, read_ground_truth
from event_video_recognition.pipeline import run_video_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and evaluation for one prepared video split.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--output-dir", default="outputs/final/test_eval")
    parser.add_argument("--device", default=None)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--count-repetitions", action="store_true")
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def read_split_videos(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def update_per_class_totals(totals: dict[str, dict[str, int]], per_class: dict) -> None:
    for label, row in per_class.items():
        totals[label]["tp"] += int(row.get("tp", 0))
        totals[label]["fp"] += int(row.get("fp", 0))
        totals[label]["fn"] += int(row.get("fn", 0))


def finalize_per_class(totals: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int]]:
    result = {}
    for label, counts in sorted(totals.items()):
        tp = counts["tp"]
        fp = counts["fp"]
        fn = counts["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        result[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return result


def finalize_per_class_iou(iou_values: dict[str, list[float]], labels: list[str]) -> dict[str, float | None]:
    result = {}
    for label in labels:
        values = iou_values.get(label, [])
        result[label] = round(sum(values) / len(values), 4) if values else None
    return result


def save_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str], path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Event-level confusion matrix")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_cfg = copy.deepcopy(cfg)
    run_cfg.setdefault("inference", {})["draw_overlay"] = bool(args.render_videos)
    if not args.count_repetitions:
        run_cfg["repetition_counting"] = {"enabled": False}

    output_dir = ensure_dir(args.output_dir)
    raw_dir = Path(cfg["data"]["raw_dir"])
    split_path = Path(args.split_path or Path(cfg["data"]["root"]) / "splits" / f"{args.split}.txt")
    video_names = read_split_videos(split_path)
    if args.max_videos is not None:
        video_names = video_names[: args.max_videos]

    iou_threshold = float(cfg.get("evaluation", {}).get("iou_threshold", 0.3))
    rows = []
    totals = {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "num_ground_truth": 0,
        "num_predicted": 0,
    }
    per_class_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_class_iou_values: dict[str, list[float]] = defaultdict(list)
    matched_true_labels: list[str] = []
    matched_pred_labels: list[str] = []
    weighted_iou_sum = 0.0
    weighted_latency_sum = 0.0
    matched_event_count = 0

    for video_name in video_names:
        video_path = raw_dir / video_name
        video_out = ensure_dir(output_dir / Path(video_name).stem)
        predicted_path = video_out / "events.json"
        if not args.reuse_existing or not predicted_path.exists():
            run_video_inference(video_path, run_cfg, args.checkpoint, video_out, args.device)

        predicted = read_events(predicted_path)
        ground_truth = read_ground_truth(cfg["data"]["annotations"], video_name)
        metrics = event_level_metrics(ground_truth, predicted, iou_threshold=iou_threshold)
        matches = match_events(ground_truth, predicted, iou_threshold=iou_threshold)
        for gt_idx, pred_idx, iou in matches:
            gt_label = ground_truth[gt_idx].label
            pred_label = predicted[pred_idx].label
            per_class_iou_values[gt_label].append(iou)
            matched_true_labels.append(gt_label)
            matched_pred_labels.append(pred_label)
        row = {"video": video_name, **metrics}
        rows.append(row)
        for key in totals:
            totals[key] += int(metrics.get(key, 0))
        update_per_class_totals(per_class_totals, metrics.get("per_class", {}))

        tp = int(metrics.get("true_positives", 0))
        if tp:
            matched_event_count += tp
            weighted_iou_sum += float(metrics.get("mean_iou", 0.0)) * tp
            weighted_latency_sum += float(metrics.get("mean_latency_sec", 0.0)) * tp

        with open(video_out / "metrics.json", "w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

    precision = totals["true_positives"] / max(1, totals["true_positives"] + totals["false_positives"])
    recall = totals["true_positives"] / max(1, totals["true_positives"] + totals["false_negatives"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    summary = {
        "split": args.split,
        "split_path": str(split_path),
        "num_videos": len(video_names),
        **totals,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(weighted_iou_sum / max(1, matched_event_count), 4),
        "mean_latency_sec": round(weighted_latency_sum / max(1, matched_event_count), 4),
        "per_class": finalize_per_class(per_class_totals),
        "per_class_iou": finalize_per_class_iou(per_class_iou_values, cfg["labels"]),
        "videos": rows,
    }

    pd.DataFrame(rows).to_csv(output_dir / "per_video_metrics.csv", index=False)
    save_confusion_matrix(matched_true_labels, matched_pred_labels, cfg["labels"], output_dir / "confusion_matrix.png")
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
