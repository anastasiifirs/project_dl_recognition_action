from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from event_video_recognition.config import load_config
from event_video_recognition.metrics import event_level_metrics, read_events, read_ground_truth, segment_iou
from event_video_recognition.visualization import plot_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predicted events against annotation CSV.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--predicted", required=True)
    parser.add_argument("--video-name", default=None)
    parser.add_argument("--output-dir", default="outputs/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted = read_events(args.predicted)
    ground_truth = read_ground_truth(cfg["data"]["annotations"], args.video_name)
    metrics = event_level_metrics(
        ground_truth,
        predicted,
        iou_threshold=float(cfg.get("evaluation", {}).get("iou_threshold", 0.3)),
    )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    plot_timeline(predicted, output_dir / "timeline_compare.png", ground_truth=ground_truth)
    save_confusion_matrix(
        ground_truth,
        predicted,
        cfg["labels"],
        output_dir / "confusion_matrix.png",
        float(cfg.get("evaluation", {}).get("iou_threshold", 0.3)),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def save_confusion_matrix(ground_truth, predicted, labels, output_path: Path, iou_threshold: float) -> None:
    labels = list(labels) + ["missed", "extra"]
    index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    matched_pred = set()
    for gt in ground_truth:
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred in enumerate(predicted):
            if pred_idx in matched_pred:
                continue
            iou = segment_iou(gt.start_sec, gt.end_sec, pred.start_sec, pred.end_sec)
            if iou > best_iou:
                best_iou = iou
                best_idx = pred_idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_pred.add(best_idx)
            matrix[index.get(gt.label, index["missed"]), index.get(predicted[best_idx].label, index["extra"])] += 1
        else:
            matrix[index.get(gt.label, index["missed"]), index["missed"]] += 1
    for pred_idx, pred in enumerate(predicted):
        if pred_idx not in matched_pred:
            matrix[index["extra"], index.get(pred.label, index["extra"])] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
