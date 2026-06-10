from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from event_video_recognition.events import Event

DEFAULT_LABELS = ["stand", "walk", "run", "jump", "push_ups", "squat", "bend", "other"]


def segment_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def _as_events(rows: Iterable[dict]) -> list[Event]:
    events: list[Event] = []
    for row in rows:
        events.append(
            Event(
                label=str(row["label"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                avg_confidence=float(row.get("avg_confidence", 1.0)),
                max_confidence=float(row.get("max_confidence", row.get("avg_confidence", 1.0))),
            )
        )
    return events


def read_events(path: str | Path) -> list[Event]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        import json

        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        return _as_events(payload.get("events", payload if isinstance(payload, list) else []))
    return _as_events(pd.read_csv(path).to_dict("records"))


def read_ground_truth(annotations_csv: str | Path, video_name: str | None = None) -> list[Event]:
    df = pd.read_csv(annotations_csv)
    if video_name:
        df = df[df["video"].astype(str) == video_name]
    return _as_events(df.to_dict("records"))


def event_level_metrics(
    ground_truth: list[Event],
    predicted: list[Event],
    iou_threshold: float = 0.3,
) -> dict[str, float | int | dict[str, float]]:
    matches = match_events(ground_truth, predicted, iou_threshold)
    matched_pred = {pred_idx for _, pred_idx, _ in matches}
    latencies = [predicted[pred_idx].start_sec - ground_truth[gt_idx].start_sec for gt_idx, pred_idx, _ in matches]

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = sum(item[2] for item in matches) / tp if tp else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    observed_labels = sorted({event.label for event in ground_truth} | {event.label for event in predicted})
    labels = list(dict.fromkeys(DEFAULT_LABELS + observed_labels))
    per_class_iou_values: dict[str, list[float]] = {label: [] for label in labels}
    for gt_idx, _, iou in matches:
        per_class_iou_values[ground_truth[gt_idx].label].append(iou)
    per_class_iou = {
        label: round(sum(values) / len(values), 4) if values else None
        for label, values in per_class_iou_values.items()
    }

    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    matched_gt = {gt_idx for gt_idx, _, _ in matches}
    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx in matched_gt:
            per_class[gt.label]["tp"] += 1
        else:
            per_class[gt.label]["fn"] += 1
    for pred_idx, pred in enumerate(predicted):
        if pred_idx not in matched_pred:
            per_class[pred.label]["fp"] += 1

    per_class_scores = {}
    for label, counts in per_class.items():
        p = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        r = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        per_class_scores[label] = {
            **counts,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        }

    return {
        "iou_threshold": iou_threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_latency_sec": round(mean_latency, 4),
        "num_ground_truth": len(ground_truth),
        "num_predicted": len(predicted),
        "per_class": per_class_scores,
        "per_class_iou": per_class_iou,
    }


def match_events(
    ground_truth: list[Event],
    predicted: list[Event],
    iou_threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for gt_idx, gt in enumerate(ground_truth):
        best_idx = None
        best_iou = 0.0
        for pred_idx, pred in enumerate(predicted):
            if pred_idx in matched_pred or pred.label != gt.label:
                continue
            iou = segment_iou(gt.start_sec, gt.end_sec, pred.start_sec, pred.end_sec)
            if iou > best_iou:
                best_idx = pred_idx
                best_iou = iou
        if best_idx is not None and best_iou >= iou_threshold:
            matched_pred.add(best_idx)
            matches.append((gt_idx, best_idx, best_iou))
    return matches


def events_to_dicts(events: list[Event]) -> list[dict]:
    return [asdict(event) for event in events]
