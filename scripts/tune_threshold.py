from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from event_video_recognition.config import ensure_dir, load_config
from event_video_recognition.events import EventRegistry, Prediction, refine_event_boundaries
from event_video_recognition.metrics import event_level_metrics, read_ground_truth
from event_video_recognition.models import default_device, load_ensemble, predict_clip_ensemble
from event_video_recognition.video import ClipBuffer, VideoFrameReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune confidence_threshold on a split without touching test.")
    parser.add_argument("--config", default="configs/x3d_s.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--output-dir", default="outputs/x3d_s/threshold_sweep_val")
    parser.add_argument("--device", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--max-videos", type=int, default=None)
    return parser.parse_args()


def read_split_videos(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def collect_raw_predictions(video_path: Path, config: dict, checkpoint: str | None, device_name: str | None) -> tuple[list[Prediction], float]:
    labels = config["labels"]
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    device = torch.device(device_name or default_device())
    runtime_config = copy.deepcopy(config)
    if checkpoint:
        runtime_config["model"].pop("checkpoints", None)
        runtime_config["model"]["checkpoint"] = checkpoint
    models = load_ensemble(runtime_config, device)

    reader = VideoFrameReader(video_path)
    total_frames = int(reader.capture.get(7))
    buffer = ClipBuffer(int(model_cfg["clip_len"]), int(model_cfg["frame_stride"]), int(model_cfg["image_size"]))
    predictions: list[Prediction] = []
    with torch.no_grad():
        for packet in tqdm(reader, total=total_frames or None, desc=f"Raw predictions {video_path.name}"):
            buffer.append(packet.frame_bgr)
            if buffer.ready() and packet.index % int(infer_cfg["infer_every_frames"]) == 0:
                probs = predict_clip_ensemble(models, buffer.as_model_tensor(device), runtime_config)
                confidence, class_id = torch.max(probs, dim=0)
                predictions.append(Prediction(packet.time_sec, labels[int(class_id.item())], float(confidence.item())))
    duration_sec = total_frames / reader.fps if total_frames else predictions[-1].time_sec if predictions else 0.0
    return predictions, duration_sec


def predictions_to_events(raw_predictions: list[Prediction], duration_sec: float, config: dict, threshold: float):
    infer_cfg = config["inference"]
    registry = EventRegistry(
        confidence_threshold=threshold,
        smoothing_window=int(infer_cfg["smoothing_window"]),
        min_event_duration_sec=float(infer_cfg["min_event_duration_sec"]),
        merge_gap_sec=float(infer_cfg["merge_gap_sec"]),
    )
    for prediction in raw_predictions:
        registry.add_prediction(prediction)
    events = registry.close(end_time_sec=duration_sec)
    if bool(infer_cfg.get("boundary_refinement", False)):
        events = [refine_event_boundaries(event, raw_predictions, threshold) for event in events]
    return events


def aggregate(rows: list[dict]) -> dict:
    totals = defaultdict(int)
    weighted_iou_sum = 0.0
    weighted_latency_sum = 0.0
    matched = 0
    for row in rows:
        for key in ["true_positives", "false_positives", "false_negatives", "num_ground_truth", "num_predicted"]:
            totals[key] += int(row.get(key, 0))
        tp = int(row.get("true_positives", 0))
        if tp:
            matched += tp
            weighted_iou_sum += float(row.get("mean_iou", 0.0)) * tp
            weighted_latency_sum += float(row.get("mean_latency_sec", 0.0)) * tp
    precision = totals["true_positives"] / max(1, totals["true_positives"] + totals["false_positives"])
    recall = totals["true_positives"] / max(1, totals["true_positives"] + totals["false_negatives"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        **dict(totals),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou": round(weighted_iou_sum / max(1, matched), 4),
        "mean_latency_sec": round(weighted_latency_sum / max(1, matched), 4),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    raw_dir = Path(cfg["data"]["raw_dir"])
    split_path = Path(args.split_path or Path(cfg["data"]["root"]) / "splits" / f"{args.split}.txt")
    video_names = read_split_videos(split_path)
    if args.max_videos is not None:
        video_names = video_names[: args.max_videos]

    raw_by_video = {}
    for video_name in video_names:
        raw_by_video[video_name] = collect_raw_predictions(raw_dir / video_name, cfg, args.checkpoint, args.device)

    summaries = []
    details = {}
    for threshold in args.thresholds:
        rows = []
        for video_name in video_names:
            raw_predictions, duration_sec = raw_by_video[video_name]
            predicted = predictions_to_events(raw_predictions, duration_sec, cfg, threshold)
            ground_truth = read_ground_truth(cfg["data"]["annotations"], video_name)
            rows.append({"video": video_name, **event_level_metrics(ground_truth, predicted, cfg["evaluation"]["iou_threshold"])})
        summary = {"threshold": threshold, "split": args.split, "num_videos": len(video_names), **aggregate(rows)}
        summaries.append(summary)
        details[str(threshold)] = rows

    best = max(summaries, key=lambda item: (item["f1"], item["mean_iou"], item["precision"]))
    pd.DataFrame(summaries).to_csv(output_dir / "threshold_sweep.csv", index=False)
    with open(output_dir / "threshold_sweep.json", "w", encoding="utf-8") as file:
        json.dump({"best": best, "summaries": summaries, "details": details}, file, ensure_ascii=False, indent=2)
    print(json.dumps({"best": best, "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
