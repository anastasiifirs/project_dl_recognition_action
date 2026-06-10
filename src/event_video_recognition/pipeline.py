from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from event_video_recognition.config import ensure_dir
from event_video_recognition.events import Event, EventRegistry, Prediction, refine_event_boundaries
from event_video_recognition.metrics import events_to_dicts
from event_video_recognition.models import default_device, load_ensemble, predict_clip_ensemble
from event_video_recognition.repetitions import add_repetition_counts
from event_video_recognition.video import ClipBuffer, VideoFrameReader
from event_video_recognition.visualization import plot_timeline


def draw_label(frame, label: str, confidence: float, repetition_count: int | None = None) -> None:
    text = f"{label} {confidence:.2f}"
    if repetition_count is not None:
        text = f"{text} | count: {repetition_count}"
    cv2.rectangle(frame, (18, 18), (620, 74), (0, 0, 0), -1)
    cv2.putText(frame, text, (30, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 230, 70), 2, cv2.LINE_AA)


def active_row_at_time(rows: list[dict], time_sec: float) -> dict | None:
    for row in rows:
        if float(row["start_sec"]) <= time_sec < float(row["end_sec"]):
            return row
    return None


def rows_to_events(rows: list[dict]) -> list[Event]:
    return [
        Event(
            label=str(row["label"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            avg_confidence=float(row.get("avg_confidence") or 0.0),
            max_confidence=float(row.get("max_confidence") or 0.0),
        )
        for row in rows
    ]


def fill_gaps_with_label(
    rows: list[dict],
    duration_sec: float | None,
    label: str = "other",
    min_gap_sec: float = 0.5,
) -> list[dict]:
    if duration_sec is None:
        return rows
    filled: list[dict] = []
    cursor = 0.0
    for row in sorted(rows, key=lambda item: float(item["start_sec"])):
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if start - cursor >= min_gap_sec:
            filled.append(
                {
                    "label": label,
                    "start_sec": round(cursor, 3),
                    "end_sec": round(start, 3),
                    "avg_confidence": 0.0,
                    "max_confidence": 0.0,
                    "repetition_count": None,
                    "repetition_confidence": None,
                    "repetition_method": "gap_fill",
                }
            )
        filled.append(row)
        cursor = max(cursor, end)
    if duration_sec - cursor >= min_gap_sec:
        filled.append(
            {
                "label": label,
                "start_sec": round(cursor, 3),
                "end_sec": round(duration_sec, 3),
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "repetition_count": None,
                "repetition_confidence": None,
                "repetition_method": "gap_fill",
            }
        )
    return filled


def render_annotated_video(video_path: str | Path, rows: list[dict], output_path: str | Path) -> None:
    reader = VideoFrameReader(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, reader.fps, (reader.width, reader.height))
    for packet in reader:
        frame = packet.frame_bgr.copy()
        row = active_row_at_time(rows, packet.time_sec)
        if row is not None:
            count = row.get("repetition_count")
            draw_label(
                frame,
                str(row["label"]),
                float(row.get("avg_confidence", row.get("max_confidence", 0.0))),
                int(count) if count not in (None, "") else None,
            )
        else:
            draw_label(frame, "no confident event", 0.0)
        writer.write(frame)
    writer.release()


def write_events_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = ["label", "start_sec", "end_sec", "avg_confidence", "max_confidence"]
    for optional in ["repetition_count", "repetition_confidence", "repetition_method"]:
        if any(optional in row for row in rows):
            fieldnames.append(optional)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_video_inference(
    video_path: str | Path,
    config: dict,
    checkpoint: str | Path | None,
    output_dir: str | Path,
    device_name: str | None = None,
) -> dict[str, str | float | int]:
    labels = config["labels"]
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    device = torch.device(device_name or default_device())
    runtime_config = dict(config)
    runtime_config["model"] = dict(model_cfg)
    if checkpoint is not None:
        runtime_config["model"].pop("checkpoints", None)
        runtime_config["model"]["checkpoint"] = str(checkpoint)
    models = load_ensemble(runtime_config, device)

    reader = VideoFrameReader(video_path)
    buffer = ClipBuffer(int(model_cfg["clip_len"]), int(model_cfg["frame_stride"]), int(model_cfg["image_size"]))
    registry = EventRegistry(
        confidence_threshold=float(infer_cfg["confidence_threshold"]),
        smoothing_window=int(infer_cfg["smoothing_window"]),
        min_event_duration_sec=float(infer_cfg["min_event_duration_sec"]),
        merge_gap_sec=float(infer_cfg["merge_gap_sec"]),
    )

    out_dir = ensure_dir(output_dir)
    events_json = out_dir / "events.json"
    events_csv = out_dir / "events.csv"
    annotated_video = out_dir / "annotated.mp4"
    timeline_png = out_dir / "timeline.png"
    stats_json = out_dir / "inference_stats.json"

    total_frames = int(reader.capture.get(cv2.CAP_PROP_FRAME_COUNT))
    infer_times: list[float] = []
    raw_predictions: list[Prediction] = []
    started = time.perf_counter()

    with torch.no_grad():
        for packet in tqdm(reader, total=total_frames or None, desc="Registering events"):
            buffer.append(packet.frame_bgr)
            if buffer.ready() and packet.index % int(infer_cfg["infer_every_frames"]) == 0:
                t0 = time.perf_counter()
                probs = predict_clip_ensemble(models, buffer.as_model_tensor(device), runtime_config)
                infer_times.append(time.perf_counter() - t0)
                confidence, class_id = torch.max(probs, dim=0)
                raw_label = labels[int(class_id.item())]
                prediction = Prediction(packet.time_sec, raw_label, float(confidence.item()))
                raw_predictions.append(prediction)
                registry.add_prediction(prediction)

    duration_sec = total_frames / reader.fps if total_frames else None
    events = registry.close(end_time_sec=duration_sec)
    if bool(infer_cfg.get("boundary_refinement", False)):
        threshold = float(infer_cfg.get("confidence_threshold", 0.0))
        events = [refine_event_boundaries(event, raw_predictions, threshold) for event in events]
    rows = events_to_dicts(events)
    rows = add_repetition_counts(rows, events, video_path, config.get("repetition_counting"))
    if infer_cfg.get("fill_gaps_with_other", False):
        rows = fill_gaps_with_label(
            rows,
            duration_sec,
            label=str(infer_cfg.get("gap_label", "other")),
            min_gap_sec=float(infer_cfg.get("min_gap_fill_sec", 0.5)),
        )
    if infer_cfg.get("draw_overlay", True):
        render_annotated_video(video_path, rows, annotated_video)
    payload = {
        "video": str(video_path),
        "fps": reader.fps,
        "model": model_cfg,
        "inference": infer_cfg,
        "repetition_counting": config.get("repetition_counting", {"enabled": False}),
        "events": rows,
    }
    with open(events_json, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    write_events_csv(events_csv, rows)
    plot_timeline(rows_to_events(rows), timeline_png)

    total_time = time.perf_counter() - started
    stats = {
        "video": str(video_path),
        "device": str(device),
        "frames": total_frames,
        "fps_video": reader.fps,
        "wall_time_sec": round(total_time, 4),
        "processing_fps": round(total_frames / total_time, 4) if total_time > 0 else 0.0,
        "num_model_calls": len(infer_times),
        "avg_inference_ms": round(1000 * sum(infer_times) / len(infer_times), 3) if infer_times else 0.0,
        "num_events": len(rows),
    }
    with open(stats_json, "w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    return {
        "events_json": str(events_json),
        "events_csv": str(events_csv),
        "annotated_video": str(annotated_video),
        "timeline_png": str(timeline_png),
        "inference_stats": str(stats_json),
        **stats,
    }
