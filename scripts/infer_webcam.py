from __future__ import annotations

import argparse

import cv2
import torch

from event_video_recognition.config import load_config
from event_video_recognition.events import EventRegistry, Prediction
from event_video_recognition.models import build_model, load_checkpoint, predict_clip
from event_video_recognition.pipeline import draw_label
from event_video_recognition.video import ClipBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam action recognition demo.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--camera", default="0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]
    model_cfg = cfg["model"]
    infer_cfg = cfg["inference"]
    device = torch.device(args.device)
    model = build_model(model_cfg["architecture"], len(labels), pretrained=bool(model_cfg.get("pretrained", True)))
    model, labels = load_checkpoint(model, args.checkpoint or model_cfg.get("checkpoint"), labels, device)
    model.eval()

    camera = int(args.camera) if str(args.camera).isdigit() else args.camera
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open camera: {args.camera}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    buffer = ClipBuffer(int(model_cfg["clip_len"]), int(model_cfg["frame_stride"]), int(model_cfg["image_size"]))
    registry = EventRegistry(
        float(infer_cfg["confidence_threshold"]),
        int(infer_cfg["smoothing_window"]),
        float(infer_cfg["min_event_duration_sec"]),
        float(infer_cfg["merge_gap_sec"]),
    )
    latest_label = "warming_up"
    latest_confidence = 0.0
    frame_idx = 0
    with torch.no_grad():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            buffer.append(frame)
            if buffer.ready() and frame_idx % int(infer_cfg["infer_every_frames"]) == 0:
                probs = predict_clip(model, buffer.as_model_tensor(device))
                confidence, class_id = torch.max(probs, dim=0)
                latest_confidence = float(confidence.item())
                latest_label = registry.add_prediction(
                    Prediction(frame_idx / fps, labels[int(class_id.item())], latest_confidence)
                )
            draw_label(frame, latest_label, latest_confidence)
            cv2.imshow("event video recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_idx += 1
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
