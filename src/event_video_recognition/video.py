from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FramePacket:
    index: int
    time_sec: float
    frame_bgr: np.ndarray


def letterbox_bgr(frame_bgr: np.ndarray, out_size: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = out_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    top = (out_size - new_h) // 2
    left = (out_size - new_w) // 2
    output[top : top + new_h, left : left + new_w] = resized
    return output


def letterbox_batch(frames_tchw: torch.Tensor, out_size: int) -> torch.Tensor:
    _, _, height, width = frames_tchw.shape
    scale = out_size / max(height, width)
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))
    resized = F.interpolate(frames_tchw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    pad_h = out_size - new_h
    pad_w = out_size - new_w
    return F.pad(
        resized,
        (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        mode="constant",
        value=0.0,
    )


class VideoFrameReader:
    def __init__(self, video_path: str | Path):
        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 1e-6 else 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self) -> Iterator[FramePacket]:
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield FramePacket(index=index, time_sec=index / self.fps, frame_bgr=frame)
            index += 1
        self.capture.release()


class ClipBuffer:
    def __init__(self, clip_len: int, frame_stride: int, image_size: int):
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.need = 1 + (clip_len - 1) * frame_stride
        self.frames: deque[np.ndarray] = deque(maxlen=self.need)

    def append(self, frame_bgr: np.ndarray) -> None:
        frame = letterbox_bgr(frame_bgr, self.image_size)
        self.frames.append(frame[..., ::-1].copy())

    def ready(self) -> bool:
        return len(self.frames) == self.need

    def as_model_tensor(self, device: torch.device) -> torch.Tensor:
        sampled = [self.frames[i] for i in range(0, self.need, self.frame_stride)]
        arr = np.stack(sampled, axis=0)
        x = torch.from_numpy(arr).float().to(device) / 255.0
        x = x.permute(0, 3, 1, 2)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
