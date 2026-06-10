from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from event_video_recognition.video import letterbox_batch


class ActionClipDataset(Dataset):
    """Dataset for event CSV or prepared clips metadata.

    Supported rows:
    - video,start_sec,end_sec,label for reading intervals from raw sessions.
    - clip_path,label for reading already cut clips.
    """

    def __init__(
        self,
        annotations_csv: str | Path,
        videos_dir: str | Path,
        labels: list[str],
        clip_len: int,
        frame_stride: int,
        image_size: int,
        random_clip: bool = True,
        augment: bool = False,
    ):
        self.annotations = pd.read_csv(annotations_csv)
        self.videos_dir = Path(videos_dir)
        self.labels = labels
        self.label_to_id = {label: idx for idx, label in enumerate(labels)}
        self.clip_len = clip_len
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.random_clip = random_clip
        self.augment = augment

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.annotations.iloc[idx]
        if "clip_path" in self.annotations.columns:
            video_path = Path(str(row["clip_path"]))
            if not video_path.is_absolute():
                video_path = Path.cwd() / video_path
            frames = self._read_clip(video_path, 0.0, float(row.get("duration_sec", 10_000.0)))
        else:
            video_path = self.videos_dir / str(row["video"])
            frames = self._read_clip(video_path, float(row["start_sec"]), float(row["end_sec"]))
        label = str(row["label"])
        return frames, torch.tensor(self.label_to_id[label], dtype=torch.long)

    def _read_clip(self, video_path: Path, start_sec: float, end_sec: float) -> torch.Tensor:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        start_frame = max(0, int(start_sec * fps))
        end_frame = max(start_frame + 1, int(end_sec * fps))
        needed = 1 + (self.clip_len - 1) * self.frame_stride
        if end_frame - start_frame < needed:
            indices = np.linspace(start_frame, end_frame, self.clip_len).astype(int)
        else:
            max_start = end_frame - needed
            if self.random_clip:
                offset = np.random.randint(start_frame, max_start + 1)
            else:
                offset = start_frame + (max_start - start_frame) // 2
            indices = np.array([offset + i * self.frame_stride for i in range(self.clip_len)])

        frames = []
        wanted = {int(frame_idx): pos for pos, frame_idx in enumerate(indices)}
        captured: dict[int, torch.Tensor] = {}
        first_idx = int(indices.min())
        last_idx = int(indices.max())
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
        current_idx = first_idx
        while current_idx <= last_idx:
            ok, frame = capture.read()
            if not ok:
                break
            if current_idx in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                captured[current_idx] = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            current_idx += 1
        capture.release()

        for frame_idx in indices:
            item = captured.get(int(frame_idx))
            if item is not None:
                frames.append(item)

        if not frames:
            raise RuntimeError(f"No frames read from {video_path}")
        while len(frames) < self.clip_len:
            frames.append(frames[-1].clone())

        batch = torch.stack(frames, dim=0)
        batch = letterbox_batch(batch, self.image_size)
        if self.augment:
            batch = self._augment(batch)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        batch = (batch - mean) / std
        return batch.permute(1, 0, 2, 3).contiguous()

    def _augment(self, batch: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < 0.5:
            batch = torch.flip(batch, dims=[3])
        if torch.rand(()) < 0.7:
            brightness = 0.85 + float(torch.rand(())) * 0.3
            batch = (batch * brightness).clamp(0.0, 1.0)
        if torch.rand(()) < 0.7:
            contrast = 0.85 + float(torch.rand(())) * 0.3
            mean = batch.mean(dim=(2, 3), keepdim=True)
            batch = ((batch - mean) * contrast + mean).clamp(0.0, 1.0)
        return batch
