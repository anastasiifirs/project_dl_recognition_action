from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np

from event_video_recognition.events import Event


@dataclass(frozen=True)
class RepetitionResult:
    count: int
    confidence: float
    method: str


POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def smooth_signal(values: list[float], window: int = 5) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    signal = np.asarray(values, dtype=np.float32)
    if len(signal) < window:
        return signal
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(signal, kernel, mode="same")


def count_peaks(signal: np.ndarray, min_distance: int, prominence_std: float) -> tuple[int, float]:
    if len(signal) < 3:
        return 0, 0.0
    centered = signal - float(np.median(signal))
    scale = float(np.std(centered))
    threshold = max(float(np.percentile(centered, 65)), scale * prominence_std)
    peaks: list[int] = []
    last_peak = -min_distance
    for idx in range(1, len(centered) - 1):
        is_peak = centered[idx] > centered[idx - 1] and centered[idx] >= centered[idx + 1]
        if not is_peak or centered[idx] < threshold:
            continue
        if idx - last_peak < min_distance:
            if peaks and centered[idx] > centered[peaks[-1]]:
                peaks[-1] = idx
                last_peak = idx
            continue
        peaks.append(idx)
        last_peak = idx
    confidence = min(1.0, max(0.0, scale / (float(np.mean(np.abs(signal))) + 1e-6)))
    return len(peaks), round(confidence, 4)


def count_valleys(signal: np.ndarray, min_distance: int, prominence_std: float) -> tuple[int, float]:
    count, confidence = count_peaks(-signal, min_distance=min_distance, prominence_std=prominence_std)
    return count, confidence


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-6:
        return 0.0
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def read_pose_signal(
    video_path: str | Path,
    event: Event,
    sample_fps: float,
    model_path: str | Path | None = None,
) -> tuple[list[float], float, str]:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is not installed") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    start_frame = max(0, int(event.start_sec * fps))
    end_frame = max(start_frame + 1, int(event.end_sec * fps))
    step = max(1, int(round(fps / max(1.0, sample_fps))))
    signal: list[float] = []
    detected = 0
    sampled = 0

    pose = None
    landmarker = None
    use_solutions = hasattr(mp, "solutions")
    if use_solutions:
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
    else:
        pose_model_path = resolve_pose_model_path(model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while frame_idx < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_idx - start_frame) % step == 0:
                sampled += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if use_solutions:
                    result = pose.process(rgb)
                    landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
                else:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect(mp_image)
                    landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
                if landmarks:
                    points = np.array([landmark_to_xyzw(lm) for lm in landmarks], dtype=np.float32)
                    value = pose_value_for_label(event.label, points)
                    if value is not None:
                        signal.append(value)
                        detected += 1
            frame_idx += 1
    finally:
        if pose is not None:
            pose.close()
        if landmarker is not None:
            landmarker.close()
        capture.release()

    detection_ratio = detected / max(1, sampled)
    return signal, detection_ratio, "mediapipe_pose"


def landmark_to_xyzw(landmark) -> list[float]:
    return [
        float(landmark.x),
        float(landmark.y),
        float(landmark.z),
        float(getattr(landmark, "visibility", getattr(landmark, "presence", 1.0))),
    ]


def resolve_pose_model_path(model_path: str | Path | None) -> Path:
    candidates = []
    if model_path:
        candidates.append(Path(model_path))
    candidates.append(Path("models/pose_landmarker_full.task"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Pose model not found, repetition counting unavailable")


def pose_value_for_label(label: str, points: np.ndarray) -> float | None:
    visibility = points[:, 3]
    if float(np.nanmean(visibility)) < 0.35:
        return None

    if label == "push_ups":
        left = angle_degrees(
            points[POSE_LANDMARKS["left_shoulder"]],
            points[POSE_LANDMARKS["left_elbow"]],
            points[POSE_LANDMARKS["left_wrist"]],
        )
        right = angle_degrees(
            points[POSE_LANDMARKS["right_shoulder"]],
            points[POSE_LANDMARKS["right_elbow"]],
            points[POSE_LANDMARKS["right_wrist"]],
        )
        return float(np.mean([left, right]))

    if label == "squat":
        left = angle_degrees(
            points[POSE_LANDMARKS["left_hip"]],
            points[POSE_LANDMARKS["left_knee"]],
            points[POSE_LANDMARKS["left_ankle"]],
        )
        right = angle_degrees(
            points[POSE_LANDMARKS["right_hip"]],
            points[POSE_LANDMARKS["right_knee"]],
            points[POSE_LANDMARKS["right_ankle"]],
        )
        return float(np.mean([left, right]))

    if label == "bend":
        shoulder_mid = (points[POSE_LANDMARKS["left_shoulder"]] + points[POSE_LANDMARKS["right_shoulder"]]) / 2.0
        hip_mid = (points[POSE_LANDMARKS["left_hip"]] + points[POSE_LANDMARKS["right_hip"]]) / 2.0
        vertical = hip_mid.copy()
        vertical[1] -= 1.0
        return angle_degrees(shoulder_mid, hip_mid, vertical)

    if label == "jump":
        hip_mid = (points[POSE_LANDMARKS["left_hip"]] + points[POSE_LANDMARKS["right_hip"]]) / 2.0
        return -float(hip_mid[1])

    return None


def estimate_pose_repetitions_for_event(
    video_path: str | Path,
    event: Event,
    config: dict,
) -> RepetitionResult:
    sample_fps = float(config.get("pose_sample_fps", config.get("sample_fps", 10.0)))
    smooth_window = int(config.get("pose_smooth_window", config.get("smooth_window", 5)))
    min_periods = config.get("min_period_sec", {})
    min_period_sec = float(min_periods.get(event.label, config.get("default_min_period_sec", 0.5)))
    prominence_std = float(config.get("pose_prominence_std", config.get("prominence_std", 0.35)))
    min_detection_ratio = float(config.get("min_pose_detection_ratio", 0.45))
    signal, detection_ratio, method = read_pose_signal(
        video_path,
        event,
        sample_fps=sample_fps,
        model_path=config.get("pose_model_path"),
    )
    if detection_ratio < min_detection_ratio or len(signal) < 3:
        raise RuntimeError(f"Pose signal is too weak for {event.label}: detection_ratio={detection_ratio:.3f}")
    smoothed = smooth_signal(signal, window=smooth_window)
    min_distance = max(1, int(round(min_period_sec * sample_fps)))
    if event.label in {"push_ups", "squat"}:
        count, motion_confidence = count_valleys(
            smoothed,
            min_distance=min_distance,
            prominence_std=prominence_std,
        )
    else:
        count, motion_confidence = count_peaks(
            smoothed,
            min_distance=min_distance,
            prominence_std=prominence_std,
        )
    confidence = round(float(np.mean([detection_ratio, motion_confidence])), 4)
    return RepetitionResult(count=count, confidence=confidence, method=method)


def read_motion_signal(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    sample_fps: float,
    resize_width: int = 160,
) -> list[float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    start_frame = max(0, int(start_sec * fps))
    end_frame = max(start_frame + 1, int(end_sec * fps))
    step = max(1, int(round(fps / max(1.0, sample_fps))))
    previous: np.ndarray | None = None
    signal: list[float] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if (frame_idx - start_frame) % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scale = resize_width / max(1, gray.shape[1])
            resized = cv2.resize(gray, (resize_width, max(1, int(gray.shape[0] * scale))))
            resized = cv2.GaussianBlur(resized, (5, 5), 0)
            if previous is not None:
                diff = cv2.absdiff(resized, previous)
                signal.append(float(np.mean(diff)))
            previous = resized
        frame_idx += 1
    capture.release()
    return signal


def estimate_repetitions_for_event(
    video_path: str | Path,
    event: Event,
    config: dict,
) -> RepetitionResult:
    method = str(config.get("method", "auto"))
    pose_labels = set(config.get("pose_labels", ["push_ups", "squat", "bend"]))
    if method in {"auto", "pose"} and event.label in pose_labels:
        try:
            return estimate_pose_repetitions_for_event(video_path, event, config)
        except FileNotFoundError:
            warnings.warn("Pose model not found, repetition counting unavailable", stacklevel=2)
            return RepetitionResult(count=0, confidence=0.0, method="unavailable")
        except Exception:
            if method == "pose":
                raise

    sample_fps = float(config.get("sample_fps", 10.0))
    smooth_window = int(config.get("smooth_window", 5))
    min_periods = config.get("min_period_sec", {})
    min_period_sec = float(min_periods.get(event.label, config.get("default_min_period_sec", 0.5)))
    prominence_std = float(config.get("prominence_std", 0.35))
    signal = read_motion_signal(video_path, event.start_sec, event.end_sec, sample_fps=sample_fps)
    smoothed = smooth_signal(signal, window=smooth_window)
    min_distance = max(1, int(round(min_period_sec * sample_fps)))
    count, confidence = count_peaks(smoothed, min_distance=min_distance, prominence_std=prominence_std)
    divisors = config.get("peak_divisor", {})
    divisor = max(1.0, float(divisors.get(event.label, 1.0)))
    if divisor > 1.0:
        count = int(round(count / divisor))
    return RepetitionResult(count=count, confidence=confidence, method="motion_energy_peaks")


def add_repetition_counts(
    rows: list[dict],
    events: list[Event],
    video_path: str | Path,
    config: dict | None,
) -> list[dict]:
    if not config or not bool(config.get("enabled", False)):
        return rows
    countable_labels = set(config.get("labels", []))
    output = [dict(row) for row in rows]
    for idx, event in enumerate(events):
        if event.label not in countable_labels:
            output[idx]["repetition_count"] = None
            output[idx]["repetition_confidence"] = None
            output[idx]["repetition_method"] = None
            continue
        result = estimate_repetitions_for_event(video_path, event, config)
        output[idx].update(asdict(result))
        output[idx]["repetition_count"] = output[idx].pop("count")
        output[idx]["repetition_confidence"] = output[idx].pop("confidence")
        output[idx]["repetition_method"] = output[idx].pop("method")
    return output
