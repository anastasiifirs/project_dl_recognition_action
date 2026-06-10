from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import torch

from event_video_recognition.config import load_config
from event_video_recognition.pipeline import run_video_inference


st.set_page_config(page_title="Event Video Recognition", layout="wide")
st.title("Регистрация событий на видеозаписи")

config_path = st.sidebar.text_input("Config", "configs/final.yaml")
checkpoint = st.sidebar.text_input("Checkpoint", "models/final_checkpoint.pt")
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    devices.append("mps")
device = st.sidebar.selectbox("Device", devices, index=0)
mode = st.sidebar.radio(
    "Processing mode",
    ["Presentation full", "Balanced", "Speed check"],
    index=0,
    help="Presentation full is recommended for final demo output with counts and continuous event labels.",
)
uploaded = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi", "mkv"])


def apply_processing_mode(cfg: dict, selected_mode: str) -> dict:
    cfg = dict(cfg)
    cfg["inference"] = dict(cfg.get("inference", {}))
    cfg["repetition_counting"] = dict(cfg.get("repetition_counting", {}))
    if selected_mode == "Presentation full":
        cfg["inference"]["infer_every_frames"] = 4
        cfg["inference"]["draw_overlay"] = True
        cfg["inference"]["fill_gaps_with_other"] = True
        cfg["repetition_counting"]["enabled"] = True
    elif selected_mode == "Balanced":
        cfg["inference"]["infer_every_frames"] = 6
        cfg["inference"]["draw_overlay"] = True
        cfg["inference"]["fill_gaps_with_other"] = True
        cfg["repetition_counting"]["enabled"] = True
    elif selected_mode == "Speed check":
        cfg["inference"]["infer_every_frames"] = 12
        cfg["inference"]["draw_overlay"] = False
        cfg["inference"]["fill_gaps_with_other"] = False
        cfg["repetition_counting"]["enabled"] = False
    return cfg


def probe_video(path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return {}
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return {
        "fps": round(fps, 3),
        "frames": frames,
        "width": width,
        "height": height,
        "duration_sec": round(frames / fps, 2) if fps > 0 else 0.0,
    }

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as file:
        file.write(uploaded.read())
        temp_video = Path(file.name)
    st.video(str(temp_video))
    info = probe_video(temp_video)
    if info:
        st.caption(
            f"Видео: {info['duration_sec']} сек, {info['frames']} кадров, "
            f"{info['width']}x{info['height']}, fps={info['fps']}"
        )
        if float(info["duration_sec"]) > 90 and mode == "Presentation full":
            st.warning("Для видео длиннее 90 секунд Presentation full может обрабатываться очень долго.")
    if device == "mps":
        st.warning("Для r3d_18 на Mac MPS может быть нестабильнее CPU. Если долго висит, попробуйте CPU.")
    if mode == "Speed check":
        st.warning("Speed check нужен только для быстрой проверки. Для демонстрации качества используйте Presentation full.")

    if st.button("Запустить inference"):
        cfg = apply_processing_mode(load_config(config_path), mode)
        output_dir = Path("outputs/streamlit") / temp_video.stem
        with st.spinner("Обрабатываю видео..."):
            result = run_video_inference(temp_video, cfg, checkpoint, output_dir, device_name=device)
        st.success("Готово")
        st.json(result)

        events_csv = Path(result["events_csv"])
        events_json = Path(result["events_json"])
        annotated = Path(result["annotated_video"])
        timeline = Path(result["timeline_png"])

        if annotated.exists():
            st.video(str(annotated))
        if timeline.exists():
            st.image(str(timeline))
        if events_csv.exists():
            df = pd.read_csv(events_csv)
            st.dataframe(df, use_container_width=True)
            if "repetition_count" in df.columns:
                counted = df[df["repetition_count"].notna()][
                    ["label", "start_sec", "end_sec", "repetition_count", "repetition_confidence"]
                ]
                if not counted.empty:
                    st.subheader("Подсчет повторений")
                    st.dataframe(counted, use_container_width=True)
            st.download_button("Скачать events.csv", events_csv.read_bytes(), "events.csv")
        if events_json.exists():
            st.download_button("Скачать events.json", events_json.read_bytes(), "events.json")
else:
    st.info("CLI остается основным способом запуска; Streamlit нужен только для демонстрации.")
