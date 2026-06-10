from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from event_video_recognition.events import Event


def plot_timeline(
    predicted: list[Event],
    output_path: str | Path,
    ground_truth: list[Event] | None = None,
    title: str = "Action events timeline",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = sorted({event.label for event in predicted + (ground_truth or [])})
    if not labels:
        labels = ["no_events"]
    colors = {label: plt.cm.tab20(idx % 20) for idx, label in enumerate(labels)}

    fig_height = 2.5 if ground_truth is None else 3.4
    fig, ax = plt.subplots(figsize=(12, fig_height))
    tracks = [("predicted", predicted)]
    if ground_truth is not None:
        tracks.append(("ground_truth", ground_truth))

    for y, (track_name, events) in enumerate(tracks):
        for event in events:
            ax.barh(
                y,
                max(0.0, event.end_sec - event.start_sec),
                left=event.start_sec,
                height=0.35,
                color=colors[event.label],
                edgecolor="black",
                alpha=0.85,
            )
            ax.text(
                event.start_sec,
                y,
                event.label,
                va="center",
                ha="left",
                fontsize=8,
                color="black",
            )
        if not events:
            ax.text(0, y, "no events", va="center", ha="left", fontsize=9, color="gray")

    ax.set_yticks(range(len(tracks)))
    ax.set_yticklabels([item[0] for item in tracks])
    ax.set_xlabel("Time, seconds")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
