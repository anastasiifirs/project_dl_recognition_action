from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final training curves.")
    parser.add_argument("--metrics", default="models/final_metrics.json")
    parser.add_argument("--output", default="models/training_curves.png")
    return parser.parse_args()


def load_history(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    history = payload.get("history", payload)
    if not isinstance(history, list):
        raise ValueError("Metrics file must contain a history list.")
    return history


def stage_transitions(history: list[dict]) -> list[tuple[int, str]]:
    transitions = []
    previous = None
    for row in history:
        stage = str(row.get("stage", "stage"))
        epoch = int(row["epoch"])
        if previous is not None and stage != previous:
            transitions.append((epoch, stage))
        previous = stage
    return transitions


def main() -> None:
    args = parse_args()
    history = load_history(args.metrics)
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [float(row["val_loss"]) for row in history]
    train_acc = [float(row["train_acc"]) for row in history]
    val_acc = [float(row["val_acc"]) for row in history]

    best_idx = max(range(len(history)), key=lambda idx: val_acc[idx])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, val_acc, marker="o", label="val_acc")
    axes[1].scatter(
        [epochs[best_idx]],
        [val_acc[best_idx]],
        marker="*",
        s=220,
        color="gold",
        edgecolor="black",
        label="best val_acc",
        zorder=5,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    for epoch, stage in stage_transitions(history):
        for ax in axes:
            ax.axvline(epoch, color="gray", linestyle="--", alpha=0.7)
        axes[0].text(epoch + 0.05, max(train_loss + val_loss), stage, color="gray", fontsize=9, va="top")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {output}")


if __name__ == "__main__":
    main()
