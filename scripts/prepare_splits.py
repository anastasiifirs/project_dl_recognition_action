from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from event_video_recognition.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test splits by source video.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--train-videos", type=int, default=None)
    parser.add_argument("--val-videos", type=int, default=None)
    parser.add_argument("--test-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-clips", action="store_true")
    return parser.parse_args()


def split_counts(total: int, args: argparse.Namespace) -> tuple[int, int, int]:
    explicit = args.train_videos is not None or args.val_videos is not None or args.test_videos is not None
    if explicit:
        train = int(args.train_videos or 0)
        val = int(args.val_videos or 0)
        test = int(args.test_videos or 0)
        if train + val + test > total:
            raise ValueError(f"Requested {train + val + test} videos, but only {total} are available.")
        if train == 0:
            train = max(1, total - val - test)
        if val == 0:
            val = max(1, round(total * args.val_ratio))
        if test == 0:
            test = max(1, total - train - val)
        return train, val, test

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if ratio_sum <= 0:
        raise ValueError("Split ratios must sum to a positive number.")
    train = max(1, round(total * args.train_ratio / ratio_sum))
    val = max(1, round(total * args.val_ratio / ratio_sum))
    test = total - train - val
    if test < 1:
        test = 1
        train = max(1, total - val - test)
    while train + val + test > total:
        train -= 1
    return train, val, test


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["data"]["root"])
    metadata_path = root / "clips_metadata.csv"
    annotations_path = Path(cfg["data"]["annotations"])

    if args.use_clips and metadata_path.exists():
        df = pd.read_csv(metadata_path)
        source_col = "source_video"
    else:
        df = pd.read_csv(annotations_path)
        source_col = "video"

    videos = sorted(df[source_col].astype(str).unique())
    if len(videos) < 3:
        raise ValueError("Need at least 3 source videos for train/val/test split.")

    shuffled = pd.Series(videos).sample(frac=1.0, random_state=args.seed).tolist()
    train_count, val_count, test_count = split_counts(len(shuffled), args)
    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count : train_count + val_count + test_count]
    splits = {"train": train, "val": val, "test": test}

    splits_dir = root / "splits"
    processed_dir = root / "processed"
    splits_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, split_videos in splits.items():
        with open(splits_dir / f"{name}.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(split_videos) + "\n")
        split_df = df[df[source_col].astype(str).isin(split_videos)].copy()
        split_df.to_csv(processed_dir / f"{name}.csv", index=False)
        label_counts = split_df["label"].value_counts().sort_index().to_dict()
        print(
            f"{name}: {len(split_videos)} videos, {len(split_df)} events -> "
            f"{processed_dir / f'{name}.csv'}"
        )
        print(f"  labels: {label_counts}")


if __name__ == "__main__":
    main()
