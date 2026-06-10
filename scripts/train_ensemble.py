from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from event_video_recognition.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train several final checkpoints with different seeds.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 123])
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_seed_config(base_config: dict, seed: int, checkpoint_path: Path) -> Path:
    cfg = dict(base_config)
    cfg["training"] = dict(cfg.get("training", {}))
    cfg["model"] = dict(cfg.get("model", {}))
    cfg["training"]["seed"] = seed
    cfg["model"]["checkpoint"] = str(checkpoint_path)
    handle = tempfile.NamedTemporaryFile("w", suffix=f"_seed{seed}.yaml", delete=False, encoding="utf-8")
    with handle:
        yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
    return Path(handle.name)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")

    for seed in args.seeds:
        set_seed(seed)
        checkpoint_path = output_dir / f"final_checkpoint_seed{seed}.pt"
        seed_config = write_seed_config(base_config, seed, checkpoint_path)
        command = [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(seed_config),
            "--output-dir",
            str(output_dir),
        ]
        if args.device:
            command.extend(["--device", args.device])
        try:
            subprocess.run(command, check=True, env=env)
            saved.append(str(checkpoint_path))
        finally:
            seed_config.unlink(missing_ok=True)

    print("Saved ensemble checkpoints:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
