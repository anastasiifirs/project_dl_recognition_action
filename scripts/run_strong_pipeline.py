from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the recommended final training pipeline.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-evaluate-all", action="store_true")
    return parser.parse_args()


def run_step(command: list[str]) -> None:
    print("\n" + "=" * 80)
    print(" ".join(command))
    print("=" * 80)
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--output-dir",
        args.output_dir,
    ]
    eval_cmd = [
        sys.executable,
        "scripts/evaluate_split.py",
        "--config",
        args.config,
        "--checkpoint",
        "models/final_checkpoint.pt",
        "--split",
        "test",
        "--output-dir",
        "outputs/final/test_eval",
    ]
    if args.device:
        train_cmd.extend(["--device", args.device])
        eval_cmd.extend(["--device", args.device])

    run_step([sys.executable, "scripts/validate_dataset.py", "--config", args.config])
    run_step(
        [
            sys.executable,
            "scripts/prepare_splits.py",
            "--config",
            args.config,
            "--train-ratio",
            "0.7",
            "--val-ratio",
            "0.15",
            "--test-ratio",
            "0.15",
            "--seed",
            "42",
        ]
    )
    run_step(train_cmd)
    if not args.skip_evaluate_all:
        run_step(eval_cmd)


if __name__ == "__main__":
    main()
