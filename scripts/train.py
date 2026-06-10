from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from event_video_recognition.config import ensure_dir, load_config
from event_video_recognition.dataset import ActionClipDataset
from event_video_recognition.models import build_model, default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug/fine-tune a video action classifier.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    freeze_backbone: bool = False,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and freeze_backbone:
        set_frozen_modules_eval(model)
    total_loss = 0.0
    correct = 0
    seen = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, y in tqdm(loader, leave=False, desc="train" if is_train else "val"):
            x = x.to(device)
            y = y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += y.numel()
    return {"loss": total_loss / max(1, seen), "acc": correct / max(1, seen)}


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_trainable_layers(model, train_strategy: str) -> None:
    for param in model.parameters():
        param.requires_grad = True

    if train_strategy == "full":
        return

    for param in model.parameters():
        param.requires_grad = False

    if train_strategy == "head":
        unfreeze_prefixes = ("fc", "blocks.5.proj")
    elif train_strategy == "last_block":
        unfreeze_prefixes = ("layer4", "fc", "blocks.4", "blocks.5")
    else:
        raise ValueError(f"Unsupported training.strategy: {train_strategy}")

    found = False
    for name, param in model.named_parameters():
        if name.startswith(unfreeze_prefixes):
            param.requires_grad = True
            found = True
    if found:
        return

    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
        return
    if hasattr(model, "blocks") and hasattr(model.blocks[-1], "proj"):
        for param in model.blocks[-1].proj.parameters():
            param.requires_grad = True
        return
    raise ValueError("Cannot find final classifier layer to unfreeze.")


def set_frozen_modules_eval(model) -> None:
    for module in model.modules():
        has_trainable_params = any(param.requires_grad for param in module.parameters(recurse=False))
        if not has_trainable_params:
            module.eval()


def make_training_stages(train_cfg: dict) -> list[dict]:
    if train_cfg.get("stages"):
        return list(train_cfg["stages"])
    return [
        {
            "name": str(train_cfg.get("strategy", "full")),
            "strategy": str(train_cfg.get("strategy", "full")),
            "epochs": int(train_cfg["epochs"]),
            "learning_rate": float(train_cfg["learning_rate"]),
            "weight_decay": float(train_cfg["weight_decay"]),
            "label_smoothing": float(train_cfg.get("label_smoothing", 0.05)),
        }
    ]


def build_class_weights(train_csv: str | Path, labels: list[str], device: torch.device) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()
    weights = []
    for label in labels:
        count = max(1, int(counts.get(label, 0)))
        weights.append(1.0 / count)
    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    return tensor / tensor.mean()


def build_balanced_sampler(train_csv: str | Path) -> WeightedRandomSampler:
    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().to_dict()
    weights = [1.0 / max(1, int(counts[str(label)])) for label in df["label"]]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    set_seed(int(train_cfg.get("seed", 42)))
    device = torch.device(args.device)

    train_csv = args.train_csv or Path(data_cfg["root"]) / "processed" / "train.csv"
    val_csv = args.val_csv or Path(data_cfg["root"]) / "processed" / "val.csv"
    train_ds = ActionClipDataset(
        train_csv,
        data_cfg["raw_dir"],
        labels,
        int(model_cfg["clip_len"]),
        int(model_cfg["frame_stride"]),
        int(model_cfg["image_size"]),
        random_clip=True,
        augment=bool(train_cfg.get("augment", True)),
    )
    val_ds = ActionClipDataset(
        val_csv,
        data_cfg["raw_dir"],
        labels,
        int(model_cfg["clip_len"]),
        int(model_cfg["frame_stride"]),
        int(model_cfg["image_size"]),
        random_clip=False,
        augment=False,
    )
    sampler = build_balanced_sampler(train_csv) if bool(train_cfg.get("balanced_sampler", True)) else None
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    val_loader = DataLoader(val_ds, batch_size=int(train_cfg["batch_size"]), shuffle=False, num_workers=0)

    model = build_model(model_cfg["architecture"], len(labels), pretrained=bool(model_cfg.get("pretrained", True)))
    model.to(device)

    history = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    output_dir = ensure_dir(args.output_dir)
    checkpoint_path = Path(model_cfg.get("checkpoint") or output_dir / "final_checkpoint.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    class_weights = build_class_weights(train_csv, labels, device)
    global_epoch = 0

    for stage in make_training_stages(train_cfg):
        stage_name = str(stage.get("name", stage.get("strategy", "stage")))
        train_strategy = str(stage.get("strategy", "full"))
        configure_trainable_layers(model, train_strategy)
        model.to(device)
        params = trainable_parameters(model)
        if not params:
            raise RuntimeError(f"No trainable parameters for stage: {stage_name}")

        optimizer = torch.optim.AdamW(
            params,
            lr=float(stage.get("learning_rate", train_cfg["learning_rate"])),
            weight_decay=float(stage.get("weight_decay", train_cfg["weight_decay"])),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(stage.get("epochs", 1))),
            eta_min=float(stage.get("min_learning_rate", 1e-6)),
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(stage.get("label_smoothing", train_cfg.get("label_smoothing", 0.05))),
        )
        print(
            {
                "stage": stage_name,
                "strategy": train_strategy,
                "trainable_parameters": sum(param.numel() for param in params),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        for _ in range(int(stage.get("epochs", 1))):
            global_epoch += 1
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                freeze_backbone=train_strategy != "full",
            )
            val_metrics = run_epoch(model, val_loader, criterion, device)
            scheduler.step()
            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "strategy": train_strategy,
                "train_loss": round(train_metrics["loss"], 5),
                "train_acc": round(train_metrics["acc"], 5),
                "val_loss": round(val_metrics["loss"], 5),
                "val_acc": round(val_metrics["acc"], 5),
                "lr": round(float(scheduler.get_last_lr()[0]), 8),
            }
            history.append(row)
            print(row)
            is_better = val_metrics["acc"] > best_val_acc or (
                val_metrics["acc"] == best_val_acc and val_metrics["loss"] < best_val_loss
            )
            if is_better:
                best_val_acc = val_metrics["acc"]
                best_val_loss = val_metrics["loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "labels": labels,
                        "architecture": model_cfg["architecture"],
                        "clip_len": model_cfg["clip_len"],
                        "frame_stride": model_cfg["frame_stride"],
                        "image_size": model_cfg["image_size"],
                        "best_epoch": global_epoch,
                        "best_stage": stage_name,
                        "best_val_acc": best_val_acc,
                        "best_val_loss": best_val_loss,
                        "dev_note": "Small-data staged fine-tuning checkpoint; validate on held-out videos before final use.",
                    },
                    checkpoint_path,
                )

    metrics_name = checkpoint_path.name.replace("_checkpoint.pt", "_metrics.json")
    if metrics_name == checkpoint_path.name:
        metrics_name = f"{checkpoint_path.stem}_metrics.json"
    metrics_path = Path(output_dir) / metrics_name
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            {"history": history, "best_val_acc": best_val_acc, "best_val_loss": best_val_loss},
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
