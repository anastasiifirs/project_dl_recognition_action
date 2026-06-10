from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torchvision.models.video import R3D_18_Weights, r3d_18


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(architecture: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if architecture == "r3d_18":
        weights = R3D_18_Weights.DEFAULT if pretrained else None
        try:
            model = r3d_18(weights=weights)
        except Exception as exc:
            if not pretrained:
                raise
            print(f"Warning: pretrained r3d_18 weights are unavailable ({exc}). Falling back to random init.")
            model = r3d_18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if architecture == "x3d_s":
        try:
            from pytorchvideo.models.hub import x3d_s
        except ImportError as exc:
            raise ImportError("Install optional dependency: pip install '.[x3d]'") from exc
        model = x3d_s(pretrained=pretrained)
        model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
        return model

    raise ValueError(f"Unsupported architecture: {architecture}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    labels: Sequence[str],
    device: torch.device,
) -> tuple[nn.Module, list[str]]:
    if checkpoint_path is None:
        return model.to(device), list(labels)
    if not Path(checkpoint_path).exists():
        print(f"Warning: checkpoint not found: {checkpoint_path}. Using current model weights.")
        return model.to(device), list(labels)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state") or checkpoint.get("model") or checkpoint
    model.load_state_dict(state_dict, strict=True)
    ckpt_labels = checkpoint.get("labels", labels) if isinstance(checkpoint, dict) else labels
    return model.to(device), list(ckpt_labels)


def apply_temperature(logits: torch.Tensor, config: dict | None) -> torch.Tensor:
    if not config:
        return logits
    model_cfg = config.get("model", config)
    if not bool(model_cfg.get("use_temperature_scaling", False)):
        return logits
    path = Path(model_cfg.get("temperature_path", "models/temperature.json"))
    if not path.exists():
        return logits
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    temperature = float(payload.get("temperature", 1.0))
    if temperature <= 0:
        return logits
    return logits / temperature


def load_ensemble(config: dict, device: torch.device) -> list[nn.Module]:
    labels = list(config["labels"])
    model_cfg = config["model"]
    checkpoints = model_cfg.get("checkpoints")
    if checkpoints is None:
        checkpoints = [model_cfg.get("checkpoint")]
    models: list[nn.Module] = []
    for checkpoint_path in checkpoints:
        model = build_model(model_cfg["architecture"], len(labels), pretrained=bool(model_cfg.get("pretrained", True)))
        model, _ = load_checkpoint(model, checkpoint_path, labels, device)
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def predict_clip_logits(model: nn.Module, clip: torch.Tensor, config: dict | None = None) -> torch.Tensor:
    logits = model(clip)[0]
    return apply_temperature(logits, config)


@torch.no_grad()
def predict_clip_ensemble(models: list[nn.Module], clip: torch.Tensor, config: dict | None = None) -> torch.Tensor:
    probs = []
    for model in models:
        logits = predict_clip_logits(model, clip, config)
        probs.append(torch.softmax(logits, dim=0))
    return torch.stack(probs, dim=0).mean(dim=0)


@torch.no_grad()
def predict_clip(model: nn.Module, clip: torch.Tensor, config: dict | None = None) -> torch.Tensor:
    logits = predict_clip_logits(model, clip, config)
    return torch.softmax(logits, dim=0)
