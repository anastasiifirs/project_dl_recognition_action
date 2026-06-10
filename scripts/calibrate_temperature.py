from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from event_video_recognition.config import load_config
from event_video_recognition.dataset import ActionClipDataset
from event_video_recognition.models import build_model, default_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate model confidence with temperature scaling.")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--output", default="models/temperature.json")
    return parser.parse_args()


def collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Collecting validation logits"):
            x = x.to(device)
            logits_batches.append(model(x).detach().cpu())
            label_batches.append(y.detach().cpu())
    return torch.cat(logits_batches, dim=0), torch.cat(label_batches, dim=0)


def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = torch.nn.Parameter(torch.log(torch.tensor(1.5)))
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-4)
        scaled = logits / temperature
        loss = F.nll_loss(F.log_softmax(scaled, dim=1), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(min=1e-4).item())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    labels = cfg["labels"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device = torch.device(args.device)
    val_csv = args.val_csv or Path(data_cfg["root"]) / "processed" / "val.csv"

    dataset = ActionClipDataset(
        val_csv,
        data_cfg["raw_dir"],
        labels,
        int(model_cfg["clip_len"]),
        int(model_cfg["frame_stride"]),
        int(model_cfg["image_size"]),
        random_clip=False,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("training", {}).get("batch_size", 2)), shuffle=False)
    model = build_model(model_cfg["architecture"], len(labels), pretrained=bool(model_cfg.get("pretrained", True)))
    model, _ = load_checkpoint(model, args.checkpoint or model_cfg.get("checkpoint"), labels, device)
    logits, y = collect_logits(model, loader, device)
    temperature = optimize_temperature(logits, y)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as file:
        json.dump({"temperature": temperature}, file, ensure_ascii=False, indent=2)
    print(f"Temperature: 1.0000 -> {temperature:.4f}")
    print(f"Saved temperature: {output}")


if __name__ == "__main__":
    main()
