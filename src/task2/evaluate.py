"""Evaluate a saved Task 2 checkpoint on a processed manifest split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ProcessedTask2Dataset, collate_task2
from .models import build_task2_model
from .train import classification_metrics, resolve_device, run_loader


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest: str | Path,
    *,
    split: str = "test",
    device_name: str = "auto",
    output: str | Path | None = None,
) -> dict:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset = ProcessedTask2Dataset(manifest, split, config["graph_variant"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_task2,
    )
    model = build_task2_model(
        config["model"],
        input_dim=checkpoint["input_dim"],
        num_labels=checkpoint["num_labels"],
        hidden_dim=config["hidden_dim"],
        layers=config["layers"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    result = run_loader(model, config["model"], loader, device, nn.BCEWithLogitsLoss())
    metrics = classification_metrics(
        result["targets"],
        result["probabilities"],
        checkpoint["label_names"],
        checkpoint["threshold"],
    )
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        np.savez_compressed(
            output.with_suffix(".predictions.npz"),
            probabilities=result["probabilities"],
            targets=result["targets"],
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("results/task2/evaluation.json"))
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        split=args.split,
        device_name=args.device,
        output=args.output,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

