"""Processed Task 2 dataset and graph/mel batch collation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .graphs import GRAPH_VARIANTS


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError("Manifest must be an object containing a records list")
    if not isinstance(value.get("label_names"), list) or not value["label_names"]:
        raise ValueError("Manifest must define non-empty label_names")
    sample_ids = [record.get("sample_id") for record in value["records"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Manifest contains duplicate sample_id values")
    return value


def resolve_record_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


class ProcessedTask2Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        graph_variant: str = "temporal_similarity",
    ) -> None:
        if graph_variant not in GRAPH_VARIANTS:
            raise ValueError(f"graph_variant must be one of {GRAPH_VARIANTS}")
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = load_manifest(self.manifest_path)
        self.label_names = self.manifest["label_names"]
        self.records = [record for record in self.manifest["records"] if record["split"] == split]
        self.graph_variant = graph_variant
        if not self.records:
            raise ValueError(f"Manifest contains no records for split {split!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        processed_path = resolve_record_path(self.manifest_path, record["processed_path"])
        sample = torch.load(processed_path, map_location="cpu", weights_only=False)
        edge_indices = sample["edge_indices"]
        if self.graph_variant not in edge_indices:
            raise ValueError(f"{processed_path} has no {self.graph_variant!r} graph")
        labels = torch.as_tensor(sample["labels"], dtype=torch.float32)
        if labels.numel() != len(self.label_names):
            raise ValueError(f"Label width mismatch in {processed_path}")
        return {
            "sample_id": sample["sample_id"],
            "graph": Data(
                x=torch.as_tensor(sample["x"], dtype=torch.float32),
                edge_index=torch.as_tensor(edge_indices[self.graph_variant], dtype=torch.long),
            ),
            "mel": torch.as_tensor(sample["mel"], dtype=torch.float32),
            "labels": labels,
        }


def collate_task2(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Task 2 batch")
    maximum_frames = max(sample["mel"].shape[-1] for sample in samples)
    mels = [
        functional.pad(sample["mel"], (0, maximum_frames - sample["mel"].shape[-1]))
        for sample in samples
    ]
    return {
        "sample_ids": [sample["sample_id"] for sample in samples],
        "graph": Batch.from_data_list([sample["graph"] for sample in samples]),
        "mel": torch.stack(mels),
        "labels": torch.stack([sample["labels"] for sample in samples]),
    }

