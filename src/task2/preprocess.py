"""Cache Task 2 node features, graph variants, and log-mel tensors from a manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import load_manifest, resolve_record_path
from .features import (
    AudioFeatureConfig,
    extract_log_mel_from_waveform,
    extract_track_features_from_waveform,
    load_audio,
)
from .graphs import build_edge_variants


def safe_sample_filename(sample_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}-{digest}.pt"


def extract_raw_sample(
    record: dict[str, Any], manifest_path: Path, config: AudioFeatureConfig
) -> dict[str, Any]:
    audio_path = resolve_record_path(manifest_path, record["audio_path"])
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found for {record['sample_id']}: {audio_path}")
    waveform = load_audio(audio_path, config.sample_rate)
    x = extract_track_features_from_waveform(waveform, config)
    mel = extract_log_mel_from_waveform(waveform, config)[None, ...]
    return {
        "sample_id": record["sample_id"],
        "audio_path": str(audio_path),
        "split": record["split"],
        "labels": torch.tensor(record["labels"], dtype=torch.float32),
        "x_raw": torch.from_numpy(x),
        "mel": torch.from_numpy(mel),
        "feature_config": config.to_dict(),
    }


def fit_train_normalizer(raw_paths: list[Path]) -> tuple[np.ndarray, np.ndarray, int]:
    total: np.ndarray | None = None
    total_squared: np.ndarray | None = None
    count = 0
    for path in raw_paths:
        sample = torch.load(path, map_location="cpu", weights_only=False)
        if sample["split"] != "train":
            continue
        x = torch.as_tensor(sample["x_raw"], dtype=torch.float64).numpy()
        total = x.sum(axis=0) if total is None else total + x.sum(axis=0)
        total_squared = (x * x).sum(axis=0) if total_squared is None else total_squared + (x * x).sum(axis=0)
        count += len(x)
    if count == 0 or total is None or total_squared is None:
        raise ValueError("Cannot fit normalization: manifest has no training nodes")
    mean = total / count
    variance = np.maximum(total_squared / count - mean * mean, 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), count


def preprocess_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    config: AudioFeatureConfig,
    top_k: int = 2,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Run resumable two-pass preprocessing and return the processed manifest path."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    if not manifest["records"]:
        raise ValueError("Input manifest contains no records")
    output_dir = Path(output_dir).resolve()
    raw_dir = output_dir / "raw_features"
    sample_dir = output_dir / "samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: list[Path] = []
    for position, record in enumerate(manifest["records"], start=1):
        raw_path = raw_dir / safe_sample_filename(record["sample_id"])
        reuse = False
        if raw_path.exists() and not overwrite:
            cached = torch.load(raw_path, map_location="cpu", weights_only=False)
            reuse = cached.get("feature_config") == config.to_dict()
        if not reuse:
            torch.save(extract_raw_sample(record, manifest_path, config), raw_path)
        raw_paths.append(raw_path)
        if position % 25 == 0 or position == len(manifest["records"]):
            print(f"Raw audio features: {position}/{len(manifest['records'])}", flush=True)

    mean, std, train_node_count = fit_train_normalizer(raw_paths)
    normalization = {
        "fitted_on": "training nodes only",
        "train_node_count": train_node_count,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    (output_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )

    processed_records = []
    for position, (record, raw_path) in enumerate(zip(manifest["records"], raw_paths), start=1):
        raw = torch.load(raw_path, map_location="cpu", weights_only=False)
        x = (torch.as_tensor(raw["x_raw"]).numpy() - mean) / std
        sample_seed = seed + int(hashlib.sha1(record["sample_id"].encode()).hexdigest()[:8], 16)
        edge_indices = build_edge_variants(x, top_k=top_k, seed=sample_seed)
        processed_path = sample_dir / safe_sample_filename(record["sample_id"])
        torch.save(
            {
                "sample_id": record["sample_id"],
                "audio_path": raw["audio_path"],
                "split": record["split"],
                "labels": raw["labels"],
                "x": torch.from_numpy(x.astype(np.float32)),
                "mel": raw["mel"].float(),
                "edge_indices": edge_indices,
                "feature_config": config.to_dict(),
                "top_k": top_k,
                "seed": sample_seed,
            },
            processed_path,
        )
        processed_records.append(
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "processed_path": str(processed_path.relative_to(output_dir)),
            }
        )
        if position % 25 == 0 or position == len(raw_paths):
            print(f"Normalized graph samples: {position}/{len(raw_paths)}", flush=True)

    processed_manifest = {
        "schema_version": 1,
        "source_manifest": str(manifest_path),
        "label_names": manifest["label_names"],
        "feature_config": config.to_dict(),
        "normalization_path": "normalization.json",
        "graph_variants": ["temporal", "temporal_similarity", "random"],
        "top_k": top_k,
        "seed": seed,
        "records": processed_records,
    }
    destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(processed_manifest, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/task2"))
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--n-mfcc", type=int, default=20)
    parser.add_argument("--n-fft", type=int, default=2_048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = AudioFeatureConfig(
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        n_mels=args.n_mels,
        n_mfcc=args.n_mfcc,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )
    destination = preprocess_manifest(
        args.manifest,
        args.output_dir,
        config=config,
        top_k=args.top_k,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Saved processed Task 2 manifest to {destination}")


if __name__ == "__main__":
    main()

