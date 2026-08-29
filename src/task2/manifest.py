"""Build a Task 2 audio/label manifest from MusicCaps metadata and Task 1 labels."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def assign_splits(
    sample_count: int,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> list[str]:
    """Return deterministic split names without touching labels or test metrics."""
    if sample_count < 7:
        raise ValueError("At least seven audio samples are required")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than one")
    indices = list(range(sample_count))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(sample_count * test_size))
    validation_count = max(1, round(sample_count * val_size))
    if test_count + validation_count >= sample_count:
        raise ValueError("Split sizes leave no training samples")
    names = ["train"] * sample_count
    for index in indices[:test_count]:
        names[index] = "test"
    for index in indices[test_count : test_count + validation_count]:
        names[index] = "validation"
    return names


def _audio_candidates(audio_dir: Path, video_id: str, start_s: int, end_s: int) -> list[Path]:
    stems = (
        video_id,
        f"{video_id}_{start_s}",
        f"{video_id}_{start_s}_{end_s}",
        f"{video_id}-{start_s}",
    )
    extensions = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
    return [audio_dir / f"{stem}{extension}" for stem in stems for extension in extensions]


def find_audio(audio_dir: Path, video_id: str, start_s: int, end_s: int) -> Path | None:
    return next(
        (candidate.resolve() for candidate in _audio_candidates(audio_dir, video_id, start_s, end_s) if candidate.is_file()),
        None,
    )


def build_musiccaps_manifest(
    metadata: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    audio_dir: Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    include_missing: bool = False,
) -> dict[str, Any]:
    """Join captions to labels and resolve local audio paths."""
    required_metadata = {"ytid", "start_s", "end_s", "caption"}
    missing = sorted(required_metadata - set(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing columns: {missing}")
    if "text" not in label_frame.columns:
        raise ValueError("Label CSV must contain a 'text' caption column")
    label_names = [column for column in label_frame.columns if column != "text"]
    if not label_names:
        raise ValueError("Label CSV must contain at least one binary label column")
    numeric = label_frame[label_names].apply(pd.to_numeric, errors="raise")
    if not set(np.unique(numeric.to_numpy())).issubset({0, 1}):
        raise ValueError("All label columns must contain only 0 and 1")
    labels = label_frame.copy()
    labels[label_names] = numeric.astype(int)
    if labels["text"].duplicated().any() or metadata["caption"].duplicated().any():
        raise ValueError("Caption join keys must be unique")
    joined = metadata.merge(labels, left_on="caption", right_on="text", how="inner", validate="one_to_one")
    records = []
    missing_audio = []
    for row in joined.to_dict("records"):
        video_id = str(row["ytid"])
        start_s, end_s = int(row["start_s"]), int(row["end_s"])
        audio_path = find_audio(audio_dir, video_id, start_s, end_s)
        sample_id = f"{video_id}_{start_s}_{end_s}"
        if audio_path is None:
            missing_audio.append(sample_id)
            if not include_missing:
                continue
            audio_path = _audio_candidates(audio_dir, video_id, start_s, end_s)[0].resolve()
        records.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "labels": [int(row[label]) for label in label_names],
                "text": str(row["caption"]),
            }
        )
    if records:
        for record, split in zip(
            records,
            assign_splits(len(records), val_size=val_size, test_size=test_size, seed=seed),
        ):
            record["split"] = split
    return {
        "schema_version": 1,
        "dataset": "MusicCaps local audio",
        "seed": seed,
        "label_names": label_names,
        "records": records,
        "audit": {
            "metadata_rows": int(len(metadata)),
            "label_rows": int(len(label_frame)),
            "caption_matches": int(len(joined)),
            "audio_found": len(records) - (len(missing_audio) if include_missing else 0),
            "audio_missing": len(missing_audio),
            "missing_sample_ids": missing_audio,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/splits/task2_manifest.json"))
    parser.add_argument("--missing-log", type=Path, default=Path("results/task2/missing_audio.json"))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()
    manifest = build_musiccaps_manifest(
        pd.read_csv(args.metadata_csv),
        pd.read_csv(args.labels_csv),
        audio_dir=args.audio_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        include_missing=args.include_missing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.missing_log.parent.mkdir(parents=True, exist_ok=True)
    args.missing_log.write_text(
        json.dumps(manifest["audit"]["missing_sample_ids"], indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest["audit"].items() if key != "missing_sample_ids"}, indent=2))
    print(f"Saved {len(manifest['records'])} records to {args.output}")


if __name__ == "__main__":
    main()
