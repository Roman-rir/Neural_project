"""Extract aligned frozen BERT tokens and GraphSAGE embeddings from trained checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.task1.predict import load_checkpoint
from src.task1.train import BertTagClassifier, load_tag_data
from src.task2.dataset import ProcessedTask2Dataset, load_manifest
from src.task2.models import build_task2_model
from src.task2.train import resolve_device


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_features(processed_manifest, source_manifest, text_checkpoint, graph_checkpoint,
                     output, device="auto"):
    processed_manifest, source_manifest = Path(processed_manifest), Path(source_manifest)
    processed, source = load_manifest(processed_manifest), load_manifest(source_manifest)
    if processed["label_names"] != source["label_names"]:
        raise ValueError("Source and processed label orders differ")
    source_rows = {r["sample_id"]: r for r in source["records"]}
    if set(source_rows) != {r["sample_id"] for r in processed["records"]}:
        raise ValueError("Source and processed sample IDs differ")
    for record in processed["records"]:
        raw = source_rows[record["sample_id"]]
        if record["split"] != raw["split"] or not raw.get("text", "").strip():
            raise ValueError("Each graph requires a caption and matching split")

    text_saved = torch.load(text_checkpoint, map_location="cpu", weights_only=True)
    graph_saved = torch.load(graph_checkpoint, map_location="cpu", weights_only=True)
    if "config" not in text_saved:
        raise ValueError("Task 3 requires a metadata-rich Task 1 checkpoint with split provenance")
    labels = processed["label_names"]
    if text_saved["tags"] != labels or graph_saved["label_names"] != labels:
        raise ValueError("Checkpoint label names/order must match the paired manifest")
    gc = graph_saved["config"]
    if gc["model"] != "gnn":
        raise ValueError("The audio checkpoint must be a trained GraphSAGE model")
    # Every frozen encoder must have trained only on this experiment's training partition.
    graph_training = load_manifest(gc["manifest"])
    old_splits = {r["sample_id"]: r["split"] for r in graph_training["records"]}
    if old_splits != {r["sample_id"]: r["split"] for r in processed["records"]}:
        raise ValueError("Graph checkpoint split assignments differ; retrain on the aligned split")
    text_splits_path = Path(text_checkpoint).with_name("split_indices.json")
    text_splits = json.loads(text_splits_path.read_text(encoding="utf-8"))
    tc = text_saved["config"]
    texts, _, _ = load_tag_data(tc["data_path"], tc.get("text_col", "text"))
    training_texts = {texts[i] for i in text_splits["train"]}
    heldout_texts = {r["text"].strip() for r in source["records"] if r["split"] != "train"}
    if training_texts & heldout_texts:
        raise ValueError("Task 1 encoder trained on Task 3 held-out captions; use aligned encoder splits")
    old_text_splits = {texts[i]: split for split, indices in text_splits.items() for i in indices}
    if any(old_text_splits.get(r["text"].strip()) != r["split"] for r in source["records"]):
        raise ValueError("Task 1 caption split assignments differ; retrain on the aligned split")
    if processed.get("feature_config") != graph_training.get("feature_config"):
        raise ValueError("Graph preprocessing feature configurations differ")
    # Same normalization is required, not merely the same feature dimensions.
    for manifest_path, manifest in ((processed_manifest, processed), (Path(gc["manifest"]), graph_training)):
        if "normalization_path" not in manifest:
            raise ValueError("Processed manifests must reference training normalization")
    if file_hash(processed_manifest.parent / processed["normalization_path"]) != file_hash(
        Path(gc["manifest"]).parent / graph_training["normalization_path"]
    ):
        raise ValueError("Graph normalization differs from the encoder training run")

    device = resolve_device(device)
    state, metadata = load_checkpoint(Path(text_checkpoint))
    tokenizer = AutoTokenizer.from_pretrained(metadata["model_name"])
    text_model = BertTagClassifier(metadata["model_name"], len(labels), freeze_bert=True).to(device)
    text_model.load_state_dict(state); text_model.eval()
    graph_model = build_task2_model("gnn", input_dim=graph_saved["input_dim"], num_labels=len(labels),
                                    hidden_dim=gc["hidden_dim"], layers=gc["layers"], dropout=gc["dropout"]).to(device)
    graph_model.load_state_dict(graph_saved["state_dict"]); graph_model.eval()
    result = {k: [] for k in ("tokens", "mask", "graph", "labels", "sample_ids", "splits", "texts")}
    with torch.no_grad():
        for split in ("train", "validation", "test"):
            dataset = ProcessedTask2Dataset(processed_manifest, split, gc["graph_variant"])
            for sample in dataset:
                row = source_rows[sample["sample_id"]]
                if sample["labels"].tolist() != row["labels"]:
                    raise ValueError("Cached labels differ from source; rerun Task 2 preprocessing")
                encoded = tokenizer(row["text"], truncation=True, padding="max_length",
                                    max_length=metadata["max_length"], return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                tokens = text_model.bert(**encoded).last_hidden_state[0].cpu()
                result["tokens"].append(tokens)
                result["mask"].append(encoded["attention_mask"][0].cpu())
                result["graph"].append(graph_model.encoder(sample["graph"].to(device))[0].cpu())
                result["labels"].append(sample["labels"])
                for key, value in (("sample_ids", row["sample_id"]), ("splits", split), ("texts", row["text"])):
                    result[key].append(value)
    for key in ("tokens", "mask", "graph", "labels"):
        result[key] = torch.stack(result[key])
    result["label_names"] = labels
    result["provenance"] = dict(synthetic=False, text_checkpoint_sha256=file_hash(text_checkpoint),
                                graph_checkpoint_sha256=file_hash(graph_checkpoint),
                                processed_manifest_sha256=file_hash(processed_manifest),
                                source_manifest_sha256=file_hash(source_manifest),
                                warning="Caption-derived proxy labels can leak targets to the text branch")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("processed-manifest", "source-manifest", "text-checkpoint", "graph-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    print(prepare_features(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
