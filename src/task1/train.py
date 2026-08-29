"""Train and evaluate the Task 1 BERT multi-label music-tag baseline."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class Task1Config:
    data_path: str
    output_dir: str = "results/task1"
    model_name: str = "distilbert-base-uncased"
    text_col: str = "text"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    lr_bert: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-5
    val_size: float = 0.15
    test_size: float = 0.15
    freeze_bert: bool = False
    patience: int = 3
    threshold: float = 0.5
    seed: int = 42
    device: str = "auto"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tag_data(path: str | Path, text_col: str = "text") -> tuple[list[str], np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    if text_col not in frame.columns:
        raise ValueError(f"Text column {text_col!r} is missing from {path}")
    tag_cols = [name for name in frame.columns if name != text_col]
    if not tag_cols:
        raise ValueError("The CSV must contain at least one binary tag column")
    numeric = frame[tag_cols].apply(pd.to_numeric, errors="raise")
    values = set(np.unique(numeric.to_numpy()).tolist())
    if not values.issubset({0, 1}):
        raise ValueError(f"Tag columns must be binary 0/1; found {sorted(values)}")
    texts = frame[text_col].fillna("").astype(str).str.strip()
    if (texts.str.len() == 0).any():
        raise ValueError("Text inputs must be non-empty")
    if texts.duplicated().any():
        raise ValueError("Duplicate captions detected; remove or group them before splitting")
    return texts.tolist(), numeric.to_numpy(dtype=np.float32), tag_cols


def make_split_indices(
    sample_count: int,
    *,
    val_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[int]]:
    if sample_count < 7:
        raise ValueError("At least seven samples are required for train/validation/test splits")
    held_out = val_size + test_size
    if val_size <= 0 or test_size <= 0 or held_out >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")
    indices = np.arange(sample_count)
    train_idx, held_idx = train_test_split(indices, test_size=held_out, random_state=seed)
    relative_test_size = test_size / held_out
    val_idx, test_idx = train_test_split(
        held_idx, test_size=relative_test_size, random_state=seed
    )
    result = {
        "train": train_idx.tolist(),
        "validation": val_idx.tolist(),
        "test": test_idx.tolist(),
    }
    flat = [item for split in result.values() for item in split]
    if len(flat) != sample_count or len(set(flat)) != sample_count:
        raise RuntimeError("Generated splits are not disjoint and exhaustive")
    return result


class TagDataset(Dataset):
    def __init__(self, texts, labels, indices, tokenizer, max_length: int) -> None:
        self.texts = [texts[index] for index in indices]
        self.labels = labels[np.asarray(indices)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {name: value.squeeze(0) for name, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item


class BertTagClassifier(nn.Module):
    """CLS pooling followed by dropout and one logit per tag."""

    def __init__(self, model_name: str, num_labels: int, freeze_bert: bool = False) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        if freeze_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.bert(**kwargs).last_hidden_state[:, 0]
        return self.classifier(self.dropout(hidden))


def evaluate_epoch(model, loader, device, threshold: float, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_probabilities, all_targets = [], []
    for batch in loader:
        targets = batch.pop("labels").to(device)
        inputs = {name: value.to(device) for name, value in batch.items()}
        with torch.set_grad_enabled(training):
            logits = model(**inputs)
            loss = loss_fn(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        total_loss += loss.item() * len(targets)
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
    probabilities = np.concatenate(all_probabilities)
    targets = np.concatenate(all_targets).astype(int)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "probabilities": probabilities,
        "targets": targets,
    }


def classification_report(targets, probabilities, tags, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    auc_pr = []
    for column in range(len(tags)):
        if len(np.unique(targets[:, column])) < 2:
            auc_pr.append(None)
        else:
            auc_pr.append(float(average_precision_score(targets[:, column], probabilities[:, column])))
    valid_auc_pr = [value for value in auc_pr if value is not None]
    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "mean_auc_pr": float(np.mean(valid_auc_pr)) if valid_auc_pr else None,
        "threshold": threshold,
        "samples": int(len(targets)),
        "per_tag": {
            tag: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auc_pr": auc_pr[i],
            }
            for i, tag in enumerate(tags)
        },
    }


def save_curve(history: list[dict], destination: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, [row["train_macro_f1"] for row in history], label="Train Macro-F1")
    plt.plot(epochs, [row["val_macro_f1"] for row in history], label="Validation Macro-F1")
    plt.plot(
        epochs,
        [row["val_micro_f1"] for row in history],
        "--",
        label="Validation Micro-F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Task 1: BERT Tag Classifier")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=150)
    plt.close()


def train(config: Task1Config) -> dict:
    set_seed(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels, tags = load_tag_data(config.data_path, config.text_col)
    splits = make_split_indices(
        len(texts), val_size=config.val_size, test_size=config.test_size, seed=config.seed
    )
    (output_dir / "split_indices.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    datasets = {
        name: TagDataset(texts, labels, indices, tokenizer, config.max_length)
        for name, indices in splits.items()
    }
    generator = torch.Generator().manual_seed(config.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=config.batch_size, shuffle=True, generator=generator
        ),
        "validation": DataLoader(datasets["validation"], batch_size=config.batch_size),
        "test": DataLoader(datasets["test"], batch_size=config.batch_size),
    }

    model = BertTagClassifier(config.model_name, len(tags), config.freeze_bert).to(device)
    parameter_groups = [
        {"params": [p for p in model.bert.parameters() if p.requires_grad], "lr": config.lr_bert},
        {"params": list(model.classifier.parameters()), "lr": config.lr_head},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=config.weight_decay)
    total_steps = len(loaders["train"]) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    history: list[dict] = []
    best_f1 = -1.0
    stale_epochs = 0
    checkpoint_path = output_dir / "best_model.pt"
    for epoch in range(1, config.epochs + 1):
        train_metrics = evaluate_epoch(
            model, loaders["train"], device, config.threshold, optimizer, scheduler
        )
        val_metrics = evaluate_epoch(model, loaders["validation"], device, config.threshold)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_micro_f1": train_metrics["micro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_micro_f1": val_metrics["micro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f} "
            f"val_micro_f1={row['val_micro_f1']:.4f}"
        )
        if row["val_macro_f1"] > best_f1:
            best_f1 = row["val_macro_f1"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(config),
                    "tags": tags,
                    "best_epoch": epoch,
                    "best_val_macro_f1": best_f1,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                print(f"Early stopping after {epoch} epochs")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    test_epoch = evaluate_epoch(model, loaders["test"], device, config.threshold)
    test_report = classification_report(
        test_epoch["targets"], test_epoch["probabilities"], tags, config.threshold
    )
    test_report["loss"] = test_epoch["loss"]
    metrics = {
        "dataset": {
            "path": config.data_path,
            "samples": len(texts),
            "num_tags": len(tags),
            "proxy_label_warning": (
                "Labels are deterministic keyword matches from the same input captions; "
                "scores measure recovery of lexical proxy rules, not independent semantic annotation."
            ),
        },
        "config": asdict(config),
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "history": history,
        "test": test_report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_curve(history, output_dir / "f1_curve.png")

    test_texts = datasets["test"].texts
    examples = []
    for index in range(min(5, len(test_texts))):
        probabilities = test_epoch["probabilities"][index]
        top_indices = np.argsort(-probabilities)[:5]
        examples.append(
            {
                "text": test_texts[index],
                "top_predicted": [
                    {"tag": tags[i], "probability": round(float(probabilities[i]), 4)}
                    for i in top_indices
                ],
                "true_tags": [tags[i] for i, value in enumerate(test_epoch["targets"][index]) if value],
            }
        )
    (output_dir / "example_predictions.json").write_text(
        json.dumps(examples, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in test_report.items() if key != "per_tag"}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="results/task1")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr-bert", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--freeze-bert", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    train(Task1Config(**vars(args)))


if __name__ == "__main__":
    main()
