"""
Task 1: BERT Multi-Label Tag Classifier
CSE425/EEE474/CSE715 - GNN-BERT Music Context Project

Fine-tunes a BERT/DistilBERT encoder with a linear + sigmoid head to predict
multi-label music tags (genre / mood / instrument etc.) from text.

Expected input CSV format (adapt `load_data` if your source differs):
    text, tag_1, tag_2, ..., tag_K
    "melancholic piano ballad with soft vocals", 0, 1, 0, ..., 1

NOTE on dataset choice:
  - MagnaTagATune's annotation file only ships binary tag labels, not free
    text. If you use it, you need a text field to classify FROM — e.g. a
    caption, a lyric snippet, or a synthetic description. Using the tag
    names themselves as input text makes the task trivially leaky.
  - MusicCaps ships real captions paired with tags/labels you derive, which
    is the cleaner fit for a *text-input* BERT baseline. Recommended if you
    want a non-degenerate Task 1.
  Either way, this script only cares about a `text` column + K binary label
  columns, so point it at whichever CSV you've built during preprocessing.

Usage:
    python train_bert_tags.py --data_path data/processed/tags_top50.csv \
        --model_name distilbert-base-uncased --epochs 10
"""

import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class Config:
    data_path: str
    model_name: str = "distilbert-base-uncased"
    text_col: str = "text"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    lr_bert: float = 2e-5
    lr_head: float = 1e-3
    val_size: float = 0.15
    test_size: float = 0.15
    freeze_bert: bool = False
    seed: int = 42
    output_dir: str = "results"
    early_stopping_patience: int = 3  # stop if val macro-F1 doesn't improve for N epochs
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(cfg: Config):
    """Loads CSV with a text column + N binary tag columns. Returns texts,
    multi-hot label matrix, and the list of tag names (for reporting)."""
    df = pd.read_csv(cfg.data_path)
    assert cfg.text_col in df.columns, f"'{cfg.text_col}' column not found in {cfg.data_path}"

    tag_cols = [c for c in df.columns if c != cfg.text_col]
    assert len(tag_cols) > 0, "No tag columns found — check your CSV format."

    texts = df[cfg.text_col].astype(str).tolist()
    labels = df[tag_cols].values.astype(np.float32)
    return texts, labels, tag_cols


class TagDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class BertTagClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, freeze_bert: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token — works for BERT/DistilBERT (index 0 of last_hidden_state)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits = self.classifier(cls)  # raw logits, sigmoid applied in loss/eval
        return logits


# --------------------------------------------------------------------------- #
# Train / eval loops
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, cfg, optimizer=None, scheduler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["labels"].to(cfg.device)

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    preds = (1 / (1 + np.exp(-all_logits)) > 0.5).astype(int)  # sigmoid + threshold

    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)

    return avg_loss, macro_f1, micro_f1, all_logits, all_labels


def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    texts, labels, tag_cols = load_data(cfg)
    num_labels = len(tag_cols)
    print(f"Loaded {len(texts)} examples, {num_labels} tags: {tag_cols[:5]}{'...' if num_labels > 5 else ''}")

    # Train / val / test split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, test_size=cfg.val_size + cfg.test_size, random_state=cfg.seed
    )
    rel_test = cfg.test_size / (cfg.val_size + cfg.test_size)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=rel_test, random_state=cfg.seed
    )
    print(f"Split -> train: {len(train_texts)}, val: {len(val_texts)}, test: {len(test_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = TagDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    val_ds = TagDataset(val_texts, val_labels, tokenizer, cfg.max_length)
    test_ds = TagDataset(test_texts, test_labels, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)

    model = BertTagClassifier(cfg.model_name, num_labels, cfg.freeze_bert).to(cfg.device)

    # Separate LR for BERT backbone vs. classifier head
    optimizer = torch.optim.AdamW([
        {"params": model.bert.parameters(), "lr": cfg.lr_bert},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ])
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_macro_f1": [],
               "val_macro_f1": [], "val_micro_f1": []}

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_macro_f1, _, _, _ = run_epoch(model, train_loader, cfg, optimizer, scheduler)
        val_loss, val_macro_f1, val_micro_f1, _, _ = run_epoch(model, val_loader, cfg)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_macro_f1"].append(train_macro_f1)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)

        print(f"Epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} "
              f"| train_macroF1 {train_macro_f1:.4f} | val_macroF1 {val_macro_f1:.4f} "
              f"| val_microF1 {val_micro_f1:.4f}")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, "best_model.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                print(f"No val macro-F1 improvement for {cfg.early_stopping_patience} epochs — "
                      f"stopping early at epoch {epoch} (saves time on a tight schedule).")
                break

    # Final test evaluation with best checkpoint
    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, "best_model.pt")))
    test_loss, test_macro_f1, test_micro_f1, test_logits, test_labels_arr = run_epoch(model, test_loader, cfg)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels_arr, (1 / (1 + np.exp(-test_logits)) > 0.5).astype(int),
        average="macro", zero_division=0
    )
    print(f"\nTEST | loss {test_loss:.4f} | macroF1 {test_macro_f1:.4f} | microF1 {test_micro_f1:.4f} "
          f"| precision {precision:.4f} | recall {recall:.4f}")

    # Save metrics
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "history": history,
            "test_macro_f1": test_macro_f1,
            "test_micro_f1": test_micro_f1,
            "test_precision": precision,
            "test_recall": recall,
        }, f, indent=2)

    # F1-vs-epoch plot (explicit deliverable)
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history["train_macro_f1"], label="Train Macro-F1")
    plt.plot(history["epoch"], history["val_macro_f1"], label="Val Macro-F1")
    plt.plot(history["epoch"], history["val_micro_f1"], label="Val Micro-F1", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.title("Task 1: BERT Tag Classifier — F1 vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "f1_curve.png"), dpi=150)
    print(f"Saved F1 curve to {cfg.output_dir}/f1_curve.png")

    # 5 example predictions (explicit deliverable)
    sigmoid_probs = 1 / (1 + np.exp(-test_logits))
    examples = []
    for i in range(min(5, len(test_texts))):
        top_idx = np.argsort(-sigmoid_probs[i])[:5]
        pred_tags = [(tag_cols[j], round(float(sigmoid_probs[i][j]), 3)) for j in top_idx]
        true_tags = [tag_cols[j] for j in range(num_labels) if test_labels_arr[i][j] == 1]
        examples.append({"text": test_texts[i], "top_predicted": pred_tags, "true_tags": true_tags})

    with open(os.path.join(cfg.output_dir, "example_predictions.json"), "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Saved 5 example predictions to {cfg.output_dir}/example_predictions.json")

    return model, history


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_bert", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    print(f"Using device: {cfg.device}")
    train(cfg)


if __name__ == "__main__":
    main()
