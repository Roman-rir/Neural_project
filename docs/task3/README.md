# Task 3: GNN–BERT fusion

This stage compares five heads on identical frozen encoder outputs: BERT-only,
GNN-only, concatenation, gated fusion, and graph-query-to-text cross-attention.
The unimodal heads are retrained controls on the paired split, not the historical
Task 1/2 scores. This implementation covers frozen-encoder fusion; encoder
fine-tuning is a later experiment.

## Run an offline terminal demonstration

From the project root with the project's Python environment active:

```powershell
python -m src.task3 --demo --epochs 2 --device cpu --output-dir tmp/task3_demo
```

Use a fresh output directory for each invocation. This generates random paired
token/graph features and three synthetic targets. It requires no audio, model
download, or GPU. Its scores are pipeline checks, never project results.
The notebook `notebooks/task3_fusion.ipynb` runs the same workflow.

## Architecture

```text
Same clip ID and split
  caption -> frozen BERT -> token embeddings [T,D] + attention mask
  graph   -> frozen GraphSAGE -> mean/max graph embedding [G]
                         |
             projections to a common hidden width
                         |
   text | graph | concat | sigmoid gate | graph-query token attention
                         |
             linear multi-label head -> BCE loss
                         |
        validation threshold and checkpoint selection
```

Cross-attention uses the graph embedding as one query and BERT tokens as keys
and values, masking padding tokens. It adds the attended result to the graph
projection. Attention is not a causal explanation.

## Prepare real aligned encoders

First complete Task 2 real-audio manifest creation and preprocessing. Local audio
is required. Both encoders must train on the same paired partition. Historical
independently split checkpoints may overlap Task 3 held-out samples and are
rejected by the preparation checks.

```powershell
python -m src.task3.align --manifest data/splits/task2_manifest.json --output-dir data/processed/task3_text
python -m src.task1.train --data-path data/processed/task3_text/tags.csv --split-path data/processed/task3_text/split_indices.json --output-dir results/task3_encoders/bert --epochs 2 --device auto
python -m src.task2.train --manifest data/processed/task2/manifest.json --output-dir results/task3_encoders/gnn --model gnn --graph-variant temporal_similarity --epochs 30 --skip-test --device auto
python -m src.task3.prepare --processed-manifest data/processed/task2/manifest.json --source-manifest data/splits/task2_manifest.json --text-checkpoint results/task3_encoders/bert/best_model.pt --graph-checkpoint results/task3_encoders/gnn/best_model.pt --output data/processed/task3/features.pt --device auto
python -m src.task3 --features data/processed/task3/features.pt --epochs 20 --output-dir results/task3/seed42 --seed 42 --device auto
```

Task 1 currently computes its test report at the end of encoder training; do not
use that report to select fusion settings. The fusion runner defaults to
validation-only reporting. Once the configuration is fixed, evaluate the selected
checkpoint without retraining (PowerShell):

```powershell
$selection = Get-Content results/task3/seed42/selection.json -Raw | ConvertFrom-Json
python -m src.task3.evaluate --checkpoint "results/task3/seed42/$($selection.checkpoint)" --features data/processed/task3/features.pt --output-dir results/task3/seed42/final_test --device auto
```

Evaluation reuses the saved threshold and rejects a changed feature file or
label order. Older Task 3 checkpoints without a feature hash require a new run.
`--evaluate-test` still supports training followed immediately by a report for
the validation winner. Keep hyperparameters fixed and repeat
with seeds 42, 43, and 44 for the planned seed comparison.

Preparation verifies IDs, captions, labels, splits, graph normalization, and
checkpoint label order. It records checkpoint and manifest SHA-256 hashes.
Keep source training CSVs, manifests, split indices, and normalization files
with the encoder checkpoints. These checks assume those provenance files have
not been edited since encoder training. The cached token tensor can require
several GB for a full MusicCaps run; preparation and training currently keep it
in CPU memory and transfer batches to the selected device.

## Outputs and interpretation

- Five checkpoints and per-epoch history files.
- `run_config.json`: training settings, resolved device, and feature SHA-256.
- `selection.json`: validation-selected checkpoint for standalone evaluation.
- `comparison.csv`: validation Macro/Micro-F1, mAP, runtime and head parameters.
- `validation_curves.png`: measured validation learning curves.
- `selected_metrics.json`: winning mode, evaluated split, metrics and provenance.
- `selected_predictions.npz`: IDs, targets, probabilities and embeddings.
- `embeddings.png`: descriptive t-SNE, colored by positive-label count.
- `cases.json`: first three evaluated examples, their targets and predictions;
  inspect mismatches as failure cases without cherry-picking.

Parameter counts include the active fusion-head modules but exclude frozen
encoders. Fusion training time excludes
encoder extraction. No real-data fusion results have been measured yet. Existing
caption-derived proxy targets still leak lexical label information into BERT;
independent annotations are required for defensible semantic performance claims.

Training supports `--hidden-dim`, `--learning-rate`, and `--patience` (default 5).
It stops a mode after that many epochs without a validation Macro-F1 improvement.
Training subsets reference the feature tensors rather than copying the entire
training partition. The complete cache still needs to fit in CPU memory.

```powershell
python -m unittest discover -s tests -v
```
