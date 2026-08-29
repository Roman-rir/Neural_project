# GNN-BERT Music Context Understanding

Course project implementation for music-context prediction using audio structure graphs and text. The project supports multi-label genre/mood/tag classification, optional valence-arousal regression, and optional caption-audio retrieval.

> **Results policy:** all metric and plot directories start empty. Add only results produced by your own experiments.

## Pipeline

```text
                  MUSIC
                    |
          +---------+---------+
          |                   |
        AUDIO                TEXT
          |                   |
    Feature Extraction       BERT
          |                   |
     Graph Construction       |
          |                   |
         GNN                  |
          |                   |
          +---------+---------+
                    |
                  FUSION
                    |
                    v
          Music Context Prediction
          genre | mood | tags | emotion
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Place audio in `data/raw/`, then build a graph:

```bash
python -m src.task2.graphs --audio data/raw/example.wav --output data/processed/example.pt
```

Each processed sample is a PyTorch dictionary containing `graph`, `text`, `labels`, and optionally `emotion`. Create JSON split files that list sample paths, then run:

```bash
python -m src.train --config config.yaml --train-split data/splits/train.json --val-split data/splits/val.json
python -m src.evaluate --config config.yaml --checkpoint results/checkpoints/best.pt --split data/splits/test.json
```

## Milestones

- Task 1: text-only BERT baseline (`--model bert`)
- Task 2: GraphSAGE audio-graph baseline (`--model gnn`)
- Task 3: GNN-BERT fusion (`--model fusion`)
- Task 4: optional contrastive retrieval (`src/contrastive.py`)

## Task 1: completed BERT baseline

Task 1 has a standalone, reproducible MusicCaps caption-to-tag pipeline in
addition to the shared multi-modal training code:

Open `notebooks/task1_bert_baseline.ipynb` for the notebook-first walkthrough,
measured artifacts, architecture, inference, and optional retraining.

```bash
python -m src.task1.data --hf-dataset --top-k 50 --output data/processed/musiccaps_tags.csv
python -m src.task1.train --data-path data/processed/musiccaps_tags.csv --output-dir results/task1
python -m src.task1.predict --checkpoint results/task1/best_model.pt --text "calm acoustic guitar with soft vocals"
```

The original completed experiment is preserved in `Task1_test/`, including its
checkpoint, F1 curve, metrics, and five examples. The current trainer saves the
exact split indices, model/tag metadata, per-tag metrics, and an inference-ready
checkpoint for future runs.

The MusicCaps labels used here are lexical proxies derived from the input
caption itself. This is useful as a Task 1 pipeline baseline, but it creates
target leakage and must not be interpreted as performance on independent music
annotations. See `docs/task1/README.md` for the architecture and complete run
guide, and `report/task1_results.md` for the result table and limitations.

## Part 1: dataset viability audit

The planning roadmap's Part 1 audit is implemented separately from the model
task above. Reproduce the metadata audit and deterministic 200-ID, no-download
availability pilot with:

```bash
python -m src.task1.audit --hf-dataset --probe-count 200 --seed 42
```

The signed-off evidence is in `results/data_audit.json`, with the scope decision
and remaining instructor-approval gate in `docs/scope.md`. Availability probing
requires `yt-dlp` and is sensitive to time, region, network, and platform
throttling; technical errors are reported separately from unavailable clips.

## Task 2: graph and CNN audio baselines

Task 2 is notebook-first. Open `notebooks/task2_gnn_cnn.ipynb` in VS Code or
Jupyter and run all cells. It defaults to a synthetic smoke dataset so the full
workflow can be verified without downloading audio. Switch `USE_SYNTHETIC` to
`False` only after placing permitted MusicCaps audio in
`data/raw/musiccaps_audio/`.

Reusable Task 2 code is consolidated under `src/task2/` and includes manifest
creation, train-only feature normalization, temporal/top-k/random graph
variants, pooled MLP, compact mel-CNN, GraphSAGE, checkpointing, predictions,
metrics, and evaluation. See `docs/task2/README.md` for the architecture and
real-data workflow.

## Collaboration

Suggested branches: `member1/audio-graph`, `member2/bert`, and `member3/gnn-fusion`. Protect `main`; merge changes through pull requests after a reproducibility check.

## Data and ethics

Do not commit copyrighted dataset audio, credentials, or generated caches. Respect each dataset licence and document all preprocessing and split decisions to avoid artist leakage.
