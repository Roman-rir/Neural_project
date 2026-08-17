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
python -m src.graph_builder --audio data/raw/example.wav --output data/processed/example.pt
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

## Collaboration

Suggested branches: `member1/audio-graph`, `member2/bert`, and `member3/gnn-fusion`. Protect `main`; merge changes through pull requests after a reproducibility check.

## Data and ethics

Do not commit copyrighted dataset audio, credentials, or generated caches. Respect each dataset licence and document all preprocessing and split decisions to avoid artist leakage.
