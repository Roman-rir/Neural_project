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

Task 4 retrieval is available in `src/task4/` and
[`notebooks/task4_retrieval.ipynb`](notebooks/task4_retrieval.ipynb).
See [`docs/task4/README.md`](docs/task4/README.md) for training and evaluation.

Task 3's standalone frozen-encoder implementation is available in
[`notebooks/task3_fusion.ipynb`](notebooks/task3_fusion.ipynb), with terminal and
real-data instructions in [`docs/task3/README.md`](docs/task3/README.md).
Run an offline five-ablation check with:

```powershell
python -m src.task3 --demo --epochs 2 --device cpu --output-dir tmp/task3_demo
```

- Task 1: text-only BERT baseline (`--model bert`)
- Task 2: GraphSAGE audio-graph baseline (`--model gnn`)
- Task 3: GNN-BERT fusion (`--model fusion`)
- Task 4: optional contrastive retrieval (`src/contrastive.py`)

## Task 1: independent AudioSet text baseline

Task 1 now preserves MusicCaps `ytid`, splits by video ID before label selection,
fits its 30-label AudioSet vocabulary on training rows only, and tunes thresholds
on validation only. The executed
[`notebooks/task1_bert_baseline.ipynb`](notebooks/task1_bert_baseline.ipynb) reviews
the completed run, controls, per-label evidence, inference and optional retraining.

CPU fine-tuning of DistilBERT for two epochs on 3,864 training captions achieved
held-out **Macro-F1 0.3840, Micro-F1 0.5902 and mAP 0.3686** on 829 test IDs.
TF-IDF reached 0.3393 / 0.6530 / 0.3603 respectively. Saved predictions reproduce
the metrics; all models share the same prepared vocabulary and ID split.

```bash
python -m src.task1.data --hf-dataset --top-k 30 --output data/processed/musiccaps_audioset_new.csv
python -m src.task1.train --data-path data/processed/musiccaps_audioset_new.csv --output-dir results/task1/audioset_new --epochs 2 --device cpu --cpu-threads 6
python -m src.task1.evaluate --run-dir results/task1/audioset_new
```

Use fresh paths; preparation and training preserve existing artifacts. Independent
AudioSet labels replace caption-derived proxy targets for this run. The original
work remains in `Task1_test/`, earlier `results/task1/` outputs,
[`notebooks/task1_lexical_proxy.ipynb`](notebooks/task1_lexical_proxy.ipynb), and
[`docs/task1/legacy_proxy.md`](docs/task1/legacy_proxy.md).
See [the run guide](docs/task1/README.md) and [measured report](report/task1_results.md)
for methods, limitations and reproduction. Task 2's real-data manifest now
inherits this vocabulary and video split; real audio/fusion models still need
training on the available-audio cohort.

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

Task 2 now has an aligned AudioSet manifest, audio-interval loading, cached
features, connected graph validation, a 20-example gallery, and one controlled
suite for GraphSAGE (temporal/similarity/random), pooled MLP and mel-CNN.
Checkpoints are bound to identical cached samples, labels and splits.

The end-to-end synthetic verification includes 30 audio clips, 90 checked graphs,
20 visualizations, five checkpoints, curves, prediction files and ablation
tables. **Real-data Task 2 is not complete:** no matching MusicCaps audio exists
in the project. Synthetic scores do not measure music-context performance.

Open [`notebooks/task2_gnn_cnn.ipynb`](notebooks/task2_gnn_cnn.ipynb) to review the
saved verification, or reproduce it in a fresh directory:

```powershell
python -m src.task2.experiment --demo --output-dir tmp/task2_demo_new --epochs 3 --hidden-dim 32 --batch-size 6 --device cpu --evaluate-test
```

See [the updated guide](docs/task2/README.md) for real-data commands and
[the completion audit](report/task2_results.md) for evidence and remaining work.

## Collaboration

Suggested branches: `member1/audio-graph`, `member2/bert`, and `member3/gnn-fusion`. Protect `main`; merge changes through pull requests after a reproducibility check.

## Data and ethics

Do not commit copyrighted dataset audio, credentials, or generated caches. Respect each dataset licence and document all preprocessing and split decisions to avoid artist leakage.
