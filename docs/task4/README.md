# Task 4: graph–text retrieval

Task 4 trains two linear projection heads over frozen BERT CLS embeddings and
GraphSAGE graph embeddings. L2-normalized projections share a space, and symmetric
InfoNCE trains each caption to retrieve its paired graph and each graph to retrieve
its caption. Other examples in the batch are negatives. Labels are not used in
the loss. Temperature defaults to 0.07.

## Terminal demo

Run from the project root in your Python environment:

```powershell
python -m src.task4 train --demo --epochs 5 --device cpu --output-dir tmp/task4_demo
python -m src.task4 evaluate --checkpoint tmp/task4_demo/best_model.pt --features tmp/task4_demo.demo.pt --output-dir tmp/task4_demo/test --device cpu
```

Use fresh output paths on subsequent runs. The demonstration uses the same random
feature fixture as Task 3. It checks execution only; it is not trained on real audio
or real BERT features. Its independent random modalities may not generalize.

## Real paired features

Follow `docs/task3/README.md` to prepare aligned encoder features, then run:

```powershell
python -m src.task4 train --features data/processed/task3/features.pt --epochs 30 --batch-size 32 --projection-dim 64 --patience 5 --output-dir results/task4/seed42 --device auto
python -m src.task4 evaluate --checkpoint results/task4/seed42/best_model.pt --features data/processed/task3/features.pt --output-dir results/task4/seed42/test --device auto
```

Training sees only training pairs. Checkpoint selection and early stopping use
mean bidirectional validation Recall@1. Test evaluation is a separate command
and reuses the selected checkpoint. Feature SHA-256 verification prevents mixing
a checkpoint and a changed feature cache. The implementation supports retrieval
within cached split galleries, not encoding a new raw audio/text query.

## Metrics and artifacts

Each evaluation uses every pair in the chosen split as both query and gallery,
with one positive per unique sample ID. It reports Recall@1/5/10 in both directions,
median ranks, gallery size, and effective K (clipped when the gallery has fewer
than K items). Small synthetic galleries make R@10 trivial. Metrics rank tied
negatives ahead of the positive to avoid optimistic ID-order artifacts. Example
lists use stable gallery-order sorting for ties and include similarity scores.

Query scoring is blocked to avoid retaining an entire N-by-N matrix. Frozen
feature loading still requires the complete Task 3 cache in CPU memory. Singleton
training batches are merged into the preceding batch because InfoNCE needs negatives.

Training saves `best_model.pt`, `config.json`, `history.json`,
`learning_curves.png`, and `validation/`. Standalone evaluation saves
`metrics.json`, `embeddings.npz` (including sample IDs), and `examples.json`
(three queries per direction). Saved embeddings allow independent recomputation.

New runs also save `initial_validation.json` and the exact initial projection
weights. Evaluations report `untrained_projection_metrics` and the trained-minus-
untrained `mean_recall_at_1_gain`; negative gains are retained. Curves show the
untrained baseline and analytical random-ranking expectation (K / gallery size,
clipped at 1). These baselines do not substitute for the planned CLAP comparison.
Old checkpoints still evaluate, but cannot provide their missing initial weights.

Metrics now include mean reciprocal rank in both directions. Examples include
`paired_rank` with the same conservative tie policy as aggregate metrics.
Recompute trained or baseline metrics without retraining or loading encoders:

```powershell
python -m src.task4.metrics --embeddings tmp/task4_demo/test/embeddings.npz
python -m src.task4.metrics --embeddings tmp/task4_demo/test/untrained_embeddings.npz
```

The verifier checks sample-ID uniqueness and uses the saved row alignment. It
cannot detect external edits that reorder only one modality while preserving IDs.

The notebook `notebooks/task4_retrieval.ipynb` runs the same APIs.

## Remaining experimental work

Real-audio retrieval, a CLAP zero-shot comparison, multiple seeds, and human
listening evaluation have not been measured. They remain necessary for the full
planned experimental comparison. Similar or duplicate captions may create false
negatives under the one-positive objective. Audit track grouping and caption
duplicates before reporting results. Classification-trained encoders and their
proxy-label limitations carry over from Task 3; retrieval scores do not remove
those limitations. No model downloads or listening judgments are fabricated.
