# Task 1 experiment artifacts

This directory preserves the original Task 1 experiment and its outputs. The
submission-ready implementation now lives in:

- `src/task1/data.py` - MusicCaps caption-to-tag proxy preprocessing
- `src/task1/train.py` - reproducible BERT training, evaluation, plots, and examples
- `src/task1/predict.py` - inference for new and legacy checkpoints

## Existing run

The saved run used `bert-base-uncased`, 5,299 MusicCaps captions, 50 lexical
proxy tags, a 70/15/15 random split, seed 42, and threshold 0.5. Its reported
test scores are Macro-F1 0.9894 and Micro-F1 0.9969. See `report/task1_results.md`
for the complete interpretation and limitation statement.

The original `best_model.pt` is a raw state dictionary. The adjacent
`model_metadata.json` records its architecture and label order so it can be
loaded by the current inference command:

```powershell
python -m src.task1.predict `
  --checkpoint Task1_test/files/results/best_model.pt `
  --text "An energetic rock track with electric guitar, bass and drums."
```

## Reproduce a new run

```powershell
python -m src.task1.data --hf-dataset --top-k 50 `
  --output data/processed/musiccaps_tags.csv

python -m src.task1.train `
  --data-path data/processed/musiccaps_tags.csv `
  --model-name distilbert-base-uncased `
  --output-dir results/task1
```

New checkpoints include model configuration and tag names internally. Each run
also saves the exact split indices, per-tag metrics, Macro/Micro-F1 history, the
F1 curve, and five example predictions.

## Important limitation

The tag columns are created by keyword matching on the same captions that BERT
receives as input. This is an accepted MusicCaps proxy-task format in the brief,
but it creates direct lexical target leakage and explains the near-perfect
score. Do not present this result as performance on independently annotated
music semantics. A stronger follow-up should derive labels from an independent
source or evaluate on held-out paraphrases with tag words removed.
