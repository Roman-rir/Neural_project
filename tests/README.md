# Tests

Run the offline test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

`tests/task1/test_pipeline.py` checks proxy-label extraction, the preserved 5,299 x 50
MusicCaps artifact, deterministic non-overlapping splits, metric calculations,
and consistency between the legacy checkpoint metadata and saved deliverables.

`tests/task1/test_audit.py` checks MusicCaps field parsing, label normalization,
duplicate/overlap diagnostics, and availability-gate handling when platform
errors make a probe inconclusive.

`tests/task1/test_academic.py` checks ID grouping and order-invariant splits,
held-out label perturbations, caption-independent targets, training-only TF-IDF
vocabulary, prepared-manifest hashes and split integrity, validation calibration,
and checkpoint inference with per-label thresholds. `test_training_safety.py`
covers configuration, frozen encoders and AP edge cases. `test_notebook.py`
checks the independent-label notebook workflow and legacy/current example support.

For the completed real-data run, recompute all metrics and verify IDs, targets,
and validation-derived thresholds without retraining:

```bash
python -m src.task1.evaluate --run-dir results/task1/audioset_cpu_20260907
```

`tests/task2/` checks segment-feature shapes, finite values, deterministic and
connected graph variants, all model output shapes, and a complete synthetic
audio -> preprocessing -> GraphSAGE -> checkpoint/predictions smoke run.

Future tasks should add tests for audio feature shapes, temporal graph edges,
model output dimensions, and one-batch smoke tests for every model mode.
