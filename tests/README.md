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

`tests/task2/` checks segment-feature shapes, finite values, deterministic and
connected graph variants, all model output shapes, and a complete synthetic
audio -> preprocessing -> GraphSAGE -> checkpoint/predictions smoke run.

Future tasks should add tests for audio feature shapes, temporal graph edges,
model output dimensions, and one-batch smoke tests for every model mode.
