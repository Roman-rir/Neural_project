# Task 2 artifacts

Real training and final evaluation are complete in [available_run1/](available_run1/),
with 3,964 clips, 15 model/seed runs and 20 real graph examples.
See [the measured report](../../report/task2_results.md),
[real comparison](available_run1/runs/comparison_aggregate.csv),
[real graph gallery](available_run1/graph_examples/index.html) and
[verification](available_run1/verification.json).
The earlier `audio_availability_audit.json` is a historical pre-download snapshot.

[synthetic_verification/](synthetic_verification/) contains a completed pipeline
check: 30 generated tone clips, 90 validated graphs, 20 graph visualizations,
five trained model checkpoints, curves, predictions and ablation tables.
Its scores are synthetic diagnostics, not MusicCaps results.

- [Graph gallery](synthetic_verification/graph_examples/index.html)
- [Model comparison](synthetic_verification/runs/comparison.csv)
- [Graph ablations](synthetic_verification/runs/graph_ablation.csv)
- [Prediction/artifact verification](synthetic_verification/verification.json)
- [Test suite result](synthetic_verification/tests_verification.json)
- [Notebook execution result](synthetic_verification/notebook_verification.json)
- [Completion report](../../report/task2_results.md)
- [Real-data run guide](../../docs/task2/README.md)

Checkpoints and tensor caches are local files ignored by Git. Preserve them with
the processed manifest to reproduce evaluation; new checkpoints reject changed
cached datasets. Reproduce generated audio and runs using the documented demo
command rather than treating these artifacts as a real dataset.
