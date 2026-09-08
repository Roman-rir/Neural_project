# Task 2 artifacts

Real-data Task 2 training is not complete. The source alignment and local audio
availability audit is in [audio_availability_audit.json](audio_availability_audit.json):
5,521 metadata records, 30 AudioSet labels, zero matching real audio clips.

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
