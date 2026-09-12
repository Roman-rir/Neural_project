# Task 3 experiment directories

| Directory | Experiment | Status |
|---|---|---|
| `available_run1` | Five frozen-encoder heads, seeds 42/43/44 | Trained and test-evaluated; main frozen report/notebook use this run. |
| `my_run` | A second five-head, three-seed frozen experiment | Trained and test-evaluated. |
| `joint_run1` | Live encoder fine-tuning | Interrupted during BERT-only training; not a completed fusion suite. |
| `joint_run2` | Live encoder fine-tuning, all five modes, seed 42 | Training and test evaluation complete; all five models passed verification. |

The joint experiment updates the final transformer block, GraphSAGE and the
active fusion/prediction head. It starts from the Task 1/2 encoders and seed-42
frozen heads; the earlier frozen experiments are preserved.

From the project root:

```powershell
.venv/Scripts/python.exe -m src.task3.pipeline check --config configs/task3.yaml
.venv/Scripts/python.exe -m src.task3.pipeline verify --config configs/task3.yaml
```

A completed joint training/evaluation suite has all five mode checkpoints,
`selection.json`, `experiment.json` with `evaluated_test: true`, a comparison
table, and a successful verification. A BERT checkpoint by itself does not
establish suite completion. See the [run guide](../../docs/task3/README.md).

`available_run1/semantic_plots` contains verified genre/mood t-SNE views from
true validation annotations. Mood coverage is limited to Angry music.
